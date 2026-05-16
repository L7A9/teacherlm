import asyncio
import json
import re
import unicodedata
from collections import Counter
from collections.abc import AsyncIterator, Awaitable
from pathlib import Path
from typing import TypeVar

from teacherlm_core.llm.language import set_current_language
from teacherlm_core.llm.runtime import set_current_llm_options
from teacherlm_core.schemas.generator_io import (
    GeneratorArtifact,
    GeneratorInput,
    GeneratorOutput,
    LearnerUpdates,
)

from .config import settings
from .schemas import MindMap, MindMapNode
from .services import (
    balancer,
    course_structure,
    hierarchy_builder,
    html_renderer,
    markdown_compiler,
    theme_extractor,
)
from .services.llm_service import get_llm_service

T = TypeVar("T")

_SIZE_CONFIGS = {
    "concise": {"n_branches": 4, "max_nodes_default": 30},
    "standard": {"n_branches": 6, "max_nodes_default": 60},
    "comprehensive": {"n_branches": 9, "max_nodes_default": 100},
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _await_llm_with_progress(
    awaitable: Awaitable[T],
    *,
    stage: str,
    timeout_s: float,
    keepalive_interval_s: float,
    progress: dict | None = None,
) -> AsyncIterator[tuple[str, T | str]]:
    task = asyncio.create_task(awaitable)
    elapsed = 0.0
    interval = max(1.0, keepalive_interval_s)
    try:
        while True:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=min(interval, max(1.0, timeout_s - elapsed)),
                )
                yield "result", result
                return
            except TimeoutError:
                elapsed += interval
                if elapsed >= timeout_s:
                    task.cancel()
                    raise TimeoutError(
                        f"{stage} exceeded {int(timeout_s)}s without a model response"
                    )
                payload = dict(progress or {})
                payload.update(
                    {
                        "stage": f"{stage}_waiting",
                        "elapsed_s": int(elapsed),
                        "timeout_s": int(timeout_s),
                    }
                )
                yield "progress", _sse("progress", payload)
    finally:
        if not task.done():
            task.cancel()


def _resolve_size(size: str) -> dict:
    return _SIZE_CONFIGS.get(size, _SIZE_CONFIGS["standard"])


def _count_nodes(node: MindMapNode) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def _total_nodes(mm: MindMap) -> int:
    return 1 + sum(_count_nodes(b) for b in mm.branches)


def _compute_depth(node: MindMapNode) -> int:
    if not node.children:
        return 1
    return 1 + max(_compute_depth(c) for c in node.children)


def _mindmap_depth(mm: MindMap) -> int:
    if not mm.branches:
        return 1
    return 1 + max(_compute_depth(b) for b in mm.branches)


def _collect_all_node_texts(mm: MindMap) -> list[str]:
    out: list[str] = [mm.central_topic]

    def walk(n: MindMapNode) -> None:
        out.append(n.text)
        for c in n.children:
            walk(c)

    for b in mm.branches:
        walk(b)
    return out


def _combine_for_topic_inference(chunks, max_chars: int = 8_000) -> str:
    parts: list[str] = []
    used = 0
    for ch in chunks:
        block = ch.text.strip()
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _combine_for_outline(chunks, max_chars: int = 36_000) -> str:
    parts: list[str] = []
    used = 0
    for ch in chunks:
        block = f"[{ch.source}] {ch.text.strip()}"
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _make_outline_batches(
    chunks,
    *,
    max_chars: int = 9_000,
    max_batches: int = 6,
) -> list[str]:
    module_batches = _make_module_outline_batches(chunks, max_chars=max_chars)
    if module_batches:
        return module_batches

    batches: list[str] = []
    parts: list[str] = []
    used = 0

    for ch in chunks:
        block = f"[{ch.source}] {ch.text.strip()}"
        if not block.strip():
            continue
        if parts and used + len(block) + 2 > max_chars:
            batches.append("\n\n".join(parts))
            if len(batches) >= max_batches:
                return batches
            parts = []
            used = 0
        if len(block) > max_chars:
            block = block[:max_chars]
        parts.append(block)
        used += len(block) + 2

    if parts and len(batches) < max_batches:
        batches.append("\n\n".join(parts))
    return batches


def _make_module_outline_batches(chunks, *, max_chars: int) -> list[str]:
    outline = next(
        (
            ch.text.strip()
            for ch in chunks
            if ch.metadata.get("context_type") == "mindmap_course_outline" and ch.text.strip()
        ),
        "",
    )
    modules = [
        ch
        for ch in chunks
        if ch.metadata.get("context_type") == "mindmap_module_pack" and ch.text.strip()
    ]
    if not modules:
        return []

    modules.sort(
        key=lambda ch: (
            int(ch.metadata.get("document_order", 0) or 0),
            str(ch.metadata.get("source_filename", ch.source)),
        )
    )
    batches: list[str] = []
    compact_outline = _compact_course_sequence(outline, max_lines=10)
    for module in modules:
        parts = []
        if compact_outline:
            parts.append("GLOBAL COURSE SEQUENCE\n" + compact_outline)
        parts.append("MODULE CONTENT\n" + _compact_module_for_llm(module.text))
        batch = "\n\n".join(parts)
        if len(batch) > max_chars:
            batch = batch[:max_chars].rsplit(" ", 1)[0].strip()
        batches.append(batch)
    return batches


def _compact_course_sequence(text: str, *, max_lines: int) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if re.match(r"^\d+\.\s+", line):
            lines.append(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _compact_module_for_llm(text: str, *, max_headings: int = 14, max_details: int = 10) -> str:
    title = _module_title(text)
    source_match = re.search(r"(?m)^Source file:\s*(.+)$", text)
    role_match = re.search(r"(?m)^Document role:\s*(.+)$", text)
    source = source_match.group(1).strip() if source_match else ""
    role = role_match.group(1).strip() if role_match else ""

    headings = _module_major_headings(text)[:max_headings]
    details = _module_key_details(text)[:max_details]

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if source:
        parts.append(f"Source: {source}")
    if role:
        parts.append(f"Role: {role}")
    if headings:
        parts.append("Major headings:")
        parts.extend(f"- {heading}" for heading in headings)
    if details:
        parts.append("Key details:")
        parts.extend(f"- {detail}" for detail in details)
    if parts:
        return "\n".join(parts)
    return text.strip()


def _module_key_details(text: str) -> list[str]:
    details: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line.startswith("Key details:"):
            continue
        detail = line.removeprefix("Key details:").strip(" -:;")
        if not detail:
            continue
        detail = re.sub(r"\bconcepts:\s*", "", detail, flags=re.IGNORECASE)
        detail = re.sub(r"\bformulas:\s*", "formulas: ", detail, flags=re.IGNORECASE)
        detail = re.sub(r"\bdates/events:\s*", "dates/events: ", detail, flags=re.IGNORECASE)
        detail = detail[:220].strip(" -:;")
        if detail and _norm(detail) not in _GENERIC_BRANCHES:
            details.append(detail)
    return _dedupe_labels(details)


def _outline_to_dict(mm: MindMap) -> dict:
    return mm.model_dump(mode="json")


def _serialize_outlines(outlines: list[MindMap]) -> str:
    payload = {
        "partial_outlines": [
            {"batch": i + 1, **_outline_to_dict(outline)}
            for i, outline in enumerate(outlines)
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_from_module_packs(chunks, *, max_nodes: int) -> MindMap | None:
    modules = [
        ch
        for ch in chunks
        if ch.metadata.get("context_type") == "mindmap_module_pack" and ch.text.strip()
    ]
    if not modules:
        return None
    modules.sort(
        key=lambda ch: (
            int(ch.metadata.get("document_order", 0) or 0),
            str(ch.metadata.get("source_filename", ch.source)),
        )
    )

    main_modules = [
        module
        for module in modules
        if str(module.metadata.get("document_role", "main")).lower() != "supporting"
    ] or modules

    branch_budget = min(7, len(main_modules))
    leaf_budget = max(2, (max_nodes - 1 - branch_budget) // max(1, branch_budget))
    branches: list[MindMapNode] = []
    module_titles: list[str] = []
    for module in main_modules[:branch_budget]:
        title = _best_module_title(module)
        module_titles.append(title)
        children = _module_study_nodes(module.text, title, max_children=leaf_budget)
        branches.append(
            MindMapNode(
                text=_short_label(_compact_module_title(title), max_len=80),
                children=children,
            )
        )

    _merge_supporting_modules(branches, modules, branch_budget=branch_budget)

    central_topic = _infer_central_topic_from_modules(main_modules, module_titles)
    if len(branches) < 3:
        headings = _dedupe_labels(
            heading
            for module in modules
            for heading in _clean_heading_labels(_module_major_headings(module.text), "")
        )
        branches = [MindMapNode(text=_short_label(heading), children=[]) for heading in headings[:10]]
    if len(branches) < 3:
        return None
    return MindMap(central_topic=central_topic, branches=branches)


def _best_module_title(module) -> str:
    metadata_title = str(module.metadata.get("document_title") or "").strip()
    text_title = _module_title(module.text)
    title = metadata_title if metadata_title else text_title
    if _is_generic_title(title):
        discovered = _discover_course_title(module.text)
        if discovered:
            title = discovered
    if _is_generic_title(title):
        title = _distinctive_module_title(module.text) or title
    return title or str(module.source)


def _is_generic_title(title: str) -> bool:
    return _norm(title) in {
        "developpement mobile",
        "plan de la seance",
        "plan de séance",
        "outline",
        "agenda",
        "introduction",
    }


def _discover_course_title(text: str) -> str:
    patterns = [
        r"(?m)^-\s*(Semaine|Week|Lecture|Chapter|Chapitre|Module)\s+\d+\s*[:\-–—]\s*(.+)$",
        r"(?m)^-\s*Plan de la séance:\s*#\s*(.+)$",
        r"(?m)^-\s*Plan de la seance:\s*#\s*(.+)$",
        r"(?m)^-\s*(Semaine|Week|Lecture|Chapter|Chapitre|Module)\s+\d+\s*:\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        title = match.group(match.lastindex or 1).strip()
        return re.sub(r"\s+", " ", title).strip(" -:;#")
    return ""


def _distinctive_module_title(text: str) -> str:
    headings = _module_major_headings(text)
    for heading in headings:
        parts = [
            _clean_label(part)
            for part in re.split(r"\s*>\s*", heading)
            if _clean_label(part)
        ]
        for part in reversed(parts):
            key = _norm(part)
            if not key or key in _GENERIC_BRANCHES or key in _GENERIC_ROOTS:
                continue
            if _is_generic_title(part):
                continue
            if key in {"lst sitd", "developpement mobile sous android", "qcm"}:
                continue
            if _is_noisy_label(part):
                continue
            return part
    return ""


def _module_title(text: str) -> str:
    match = re.search(r"(?m)^Module\s+\d+\s*:\s*(.+)$", text)
    return match.group(1).strip() if match else ""


def _module_major_headings(text: str) -> list[str]:
    in_headings = False
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "Major headings:":
            in_headings = True
            continue
        if in_headings and line == "Study outline details:":
            break
        if not in_headings or not line.startswith("- "):
            continue
        label = re.sub(r"\s+", " ", line[2:]).strip(" -:;")
        if label and _norm(label) not in _GENERIC_BRANCHES:
            headings.append(label)
    return _dedupe_labels(headings)


def _module_study_nodes(text: str, module_title: str, *, max_children: int) -> list[MindMapNode]:
    labels = _clean_heading_labels(_module_major_headings(text), module_title)
    details = _module_key_details(text)
    children: list[MindMapNode] = []
    used_details: set[str] = set()

    for label in labels:
        if len(children) >= max_children:
            break
        child_details = _matching_details(label, details, used_details, limit=2)
        children.append(
            MindMapNode(
                text=_short_label(label),
                children=[MindMapNode(text=_short_label(detail), children=[]) for detail in child_details],
            )
        )

    if len(children) < max_children:
        for detail in details:
            key = _norm(detail)
            if key in used_details or _is_noisy_label(detail):
                continue
            children.append(MindMapNode(text=_short_label(detail), children=[]))
            used_details.add(key)
            if len(children) >= max_children:
                break

    return children


def _clean_heading_labels(headings: list[str], module_title: str) -> list[str]:
    labels: list[str] = []
    module_key = _norm(_compact_module_title(module_title))
    for heading in headings:
        parts = [
            _clean_label(part)
            for part in re.split(r"\s*>\s*", heading)
            if _clean_label(part)
        ]
        for part in parts:
            key = _norm(part)
            if not key or key == module_key or key in _GENERIC_BRANCHES:
                continue
            if _is_noisy_label(part):
                continue
            labels.append(part)
    return _dedupe_labels(labels)


def _clean_label(label: str) -> str:
    label = re.sub(r"[*_`#]+", "", label)
    label = re.sub(r"\s+", " ", label)
    label = re.sub(
        r"^(?:semaine|week|lecture|lesson|chapter|chapitre|module|unit|cours)\s+\d+\s*[:\-–—]?\s*",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\s+\d{1,3}$", "", label)
    label = re.sub(r"^(?:\d+\.\s*)+", "", label)
    return label.strip(" -:;")


def _is_noisy_label(label: str) -> bool:
    key = _norm(label)
    if not key or len(key) < 4:
        return True
    noisy_exact = {
        "people who bought",
        "recommended to user",
        "recommended system",
        "layout attribution critical",
        "similar users",
        "read by her recommended to him",
        "statistiques avancees non pertinent",
        "conclusion",
        "introduction",
        "plan de la seance",
    }
    if key in noisy_exact:
        return True
    return bool(
        re.search(
            r"\b(slide|logo|layout|attribution|page|copyright|questions?|thank you)\b",
            key,
        )
    )


def _matching_details(label: str, details: list[str], used: set[str], *, limit: int) -> list[str]:
    label_tokens = _tokens(label)
    matches: list[str] = []
    for detail in details:
        key = _norm(detail)
        if key in used or _is_noisy_label(detail):
            continue
        detail_tokens = _tokens(detail)
        if label_tokens & detail_tokens:
            matches.append(detail)
            used.add(key)
        if len(matches) >= limit:
            break
    return matches


def _merge_supporting_modules(
    branches: list[MindMapNode],
    modules,
    *,
    branch_budget: int,
) -> None:
    if not branches:
        return
    supporting = [
        module
        for module in modules[branch_budget:]
        if str(module.metadata.get("document_role", "main")).lower() == "supporting"
    ]
    if not supporting:
        return
    for module in supporting:
        labels = _clean_heading_labels(_module_major_headings(module.text), _best_module_title(module))
        target = _best_matching_branch(branches, labels)
        existing = {_norm(child.text) for child in target.children}
        for label in labels:
            key = _norm(label)
            if key in existing or _is_noisy_label(label):
                continue
            target.children.append(MindMapNode(text=_short_label(label), children=[]))
            existing.add(key)
            if len(target.children) >= 9:
                break


def _best_matching_branch(branches: list[MindMapNode], labels: list[str]) -> MindMapNode:
    label_tokens = _tokens(" ".join(labels))
    best = branches[-1]
    best_score = -1
    for branch in branches:
        branch_tokens = _tokens(" ".join(_node_texts(branch)))
        score = len(label_tokens & branch_tokens)
        if score > best_score:
            best = branch
            best_score = score
    return best


def _compact_module_title(title: str) -> str:
    title = re.sub(
        r"^(?:semaine|week|lecture|lesson|chapter|chapitre|module|unit|cours)\s+\d+\s*[:\-–—]?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s+", " ", title).strip(" -:;")
    return title[:80] or "Module"


def _infer_topic_from_module_titles(titles: list[str]) -> str:
    cleaned = [_compact_module_title(title) for title in titles if title.strip()]
    if not cleaned:
        return "Carte du cours"
    tokens_by_title = [_tokens(title) for title in cleaned]
    token_counts: dict[str, int] = {}
    for tokens in tokens_by_title:
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
    repeated = {token for token, count in token_counts.items() if count >= 2}
    if repeated:
        phrase = _phrase_around_repeated_token(cleaned[0], repeated)
        if phrase:
            return _short_label(phrase, max_len=60)
    return _short_label(cleaned[0], max_len=60)


def _infer_central_topic_from_modules(modules, titles: list[str]) -> str:
    metadata_titles = [
        str(module.metadata.get("document_title") or "").strip()
        for module in modules
        if str(module.metadata.get("document_title") or "").strip()
    ]
    counts = Counter(_compact_module_title(title) for title in metadata_titles)
    for title, count in counts.most_common():
        if count >= 2 and _norm(title) not in {
            "plan de la seance",
            "plan de séance",
            "outline",
            "agenda",
            "introduction",
        }:
            return _short_label(title, max_len=60)
    return _infer_topic_from_module_titles(titles)


def _phrase_around_repeated_token(title: str, repeated: set[str]) -> str:
    words = re.findall(r"[\wÀ-ÿ]+", title)
    norm_words = [_norm(word) for word in words]
    preferred = sorted(
        [
            (index, token)
            for index, token in enumerate(norm_words)
            if token in repeated
        ],
        key=lambda item: item[0],
    )
    for index, token in preferred:
        if index + 2 < len(words) and norm_words[index + 1] in {"de", "of"}:
            phrase = " ".join(words[index : index + 3]).strip()
            if len(_tokens(phrase)) >= 2:
                return phrase
        if index >= 2 and norm_words[index - 1] in {"de", "of"}:
            phrase = " ".join(words[index - 2 : index + 1]).strip()
            if len(_tokens(phrase)) >= 2:
                return phrase
        if token not in repeated:
            continue
        start = max(0, index - 2)
        while start < index and norm_words[start] in {
            "de",
            "des",
            "du",
            "la",
            "le",
            "les",
            "and",
            "of",
            "the",
        }:
            start += 1
        end = min(len(words), index + 2)
        phrase = " ".join(words[start:end]).strip()
        if len(_tokens(phrase)) >= 2:
            return phrase
    return ""


def _short_label(label: str, *, max_len: int = 80) -> str:
    label = _clean_label(label)
    if len(label) <= max_len:
        return label
    cut = label[:max_len].rsplit(" ", 1)[0].strip(" -:;")
    return cut or label[:max_len].strip(" -:;")


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        key = _norm(label)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _node_texts(node: MindMapNode) -> list[str]:
    out = [node.text]
    for child in node.children:
        out.extend(_node_texts(child))
    return out


def _tokens(text: str) -> set[str]:
    stop = {
        "les", "des", "dans", "pour", "avec", "une", "un", "du", "de", "la",
        "le", "et", "en", "au", "aux", "sur", "par", "the", "and", "for",
        "with", "from", "into", "this", "that",
    }
    return {
        t for t in re.findall(r"[\wÀ-ÿ]+", _norm(text))
        if len(t) > 2 and t not in stop
    }


def _signature(node: MindMapNode) -> set[str]:
    return _tokens(" ".join(_node_texts(node)))


_GENERIC_ROOTS = {
    "informatique",
    "computer science",
    "machine learning",
    "mathematiques",
    "mathematics",
    "python",
    "blockchain",
    "geography",
    "geographie",
    "history",
    "histoire",
    "science",
    "course",
    "cours",
}


_GENERIC_BRANCHES = {
    "informatique",
    "machine learning",
    "mathematiques",
    "mathematics",
    "python",
    "blockchain",
    "concepts",
    "applications",
    "autres",
    "other",
    "overview",
    "introduction",
    "conclusion",
}


def _topic_from_sources(chunks) -> str | None:
    text = "\n".join(ch.text for ch in chunks[:12])
    headings = re.findall(r"(?m)^\s*#{1,3}\s+(.{4,80})$", text)
    for heading in headings:
        label = re.sub(r"\s+", " ", heading).strip(" -:")
        if _norm(label) not in _GENERIC_ROOTS:
            return label[:60]
    return None


def _refine_mindmap(mm: MindMap, chunks) -> MindMap:
    source_topic = _topic_from_sources(chunks)
    if source_topic and _norm(mm.central_topic) in _GENERIC_ROOTS:
        mm.central_topic = source_topic

    kept: list[MindMapNode] = []
    seen_signatures: list[set[str]] = []
    for branch in mm.branches:
        label = _norm(branch.text)
        sig = _signature(branch)

        if len(mm.branches) > 3 and label in _GENERIC_BRANCHES and mm.central_topic not in branch.text:
            continue

        duplicate = False
        for prior in seen_signatures:
            overlap = len(sig & prior)
            denom = max(1, min(len(sig), len(prior)))
            if overlap / denom >= 0.55:
                duplicate = True
                break
        if duplicate:
            continue

        kept.append(branch)
        seen_signatures.append(sig)

    if len(kept) >= 3:
        mm.branches = kept
    return mm


async def run(payload: GeneratorInput) -> AsyncIterator[str]:
    """SSE generator: yields `progress` events through the pipeline and a
    final `done` event with the full GeneratorOutput payload."""
    set_current_llm_options(payload.options or {})
    set_current_language((payload.options or {}).get("language"))
    size_config = _resolve_size(payload.options.get("size", settings.DEFAULT_SIZE))
    max_nodes = int(
        payload.options.get("max_nodes", size_config["max_nodes_default"])
    )

    yield _sse("progress", {"stage": "starting", "size": size_config})

    # 1. Read the course in batches, extract local structure from each part,
    # then synthesize one coherent study hierarchy. This keeps coverage broad
    # without turning the final map into a literal slide-heading dump.
    mindmap: MindMap | None = None
    llm_available = True
    has_module_packs = any(
        ch.metadata.get("context_type") == "mindmap_module_pack"
        for ch in payload.context_chunks
    )
    llm_refine = str(payload.options.get("llm_refine", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    if has_module_packs and not llm_refine:
        mindmap = _build_from_module_packs(
            payload.context_chunks,
            max_nodes=max_nodes,
        )
        if mindmap is not None:
            yield _sse(
                "progress",
                {
                    "stage": "course_structure_synthesized",
                    "central_topic": mindmap.central_topic,
                    "branches": [b.text for b in mindmap.branches],
                },
            )

    if mindmap is None:
        try:
            module_batch_count = len(
                [
                    ch
                    for ch in payload.context_chunks
                    if ch.metadata.get("context_type") == "mindmap_module_pack"
                ]
            )
            batch_count = module_batch_count or {"concise": 4, "standard": 6, "comprehensive": 8}.get(
                payload.options.get("size", settings.DEFAULT_SIZE),
                6,
            )
            batches = _make_outline_batches(
                payload.context_chunks,
                max_batches=batch_count,
            )
            partial_outlines: list[MindMap] = []
            llm = get_llm_service()
            available, availability_error = await llm.is_available()
            if not available:
                llm_available = False
                yield _sse(
                    "progress",
                    {
                        "stage": "llm_unavailable",
                        "error": availability_error,
                        "fallback": "module_pack",
                    },
                )
                batches = []
            for idx, batch in enumerate(batches, start=1):
                yield _sse(
                    "progress",
                    {
                        "stage": "batch_extracting",
                        "batch": idx,
                        "total_batches": len(batches),
                    },
                )
                try:
                    outline = None
                    async for kind, value in _await_llm_with_progress(
                        llm.build_batch_outline(batch),
                        stage="batch_extracting",
                        timeout_s=settings.LLM_CALL_TIMEOUT_S,
                        keepalive_interval_s=settings.LLM_KEEPALIVE_INTERVAL_S,
                        progress={"batch": idx, "total_batches": len(batches)},
                    ):
                        if kind == "progress":
                            yield value
                        else:
                            outline = value
                    if outline is None:
                        continue
                    partial_outlines.append(
                        MindMap(
                            central_topic=outline.central_topic,
                            branches=outline.branches,
                        )
                    )
                except Exception as batch_exc:
                    yield _sse(
                        "progress",
                        {
                            "stage": "batch_extract_failed",
                            "batch": idx,
                            "error": str(batch_exc),
                        },
                    )
                    if isinstance(batch_exc, TimeoutError):
                        break

            if partial_outlines:
                yield _sse(
                    "progress",
                    {
                        "stage": "course_synthesizing",
                        "partial_outlines": len(partial_outlines),
                    },
                )
                synthesized = None
                async for kind, value in _await_llm_with_progress(
                    llm.synthesize_course_outline(
                        _serialize_outlines(partial_outlines),
                        n_branches=size_config["n_branches"],
                    ),
                    stage="course_synthesizing",
                    timeout_s=settings.LLM_CALL_TIMEOUT_S,
                    keepalive_interval_s=settings.LLM_KEEPALIVE_INTERVAL_S,
                    progress={"partial_outlines": len(partial_outlines)},
                ):
                    if kind == "progress":
                        yield value
                    else:
                        synthesized = value
                if synthesized is None:
                    raise RuntimeError("course synthesis did not return an outline")
                mindmap = MindMap(
                    central_topic=synthesized.central_topic,
                    branches=synthesized.branches,
                )
                yield _sse(
                    "progress",
                    {
                        "stage": "concept_outline_built",
                        "central_topic": mindmap.central_topic,
                        "branches": [b.text for b in mindmap.branches],
                    },
                )
        except Exception as exc:
            yield _sse("progress", {"stage": "concept_outline_fallback", "error": str(exc)})

    # 2. Fallbacks: first try a one-shot whole-course outline, then the parsed
    # Markdown structure, then the older theme-by-theme expansion.
    if mindmap is None:
        try:
            if not llm_available:
                raise RuntimeError("LLM provider unavailable; using non-LLM fallback")
            outline_text = _combine_for_outline(payload.context_chunks)
            outline = None
            async for kind, value in _await_llm_with_progress(
                get_llm_service().build_course_outline(
                    outline_text,
                    n_branches=size_config["n_branches"],
                ),
                stage="outline_building",
                timeout_s=settings.LLM_CALL_TIMEOUT_S,
                keepalive_interval_s=settings.LLM_KEEPALIVE_INTERVAL_S,
            ):
                if kind == "progress":
                    yield value
                else:
                    outline = value
            if outline is None:
                raise RuntimeError("outline builder did not return an outline")
            mindmap = MindMap(
                central_topic=outline.central_topic,
                branches=outline.branches,
            )
            yield _sse(
                "progress",
                {
                    "stage": "outline_built",
                    "central_topic": mindmap.central_topic,
                    "branches": [b.text for b in mindmap.branches],
                },
            )
        except Exception as exc:
            yield _sse("progress", {"stage": "outline_fallback", "error": str(exc)})

            try:
                mindmap = course_structure.build_from_chunks(
                    payload.context_chunks,
                    max_nodes=max_nodes,
                )
                if mindmap is not None:
                    yield _sse(
                        "progress",
                        {
                            "stage": "course_structure_built",
                            "central_topic": mindmap.central_topic,
                            "branches": [b.text for b in mindmap.branches],
                        },
                    )
            except Exception as structure_exc:
                yield _sse(
                    "progress",
                    {
                        "stage": "course_structure_fallback",
                        "error": str(structure_exc),
                    },
                )

            if mindmap is None:
                mindmap = _build_from_module_packs(
                    payload.context_chunks,
                    max_nodes=max_nodes,
                )
                if mindmap is not None:
                    yield _sse(
                        "progress",
                        {
                            "stage": "module_pack_fallback_built",
                            "central_topic": mindmap.central_topic,
                            "branches": [b.text for b in mindmap.branches],
                        },
                    )

            if mindmap is None:
                topic_text = _combine_for_topic_inference(payload.context_chunks)
                central_topic = None
                async for kind, value in _await_llm_with_progress(
                    get_llm_service().infer_central_topic(topic_text),
                    stage="central_topic",
                    timeout_s=settings.LLM_CALL_TIMEOUT_S,
                    keepalive_interval_s=settings.LLM_KEEPALIVE_INTERVAL_S,
                ):
                    if kind == "progress":
                        yield value
                    else:
                        central_topic = value
                if central_topic is None:
                    raise RuntimeError("central topic inference did not return text")
                yield _sse("progress", {"stage": "central_topic", "central_topic": central_topic})

                themes = await theme_extractor.extract(
                    payload.context_chunks, n_branches=size_config["n_branches"]
                )
                yield _sse("progress", {"stage": "themes_extracted", "themes": themes})

                mindmap = await hierarchy_builder.build(
                    themes, payload.context_chunks, size_config, central_topic
                )

    mindmap = _refine_mindmap(mindmap, payload.context_chunks)
    yield _sse(
        "progress",
        {
            "stage": "hierarchy_built",
            "node_count": _total_nodes(mindmap),
            "branches": [b.text for b in mindmap.branches],
        },
    )

    # 4. Balance
    mindmap = balancer.balance(mindmap, max_nodes=max_nodes)
    node_count = _total_nodes(mindmap)
    yield _sse("progress", {"stage": "balanced", "node_count": node_count})

    # 5. Compile to markdown (Markmap input format)
    markdown = markdown_compiler.compile(mindmap)

    # 6. Render artifacts (JSON for inline render + standalone HTML for offline)
    json_path, html_path = await html_renderer.render(
        mindmap, markdown, settings.ARTIFACTS_DIR
    )
    json_filename = Path(json_path).name
    html_filename = Path(html_path).name

    artifacts = [
        GeneratorArtifact(
            type="mindmap",
            url=f"{settings.PUBLIC_BASE_URL}/artifacts/{json_filename}",
            filename=json_filename,
        ),
        GeneratorArtifact(
            type="html",
            url=f"{settings.PUBLIC_BASE_URL}/artifacts/{html_filename}",
            filename=html_filename,
        ),
    ]
    for art in artifacts:
        yield _sse("artifact", art.model_dump())

    # 7. Teacher-voice response
    response_text = (
        f"I've built a mind map of '{mindmap.central_topic}' from your materials. "
        f"It covers {len(mindmap.branches)} main themes and "
        f"{node_count} total concepts. "
        f"Click any node to ask me to explain it in detail."
    )

    output = GeneratorOutput(
        response=response_text,
        generator_id="mindmap_gen",
        output_type="mindmap",
        artifacts=artifacts,
        sources=payload.context_chunks[:10],
        learner_updates=LearnerUpdates(
            concepts_covered=_collect_all_node_texts(mindmap)
        ),
        metadata={
            "markdown": markdown,
            "node_count": node_count,
            "depth": _mindmap_depth(mindmap),
            "central_topic": mindmap.central_topic,
            "main_branches": [b.text for b in mindmap.branches],
        },
    )

    yield _sse("done", output.model_dump(mode="json"))

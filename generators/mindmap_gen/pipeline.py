import asyncio
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import AsyncIterator, Awaitable
from pathlib import Path
from typing import TypeVar

from teacherlm_core.llm.language import language_name, set_current_language
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
    "standard": {"n_branches": 6, "max_nodes_default": 110},
    "comprehensive": {"n_branches": 9, "max_nodes_default": 150},
}
_FRESH_LAYOUT_STYLES = [
    "group the course by learning path from foundations to applications",
    "group the course by major concepts and supporting examples",
    "group the course by student study questions and practical uses",
    "group the course by components, processes, and relationships",
    "group the course by exam-review themes and common confusions",
]


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


def _truthy_option(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _generation_id(options: dict) -> str:
    raw = str(options.get("generation_id") or "").strip()
    if raw:
        return raw
    return hashlib.sha1(json.dumps(options, sort_keys=True, default=str).encode()).hexdigest()


def _fresh_generation_hint(options: dict) -> str:
    generation_id = _generation_id(options)
    index = int(hashlib.sha1(generation_id.encode()).hexdigest(), 16) % len(_FRESH_LAYOUT_STYLES)
    return (
        "Fresh regeneration request.\n"
        f"Generation id: {generation_id}\n"
        f"Preferred organization for this run: {_FRESH_LAYOUT_STYLES[index]}.\n"
        "Create a newly organized mind map from the same grounded course evidence. "
        "Do not copy a previous branch ordering or wording when another faithful "
        "organization is possible. Keep every node supported by the provided content."
    )


def _with_generation_hint(text: str, hint: str) -> str:
    if not hint:
        return text
    return f"{hint}\n\nCOURSE EVIDENCE:\n{text}"


def _use_module_pack_fast_path(
    *,
    has_module_packs: bool,
    llm_refine: bool,
    force_regenerate: bool,
) -> bool:
    return has_module_packs and not llm_refine and not force_regenerate


def _apply_fresh_layout_variation(mm: MindMap, generation_id: str) -> MindMap:
    """Deterministic variation for repeat clicks over the same grounded content."""

    digest = int(hashlib.sha1(generation_id.encode()).hexdigest(), 16)
    sequence = _generation_sequence_index(generation_id)
    layout_seed = digest if sequence is None else digest + sequence * 1_315_423_911
    for index, branch in enumerate(mm.branches):
        _vary_node_wording(branch, layout_seed + index)

    if len(mm.branches) < 2:
        return mm

    if sequence is None:
        offset = digest % len(mm.branches)
        if offset == 0:
            offset = 1
    else:
        offset = ((sequence - 1) % (len(mm.branches) - 1)) + 1
    mm.branches = [*mm.branches[offset:], *mm.branches[:offset]]
    for index, branch in enumerate(mm.branches):
        _vary_child_positions(branch, layout_seed >> (index % 16))
    return mm


def _generation_sequence_index(generation_id: str) -> int | None:
    match = re.match(r"^mindmap:[^:]+:(\d+):", generation_id)
    if not match:
        return None
    return max(1, int(match.group(1)))


def _vary_node_wording(node: MindMapNode, seed: int) -> None:
    node.text = _variant_label(node.text, seed)
    for index, child in enumerate(node.children):
        _vary_node_wording(child, seed + (index + 1) * 17)


def _vary_child_positions(node: MindMapNode, seed: int) -> None:
    if len(node.children) > 1:
        offset = seed % len(node.children)
        if offset:
            node.children = [*node.children[offset:], *node.children[:offset]]
        if len(node.children) > 3 and seed & 1:
            node.children = [node.children[0], *reversed(node.children[1:])]
    for index, child in enumerate(node.children):
        _vary_child_positions(child, seed + index + 3)


def _variant_label(label: str, seed: int) -> str:
    variants = _label_variants(label)
    if not variants:
        return label
    variant = variants[seed % len(variants)]
    if _norm(variant) == _norm(label):
        return label
    return _short_label(variant, max_len=80)


def _label_variants(label: str) -> list[str]:
    text = re.sub(r"\s+", " ", label).strip()
    variants: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\s+", " ", value).strip(" -:;")
        if value and _norm(value) != _norm(text):
            variants.append(value)

    match = re.match(r"(?i)^fondements?\s+(de|du|des|d['’])\s+(.+)$", text)
    if match:
        add(f"Bases {match.group(1)} {match.group(2)}")

    match = re.match(r"(?i)^approches?\s+bas[ée]es\s+sur\s+(.+)$", text)
    if match:
        add(f"Methodes basees sur {match.group(1)}")

    match = re.match(r"(?i)^l['’]ere\s+du\s+(.+?)\s+dans\s+(.+)$", text)
    if match:
        add(f"{match.group(1)} pour {match.group(2)}")

    match = re.match(r"(?i)^qu['’]?est-ce qu['’]?(?:un|une)?\s+(.+?)\??$", text)
    if match:
        add(f"Comprendre {match.group(1)}")

    match = re.match(r"(?i)^pourquoi\s+(.+?)\??$", text)
    if match:
        add(f"Raisons: {match.group(1)}")

    match = re.match(r"(?i)^comment\s+(.+?)\??$", text)
    if match:
        add(f"Mecanisme: {match.group(1)}")

    match = re.match(r"(?i)^introduction\s+(?:au|aux|a la|to)\s+(.+)$", text)
    if match:
        add(f"Premiers reperes sur {match.group(1)}")

    match = re.match(r"(?i)^rappel\s*:\s*(.+)$", text)
    if match:
        add(f"Revoir {match.group(1)}")

    match = re.match(r"(?i)^(?:le|la|les)\s+principe[s]?\s+(?:du|de la|des|de)\s+(.+)$", text)
    if match:
        add(f"Principes: {match.group(1)}")

    match = re.match(r"(?i)^types?\s+(?:de|du|des)\s+(.+)$", text)
    if match:
        add(f"Categories: {match.group(1)}")

    if " : " in text:
        head, tail = [part.strip() for part in text.split(" : ", 1)]
        if head and tail and len(tail) > 3:
            add(f"{tail} ({head})")

    comma_parts = [part.strip() for part in re.split(r",|\s+et\s+", text) if part.strip()]
    if len(comma_parts) >= 3:
        rotated = [*comma_parts[1:], comma_parts[0]]
        add(", ".join(rotated[:-1]) + f" et {rotated[-1]}")

    return _dedupe_labels(variants)


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
    if _is_generic_title(title) or _is_wrapper_module_title(title):
        title = _discover_course_title(module.text) or title
    if _is_generic_title(title) or _is_wrapper_module_title(title):
        title = _distinctive_module_title(module.text) or title
    return title or str(module.source)


def _is_generic_title(title: str) -> bool:
    return _norm(title) in {
        "developpement mobile",
        "plan de la seance",
        "outline",
        "agenda",
        "introduction",
        "conclusion",
        "course",
        "cours",
        "module",
        "lecture",
    }


_COURSE_SEQUENCE_PREFIX_RE = re.compile(
    r"^(?:semaine|week|lecture|lesson|chapter|chapitre|module|unit|cours)"
    r"\s+\d+\s*[:\-\u2013\u2014]?\s*",
    flags=re.IGNORECASE,
)
_WRAPPER_TITLE_KEYS = {
    "organized",
    "v2",
    "v3",
    "v2 organized",
    "v3 organized",
    "clean",
    "cleaned",
    "converted",
    "slides",
    "presentation",
    "source material",
}


def _strip_course_sequence_prefix(title: str) -> str:
    return _COURSE_SEQUENCE_PREFIX_RE.sub("", title).strip()


def _is_wrapper_module_title(title: str) -> bool:
    raw = _norm(title)
    compact = _norm(_strip_course_sequence_prefix(title))
    if not raw:
        return True
    if _is_generic_title(title) or compact in _WRAPPER_TITLE_KEYS:
        return True
    return bool(
        re.fullmatch(
            r"(?:lecture|week|semaine|chapter|chapitre|module|cours)\s*\d+"
            r"(?:\s+(?:v\d+|organized|clean|cleaned|slides|presentation))*",
            raw,
        )
    )


def _discover_course_title(text: str) -> str:
    sequence_pattern = re.compile(
        r"(?:^|>\s*)"
        r"(?:Semaine|Week|Lecture|Chapter|Chapitre|Module)\s+\d+"
        r"\s*[:\-\u2013\u2014]\s*([^>\n]+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    plan_pattern = re.compile(
        r"(?im)^\s*-\s*plan de la s\S*ance\s*:\s*#?\s*([^>\n]+)$"
    )
    for pattern in (sequence_pattern, plan_pattern):
        for match in pattern.finditer(text):
            title = _clean_label(match.group(1))
            if not title or _is_noisy_label(title) or _is_wrapper_module_title(title):
                continue
            return title
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
            if _is_wrapper_module_title(part):
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
    paths = _module_heading_paths(text, module_title)
    details = _module_key_details(text)
    children = _nodes_from_heading_paths(paths, max_children=max_children)
    if children:
        _attach_details_to_tree(children, details)
        return children[:max_children]

    labels = _clean_heading_labels(_module_major_headings(text), module_title)
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


def _module_heading_paths(text: str, module_title: str) -> list[list[str]]:
    module_keys = {
        _norm(module_title),
        _norm(_compact_module_title(module_title)),
    }
    paths: list[list[str]] = []
    for heading in _module_major_headings(text):
        parts: list[str] = []
        for raw_part in re.split(r"\s*>\s*", heading):
            label = _clean_label(raw_part)
            key = _norm(label)
            if not label or key in module_keys or key in _GENERIC_BRANCHES:
                continue
            if _is_noisy_label(label) or _is_wrapper_module_title(label):
                continue
            parts.append(label)
        if parts:
            paths.append(parts[:4])
    return _dedupe_paths(paths)


def _dedupe_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for path in paths:
        key = tuple(_norm(part) for part in path if _norm(part))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _nodes_from_heading_paths(paths: list[list[str]], *, max_children: int) -> list[MindMapNode]:
    roots: list[MindMapNode] = []

    def add_child(children: list[MindMapNode], label: str, *, limit: int) -> MindMapNode | None:
        key = _norm(label)
        for child in children:
            if _norm(child.text) == key:
                return child
        if len(children) >= limit:
            return None
        node = MindMapNode(text=_short_label(label), children=[])
        children.append(node)
        return node

    for path in paths:
        siblings = roots
        for depth, label in enumerate(path):
            node = add_child(
                siblings,
                label,
                limit=max_children if depth == 0 else 7,
            )
            if node is None:
                break
            siblings = node.children
    return roots


def _iter_tree_nodes(nodes: list[MindMapNode]) -> list[MindMapNode]:
    out: list[MindMapNode] = []

    def walk(node: MindMapNode) -> None:
        out.append(node)
        for child in node.children:
            walk(child)

    for node in nodes:
        walk(node)
    return out


def _attach_details_to_tree(nodes: list[MindMapNode], details: list[str]) -> None:
    if not details:
        return
    tree_nodes = _iter_tree_nodes(nodes)
    used: set[str] = set()
    for detail in details:
        key = _norm(detail)
        if not key or key in used or _is_noisy_label(detail):
            continue
        detail_tokens = _tokens(detail)
        best_node: MindMapNode | None = None
        best_score = 0
        for node in tree_nodes:
            if len(node.children) >= 7 or _norm(node.text) == key:
                continue
            score = len(_tokens(node.text) & detail_tokens)
            if score > best_score:
                best_node = node
                best_score = score
        if best_node is None or best_score <= 0:
            continue
        best_node.children.append(MindMapNode(text=_short_label(detail), children=[]))
        used.add(key)


def _clean_heading_labels(headings: list[str], module_title: str) -> list[str]:
    labels: list[str] = []
    module_keys = {
        _norm(module_title),
        _norm(_compact_module_title(module_title)),
    }
    for heading in headings:
        parts = [
            _clean_label(part)
            for part in re.split(r"\s*>\s*", heading)
            if _clean_label(part)
        ]
        for part in parts:
            key = _norm(part)
            if not key or key in module_keys or key in _GENERIC_BRANCHES:
                continue
            if _is_noisy_label(part) or _is_wrapper_module_title(part):
                continue
            labels.append(part)
    return _dedupe_labels(labels)


def _clean_label(label: str) -> str:
    label = re.sub(r"[*_`#]+", "", label)
    label = re.sub(r"\s+", " ", label)
    label = _strip_course_sequence_prefix(label)
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
        "source material",
        "source plan item",
        "table des matieres",
        "conclusion",
        "introduction",
        "plan de la seance",
    }
    if key in noisy_exact:
        return True
    return bool(
        re.search(
            r"\b(slide|logo|layout|attribution|page|copyright|questions?|thank you|navigation|footer)\b",
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
    title = _strip_course_sequence_prefix(title)
    title = re.sub(r"\s+", " ", title).strip(" -:;")
    return title[:80] or "Module"


def _infer_topic_from_module_titles(titles: list[str]) -> str:
    cleaned = [
        _compact_module_title(title)
        for title in titles
        if title.strip() and not _is_wrapper_module_title(title)
    ]
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
    counts = Counter(
        _compact_module_title(title)
        for title in metadata_titles
        if not _is_wrapper_module_title(title)
    )
    for title, count in counts.most_common():
        if count >= 2 and not _is_generic_title(title) and not _is_wrapper_module_title(title):
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
    "overview",
    "outline",
    "introduction",
    "conclusion",
    "module",
    "lecture",
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
    "module",
    "lecture",
    "organized",
    "source material",
}


def _topic_from_sources(chunks) -> str | None:
    text = "\n".join(ch.text for ch in chunks[:12])
    headings = re.findall(r"(?m)^\s*#{1,3}\s+(.{4,80})$", text)
    for heading in headings:
        label = re.sub(r"\s+", " ", heading).strip(" -:")
        if _norm(label) not in _GENERIC_ROOTS:
            return label[:60]
    return None


def _branch_needs_enrichment(branch: MindMapNode) -> bool:
    if any(child.children for child in branch.children):
        return False
    return len(branch.children) < 2


def _rich_branch_count(mm: MindMap) -> int:
    return sum(1 for branch in mm.branches if not _branch_needs_enrichment(branch))


def _enrichment_maps(chunks, *, max_nodes: int) -> list[MindMap]:
    maps: list[MindMap] = []
    for builder in (
        lambda: _build_from_module_packs(chunks, max_nodes=max_nodes),
        lambda: course_structure.build_from_chunks(chunks, max_nodes=max_nodes),
    ):
        try:
            candidate = builder()
        except Exception:
            continue
        if candidate is not None and _total_nodes(candidate) > 1 + len(candidate.branches):
            maps.append(candidate)
    return maps


def _branch_enrichment_score(target: MindMapNode, source: MindMapNode) -> int:
    if not source.children:
        return -1
    if _norm(target.text) == _norm(source.text):
        return 100 + len(source.children)
    target_label_tokens = _tokens(target.text)
    source_label_tokens = _tokens(source.text)
    target_tokens = _signature(target)
    source_tokens = _signature(source)
    return (len(target_label_tokens & source_label_tokens) * 8) + len(
        target_tokens & source_tokens
    )


def _append_distinct_children(
    target: MindMapNode,
    source: MindMapNode,
    *,
    limit: int = 7,
) -> int:
    existing = {_norm(child.text) for child in target.children}
    added = 0
    for child in source.children:
        key = _norm(child.text)
        if not key or key in existing:
            continue
        target.children.append(child.model_copy(deep=True))
        existing.add(key)
        added += 1
        if len(target.children) >= limit:
            break
    return added


def _merge_enrichment_map(target: MindMap, source: MindMap) -> int:
    used_sources: set[int] = set()
    added = 0
    for index, branch in enumerate(target.branches):
        if not _branch_needs_enrichment(branch):
            continue

        best_index = -1
        best_score = 0
        for source_index, source_branch in enumerate(source.branches):
            if source_index in used_sources:
                continue
            score = _branch_enrichment_score(branch, source_branch)
            if score > best_score:
                best_index = source_index
                best_score = score

        if best_index < 0 and index < len(source.branches) and source.branches[index].children:
            best_index = index
        if best_index < 0:
            continue

        source_branch = source.branches[best_index]
        count = _append_distinct_children(branch, source_branch)
        if count:
            used_sources.add(best_index)
            added += count
    return added


def _source_study_labels(chunks, *, max_labels: int = 80) -> list[str]:
    labels: list[str] = []
    for chunk in chunks:
        headings = _clean_heading_labels(_module_major_headings(chunk.text), "")
        labels.extend(headings)
        for raw_line in chunk.text.splitlines():
            line = raw_line.strip()
            match = re.match(r"^\s*#{1,4}\s+(.+)$", line)
            if not match:
                match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", line)
            if not match:
                continue
            label = _short_label(match.group(1))
            if label and not _is_noisy_label(label):
                labels.append(label)
            if len(labels) >= max_labels * 2:
                break
        if len(labels) >= max_labels * 2:
            break

    cleaned = [
        label
        for label in _dedupe_labels(labels)
        if _norm(label) not in _GENERIC_BRANCHES and not _is_noisy_label(label)
    ]
    return cleaned[:max_labels]


def _add_source_label_children(mm: MindMap, chunks, *, max_nodes: int) -> int:
    labels = _source_study_labels(chunks)
    if not labels:
        return 0

    branch_keys = {_norm(branch.text) for branch in mm.branches}
    used: set[str] = set()
    added = 0
    for branch in mm.branches:
        if not _branch_needs_enrichment(branch):
            continue
        branch_tokens = _tokens(branch.text)
        existing = {_norm(child.text) for child in branch.children}
        matches = [
            label
            for label in labels
            if _norm(label) not in branch_keys
            and _norm(label) not in existing
            and _norm(label) not in used
            and (branch_tokens & _tokens(label))
        ]
        for label in matches[: max(0, 3 - len(branch.children))]:
            branch.children.append(MindMapNode(text=_short_label(label), children=[]))
            used.add(_norm(label))
            added += 1
            if _total_nodes(mm) >= max_nodes:
                return added
    return added


def _ensure_rich_mindmap(mm: MindMap, chunks, *, max_nodes: int) -> MindMap:
    if _mindmap_depth(mm) >= 3 and _rich_branch_count(mm) >= max(2, len(mm.branches) // 2):
        return mm

    for candidate in _enrichment_maps(chunks, max_nodes=max_nodes):
        _merge_enrichment_map(mm, candidate)
        if _mindmap_depth(mm) >= 3 and _rich_branch_count(mm) >= max(2, len(mm.branches) // 2):
            return mm

    _add_source_label_children(mm, chunks, max_nodes=max_nodes)
    return mm


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


def _mindmap_response_text(mm: MindMap, node_count: int, language_code: str | None) -> str:
    code = str(language_code or "").strip().casefold()
    simple_templates = {
        "fr-fr": (
            "J'ai construit une carte mentale de '{topic}' a partir de tes supports. "
            "Elle organise les idees importantes en une structure de revision riche."
        ),
        "es": (
            "He creado un mapa mental de '{topic}' a partir de tus materiales. "
            "Organiza las ideas importantes en una estructura de estudio rica."
        ),
        "it": (
            "Ho creato una mappa mentale di '{topic}' dai tuoi materiali. "
            "Organizza le idee importanti in una struttura di studio ricca."
        ),
        "pt-br": (
            "Criei um mapa mental de '{topic}' a partir dos seus materiais. "
            "Ele organiza as ideias importantes em uma estrutura de estudo rica."
        ),
        "de": (
            "Ich habe aus deinen Materialien eine Mindmap zu '{topic}' erstellt. "
            "Sie ordnet die wichtigen Ideen in eine reichhaltige Lernstruktur."
        ),
    }
    simple_template = simple_templates.get(
        code,
        (
            "I've built a mind map of '{topic}' from your materials. "
            "It organizes the important ideas into a rich study structure."
        ),
    )
    return simple_template.format(topic=mm.central_topic)

    templates = {
        "fr-fr": (
            "J'ai construit une carte mentale de '{topic}' a partir de tes supports. "
            "Elle couvre {branches} themes principaux et {nodes} concepts au total. "
            "Clique sur n'importe quel noeud pour me demander de l'expliquer en detail."
        ),
        "es": (
            "He creado un mapa mental de '{topic}' a partir de tus materiales. "
            "Cubre {branches} temas principales y {nodes} conceptos en total. "
            "Haz clic en cualquier nodo para pedirme que lo explique en detalle."
        ),
        "it": (
            "Ho creato una mappa mentale di '{topic}' dai tuoi materiali. "
            "Copre {branches} temi principali e {nodes} concetti in totale. "
            "Fai clic su qualsiasi nodo per chiedermi di spiegarlo in dettaglio."
        ),
        "pt-br": (
            "Criei um mapa mental de '{topic}' a partir dos seus materiais. "
            "Ele cobre {branches} temas principais e {nodes} conceitos no total. "
            "Clique em qualquer no para me pedir uma explicacao detalhada."
        ),
        "de": (
            "Ich habe aus deinen Materialien eine Mindmap zu '{topic}' erstellt. "
            "Sie deckt {branches} Hauptthemen und insgesamt {nodes} Konzepte ab. "
            "Klicke auf einen Knoten, damit ich ihn dir im Detail erklaere."
        ),
        "ja": (
            "'{topic}'について、アップロード資料に基づくマインドマップを作成しました。"
            "主なテーマは{branches}個、概念は合計{nodes}個あります。"
            "詳しく説明してほしいノードをクリックしてください。"
        ),
        "cmn": (
            "我已根据你上传的资料生成关于“{topic}”的思维导图。"
            "它包含 {branches} 个主要主题，共 {nodes} 个概念。"
            "点击任意节点，我会进一步详细解释。"
        ),
        "hi": (
            "मैंने आपकी सामग्री से '{topic}' का माइंड मैप बना दिया है। "
            "इसमें {branches} मुख्य विषय और कुल {nodes} अवधारणाएं शामिल हैं। "
            "किसी भी नोड पर क्लिक करें, मैं उसे विस्तार से समझाऊंगा।"
        ),
    }
    template = templates.get(
        code,
        (
            "I've built a mind map of '{topic}' from your materials. "
            "It covers {branches} main themes and {nodes} total concepts. "
            "Click any node to ask me to explain it in detail."
        ),
    )
    return template.format(
        topic=mm.central_topic,
        branches=len(mm.branches),
        nodes=node_count,
    )


async def run(payload: GeneratorInput) -> AsyncIterator[str]:
    """SSE generator: yields `progress` events through the pipeline and a
    final `done` event with the full GeneratorOutput payload."""
    options = dict(payload.options or {})
    set_current_llm_options(options)
    set_current_language(options.get("language"))
    size_config = _resolve_size(options.get("size", settings.DEFAULT_SIZE))
    max_nodes = int(
        options.get("max_nodes", size_config["max_nodes_default"])
    )
    force_regenerate = _truthy_option(options.get("force_regenerate")) or _truthy_option(
        options.get("regenerate")
    )
    generation_id = _generation_id(options)
    generation_hint = _fresh_generation_hint(options) if force_regenerate else ""

    yield _sse(
        "progress",
        {
            "stage": "starting",
            "size": size_config,
            "force_regenerate": force_regenerate,
            "generation_id": generation_id if force_regenerate else None,
        },
    )

    # 1. Read the course in batches, extract local structure from each part,
    # then synthesize one coherent study hierarchy. This keeps coverage broad
    # without turning the final map into a literal slide-heading dump.
    mindmap: MindMap | None = None
    llm_available = True
    has_module_packs = any(
        ch.metadata.get("context_type") == "mindmap_module_pack"
        for ch in payload.context_chunks
    )
    llm_refine = str(options.get("llm_refine", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    built_via_llm = False
    if _use_module_pack_fast_path(
        has_module_packs=has_module_packs,
        llm_refine=llm_refine,
        force_regenerate=force_regenerate,
    ):
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
                options.get("size", settings.DEFAULT_SIZE),
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
                        llm.build_batch_outline(_with_generation_hint(batch, generation_hint)),
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
                        _with_generation_hint(
                            _serialize_outlines(partial_outlines),
                            generation_hint,
                        ),
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
                built_via_llm = True
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
                    _with_generation_hint(outline_text, generation_hint),
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
            built_via_llm = True
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
                built_via_llm = True

    if force_regenerate and not built_via_llm:
        mindmap = _apply_fresh_layout_variation(mindmap, generation_id)
    mindmap = _refine_mindmap(mindmap, payload.context_chunks)
    pre_enrichment_count = _total_nodes(mindmap)
    mindmap = _ensure_rich_mindmap(
        mindmap,
        payload.context_chunks,
        max_nodes=max_nodes,
    )
    enriched_count = _total_nodes(mindmap)
    if enriched_count > pre_enrichment_count:
        yield _sse(
            "progress",
            {
                "stage": "hierarchy_enriched",
                "node_count": enriched_count,
            },
        )

    target_language_code = str(options.get("language") or "").strip()
    target_language = language_name(target_language_code)
    if target_language:
        yield _sse(
            "progress",
            {
                "stage": "language_adapting",
                "language": target_language,
            },
        )
        try:
            translated_mindmap = None
            async for kind, value in _await_llm_with_progress(
                get_llm_service().translate_mindmap_labels(
                    mindmap,
                    target_language_code,
                ),
                stage="language_adapting",
                timeout_s=settings.LLM_CALL_TIMEOUT_S,
                keepalive_interval_s=settings.LLM_KEEPALIVE_INTERVAL_S,
                progress={"language": target_language},
            ):
                if kind == "progress":
                    yield value
                else:
                    translated_mindmap = value
            if translated_mindmap is not None:
                mindmap = translated_mindmap
        except Exception as language_exc:
            yield _sse(
                "progress",
                {
                    "stage": "language_adapt_fallback",
                    "language": target_language,
                    "error": str(language_exc),
                },
            )

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
    response_text = _mindmap_response_text(
        mindmap,
        node_count,
        target_language_code,
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
            "language": target_language_code or None,
        },
    )

    yield _sse("done", output.model_dump(mode="json"))

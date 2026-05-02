import json
import re
import unicodedata
from collections.abc import AsyncIterator
from pathlib import Path

from teacherlm_core.llm.language import set_current_language
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

_SIZE_CONFIGS = {
    "concise": {"n_branches": 4, "max_nodes_default": 30},
    "standard": {"n_branches": 6, "max_nodes_default": 60},
    "comprehensive": {"n_branches": 9, "max_nodes_default": 100},
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
    try:
        batch_count = {"concise": 4, "standard": 6, "comprehensive": 8}.get(
            payload.options.get("size", settings.DEFAULT_SIZE),
            6,
        )
        batches = _make_outline_batches(
            payload.context_chunks,
            max_batches=batch_count,
        )
        partial_outlines: list[MindMap] = []
        llm = get_llm_service()
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
                outline = await llm.build_batch_outline(batch)
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

        if partial_outlines:
            yield _sse(
                "progress",
                {
                    "stage": "course_synthesizing",
                    "partial_outlines": len(partial_outlines),
                },
            )
            synthesized = await llm.synthesize_course_outline(
                _serialize_outlines(partial_outlines),
                n_branches=size_config["n_branches"],
            )
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
            outline_text = _combine_for_outline(payload.context_chunks)
            outline = await get_llm_service().build_course_outline(
                outline_text,
                n_branches=size_config["n_branches"],
            )
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
                topic_text = _combine_for_topic_inference(payload.context_chunks)
                central_topic = await get_llm_service().infer_central_topic(topic_text)
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

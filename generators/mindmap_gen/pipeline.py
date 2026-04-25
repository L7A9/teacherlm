import json
from collections.abc import AsyncIterator
from pathlib import Path

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


async def run(payload: GeneratorInput) -> AsyncIterator[str]:
    """SSE generator: yields `progress` events through the pipeline and a
    final `done` event with the full GeneratorOutput payload."""
    size_config = _resolve_size(payload.options.get("size", settings.DEFAULT_SIZE))
    max_nodes = int(
        payload.options.get("max_nodes", size_config["max_nodes_default"])
    )

    yield _sse("progress", {"stage": "starting", "size": size_config})

    # 1. Central topic
    topic_text = _combine_for_topic_inference(payload.context_chunks)
    central_topic = await get_llm_service().infer_central_topic(topic_text)
    yield _sse("progress", {"stage": "central_topic", "central_topic": central_topic})

    # 2. Themes (main branches)
    themes = await theme_extractor.extract(
        payload.context_chunks, n_branches=size_config["n_branches"]
    )
    yield _sse("progress", {"stage": "themes_extracted", "themes": themes})

    # 3. Build hierarchy
    mindmap = await hierarchy_builder.build(
        themes, payload.context_chunks, size_config, central_topic
    )
    yield _sse(
        "progress",
        {"stage": "hierarchy_built", "node_count": _total_nodes(mindmap)},
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
        f"I've built a mind map of '{central_topic}' from your materials. "
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
            "central_topic": central_topic,
            "main_branches": [b.text for b in mindmap.branches],
        },
    )

    yield _sse("done", output.model_dump(mode="json"))

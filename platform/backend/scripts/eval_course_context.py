from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
for path in (BACKEND_DIR, REPO_ROOT / "packages" / "teacherlm_core"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from services.course_context_service import get_course_context_service

    service = get_course_context_service()
    outputs = [item.strip() for item in args.output_types.split(",") if item.strip()]
    report: dict[str, Any] = {"conversation_id": args.conversation_id, "contexts": []}
    for output_type in outputs:
        started = time.perf_counter()
        chunks = await service.get_generator_context(
            conversation_id=args.conversation_id,
            output_type=output_type,
            query=args.query or args.topic or "",
            topic=args.topic,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        report["contexts"].append(
            {
                "output_type": output_type,
                "topic": args.topic,
                "chunk_count": len(chunks),
                "section_count": len({chunk.metadata.get("section_id") for chunk in chunks if chunk.metadata.get("section_id")}),
                "context_types": sorted({str(chunk.metadata.get("context_type", "chunk")) for chunk in chunks}),
                "latency_ms": round(latency_ms, 2),
                "previews": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "section_id": chunk.metadata.get("section_id"),
                        "preview": " ".join(chunk.text.split())[: args.preview_chars],
                    }
                    for chunk in chunks[: args.preview_count]
                ],
            }
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect course-context policies per output type.")
    parser.add_argument("conversation_id")
    parser.add_argument("--output-types", default="text,quiz,mindmap,presentation,podcast,chart")
    parser.add_argument("--query", default="")
    parser.add_argument("--topic")
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=260)
    parser.add_argument("--out", type=Path)
    return parser


async def _amain() -> None:
    args = _parser().parse_args()
    report = await _run(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    asyncio.run(_amain())

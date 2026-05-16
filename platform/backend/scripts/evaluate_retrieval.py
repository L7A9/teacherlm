from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
for path in (BACKEND_DIR, REPO_ROOT / "packages" / "teacherlm_core"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from teacherlm_core.retrieval.evaluation import (  # noqa: E402
    DEFAULT_K_VALUES,
    RetrievalCase,
    evaluate_case,
    result_to_dict,
    summarize_results,
)


def _parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("k values must be positive integers")
    return values


def _load_eval_file(path: Path) -> tuple[str, list[RetrievalCase]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    conversation_id = str(data.get("conversation_id") or "").strip()
    if not conversation_id:
        raise ValueError("eval file must contain conversation_id")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("eval file must contain a non-empty cases array")
    return conversation_id, [
        RetrievalCase.from_dict(raw, idx) for idx, raw in enumerate(raw_cases)
    ]


async def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    from services.retrieval_orchestrator import RetrievalOrchestrator

    conversation_id, cases = _load_eval_file(args.eval_file)
    orchestrator = RetrievalOrchestrator()
    results = []

    for case in cases:
        started = time.perf_counter()
        if case.mode:
            chunks = await orchestrator.retrieve(
                mode=case.mode,
                query=case.query,
                conversation_id=conversation_id,
            )
        else:
            started = time.perf_counter()
            chunks = await orchestrator.retrieve_for(
                output_type=case.output_type or args.output_type,
                query=case.query,
                conversation_id=conversation_id,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        result = evaluate_case(
            case,
            retrieved_ids=[chunk.chunk_id for chunk in chunks],
            retrieved_sources=[chunk.source for chunk in chunks],
            retrieved_section_ids=[
                str(chunk.metadata.get("section_id", "")) for chunk in chunks
            ],
            k_values=args.k_values,
            latency_ms=latency_ms,
        )
        results.append(result)

    return {
        "conversation_id": conversation_id,
        "k_values": list(args.k_values),
        "summary": summarize_results(results, k_values=args.k_values),
        "cases": [result_to_dict(result) for result in results],
    }


async def _dump_corpus(args: argparse.Namespace) -> dict[str, Any]:
    from db.session import session_scope
    from services.course_content_store import get_course_content_store

    async with session_scope() as session:
        chunks = await get_course_content_store().get_chunks(session, uuid.UUID(args.conversation_id))
    return {
        "conversation_id": args.conversation_id,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "metadata": chunk.metadata,
                "preview": " ".join(chunk.text.split())[: args.preview_chars],
            }
            for chunk in chunks
        ],
    }


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TeacherLM retrieval against a gold chunk dataset."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("run", help="run retrieval metrics")
    eval_parser.add_argument("eval_file", type=Path)
    eval_parser.add_argument("--output-type", default="text")
    eval_parser.add_argument("--k-values", type=_parse_k_values, default=DEFAULT_K_VALUES)
    eval_parser.add_argument("--out", type=Path)

    corpus_parser = sub.add_parser(
        "dump-corpus",
        help="dump chunk ids and previews to help build a gold eval file",
    )
    corpus_parser.add_argument("conversation_id")
    corpus_parser.add_argument("--preview-chars", type=int, default=260)
    corpus_parser.add_argument("--out", type=Path)
    return parser


async def _amain() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        report = await _run_eval(args)
    elif args.command == "dump-corpus":
        report = await _dump_corpus(args)
    else:
        raise ValueError(f"unknown command {args.command!r}")
    _write_report(report, args.out)


if __name__ == "__main__":
    asyncio.run(_amain())

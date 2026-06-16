from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
for path in (BACKEND_DIR, REPO_ROOT / "packages" / "teacherlm_core"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from teacherlm_core.retrieval.bm25 import BM25Index  # noqa: E402
from teacherlm_core.retrieval.evaluation import (  # noqa: E402
    RetrievalCase,
    evaluate_case,
    result_to_dict,
    summarize_results,
)
from teacherlm_core.retrieval.hybrid_retriever import RRF_K  # noqa: E402
from teacherlm_core.schemas.chunk import Chunk  # noqa: E402


DEFAULT_METRICS = ("recall@5", "mrr@5", "ndcg@5")
VARIANT_LABELS = {
    "semantic_only": "Semantic only",
    "bm25_only": "BM25 only",
    "hybrid_rrf": "Hybrid RRF",
}


def _parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part) for part in raw.split(",") if part.strip()}))
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("k values must be positive integers")
    return values


def _parse_metrics(raw: str) -> tuple[str, ...]:
    metrics = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not metrics:
        raise argparse.ArgumentTypeError("at least one metric is required")
    return metrics


def _parse_source_file_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or []


def _load_eval_file(path: Path) -> tuple[uuid.UUID, list[RetrievalCase]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    conversation_id = str(data.get("conversation_id") or "").strip()
    if not conversation_id:
        raise ValueError("eval file must contain conversation_id")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("eval file must contain a non-empty cases array")
    return uuid.UUID(conversation_id), [
        RetrievalCase.from_dict(raw, index) for index, raw in enumerate(raw_cases)
    ]


async def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    from db.session import session_scope
    from services.course_content_store import get_course_content_store
    from services.vector_service import get_vector_service

    conversation_id, cases = _load_eval_file(args.eval_file)
    source_file_ids = _parse_source_file_ids(args.source_file_ids)
    top_k = max(args.k_values)
    candidate_k = max(args.candidate_k, top_k)

    async with session_scope() as session:
        corpus = await get_course_content_store().get_chunks(
            session,
            conversation_id,
            source_file_ids=source_file_ids,
        )

    if not corpus:
        raise RuntimeError(
            "no searchable chunks found for this conversation; upload and index files first"
        )

    bm25 = BM25Index(corpus)
    vector_service = get_vector_service()

    grouped_results: dict[str, list[Any]] = {
        "semantic_only": [],
        "bm25_only": [],
        "hybrid_rrf": [],
    }

    for case in cases:
        dense_hits, dense_latency = await _timed(
            _semantic_only(
                vector_service,
                conversation_id,
                case.query,
                limit=candidate_k,
                source_file_ids=source_file_ids,
            )
        )
        sparse_hits, sparse_latency = await _timed(
            _bm25_only(bm25, case.query, limit=candidate_k)
        )
        hybrid_hits = _rrf_fuse([dense_hits, sparse_hits], top_k=top_k)

        grouped_results["semantic_only"].append(
            evaluate_case(
                case,
                retrieved_ids=[chunk.chunk_id for chunk in dense_hits[:top_k]],
                retrieved_sources=[chunk.source for chunk in dense_hits[:top_k]],
                retrieved_section_ids=_section_ids(dense_hits[:top_k]),
                k_values=args.k_values,
                latency_ms=dense_latency,
            )
        )
        grouped_results["bm25_only"].append(
            evaluate_case(
                case,
                retrieved_ids=[chunk.chunk_id for chunk in sparse_hits[:top_k]],
                retrieved_sources=[chunk.source for chunk in sparse_hits[:top_k]],
                retrieved_section_ids=_section_ids(sparse_hits[:top_k]),
                k_values=args.k_values,
                latency_ms=sparse_latency,
            )
        )
        grouped_results["hybrid_rrf"].append(
            evaluate_case(
                case,
                retrieved_ids=[chunk.chunk_id for chunk in hybrid_hits],
                retrieved_sources=[chunk.source for chunk in hybrid_hits],
                retrieved_section_ids=_section_ids(hybrid_hits),
                k_values=args.k_values,
                latency_ms=dense_latency + sparse_latency,
            )
        )

    summaries = {
        variant: summarize_results(results, k_values=args.k_values)
        for variant, results in grouped_results.items()
    }
    comparison_rows = _comparison_rows(summaries, args.metrics)
    report = {
        "conversation_id": str(conversation_id),
        "eval_file": str(args.eval_file),
        "source_file_ids": source_file_ids,
        "case_count": len(cases),
        "corpus_chunk_count": len(corpus),
        "k_values": list(args.k_values),
        "candidate_k": candidate_k,
        "variants": {
            "semantic_only": "Dense vector search in Qdrant using the configured embedding model.",
            "bm25_only": "Lexical BM25 search over the same PostgreSQL chunk corpus.",
            "hybrid_rrf": f"Dense candidates and BM25 candidates fused with RRF using k={RRF_K}.",
        },
        "summary": summaries,
        "comparison": comparison_rows,
        "mermaid": _mermaid_xychart(comparison_rows, args.metrics, args.chart_title),
        "cases": {
            variant: [result_to_dict(result) for result in results]
            for variant, results in grouped_results.items()
        },
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        _write_csv(args.csv_out, comparison_rows, args.metrics)
    if args.mermaid_out:
        args.mermaid_out.parent.mkdir(parents=True, exist_ok=True)
        args.mermaid_out.write_text(report["mermaid"] + "\n", encoding="utf-8")

    return report


async def _semantic_only(
    vector_service: Any,
    conversation_id: uuid.UUID,
    query: str,
    *,
    limit: int,
    source_file_ids: list[str] | None,
) -> list[Chunk]:
    hits = await vector_service.search(
        conversation_id,
        query,
        top_k=limit,
        file_ids=source_file_ids,
    )
    return [
        Chunk(
            text=hit.text,
            source=hit.source,
            score=hit.score,
            chunk_id=hit.chunk_id,
            metadata=hit.metadata,
        )
        for hit in hits
    ]


async def _bm25_only(bm25: BM25Index, query: str, *, limit: int) -> list[Chunk]:
    return bm25.query(query, top_k=limit)


async def _timed(awaitable: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = await awaitable
    return result, (time.perf_counter() - started) * 1000.0


def _rrf_fuse(rankings: list[list[Chunk]], *, top_k: int) -> list[Chunk]:
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            fused_scores[chunk.chunk_id] = (
                fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            )
            chunk_by_id.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        Chunk(
            text=chunk_by_id[chunk_id].text,
            source=chunk_by_id[chunk_id].source,
            score=score,
            chunk_id=chunk_id,
            metadata=chunk_by_id[chunk_id].metadata,
        )
        for chunk_id, score in ordered
    ]


def _section_ids(chunks: Iterable[Chunk]) -> list[str]:
    return [str(chunk.metadata.get("section_id", "")) for chunk in chunks]


def _comparison_rows(
    summaries: dict[str, dict[str, Any]],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for variant, summary in summaries.items():
        values = summary.get("metrics", {})
        row: dict[str, Any] = {
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
        }
        for metric in metrics:
            row[metric] = round(float(values.get(metric, 0.0)), 4)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "label", *metrics])
        writer.writeheader()
        writer.writerows(rows)


def _mermaid_xychart(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    title: str,
) -> str:
    labels = ", ".join(f'"{row["label"]}"' for row in rows)
    metric_lines = []
    for metric in metrics:
        values = ", ".join(f'{float(row.get(metric, 0.0)):.3f}' for row in rows)
        metric_lines.append(f'    bar "{metric}" [{values}]')

    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            f'    title "{title}"',
            f"    x-axis [{labels}]",
            '    y-axis "Score" 0 --> 1',
            *metric_lines,
            "```",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare semantic-only, BM25-only, and hybrid RRF retrieval on a "
            "TeacherLM gold retrieval eval file."
        )
    )
    parser.add_argument("eval_file", type=Path)
    parser.add_argument("--k-values", type=_parse_k_values, default=(5,))
    parser.add_argument("--metrics", type=_parse_metrics, default=DEFAULT_METRICS)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument(
        "--source-file-ids",
        help="Optional comma-separated source_file_id filter, matching frontend source selection.",
    )
    parser.add_argument(
        "--chart-title",
        default="TeacherLM RAG Retrieval Comparison",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--mermaid-out", type=Path)
    return parser


async def _amain() -> None:
    args = _build_parser().parse_args()
    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be positive")
    report = await _run_eval(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_amain())

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10, 20)


@dataclass(slots=True)
class RetrievalCase:
    id: str
    query: str
    relevant_chunk_ids: set[str]
    relevant_section_ids: set[str] = field(default_factory=set)
    expected_source_document: str | None = None
    answer_facts: list[str] = field(default_factory=list)
    mode: str | None = None
    output_type: str | None = None
    relevant_source_contains: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int) -> "RetrievalCase":
        query = str(raw.get("query") or "").strip()
        if not query:
            raise ValueError(f"eval case #{index + 1} is missing query")

        relevant_ids = {
            str(item).strip()
            for item in raw.get("relevant_chunk_ids", raw.get("gold_chunk_ids", []))
            if str(item).strip()
        }
        relevant_sections = {
            str(item).strip()
            for item in raw.get("expected_section_ids", raw.get("relevant_section_ids", []))
            if str(item).strip()
        }
        source_contains = [
            str(item).strip()
            for item in raw.get("relevant_source_contains", [])
            if str(item).strip()
        ]
        expected_source = raw.get("expected_source_document")
        answer_facts = [
            str(item).strip()
            for item in raw.get("answer_facts", [])
            if str(item).strip()
        ]
        if not relevant_ids and not relevant_sections and not source_contains and not expected_source:
            raise ValueError(
                f"eval case #{index + 1} needs relevant_chunk_ids, expected_section_ids, "
                "expected_source_document, or relevant_source_contains"
            )

        return cls(
            id=str(raw.get("id") or f"case_{index + 1}"),
            query=query,
            relevant_chunk_ids=relevant_ids,
            relevant_section_ids=relevant_sections,
            expected_source_document=str(expected_source).strip() if expected_source else None,
            answer_facts=answer_facts,
            mode=raw.get("mode"),
            output_type=raw.get("output_type"),
            relevant_source_contains=source_contains,
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(slots=True)
class CaseResult:
    case: RetrievalCase
    retrieved_ids: list[str]
    retrieved_sources: list[str]
    retrieved_section_ids: list[str]
    matched_relevant: set[str]
    metrics: dict[str, float]


def evaluate_case(
    case: RetrievalCase,
    *,
    retrieved_ids: list[str],
    retrieved_sources: list[str] | None = None,
    retrieved_section_ids: list[str] | None = None,
    cited_ids: list[str] | None = None,
    latency_ms: float | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> CaseResult:
    retrieved_sources = retrieved_sources or []
    retrieved_section_ids = retrieved_section_ids or []
    cited_ids = cited_ids or []
    relevant = _resolve_relevant(case, retrieved_ids, retrieved_sources)
    metrics: dict[str, float] = {}

    for k in k_values:
        top_ids = retrieved_ids[:k]
        hits = [chunk_id for chunk_id in top_ids if chunk_id in relevant]
        top_sections = retrieved_section_ids[:k]
        section_hits = [
            section_id for section_id in top_sections if section_id in case.relevant_section_ids
        ]
        metrics[f"hit_rate@{k}"] = 1.0 if hits else 0.0
        metrics[f"precision@{k}"] = len(hits) / k if k else 0.0
        metrics[f"recall@{k}"] = len(set(hits)) / len(relevant) if relevant else 0.0
        metrics[f"mrr@{k}"] = _mrr_at(top_ids, relevant)
        metrics[f"ndcg@{k}"] = _ndcg_at(top_ids, relevant)
        metrics[f"section_recall@{k}"] = (
            len(set(section_hits)) / len(case.relevant_section_ids)
            if case.relevant_section_ids
            else 0.0
        )

    metrics["citation_precision"] = _citation_precision(cited_ids, relevant)
    if case.expected_source_document:
        metrics["source_document_hit"] = (
            1.0 if any(case.expected_source_document in source for source in retrieved_sources) else 0.0
        )
    if latency_ms is not None:
        metrics["latency_ms"] = float(latency_ms)

    return CaseResult(
        case=case,
        retrieved_ids=retrieved_ids,
        retrieved_sources=retrieved_sources,
        retrieved_section_ids=retrieved_section_ids,
        matched_relevant=set(retrieved_ids) & relevant,
        metrics=metrics,
    )


def summarize_results(
    results: list[CaseResult],
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    if not results:
        return {"case_count": 0, "metrics": {}}

    metric_names = [
        f"{name}@{k}"
        for k in k_values
        for name in ("hit_rate", "precision", "recall", "mrr", "ndcg", "section_recall")
    ]
    metric_names.extend(["citation_precision", "source_document_hit", "latency_ms"])
    means = {
        name: sum(result.metrics.get(name, 0.0) for result in results) / len(results)
        for name in metric_names
        if any(name in result.metrics for result in results)
    }
    return {
        "case_count": len(results),
        "metrics": means,
        "failed_cases": [
            {
                "id": result.case.id,
                "query": result.case.query,
                "expected": sorted(result.case.relevant_chunk_ids),
                "expected_sections": sorted(result.case.relevant_section_ids),
                "retrieved": result.retrieved_ids[: max(k_values)],
            }
            for result in results
            if result.metrics.get(f"hit_rate@{max(k_values)}", 0.0) == 0.0
        ],
    }


def result_to_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "id": result.case.id,
        "query": result.case.query,
        "mode": result.case.mode,
        "output_type": result.case.output_type,
        "relevant_chunk_ids": sorted(result.case.relevant_chunk_ids),
        "relevant_section_ids": sorted(result.case.relevant_section_ids),
        "expected_source_document": result.case.expected_source_document,
        "answer_facts": result.case.answer_facts,
        "retrieved_ids": result.retrieved_ids,
        "retrieved_sources": result.retrieved_sources,
        "retrieved_section_ids": result.retrieved_section_ids,
        "matched_relevant": sorted(result.matched_relevant),
        "metrics": result.metrics,
    }


def _resolve_relevant(
    case: RetrievalCase,
    retrieved_ids: list[str],
    retrieved_sources: list[str],
) -> set[str]:
    relevant = set(case.relevant_chunk_ids)
    if not case.relevant_source_contains:
        source_terms = [case.expected_source_document] if case.expected_source_document else []
    else:
        source_terms = case.relevant_source_contains
    if not source_terms:
        return relevant

    for chunk_id, source in zip(retrieved_ids, retrieved_sources, strict=False):
        if any(needle and needle in source for needle in source_terms):
            relevant.add(chunk_id)
    return relevant


def _mrr_at(retrieved_ids: list[str], relevant: set[str]) -> float:
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / idx
    return 0.0


def _ndcg_at(retrieved_ids: list[str], relevant: set[str]) -> float:
    dcg = 0.0
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), len(retrieved_ids))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _citation_precision(cited_ids: list[str], relevant: set[str]) -> float:
    if not cited_ids:
        return 0.0
    if not relevant:
        return 0.0
    return len([chunk_id for chunk_id in cited_ids if chunk_id in relevant]) / len(cited_ids)

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from teacherlm_core.llm.providers import LLMMessage, LLMProviderError, complete_text
from teacherlm_core.retrieval import BM25Index, build_hyde_prompt, rrf_fuse, should_skip_hyde, tokenize
from teacherlm_core.retrieval.reranker import CrossEncoderReranker
from teacherlm_core.schemas.chunk import Chunk

from local_api.db import get_store
from local_api.services.knowledge_graph import get_knowledge_graph_service
from local_api.services.settings import get_settings_service
from local_api.services.vector_service import get_vector_service


_OUTPUT_TO_MODE = {
    "text": "semantic_topk",
    "chat": "semantic_topk",
    "quiz": "coverage_broad",
    "podcast": "narrative_arc",
    "mindmap": "topic_clusters",
    "presentation": "topic_clusters",
    "report": "topic_clusters",
    "chart": "relationship_dense",
    "diagram": "relationship_dense",
}

_BROAD_NO_TOPIC_OUTPUTS = {"quiz", "mindmap", "presentation", "podcast"}
_TOPIC_OUTPUTS_WITH_SECTION_CONTEXT = {"quiz", "podcast", "presentation"}


class RetrievalService:
    def __init__(self) -> None:
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_model = ""

    async def retrieve_for(
        self,
        *,
        conversation_id: str,
        user_message: str,
        output_type: str = "text",
        source_file_ids: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        settings = get_settings_service().effective_retrieval_settings()
        options = options or {}
        rows = get_store().list_chunks(conversation_id, source_file_ids=source_file_ids or None)
        all_chunks = [_row_to_chunk(row) for row in rows]
        if output_type == "mindmap":
            full_context = [
                _with_metadata(
                    chunk,
                    {
                        "retrieval_via": "mindmap_full_selected_files",
                        "retrieval_mode": "full_document",
                        "mindmap_full_context": True,
                    },
                )
                for chunk in all_chunks
            ]
            if not full_context:
                return []
            structural_context = _mindmap_course_context(
                all_chunks,
                conversation_id=conversation_id,
                source_file_ids=source_file_ids or None,
            )
            graph_context = get_knowledge_graph_service().mindmap_context_chunks(
                conversation_id,
                source_file_ids=source_file_ids or None,
            )
            return _dedupe_chunks([*full_context, *structural_context, *graph_context])

        chunks = [chunk for chunk in all_chunks if not _is_low_information_text(chunk.text)]
        if not chunks:
            chunks = [_row_to_chunk(row) for row in rows]
        if not chunks:
            return []

        mode = _OUTPUT_TO_MODE.get(output_type, "semantic_topk")
        query = str(options.get("topic") or user_message or "").strip()
        top_k = int(options.get("top_k") or settings.retrieval_top_k)
        candidate_k = max(
            top_k,
            settings.retrieval_dense_candidate_k,
            settings.retrieval_sparse_candidate_k,
            settings.retrieval_top_k,
            50,
        )

        if output_type in _BROAD_NO_TOPIC_OUTPUTS and not str(options.get("topic") or "").strip():
            return _generator_context(
                chunks,
                output_type=output_type,
                top_k=top_k,
                conversation_id=conversation_id,
                source_file_ids=source_file_ids or None,
            )

        if output_type in {"text", "chat"} and _is_course_overview_query(query):
            overview = _dedupe_chunks(
                [
                    *_mindmap_course_context(chunks),
                    *_full_course_outline(chunks),
                    *_representative_chunks(chunks, top_k=max(top_k, 12), mode="coverage_broad"),
                ]
            )
            if overview:
                return _expand_context(
                    overview[:top_k],
                    chunks,
                    mode="coverage_broad",
                    conversation_id=conversation_id,
                    max_chars=settings.index_status.get("retrieval_expansion_max_chars", 4500)
                    if isinstance(settings.index_status, dict)
                    else 4500,
                )

        if mode in {"coverage_broad", "narrative_arc", "topic_clusters"} and not query:
            return _generator_context(
                chunks,
                output_type=output_type,
                top_k=top_k,
                conversation_id=conversation_id,
                source_file_ids=source_file_ids or None,
            )

        fused = await self._retrieve_candidates(
            conversation_id=conversation_id,
            query=query,
            chunks=chunks,
            mode=mode,
            top_k=top_k,
            candidate_k=candidate_k,
            source_file_ids=source_file_ids or None,
            options=options,
        )
        if not fused:
            fused = _generator_context(
                chunks,
                output_type=output_type,
                top_k=top_k,
                conversation_id=conversation_id,
                source_file_ids=source_file_ids or None,
            )

        fused = await self._maybe_rerank(query, fused, top_k=max(top_k, settings.retrieval_top_k))

        if output_type in _TOPIC_OUTPUTS_WITH_SECTION_CONTEXT and query:
            focused = _dedupe_chunks([*_section_summaries_for_hits(fused, chunks), *fused])
            if focused:
                fused = focused
        return _expand_context(
            fused[:top_k],
            chunks,
            mode=mode,
            conversation_id=conversation_id,
            max_chars=4500,
        )

    async def _retrieve_candidates(
        self,
        *,
        conversation_id: str,
        query: str,
        chunks: list[Chunk],
        mode: str,
        top_k: int,
        candidate_k: int,
        source_file_ids: list[str] | None,
        options: dict[str, Any],
    ) -> list[Chunk]:
        settings = get_settings_service().effective_retrieval_settings()
        rankings: list[list[Chunk]] = []
        if query:
            bm25_hits = BM25Index(chunks).query(query, top_k=max(candidate_k, settings.retrieval_sparse_candidate_k))
            dense_hits = await get_vector_service().search(
                conversation_id,
                query,
                top_k=max(candidate_k, settings.retrieval_dense_candidate_k),
                source_file_ids=source_file_ids,
            )
            rankings.extend([bm25_hits, dense_hits])
            if settings.retrieval_graph_enabled:
                rankings.append(
                    get_knowledge_graph_service().graph_relevant_chunks(
                        conversation_id,
                        query,
                        limit=max(settings.retrieval_top_k, top_k, 12),
                        source_file_ids=source_file_ids,
                    )
                )
            comparison = await self._comparison_candidates(
                conversation_id,
                query,
                chunks,
                source_file_ids=source_file_ids,
                candidate_k=candidate_k,
            )
            if comparison:
                rankings.append(comparison)

        hyde_text = await self._maybe_hyde(conversation_id, query, options)
        if hyde_text:
            hyde_bm25 = [
                _with_metadata(chunk, {"retrieval_via": "hyde", "hyde_used": True})
                for chunk in BM25Index(chunks).query(hyde_text, top_k=candidate_k)
            ]
            hyde_dense = [
                _with_metadata(chunk, {"retrieval_via": "hyde_dense", "hyde_used": True})
                for chunk in await get_vector_service().search(
                    conversation_id,
                    hyde_text,
                    top_k=candidate_k,
                    source_file_ids=source_file_ids,
                )
            ]
            rankings.extend([hyde_bm25, hyde_dense])

        if mode == "coverage_broad":
            rankings.append(await _coverage_broad(conversation_id, query, chunks, top_k=candidate_k, source_file_ids=source_file_ids))
        elif mode == "narrative_arc":
            rankings.append(await _narrative_arc(conversation_id, query, chunks, top_k=candidate_k, source_file_ids=source_file_ids))
        elif mode == "topic_clusters":
            rankings.append(await _topic_clusters(conversation_id, query, chunks, top_k=candidate_k, source_file_ids=source_file_ids))
        elif mode == "relationship_dense":
            rankings.append(await _relationship_dense(conversation_id, query, chunks, top_k=candidate_k, source_file_ids=source_file_ids))

        fused = rrf_fuse([ranking for ranking in rankings if ranking], top_k=candidate_k)
        if query:
            fused = _merge_formula_hits(query, fused, chunks, target=candidate_k)
        return fused

    async def _comparison_candidates(
        self,
        conversation_id: str,
        query: str,
        chunks: list[Chunk],
        *,
        source_file_ids: list[str] | None,
        candidate_k: int,
    ) -> list[Chunk]:
        terms = _comparison_terms(query)
        if len(terms) < 2:
            return []
        term_hits: dict[str, list[Chunk]] = {}
        per_term_k = max(8, min(24, candidate_k // len(terms) + 4))
        for term in terms:
            bm25_hits = BM25Index(chunks).query(term.query, top_k=per_term_k)
            dense_hits = await get_vector_service().search(
                conversation_id,
                term.query,
                top_k=per_term_k,
                source_file_ids=source_file_ids,
            )
            term_hits[term.label] = [
                _with_metadata(hit, {"matched_query_term": term.label})
                for hit in rrf_fuse([bm25_hits, dense_hits], top_k=per_term_k)
            ]
        full_query_hits = rrf_fuse(
            [
                BM25Index(chunks).query(query, top_k=candidate_k),
                await get_vector_service().search(conversation_id, query, top_k=candidate_k, source_file_ids=source_file_ids),
            ],
            top_k=candidate_k,
        )
        return _balanced_term_merge(term_hits, full_query_hits, target=candidate_k)

    async def _maybe_rerank(self, query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
        settings = get_settings_service().effective_retrieval_settings()
        if not settings.retrieval_rerank_enabled or not query.strip() or not chunks:
            return chunks[:top_k]
        try:
            if self._reranker is None or self._reranker_model != settings.retrieval_reranker_model:
                self._reranker = CrossEncoderReranker(settings.retrieval_reranker_model)
                self._reranker_model = settings.retrieval_reranker_model
            labels = _comparison_labels(chunks)
            if len(labels) >= 2:
                return await self._rerank_comparison_groups(query, chunks, labels, top_k=top_k)
            return await self._reranker.rerank(query, chunks, top_k=top_k)
        except Exception:  # noqa: BLE001
            return chunks[:top_k]

    async def _rerank_comparison_groups(
        self,
        query: str,
        chunks: list[Chunk],
        labels: list[str],
        *,
        top_k: int,
    ) -> list[Chunk]:
        if self._reranker is None:
            return chunks[:top_k]
        per_group_top_k = max(2, top_k // len(labels) + 2)
        ranked_groups: dict[str, list[Chunk]] = {}
        for label in labels:
            group = [chunk for chunk in chunks if str(chunk.metadata.get("matched_query_term", "")) == label]
            ranked_groups[label] = await self._reranker.rerank(query, group, top_k=per_group_top_k)
        selected: list[Chunk] = []
        seen: set[str] = set()
        max_depth = max((len(group) for group in ranked_groups.values()), default=0)
        for index in range(max_depth):
            for label in labels:
                group = ranked_groups[label]
                if index >= len(group):
                    continue
                chunk = group[index]
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                selected.append(chunk)
                if len(selected) >= top_k:
                    return selected
        if len(selected) < top_k:
            selected.extend(chunk for chunk in chunks if chunk.chunk_id not in seen and not chunk.metadata.get("matched_query_term"))
        return selected[:top_k]

    async def _maybe_hyde(
        self,
        conversation_id: str,
        query: str,
        options: dict[str, Any],
    ) -> str | None:
        settings = get_settings_service().effective_retrieval_settings()
        if not query or not options.get("hyde_enabled", settings.retrieval_hyde_enabled):
            return None
        if should_skip_hyde(query):
            get_store().log_hyde_trace(
                {
                    "conversation_id": conversation_id,
                    "query": query,
                    "status": "skipped_formula_or_empty",
                    "metadata": {"reason": "formula_exact_match"},
                }
            )
            return None
        explicit_hyde = str(options.get("hyde_text") or "").strip()
        if explicit_hyde:
            hyde_text = explicit_hyde[:900]
            status = "provided"
            provider_id = None
            metadata_error = ""
        else:
            provider = get_settings_service().get_default_chat_provider_config()
            provider_id = provider.provider_id if provider else None
            try:
                if provider is None:
                    raise LLMProviderError("no default provider")
                hyde_text = await complete_text(
                    provider,
                    [
                        LLMMessage(role="system", content="You write retrieval-only hypothetical course excerpts."),
                        LLMMessage(role="user", content=build_hyde_prompt(query, 900)),
                    ],
                    temperature=0.1,
                )
                status = "ok"
                metadata_error = ""
            except Exception as exc:  # noqa: BLE001
                hyde_text = _deterministic_hyde(query, 900)
                status = "fallback"
                provider_id = provider_id or "deterministic"
                metadata_error = str(exc)

        hyde_text = hyde_text.strip()[:900]
        if not hyde_text:
            return None
        digest = hashlib.sha256(hyde_text.encode("utf-8")).hexdigest()
        get_store().log_hyde_trace(
            {
                "conversation_id": conversation_id,
                "query": query,
                "provider_id": provider_id,
                "hyde_preview": hyde_text[:240],
                "hyde_hash": digest,
                "status": status,
                "metadata": {"visible_to_student": False, "error": metadata_error if status == "fallback" else ""},
            }
        )
        return hyde_text


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        text=row["text"],
        source=row["source_filename"],
        score=0.0,
        chunk_id=row["id"],
        metadata=row.get("metadata", {}),
    )


def _generator_context(
    chunks: list[Chunk],
    *,
    output_type: str,
    top_k: int,
    conversation_id: str | None = None,
    source_file_ids: list[str] | None = None,
) -> list[Chunk]:
    if output_type == "podcast":
        return _narrative_arc_sync("", chunks, top_k=max(top_k, 10))
    if output_type == "mindmap":
        context = [
            *_mindmap_course_context(chunks, conversation_id=conversation_id, source_file_ids=source_file_ids),
            *_topic_clusters_sync("", chunks, top_k=max(top_k, 12)),
        ]
        return _dedupe_chunks(context)[: max(top_k, 12)]
    if output_type in {"quiz", "presentation"}:
        return _coverage_broad_sync("", chunks, top_k=max(top_k, 12))
    return _representative_chunks(chunks, top_k=top_k, mode="coverage_broad")


async def _coverage_broad(
    conversation_id: str,
    query: str,
    chunks: list[Chunk],
    *,
    top_k: int,
    source_file_ids: list[str] | None,
    diversity_lambda: float = 0.7,
) -> list[Chunk]:
    pool = await _rank_pool(conversation_id, query, chunks, top_k=max(top_k * 3, 30), source_file_ids=source_file_ids)
    if not pool:
        pool = _representative_chunks(chunks, top_k=max(top_k * 3, 30), mode="coverage_broad")
    return _coverage_from_pool(query, pool, top_k=top_k, diversity_lambda=diversity_lambda)


def _coverage_broad_sync(query: str, chunks: list[Chunk], *, top_k: int, diversity_lambda: float = 0.7) -> list[Chunk]:
    pool = _representative_chunks(chunks, top_k=max(top_k * 3, 30), mode="coverage_broad")
    return _coverage_from_pool(query, pool, top_k=top_k, diversity_lambda=diversity_lambda)


def _coverage_from_pool(query: str, pool: list[Chunk], *, top_k: int, diversity_lambda: float) -> list[Chunk]:
    query_tokens = set(tokenize(query))
    candidates: list[tuple[Chunk, set[str]]] = [(chunk, set(tokenize(_searchable_text(chunk)))) for chunk in pool]
    selected: list[tuple[Chunk, set[str]]] = []
    while candidates and len(selected) < top_k:
        best_idx = 0
        best_score = -1e9
        for index, (_chunk, tokens) in enumerate(candidates):
            relevance = _jaccard(query_tokens, tokens) if query_tokens else 0.5
            diversity = max((_jaccard(tokens, chosen_tokens) for _, chosen_tokens in selected), default=0.0)
            score = diversity_lambda * relevance - (1 - diversity_lambda) * diversity
            if score > best_score:
                best_score = score
                best_idx = index
        selected.append(candidates.pop(best_idx))
    return [
        _with_metadata(chunk, {"retrieval_via": "coverage_mmr", "retrieval_mode": "coverage_broad"})
        for chunk, _tokens in selected
    ]


async def _narrative_arc(
    conversation_id: str,
    query: str,
    chunks: list[Chunk],
    *,
    top_k: int,
    source_file_ids: list[str] | None,
) -> list[Chunk]:
    middle = await _rank_pool(conversation_id, query, chunks, top_k=max(6, top_k), source_file_ids=source_file_ids) if query else _representative_chunks(chunks, top_k=max(6, top_k), mode="narrative_arc")
    return _narrative_arc_from_middle(chunks, middle, top_k=top_k)


def _narrative_arc_sync(query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
    middle = _representative_chunks(chunks, top_k=max(6, top_k), mode="narrative_arc")
    return _narrative_arc_from_middle(chunks, middle, top_k=top_k)


def _narrative_arc_from_middle(chunks: list[Chunk], middle: list[Chunk], *, top_k: int) -> list[Chunk]:
    selected = _dedupe_chunks([*_pick_intro(chunks), *middle, *_pick_conclusion(chunks)])
    return [
        _with_metadata(chunk, {"retrieval_via": "narrative_arc", "retrieval_mode": "narrative_arc"})
        for chunk in selected[:top_k]
    ]


async def _topic_clusters(
    conversation_id: str,
    query: str,
    chunks: list[Chunk],
    *,
    top_k: int,
    source_file_ids: list[str] | None,
) -> list[Chunk]:
    pool = await _rank_pool(conversation_id, query, chunks, top_k=max(top_k * 3, 30), source_file_ids=source_file_ids) if query else chunks
    return _topic_clusters_from_pool(pool, top_k=top_k)


def _topic_clusters_sync(query: str, chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
    return _topic_clusters_from_pool(chunks, top_k=top_k)


def _topic_clusters_from_pool(pool: list[Chunk], *, top_k: int) -> list[Chunk]:
    buckets: dict[str, list[Chunk]] = {}
    for chunk in pool:
        buckets.setdefault(_cluster_key(chunk), []).append(chunk)
    reps: list[Chunk] = []
    for key, group in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        best = max(group, key=lambda chunk: (chunk.score, len(_searchable_text(chunk))))
        reps.append(_with_metadata(best, {"retrieval_via": "topic_cluster", "retrieval_mode": "topic_clusters", "cluster_key": key}))
        if len(reps) >= top_k:
            break
    return reps


async def _relationship_dense(
    conversation_id: str,
    query: str,
    chunks: list[Chunk],
    *,
    top_k: int,
    source_file_ids: list[str] | None,
) -> list[Chunk]:
    pool = await _rank_pool(conversation_id, query, chunks, top_k=max(top_k * 3, 30), source_file_ids=source_file_ids) if query else chunks
    scored: list[tuple[float, Chunk]] = []
    for chunk in pool:
        text = _searchable_text(chunk)
        tokens = tokenize(text)
        if not tokens:
            continue
        entity_hits = len(re.findall(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)+\b", text))
        verb_hits = len(re.findall(r"\b\w+(?:ed|ing|es)\b", text, flags=re.IGNORECASE))
        formula_hits = int(chunk.metadata.get("equation_count") or 0)
        table_hits = int(chunk.metadata.get("table_count") or 0)
        density = (entity_hits + verb_hits + formula_hits + table_hits) / len(tokens)
        scored.append((density, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        _with_metadata(
            chunk.model_copy(update={"score": float(score)}),
            {"retrieval_via": "relationship_dense", "retrieval_mode": "relationship_dense"},
        )
        for score, chunk in scored[:top_k]
    ]


async def _rank_pool(
    conversation_id: str,
    query: str,
    chunks: list[Chunk],
    *,
    top_k: int,
    source_file_ids: list[str] | None,
) -> list[Chunk]:
    if not query.strip():
        return _representative_chunks(chunks, top_k=top_k, mode="representative")
    settings = get_settings_service().effective_retrieval_settings()
    rankings = [
        BM25Index(chunks).query(query, top_k=top_k),
        await get_vector_service().search(conversation_id, query, top_k=top_k, source_file_ids=source_file_ids),
    ]
    if settings.retrieval_graph_enabled:
        rankings.append(
            get_knowledge_graph_service().graph_relevant_chunks(
                conversation_id,
                query,
                limit=max(6, top_k // 2),
                source_file_ids=source_file_ids,
            )
        )
    return rrf_fuse([ranking for ranking in rankings if ranking], top_k=top_k)


def _representative_chunks(chunks: list[Chunk], *, top_k: int, mode: str) -> list[Chunk]:
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk)
    selected: list[Chunk] = []
    while len(selected) < top_k and any(by_source.values()):
        for source in sorted(by_source):
            if not by_source[source]:
                continue
            chunk = by_source[source].pop(0)
            selected.append(_with_metadata(chunk, {"retrieval_via": "representative", "retrieval_mode": mode}))
            if len(selected) >= top_k:
                break
    return selected


def _expand_context(
    selected: list[Chunk],
    all_chunks: list[Chunk],
    *,
    mode: str,
    conversation_id: str,
    max_chars: int,
) -> list[Chunk]:
    settings = get_settings_service().effective_retrieval_settings()
    if not settings.retrieval_graph_enabled and mode == "coverage_broad":
        return selected
    if mode == "coverage_broad":
        return selected
    by_id = {chunk.chunk_id: chunk for chunk in all_chunks}
    expanded: list[Chunk] = []
    for chunk in selected:
        neighbor_texts: list[str] = []
        metadata = chunk.metadata or {}
        for key in ("prev_chunk_id", "next_chunk_id"):
            neighbor = by_id.get(str(metadata.get(key) or ""))
            if neighbor and neighbor.source == chunk.source:
                neighbor_texts.append(neighbor.text[:500])
        if settings.retrieval_graph_enabled:
            related_ids = get_knowledge_graph_service().graph_related_chunk_ids(conversation_id, [chunk.chunk_id], limit=4)
            for related_id in related_ids:
                neighbor = by_id.get(related_id)
                if neighbor:
                    neighbor_texts.append(neighbor.text[:500])
        parts: list[str] = []
        heading_path = metadata.get("heading_path")
        if heading_path:
            if isinstance(heading_path, list):
                parts.append("Section path: " + " > ".join(str(item) for item in heading_path))
            else:
                parts.append("Section path: " + str(heading_path))
        summary = str(metadata.get("section_summary", "") or "").strip()
        if summary:
            parts.append("Section summary: " + summary)
        parts.extend(_dedupe_text(neighbor_texts))
        parts.append("Focused chunk:\n" + chunk.text)
        expanded.append(
            Chunk(
                text="\n\n".join(part for part in parts if part).strip()[:max_chars],
                source=chunk.source,
                score=chunk.score,
                chunk_id=chunk.chunk_id,
                metadata={**metadata, "retrieval_expanded": True, "retrieval_mode": mode},
            )
        )
    return expanded


def _section_summaries_for_hits(hits: list[Chunk], all_chunks: list[Chunk]) -> list[Chunk]:
    wanted = {str(chunk.metadata.get("section_id", "")).strip() for chunk in hits if chunk.metadata.get("section_id")}
    if not wanted:
        return []
    out: list[Chunk] = []
    seen: set[str] = set()
    for chunk in all_chunks:
        section_id = str(chunk.metadata.get("section_id", "")).strip()
        if not section_id or section_id not in wanted or section_id in seen:
            continue
        seen.add(section_id)
        heading = chunk.metadata.get("heading_path") or chunk.metadata.get("section_title") or chunk.source
        text = f"Section path: {heading}\n\nSection summary: {chunk.metadata.get('section_summary') or chunk.text[:700]}"
        out.append(
            Chunk(
                text=text[:1500],
                source=chunk.source,
                score=1.0,
                chunk_id=f"topic-section:{section_id}",
                metadata={**chunk.metadata, "context_type": "topic_section", "retrieval_via": "section_summary"},
            )
        )
    return out


def _mindmap_course_context(
    chunks: list[Chunk],
    *,
    conversation_id: str | None = None,
    source_file_ids: list[str] | None = None,
) -> list[Chunk]:
    db_context = _mindmap_course_context_from_sections(conversation_id, source_file_ids=source_file_ids)
    if db_context:
        return db_context

    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        source_id = str(chunk.metadata.get("source_file_id") or chunk.source)
        groups.setdefault(source_id, []).append(chunk)
    out: list[Chunk] = []
    for order, group in enumerate(groups.values()):
        first = group[0]
        headings = _dedupe_text(
            str(chunk.metadata.get("section_title") or chunk.metadata.get("heading_path") or "").strip()
            for chunk in group
        )
        concepts = _dedupe_text(
            str(concept)
            for chunk in group
            for concept in (chunk.metadata.get("key_concepts") or [])
        )
        text = "\n".join(
            [
                f"Module {order + 1}: {first.source}",
                f"Source file: {first.source}",
                "Major headings:",
                *[f"- {heading}" for heading in headings[:14] if heading],
                "Key details:",
                *[f"- concepts: {concept}" for concept in concepts[:12] if concept],
            ]
        )
        out.append(
            Chunk(
                text=text,
                source=first.source,
                score=1.0,
                chunk_id=f"mindmap-module:{first.metadata.get('source_file_id') or first.source}",
                metadata={
                    "context_type": "mindmap_module_pack",
                    "source_filename": first.source,
                    "source_file_id": first.metadata.get("source_file_id"),
                    "document_order": order,
                    "key_concepts": concepts[:12],
                },
            )
        )
    return out


def _mindmap_course_context_from_sections(
    conversation_id: str | None,
    *,
    source_file_ids: list[str] | None = None,
) -> list[Chunk]:
    if not conversation_id:
        return []

    params: list[Any] = [conversation_id]
    source_filter = ""
    if source_file_ids:
        placeholders = ",".join("?" for _ in source_file_ids)
        source_filter = f" AND d.source_file_id IN ({placeholders})"
        params.extend(source_file_ids)

    documents = get_store().query(
        f"""
        SELECT
          d.id,
          d.title,
          d.source_file_id,
          COALESCE(f.filename, d.title) AS source_filename,
          COALESCE(f.created_at, d.created_at) AS sort_key
        FROM course_documents d
        LEFT JOIN uploaded_files f ON f.id = d.source_file_id
        WHERE d.conversation_id = ?{source_filter}
        ORDER BY sort_key ASC, source_filename ASC
        """,
        params,
    )
    if not documents:
        return []

    section_params: list[Any] = [conversation_id]
    section_filter = ""
    if source_file_ids:
        placeholders = ",".join("?" for _ in source_file_ids)
        section_filter = f" AND source_file_id IN ({placeholders})"
        section_params.extend(source_file_ids)
    sections = get_store().query(
        f"""
        SELECT *
        FROM course_sections
        WHERE conversation_id = ?{section_filter}
        ORDER BY source_file_id ASC, order_index ASC
        """,
        section_params,
    )

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        by_file[str(section.get("source_file_id") or "")].append(section)

    out: list[Chunk] = []
    for order, document in enumerate(documents):
        source_file_id = str(document.get("source_file_id") or "")
        source_filename = str(document.get("source_filename") or document.get("title") or "Course file")
        title = _clean_mindmap_label(str(document.get("title") or source_filename)) or source_filename
        document_sections = by_file.get(source_file_id, [])
        selected = _select_mindmap_section_rows(document_sections, limit=_per_document_section_budget(len(document_sections)))
        headings = [
            heading
            for section in _select_mindmap_section_rows(document_sections, limit=18)
            if (heading := _section_heading_label(section))
        ]

        lines = [
            f"Module {order + 1}: {title}",
            f"Source file: {source_filename}",
            f"Document role: {_document_role(source_filename, title)}",
            "",
            "Major headings:",
            *[f"- {heading}" for heading in _dedupe_text(headings)],
            "",
            "Study outline details:",
        ]
        for section in selected:
            heading = _section_heading_label(section)
            if not heading:
                continue
            summary = _section_summary(section)
            facts = _section_facts(summary or str(section.get("text") or ""))
            detail = summary
            if facts:
                detail = f"{detail}\n  Key details: {facts}" if detail else f"Key details: {facts}"
            if detail:
                lines.append(f"- {heading}: {detail}".strip())

        text = "\n".join(line for line in lines if line is not None).strip()
        if len(text) > 6500:
            text = text[:6500].rsplit(" ", 1)[0].strip()
        out.append(
            Chunk(
                text=text,
                source=source_filename,
                score=1.0,
                chunk_id=f"mindmap-module:{source_file_id or document.get('id')}",
                metadata={
                    "context_type": "mindmap_module_pack",
                    "document_id": str(document.get("id") or ""),
                    "source_filename": source_filename,
                    "source_file_id": source_file_id,
                    "document_title": title,
                    "document_order": order,
                    "document_role": _document_role(source_filename, title),
                    "section_count": len(document_sections),
                    "selected_section_count": len(selected),
                },
            )
        )

    if out:
        outline_lines = ["Course document sequence:"]
        for index, module in enumerate(out, start=1):
            outline_lines.append(f"{index}. {module.metadata.get('document_title') or module.source} ({module.source}) [{module.metadata.get('document_role', 'main')}]")
            for heading in _module_pack_headings(module.text)[:8]:
                outline_lines.append(f"   - {heading}")
        out.insert(
            0,
            Chunk(
                text="\n".join(outline_lines),
                source="course_outline",
                score=1.0,
                chunk_id=f"mindmap-outline:{conversation_id}",
                metadata={"context_type": "mindmap_course_outline", "document_count": len(documents)},
            ),
        )
    return out


def _select_mindmap_section_rows(sections: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    clean = [section for section in sections if not _is_noisy_mindmap_section(section)]
    if len(clean) <= limit:
        return clean

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for section in clean:
        path = _section_heading_path(section)
        if len(path) <= 2:
            _append_unique_mindmap_section(selected, seen, section)
        if len(selected) >= max(3, limit // 2):
            break

    stride = max(1, len(clean) // max(1, limit - len(selected)))
    for section in clean[::stride]:
        _append_unique_mindmap_section(selected, seen, section)
        if len(selected) >= limit:
            break

    for section in clean:
        _append_unique_mindmap_section(selected, seen, section)
        if len(selected) >= limit:
            break
    return selected


def _append_unique_mindmap_section(selected: list[dict[str, Any]], seen: set[str], section: dict[str, Any]) -> None:
    section_id = str(section.get("id") or "")
    if section_id in seen:
        return
    seen.add(section_id)
    selected.append(section)


def _per_document_section_budget(section_count: int) -> int:
    if section_count <= 10:
        return max(4, section_count)
    if section_count <= 30:
        return 10
    return 14


def _section_heading_path(section: dict[str, Any]) -> list[str]:
    raw = section.get("heading_path_json")
    try:
        values = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        values = []
    return [_clean_mindmap_label(str(value)) for value in values if _clean_mindmap_label(str(value))]


def _section_heading_label(section: dict[str, Any]) -> str:
    path = _section_heading_path(section)
    if not path:
        return ""
    return " > ".join(path[:4])


def _section_summary(section: dict[str, Any]) -> str:
    text = str(section.get("summary") or section.get("text") or "")
    text = _clean_mindmap_detail(text)
    if len(text) > 520:
        text = text[:520].rsplit(" ", 1)[0].strip(" -:;")
    return text


def _section_facts(text: str) -> str:
    facts: list[str] = []
    for raw in text.splitlines():
        line = _clean_mindmap_label(raw)
        if not line or _is_noisy_mindmap_label(line):
            continue
        if re.match(r"^(?:[-*]|\d+[.)]|▶)\s*", raw.strip()) or ":" in line:
            facts.append(line)
        if len(facts) >= 4:
            break
    return "; ".join(_dedupe_text(facts))


def _is_noisy_mindmap_section(section: dict[str, Any]) -> bool:
    heading = _section_heading_label(section)
    summary = _section_summary(section)
    if not heading:
        return True
    leaf = heading.split(">")[-1].strip()
    if _is_noisy_mindmap_label(leaf):
        return True
    if _is_cover_or_bibliography(heading, summary):
        return True
    useful_tokens = _mindmap_tokens(f"{heading} {summary}")
    return len(useful_tokens) < 2


def _is_cover_or_bibliography(heading: str, summary: str) -> bool:
    key = _norm_mindmap_text(f"{heading} {summary}")
    if re.search(r"\b(reference|references|bibliographie|lectures recommandees|thank you|merci)\b", key):
        return True
    cover_hits = [
        "pr ",
        "professeur",
        "universite",
        "ecole normale",
        "master",
        "novembre",
        "moulay",
    ]
    return sum(1 for token in cover_hits if token in key) >= 3 and len(_mindmap_tokens(heading)) <= 4


def _document_role(filename: str, title: str) -> str:
    text = _norm_mindmap_text(f"{filename} {title}")
    if re.search(r"\b(appendix|annexe|references|bibliographie|guide|rubric|syllabus)\b", text):
        return "supporting"
    return "main"


def _module_pack_headings(text: str) -> list[str]:
    headings: list[str] = []
    in_headings = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "Major headings:":
            in_headings = True
            continue
        if in_headings and line == "Study outline details:":
            break
        if in_headings and line.startswith("- "):
            label = _clean_mindmap_label(line[2:])
            if label:
                headings.append(label)
    return _dedupe_text(headings)


def _full_course_outline(chunks: list[Chunk]) -> list[Chunk]:
    modules = _mindmap_course_context(chunks)
    if not modules:
        return []
    lines = ["Course: " + (modules[0].source if modules else "Uploaded materials")]
    for index, module in enumerate(modules, start=1):
        lines.append(f"{index}. {module.source}")
    return [
        Chunk(
            text="\n".join(lines),
            source="course_outline",
            score=1.0,
            chunk_id="course_outline",
            metadata={"context_type": "mindmap_course_outline"},
        )
    ]


def _pick_intro(chunks: list[Chunk], n: int = 2) -> list[Chunk]:
    hits = [
        chunk
        for chunk in chunks
        if any(token in _searchable_text(chunk).casefold() for token in ("introduction", "overview", "abstract", "definition"))
    ]
    return (hits or chunks[: max(1, len(chunks) // 10)])[:n]


def _pick_conclusion(chunks: list[Chunk], n: int = 2) -> list[Chunk]:
    hits = [
        chunk
        for chunk in chunks
        if any(token in _searchable_text(chunk).casefold() for token in ("conclusion", "summary", "in summary", "takeaway"))
    ]
    return (hits[-n:] if hits else chunks[-max(1, len(chunks) // 10) :])[-n:]


def _cluster_key(chunk: Chunk) -> str:
    metadata = chunk.metadata
    for key in ("heading_path", "section_title"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            return str(value[0]).casefold()
        if value:
            return str(value).split(">")[0].strip().casefold()
    concepts = metadata.get("key_concepts") or []
    if concepts:
        return str(concepts[0]).casefold()
    return chunk.source.casefold()


def _searchable_text(chunk: Chunk) -> str:
    metadata = chunk.metadata
    parts = [chunk.text, str(metadata.get("section_title", ""))]
    for key in ("heading_path", "key_concepts", "generated_questions"):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return "\n".join(parts)


def _merge_formula_hits(query: str, hits: list[Chunk], all_chunks: list[Chunk], *, target: int) -> list[Chunk]:
    if not _FORMULA_QUERY_RE.search(query):
        return hits
    formula_hits = _formula_chunks(query, all_chunks, limit=max(4, target // 3))
    return _dedupe_chunks([*formula_hits, *hits])[:target]


def _formula_chunks(query: str, chunks: list[Chunk], *, limit: int) -> list[Chunk]:
    query_tokens = {
        token.casefold()
        for token in _TERM_WORD_RE.findall(query)
        if len(token) > 2 and token.casefold() not in _COMPARISON_STOPWORDS
    }
    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        text = chunk.text or ""
        if not _MATH_TEXT_RE.search(text):
            continue
        haystack = f"{text} {chunk.metadata.get('heading_path', '')}".casefold()
        overlap = sum(1 for token in query_tokens if token in haystack)
        math_density = len(_MATH_TEXT_RE.findall(text))
        score = float(overlap * 4 + min(math_density, 6))
        if overlap == 0 and query_tokens:
            score *= 0.35
        if score <= 0:
            continue
        scored.append((score, _with_metadata(chunk.model_copy(update={"score": max(chunk.score, score)}), {"retrieval_via": "formula_boost"})))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _score, chunk in scored[:limit]]


class _QueryTerm:
    def __init__(self, label: str, query: str) -> None:
        self.label = label
        self.query = query


def _comparison_terms(query: str) -> list[_QueryTerm]:
    if not _COMPARISON_MARKERS_RE.search(query):
        return []
    labels = _extract_compared_labels(query)
    if len(labels) < 2:
        labels = sorted({match.group(0) for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,9}\b", query)})
    if len(labels) < 2:
        return []
    out: list[_QueryTerm] = []
    seen: set[str] = set()
    for label in labels:
        cleaned = _clean_comparison_label(label)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        out.append(_QueryTerm(cleaned, cleaned))
    return out if len(out) >= 2 else []


def _extract_compared_labels(query: str) -> list[str]:
    compact = re.sub(r"\s+", " ", query).strip(" ?!.")
    patterns = [
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:$|[?.,;]|\s+in\s+|\s+for\s+)",
        r"\bcompare\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+?)(?:$|[?.,;]|\s+in\s+|\s+for\s+)",
        r"\bdifference\s+between\s+(.+?)\s+and\s+(.+?)(?:$|[?.,;]|\s+in\s+|\s+for\s+)",
        r"\bentre\s+(.+?)\s+et\s+(.+?)(?:$|[?.,;]|\s+dans\s+|\s+pour\s+)",
        r"\bcomparer\s+(.+?)\s+(?:et|avec|a|a)\s+(.+?)(?:$|[?.,;]|\s+dans\s+|\s+pour\s+)",
    ]
    labels: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            labels.extend(match.groups())
            break
    if not labels and re.search(r"\b(?:vs\.?|versus)\b", compact, flags=re.IGNORECASE):
        labels = re.split(r"\b(?:vs\.?|versus)\b", compact, maxsplit=1, flags=re.IGNORECASE)
    if not labels:
        return []
    expanded: list[str] = []
    for label in labels:
        expanded.extend(re.split(r"\s*,\s*|\s*/\s*", label))
    return expanded


def _clean_comparison_label(label: str) -> str:
    words = [word.strip(".,;:?!()[]{}\"'") for word in _TERM_WORD_RE.findall(label)]
    kept = [word for word in words if word and word.casefold() not in _COMPARISON_STOPWORDS]
    if not kept:
        return ""
    if len(kept) > 5:
        kept = kept[-5:]
    return " ".join(kept).strip()


def _balanced_term_merge(term_hits: dict[str, list[Chunk]], full_query_hits: list[Chunk], *, target: int) -> list[Chunk]:
    selected: list[Chunk] = []
    seen: set[str] = set()
    max_depth = max((len(hits) for hits in term_hits.values()), default=0)
    for index in range(max_depth):
        for label, hits in term_hits.items():
            if index >= len(hits):
                continue
            chunk = hits[index]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            selected.append(_with_metadata(chunk, {"matched_query_term": label}))
            if len(selected) >= target:
                return selected
    for chunk in full_query_hits:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        selected.append(chunk)
        if len(selected) >= target:
            break
    return selected


def _comparison_labels(chunks: list[Chunk]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        label = str(chunk.metadata.get("matched_query_term", "")).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _with_metadata(chunk: Chunk, metadata: dict[str, object]) -> Chunk:
    return chunk.model_copy(update={"metadata": {**dict(chunk.metadata or {}), **metadata}})


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        out.append(chunk)
    return out


def _dedupe_text(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip(" -:;")
        key = text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _clean_mindmap_detail(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html_unescape(str(text or "")))
    text = _strip_mindmap_media_markers(text)
    text = re.sub(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -:;")
    return _repair_mindmap_label_punctuation(text)


def _clean_mindmap_label(label: str) -> str:
    label = html_unescape(str(label or ""))
    label = re.sub(r"<[^>]+>", " ", label)
    label = _strip_mindmap_media_markers(label)
    label = re.sub(r"\${1,2}.*?\${1,2}", " ", label)
    label = re.sub(r"[*_`#]+", "", label)
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"^(?:\d+\.\s*)+", "", label)
    label = re.sub(r"\s+\d{1,3}$", "", label)
    label = label.strip(" -:;,.\u2022\u25b6")
    if len(label) > 100:
        label = label[:100].rsplit(" ", 1)[0].strip(" -:;")
    return _repair_mindmap_label_punctuation(label)


def _strip_mindmap_media_markers(text: str) -> str:
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", " ", text)
    text = re.sub(r"\[\s*Image[^\]]*\]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bImage\s*\d*\b", " ", text, flags=re.IGNORECASE)
    return text


def _repair_mindmap_label_punctuation(label: str) -> str:
    label = str(label or "").strip(" -:;,.\u2022\u25b6")
    label = re.sub(r"\(\s*\)", "", label)
    label = re.sub(r"\[\s*]", "", label)
    label = re.sub(r"\s+", " ", label).strip(" -:;,.\u2022\u25b6")
    if label.count("(") > label.count(")"):
        open_index = label.rfind("(")
        if open_index >= max(0, len(label) - 24):
            label = label[:open_index].strip(" -:;,.")
    if label.count("[") > label.count("]"):
        open_index = label.rfind("[")
        if open_index >= max(0, len(label) - 24):
            label = label[:open_index].strip(" -:;,.")
    return label


def html_unescape(value: str) -> str:
    return (
        value.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )


def _norm_mindmap_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[_\-.]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized.casefold()).strip()


def _is_generic_mindmap_root_label(label: str) -> bool:
    key = _norm_mindmap_text(label)
    return bool(
        key in {"conclusion", "conclusion et prochaines etapes", "synthese"}
        or re.search(r"\b(?:conclusion|prochaines etapes|synthese|recapitulatif|plan de la seance)\b", key)
    )


def _mindmap_tokens(text: str) -> set[str]:
    stop = {
        "les",
        "des",
        "dans",
        "pour",
        "avec",
        "une",
        "un",
        "du",
        "de",
        "la",
        "le",
        "et",
        "en",
        "au",
        "aux",
        "sur",
        "par",
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "this",
        "that",
        "cours",
        "lecture",
    }
    return {
        token
        for token in re.findall(r"[\w\u00c0-\u00ff]+", _norm_mindmap_text(text))
        if len(token) > 2 and token not in stop and not token.isdigit()
    }


def _is_noisy_mindmap_label(label: str) -> bool:
    key = _norm_mindmap_text(label)
    if not key or len(key) < 4:
        return True
    noisy_exact = {
        "abdelaaziz",
        "hessane",
        "master",
        "normale",
        "novembre",
        "moulay ismail",
        "ecole normale superieure",
        "universite moulay ismail",
        "plan de la seance",
        "table des matieres",
        "references",
        "lectures recommandees",
        "source material",
        "source plan item",
        "key details",
        "resume",
        "ordre",
        "liste ordonnee",
        "scenario",
        "etape",
        "people who bought",
        "recommended to user",
        "similar users",
        "recommended system",
    }
    if key in noisy_exact:
        return True
    if _is_generic_mindmap_root_label(label):
        return True
    if re.search(r"\ba\s*verifier\b", key):
        return True
    if re.search(r"(?:^|\s)image(?:\s|$)", key):
        return True
    if re.search(r"\b(slide|logo|layout|attribution|page|copyright|navigation|footer|figure)\b", key):
        return True
    tokens = _mindmap_tokens(label)
    return len(tokens) < 1


def _is_low_information_text(text: str) -> bool:
    compact = " ".join(str(text or "").split())
    if not compact:
        return True
    if len(compact) < 18:
        return True
    if _LOW_INFORMATION_RE.fullmatch(compact.strip()):
        return True
    alpha_chars = sum(1 for char in compact if char.isalpha())
    if alpha_chars < 8 and len(compact) < 80:
        return True
    words = re.findall(r"[\w\u0600-\u06ff]+", compact)
    return len(set(word.casefold() for word in words)) <= 2 and len(compact) < 80


def _deterministic_hyde(query: str, max_chars: int) -> str:
    tokens = [token for token, _count in Counter(tokenize(query)).most_common(12) if len(token) > 2]
    topic = ", ".join(tokens) if tokens else query
    text = (
        "A course excerpt for this question would define the key terms, show "
        f"how {topic} relate to the lesson, and include examples, formulas, "
        "tables, or steps from the uploaded material."
    )
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


_COURSE_OVERVIEW_INTENTS = {
    "explain",
    "summarize",
    "summary",
    "overview",
    "teach",
    "review",
    "understand",
    "roadmap",
    "prepare",
    "study",
    "start",
    "begin",
    "exam",
    "resume",
    "explique",
    "etudier",
    "commencer",
}
_COURSE_TARGET_TERMS = {
    "course",
    "class",
    "lesson",
    "lecture",
    "chapter",
    "module",
    "material",
    "materials",
    "document",
    "documents",
    "file",
    "files",
    "uploaded",
    "cours",
    "chapitre",
    "support",
}
_BROAD_PRONOUN_TARGETS = {"this", "that", "it", "these", "those", "ce", "cet", "cette", "ces"}


def _is_course_overview_query(query: str) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return False
    if any(
        re.search(pattern, normalized)
        for pattern in (
            r"\bwhat\s+should\s+i\s+study\s+first\b",
            r"\bwhere\s+should\s+i\s+start\b",
            r"\bgive\s+me\s+a\s+(?:beginner\s+)?roadmap\b",
            r"\bprepare\s+me\s+for\s+(?:the\s+)?exam\b",
            r"\bteach\s+me\s+(?:this|the)\s+course\b",
            r"\bhelp\s+me\s+(?:study|revise|review)\b",
        )
    ):
        return True
    tokens = set(normalized.split())
    has_intent = bool(tokens & _COURSE_OVERVIEW_INTENTS)
    has_course_target = bool(tokens & _COURSE_TARGET_TERMS)
    has_broad_pronoun = bool(tokens & _BROAD_PRONOUN_TARGETS)
    if has_intent and has_course_target:
        return True
    if has_intent and has_broad_pronoun and len(tokens) <= 8:
        return True
    return bool(
        re.search(r"\bwhat\s+(?:is\s+)?(?:this|that)\s+(?:course|class|lecture|lesson)\s+about\b", normalized)
        or re.search(r"\bwhat\s+are\s+these\s+(?:documents|files|materials)\s+about\b", normalized)
        or re.search(r"\bde\s+quoi\s+parle\s+(?:ce|cet|cette)\s+(?:cours|document|chapitre)\b", normalized)
    )


def _normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKD", query.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s\u0600-\u06ff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_COMPARISON_MARKERS_RE = re.compile(
    r"\b(compare|comparison|difference|differentiate|versus|vs|between|"
    r"comparer|comparaison|difference|entre)\b",
    re.IGNORECASE,
)
_TERM_WORD_RE = re.compile(r"[\w\u0600-\u06ff][\w\u0600-\u06ff+/#.-]*")
_COMPARISON_STOPWORDS = {
    "what",
    "whats",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "from",
    "course",
    "material",
    "materials",
    "explain",
    "compare",
    "comparison",
    "difference",
    "differentiate",
    "between",
    "versus",
    "vs",
    "and",
    "with",
    "me",
    "please",
    "quelle",
    "quel",
    "est",
    "sont",
    "la",
    "le",
    "les",
    "des",
    "du",
    "de",
    "un",
    "une",
    "dans",
    "pour",
    "cours",
    "document",
    "documents",
    "explique",
    "comparer",
    "comparaison",
    "entre",
    "et",
}
_FORMULA_QUERY_RE = re.compile(
    r"\b(formula|equation|derive|derivation|calculate|compute|symbol|formule|equation|calculer|calcule)\b|[=+\-*/^_]",
    re.IGNORECASE,
)
_MATH_TEXT_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.+?\\\)|\\(?:frac|sum|sqrt|hat|bar|vec|int|prod)\b|"
    r"[=]|[A-Za-z]\s*[_^]\s*[A-Za-z0-9])",
    re.DOTALL,
)
_LOW_INFORMATION_RE = re.compile(
    r"^(?:\d{1,4}|\d{1,2}\s*/\s*\d{1,2}|(?:19|20)\d{2}(?:\s*/\s*(?:19|20)?\d{2})?|"
    r"page\s+\d+|questions?\s*\??|merci|thank\s+you|table\s+des\s+matieres|contents?)$",
    re.IGNORECASE,
)


_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service

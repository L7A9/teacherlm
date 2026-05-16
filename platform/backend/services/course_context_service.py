from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.retrieval.hybrid_retriever import HybridRetriever
from teacherlm_core.retrieval.retrieval_modes import (
    coverage_broad,
    narrative_arc,
    relationship_dense,
    semantic_topk,
    topic_clusters,
)
from teacherlm_core.schemas.chunk import Chunk

from config import Settings, get_settings
from db.session import session_scope
from services.course_content_store import CourseContentStore, get_course_content_store
from services.vector_service import VectorService, _collection_name, get_vector_service


logger = logging.getLogger(__name__)

RetrievalMode = Literal[
    "semantic_topk",
    "coverage_broad",
    "narrative_arc",
    "topic_clusters",
    "relationship_dense",
]


class CourseContextService:
    """Course-aware context API used by generators through the backend."""

    def __init__(
        self,
        settings: Settings | None = None,
        vector_service: VectorService | None = None,
        content_store: CourseContentStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vectors = vector_service or get_vector_service()
        self._store = content_store or get_course_content_store()

    async def get_relevant_chunks(
        self,
        conversation_id: uuid.UUID | str,
        query: str,
        mode: RetrievalMode = "semantic_topk",
    ) -> list[Chunk]:
        async with session_scope() as session:
            all_chunks = await self._store.get_chunks(
                session,
                uuid.UUID(str(conversation_id)),
                limit=self._settings.course_context_max_chunks,
            )
        return await self._retrieve_from_chunks(
            conversation_id=conversation_id,
            query=query,
            mode=mode,
            all_chunks=all_chunks,
        )

    async def get_full_course_outline(self, conversation_id: uuid.UUID | str) -> list[Chunk]:
        async with session_scope() as session:
            sections = await self._store.get_sections(session, uuid.UUID(str(conversation_id)))
        if not sections:
            return []
        lines = []
        for section in sections:
            indent = "  " * max(0, section.level - 1)
            path = " > ".join(section.heading_path or [section.title])
            lines.append(f"{indent}- {path}")
        return [
            Chunk(
                text="Course outline:\n" + "\n".join(lines),
                source="course_outline",
                score=1.0,
                chunk_id=f"outline:{conversation_id}",
                metadata={"context_type": "course_outline", "section_count": len(sections)},
            )
        ]

    async def get_course_sections(self, conversation_id: uuid.UUID | str) -> list[Chunk]:
        async with session_scope() as session:
            sections = await self._store.get_sections(session, uuid.UUID(str(conversation_id)))
        return [self._section_to_chunk(section, context_type="course_section") for section in sections]

    async def get_section_content(self, section_id: uuid.UUID | str) -> Chunk | None:
        async with session_scope() as session:
            section = await self._store.get_section(session, section_id)
        return self._section_to_chunk(section, context_type="section_content") if section else None

    async def get_representative_course_context(
        self,
        conversation_id: uuid.UUID | str,
    ) -> list[Chunk]:
        async with session_scope() as session:
            sections = await self._store.get_sections(session, uuid.UUID(str(conversation_id)))
        if not sections:
            return []
        major = [section for section in sections if section.level <= 2] or sections
        target = self._settings.course_context_section_budget
        if len(major) > target:
            stride = max(1, len(major) // target)
            major = major[::stride][:target]
        return [self._section_summary_chunk(section, "representative_section") for section in major]

    async def get_mindmap_course_context(self, conversation_id: uuid.UUID | str) -> list[Chunk]:
        async with session_scope() as session:
            documents = await self._store.get_documents(session, uuid.UUID(str(conversation_id)))
            sections = await self._store.get_sections(session, uuid.UUID(str(conversation_id)))
        if not documents:
            return []

        sections_by_doc: dict[str, list[object]] = defaultdict(list)
        for section in sections:
            sections_by_doc[str(section.document_id)].append(section)

        ordered_documents = sorted(
            documents,
            key=lambda doc: _document_sort_key(
                filename=str(getattr(doc, "source_filename", "")),
                title=str(getattr(doc, "title", "")),
                created_at=getattr(doc, "created_at", None),
            ),
        )
        packs = [
            self._mindmap_module_pack(
                document=document,
                sections=sections_by_doc.get(str(document.id), []),
                order=index,
            )
            for index, document in enumerate(ordered_documents)
        ]

        outline = self._mindmap_global_outline(ordered_documents, sections_by_doc)
        return _dedupe_chunks([outline, *packs])

    async def get_topic_context(
        self,
        conversation_id: uuid.UUID | str,
        topic: str,
    ) -> list[Chunk]:
        chunks = await self.get_relevant_chunks(conversation_id, topic, "semantic_topk")
        section_ids = _section_ids(chunks)
        async with session_scope() as session:
            sections = await self._store.get_sections(
                session,
                uuid.UUID(str(conversation_id)),
                section_ids=list(section_ids),
            )
        summaries = [self._section_summary_chunk(section, "topic_section") for section in sections]
        return _dedupe_chunks([*summaries, *chunks])

    async def get_equations(self, conversation_id: uuid.UUID | str) -> list[Chunk]:
        return await self._typed_section_items(conversation_id, "equations", "equations")

    async def get_tables(self, conversation_id: uuid.UUID | str) -> list[Chunk]:
        return await self._typed_section_items(conversation_id, "tables", "tables")

    async def get_timeline_events(self, conversation_id: uuid.UUID | str) -> list[Chunk]:
        return await self._typed_section_items(conversation_id, "timeline_events", "timeline_events")

    async def get_generator_context(
        self,
        *,
        conversation_id: uuid.UUID | str,
        output_type: str,
        query: str,
        topic: str | None = None,
    ) -> list[Chunk]:
        if output_type in {"text", "chat"}:
            return await self.get_relevant_chunks(conversation_id, query, "semantic_topk")

        if output_type == "podcast":
            if topic:
                return await self.get_topic_context(conversation_id, topic)
            return _dedupe_chunks(
                [
                    *(await self.get_full_course_outline(conversation_id)),
                    *(await self.get_representative_course_context(conversation_id)),
                ]
            )

        if output_type == "quiz":
            if topic:
                return await self.get_topic_context(conversation_id, topic)
            return await self.get_representative_course_context(conversation_id)

        if output_type == "mindmap":
            if topic:
                return await self.get_topic_context(conversation_id, topic)
            return await self.get_mindmap_course_context(conversation_id)

        if output_type == "presentation":
            if topic:
                return _dedupe_chunks(
                    [
                        *(await self.get_topic_context(conversation_id, topic)),
                        *(await self.get_equations(conversation_id)),
                        *(await self.get_tables(conversation_id)),
                    ]
                )
            return _dedupe_chunks(
                [
                    *(await self.get_representative_course_context(conversation_id)),
                    *(await self.get_equations(conversation_id)),
                    *(await self.get_tables(conversation_id)),
                ]
            )

        if output_type in {"chart", "diagram"}:
            return await self.get_relevant_chunks(
                conversation_id,
                query or topic or "",
                "relationship_dense",
            )

        return await self.get_relevant_chunks(conversation_id, query, "semantic_topk")

    async def _retrieve_from_chunks(
        self,
        *,
        conversation_id: uuid.UUID | str,
        query: str,
        mode: RetrievalMode,
        all_chunks: list[Chunk],
    ) -> list[Chunk]:
        if not all_chunks:
            return []
        if not query.strip():
            return self._broad_sample(all_chunks, self._settings.course_context_chunk_budget)

        collection = _collection_name(conversation_id)
        if not await self._vectors._client.collection_exists(collection):
            return self._bm25_only(query, all_chunks)

        embedder = await self._vectors._get_embedder()
        retriever = HybridRetriever(
            qdrant_client=self._vectors._client,
            collection_name=collection,
            embedder=embedder,
            dense_top_k=self._settings.retrieval_dense_candidate_k,
            sparse_top_k=self._settings.retrieval_sparse_candidate_k,
        )
        retriever.index_bm25(all_chunks)

        k = self._settings.retrieval_top_k
        match mode:
            case "semantic_topk":
                return await self._semantic_topk(query, retriever, k)
            case "coverage_broad":
                return await coverage_broad(query, retriever, k=max(k * 2, 16))
            case "narrative_arc":
                return await narrative_arc(query, retriever, all_chunks)
            case "topic_clusters":
                return await topic_clusters(query, retriever, n_clusters=max(6, min(12, k)))
            case "relationship_dense":
                return await relationship_dense(query, retriever)

    async def _typed_section_items(
        self,
        conversation_id: uuid.UUID | str,
        attr: str,
        context_type: str,
    ) -> list[Chunk]:
        async with session_scope() as session:
            sections = await self._store.get_sections(session, uuid.UUID(str(conversation_id)))
        out: list[Chunk] = []
        for section in sections:
            items = getattr(section, attr)
            if not items:
                continue
            if attr == "tables":
                body = "\n\n".join("\n".join(table.get("rows", [])) for table in items)
            else:
                body = "\n".join(str(item) for item in items)
            out.append(
                Chunk(
                    text=f"{' > '.join(section.heading_path)}\n\n{body}",
                    source="course",
                    score=1.0,
                    chunk_id=f"{context_type}:{section.id}",
                    metadata={
                        "context_type": context_type,
                        "section_id": str(section.id),
                        "document_id": str(section.document_id),
                        "heading_path": " > ".join(section.heading_path),
                    },
                )
            )
        return out[: self._settings.course_context_section_budget]

    @staticmethod
    def _section_to_chunk(section: object, *, context_type: str) -> Chunk:
        text = getattr(section, "text")
        heading_path = list(getattr(section, "heading_path") or [getattr(section, "title")])
        return Chunk(
            text=f"{' > '.join(heading_path)}\n\n{text}",
            source="course",
            score=1.0,
            chunk_id=f"{context_type}:{getattr(section, 'id')}",
            metadata={
                "context_type": context_type,
                "section_id": str(getattr(section, "id")),
                "document_id": str(getattr(section, "document_id")),
                "heading_path": " > ".join(heading_path),
                "key_concepts": list(getattr(section, "key_concepts") or []),
            },
        )

    @staticmethod
    def _section_summary_chunk(section: object, context_type: str) -> Chunk:
        heading_path = list(getattr(section, "heading_path") or [getattr(section, "title")])
        facts: list[str] = []
        concepts = list(getattr(section, "key_concepts") or [])[:8]
        if concepts:
            facts.append("Key concepts: " + ", ".join(concepts))
        equations = list(getattr(section, "equations") or [])[:4]
        if equations:
            facts.append("Equations:\n" + "\n".join(equations))
        tables = list(getattr(section, "tables") or [])[:2]
        if tables:
            facts.append("Tables:\n" + "\n\n".join("\n".join(t.get("rows", [])) for t in tables))
        summary = getattr(section, "summary", "") or " ".join(getattr(section, "text").split())[:700]
        return Chunk(
            text="\n\n".join([f"{' > '.join(heading_path)}", summary, *facts]).strip(),
            source="course",
            score=1.0,
            chunk_id=f"{context_type}:{getattr(section, 'id')}",
            metadata={
                "context_type": context_type,
                "section_id": str(getattr(section, "id")),
                "document_id": str(getattr(section, "document_id")),
                "heading_path": " > ".join(heading_path),
                "key_concepts": concepts,
            },
        )

    def _mindmap_global_outline(
        self,
        documents: list[object],
        sections_by_doc: dict[str, list[object]],
    ) -> Chunk:
        lines = ["Course document sequence:"]
        for index, document in enumerate(documents, start=1):
            filename = str(getattr(document, "source_filename", ""))
            title = str(getattr(document, "title", "") or filename)
            role = "supporting" if _is_supplemental_document(filename, title) else "main"
            lines.append(f"{index}. {title} ({filename}) [{role}]")
            major = _select_mindmap_sections(sections_by_doc.get(str(getattr(document, "id")), []), limit=10)
            for section in major[:8]:
                heading = _clean_heading_label(_heading_text(section))
                if heading:
                    lines.append(f"   - {heading}")

        conversation_id = str(getattr(documents[0], "conversation_id", "course")) if documents else "course"
        return Chunk(
            text="\n".join(lines),
            source="course_outline",
            score=1.0,
            chunk_id=f"mindmap-outline:{conversation_id}",
            metadata={
                "context_type": "mindmap_course_outline",
                "document_count": len(documents),
            },
        )

    def _mindmap_module_pack(
        self,
        *,
        document: object,
        sections: list[object],
        order: int,
    ) -> Chunk:
        filename = str(getattr(document, "source_filename", ""))
        title = str(getattr(document, "title", "") or filename)
        role = "supporting" if _is_supplemental_document(filename, title) else "main"
        selected = _select_mindmap_sections(sections, limit=_per_document_section_budget(len(sections)))

        lines = [
            f"Module {order + 1}: {title}",
            f"Source file: {filename}",
            f"Document role: {role}",
            "",
            "Major headings:",
        ]
        for section in _select_mindmap_sections(sections, limit=18):
            heading = _clean_heading_label(_heading_text(section))
            if heading:
                lines.append(f"- {heading}")

        lines.extend(["", "Study outline details:"])
        for section in selected:
            heading_path = " > ".join(getattr(section, "heading_path", None) or [getattr(section, "title", "")])
            heading = _clean_heading_label(heading_path)
            if not heading:
                continue
            facts = _section_facts(section)
            summary = str(getattr(section, "summary", "") or "").strip()
            if not summary:
                summary = " ".join(str(getattr(section, "text", "")).split())[:500]
            detail = summary
            if facts:
                detail = f"{detail}\n  Key details: {facts}" if detail else f"Key details: {facts}"
            lines.append(f"- {heading}: {detail}".strip())

        text = "\n".join(line for line in lines if line is not None).strip()
        max_chars = 6500
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].strip()
        return Chunk(
            text=text,
            source=filename,
            score=1.0,
            chunk_id=f"mindmap-module:{getattr(document, 'id')}",
            metadata={
                "context_type": "mindmap_module_pack",
                "document_id": str(getattr(document, "id")),
                "source_filename": filename,
                "document_title": title,
                "document_order": order,
                "document_role": role,
                "section_count": len(sections),
                "selected_section_count": len(selected),
            },
        )

    def _bm25_only(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        from teacherlm_core.retrieval.bm25 import BM25Index

        logger.info("Qdrant collection unavailable; using BM25-only retrieval")
        hits = BM25Index(chunks).query(query, top_k=max(self._settings.retrieval_top_k, 20))
        terms = _comparison_terms(query)
        if len(terms) < 2:
            return hits[: self._settings.retrieval_top_k]
        term_hits = {
            term.label: BM25Index(chunks).query(term.query, top_k=max(self._settings.retrieval_top_k, 8))
            for term in terms
        }
        return _balanced_term_merge(term_hits, hits, target=max(self._settings.retrieval_top_k, 20))

    async def _semantic_topk(
        self,
        query: str,
        retriever: HybridRetriever,
        k: int,
    ) -> list[Chunk]:
        candidate_k = max(k, self._settings.retrieval_rerank_candidate_k)
        full_query_hits = await semantic_topk(query, retriever, k=candidate_k)

        terms = _comparison_terms(query)
        if len(terms) < 2:
            return full_query_hits

        per_term_k = max(k, min(24, candidate_k // len(terms) + 4))
        term_hits: dict[str, list[Chunk]] = {}
        for term in terms:
            hits = await semantic_topk(term.query, retriever, k=per_term_k)
            term_hits[term.label] = [
                _with_metadata(hit, {"matched_query_term": term.label})
                for hit in hits
            ]
        return _balanced_term_merge(term_hits, full_query_hits, target=candidate_k)

    @staticmethod
    def _broad_sample(chunks: list[Chunk], target: int) -> list[Chunk]:
        if len(chunks) <= target:
            return list(chunks)
        by_section: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_section[str(chunk.metadata.get("section_id", ""))].append(chunk)
        selected: list[Chunk] = []
        for group in by_section.values():
            selected.append(group[0])
            if len(selected) >= target:
                return selected
        stride = max(1, len(chunks) // target)
        return _dedupe_chunks([*selected, *chunks[::stride]])[:target]


def _section_ids(chunks: list[Chunk]) -> set[str]:
    return {str(chunk.metadata.get("section_id", "")).strip() for chunk in chunks if chunk.metadata.get("section_id")}


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        out.append(chunk)
    return out


_SUPPLEMENTAL_DOCUMENT_RE = re.compile(
    r"\b(guide|reference|references|appendix|annexe|corrige|corrigé|solutions?|exercises?|exercices?|worksheet|fiche|bibliography|bibliographie)\b",
    re.IGNORECASE,
)
_SEQUENCE_RE = re.compile(
    r"\b(?:lecture|lec|lesson|week|semaine|chapter|chapitre|unit|module|cours|part|partie)[\s_\-]*(\d{1,4}|[ivxlcdm]{1,8})\b",
    re.IGNORECASE,
)
_ANY_NUMBER_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
_DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})[\-_./ ](0?[1-9]|1[0-2])[\-_./ ](0?[1-9]|[12]\d|3[01])\b")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_NOISY_HEADING_RE = re.compile(
    r"^(?:table\s+des\s+mati[eè]res|contents?|references?|ressources|questions?\s*\??|merci|thank\s+you|page\s+\d+|\d+\s*)$",
    re.IGNORECASE,
)
_FORMULA_HEADING_RE = re.compile(r"^[\s$\\{}_^=+\-*/().,\dA-Za-z]+$")


def _document_sort_key(
    *,
    filename: str,
    title: str,
    created_at: datetime | None,
) -> tuple[int, int, str, float, str]:
    combined = f"{filename} {title}"
    supplemental = 1 if _is_supplemental_document(filename, title) else 0
    sequence = _sequence_number(combined)
    if sequence is None:
        sequence = _date_number(combined)
    if sequence is None:
        sequence = _fallback_number(filename)
    sequence_rank = 0 if sequence is not None else 1
    timestamp = created_at.timestamp() if isinstance(created_at, datetime) else 0.0
    return (supplemental, sequence_rank, _sequence_sort_value(sequence), timestamp, filename.lower())


def _sequence_number(text: str) -> int | None:
    match = _SEQUENCE_RE.search(text)
    if not match:
        return None
    token = match.group(1).lower()
    if token.isdigit():
        return int(token)
    return _roman_to_int(token)


def _date_number(text: str) -> int | None:
    match = _DATE_RE.search(text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return year * 10_000 + month * 100 + day


def _fallback_number(text: str) -> int | None:
    match = _ANY_NUMBER_RE.search(text)
    return int(match.group(1)) if match else None


def _sequence_sort_value(value: int | None) -> str:
    return f"{value:08d}" if value is not None else "zzzzzzzz"


def _roman_to_int(value: str) -> int | None:
    total = 0
    prev = 0
    for char in reversed(value.lower()):
        current = _ROMAN_VALUES.get(char)
        if current is None:
            return None
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total or None


def _is_supplemental_document(filename: str, title: str) -> bool:
    return bool(_SUPPLEMENTAL_DOCUMENT_RE.search(f"{filename} {title}"))


def _select_mindmap_sections(sections: list[object], *, limit: int) -> list[object]:
    clean = [section for section in sections if not _is_noisy_section(section)]
    if len(clean) <= limit:
        return clean

    selected: list[object] = []
    seen: set[str] = set()

    for section in clean:
        if int(getattr(section, "level", 1) or 1) <= 2:
            _append_unique_section(selected, seen, section)
        if len(selected) >= max(3, limit // 2):
            break

    stride = max(1, len(clean) // max(1, limit - len(selected)))
    for section in clean[::stride]:
        _append_unique_section(selected, seen, section)
        if len(selected) >= limit:
            break

    for section in clean:
        _append_unique_section(selected, seen, section)
        if len(selected) >= limit:
            break
    return selected


def _append_unique_section(selected: list[object], seen: set[str], section: object) -> None:
    section_id = str(getattr(section, "id", ""))
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


def _is_noisy_section(section: object) -> bool:
    title = _clean_heading_label(_heading_text(section))
    leaf_title = _clean_heading_label(str((getattr(section, "heading_path", None) or [getattr(section, "title", "")])[-1]))
    if not title:
        return True
    normalized = title.strip(" :-–—").lower()
    normalized_leaf = leaf_title.strip(" :-–—").lower()
    if _NOISY_HEADING_RE.match(normalized) or _NOISY_HEADING_RE.match(normalized_leaf):
        return True
    text = " ".join(str(getattr(section, "text", "")).split())
    if len(text) < 24 and not getattr(section, "key_concepts", None):
        return True
    if len(title) > 140:
        return True
    if "$" in title and _FORMULA_HEADING_RE.match(title):
        return True
    return False


def _heading_text(section: object) -> str:
    heading_path = getattr(section, "heading_path", None) or []
    if heading_path:
        return " > ".join(str(item) for item in heading_path if str(item).strip())
    return str(getattr(section, "title", "")).strip()


def _clean_heading_label(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("**", "").strip(" -:;")
    return text


def _section_facts(section: object) -> str:
    facts: list[str] = []
    concepts = [str(item).strip() for item in (getattr(section, "key_concepts", None) or []) if str(item).strip()]
    if concepts:
        facts.append("concepts: " + ", ".join(concepts[:6]))
    equations = [str(item).strip() for item in (getattr(section, "equations", None) or []) if str(item).strip()]
    if equations:
        facts.append("formulas: " + "; ".join(equations[:2]))
    events = [str(item).strip() for item in (getattr(section, "timeline_events", None) or []) if str(item).strip()]
    if events:
        facts.append("dates/events: " + "; ".join(events[:4]))
    tables = list(getattr(section, "tables", None) or [])
    if tables:
        rows = []
        for table in tables[:1]:
            rows.extend(str(row).strip() for row in table.get("rows", [])[:3] if str(row).strip())
        if rows:
            facts.append("table: " + " | ".join(rows))
    return "; ".join(facts)


class _QueryTerm:
    def __init__(self, label: str, query: str) -> None:
        self.label = label
        self.query = query


def _comparison_terms(query: str) -> list[_QueryTerm]:
    normalized = query.lower()
    terms: list[_QueryTerm] = []

    concept_queries = [
        (
            "svd",
            ("svd", "singular value", "matrix factorization", "factorisation matricielle"),
            "SVD matrix factorization latent factors user item P Q dot product linear rating prediction",
        ),
        (
            "ncf",
            ("ncf", "neural collaborative filtering", "collaborative filtering neural"),
            "NCF replace dot product matrix factorization MLP neural layers embeddings non linear f p_u q_i theta",
        ),
        (
            "rnn",
            ("rnn", "recurrent neural", "recurrent network", "gru", "lstm"),
            "RNN at each time step h_t x_t y_t item embedding hidden state memory prediction sequence",
        ),
        (
            "transformer",
            ("transformer", "attention", "self-attention"),
            "Transformer self-attention sequence modeling temporal dependencies recommendation",
        ),
    ]

    for label, aliases, expanded_query in concept_queries:
        if any(alias in normalized for alias in aliases):
            terms.append(_QueryTerm(label, expanded_query))

    acronym_matches = {match.lower() for match in re.findall(r"\b[A-Z][A-Z0-9]{1,7}\b", query)}
    seen = {term.label for term in terms}
    for acronym in sorted(acronym_matches - seen):
        terms.append(_QueryTerm(acronym, acronym))

    comparison_markers = ("difference", "compare", "versus", " vs ", "between", "différence", "comparer")
    if len(terms) >= 2 and any(marker in normalized for marker in comparison_markers):
        return terms
    return []


def _balanced_term_merge(
    term_hits: dict[str, list[Chunk]],
    full_query_hits: list[Chunk],
    *,
    target: int,
) -> list[Chunk]:
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


def _with_metadata(chunk: Chunk, metadata: dict[str, object]) -> Chunk:
    return chunk.model_copy(update={"metadata": {**dict(chunk.metadata or {}), **metadata}})


_context_service: CourseContextService | None = None


def get_course_context_service() -> CourseContextService:
    global _context_service
    if _context_service is None:
        _context_service = CourseContextService()
    return _context_service

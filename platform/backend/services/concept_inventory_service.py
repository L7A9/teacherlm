from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha1
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs

from config import Settings, get_settings
from db.models import CourseConceptRecord, KnowledgeAttemptRecord, KnowledgeCheckRecord, SearchChunkRecord


logger = logging.getLogger(__name__)


BloomLevel = Literal["remember", "understand", "apply", "analyze"]
_BLOOM_LEVELS: set[str] = {"remember", "understand", "apply", "analyze"}
_MAX_CONCEPTS = 80
_LOCAL_FALLBACK_MODEL = "gemma4:e2b"
_DEFINITION_RE = re.compile(r"^\s*(?:[-*]\s*)?([^:\n]{2,90})\s*:\s+(.{12,})$")
_BOLD_TERM_RE = re.compile(r"\*\*([^*\n]{2,80})\*\*")
_WORD_RE = re.compile(r"[a-z0-9]+")
_NOISE_RE = re.compile(
    r"\b(author|authors|copyright|references|bibliography|table of contents|"
    r"contents|page|figure|fig|university|department|faculty|school|"
    r"introduction|overview|agenda|plan|objectives?|objectifs?|conclusion|"
    r"summary|resume|r[ée]sum[ée]|chapter|chapitre|lecture|course|cours|"
    r"week|semaine|session|s[ée]ance|part|partie|module|unit|slide)\b",
    re.IGNORECASE,
)
_SECTION_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"(?:week|semaine|chapter|chapitre|lecture|session|s[ée]ance|module|part|partie)\s+\d+|"
    r"\d+(?:\.\d+)*\.?|"
    r"[ivxlcdm]+\."
    r")\s*[:.)-]?\s*",
    re.IGNORECASE,
)
_FRAGMENT_END_RE = re.compile(r"\b(?:of|de|des|du|la|le|les|and|or|et|ou|for|with|avec|to|à|a|an|the)\s*$", re.IGNORECASE)
_BAD_NAME_CHARS_RE = re.compile(r"[$\\{}<>]|</?\w+|<mark|</mark|</b|<b", re.IGNORECASE)
_BAD_MARKDOWN_RE = re.compile(r"\*\*|__|`|\[|\]")
_LIST_OR_EXAMPLE_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)*\s+|"
    r"\d+(?:\.\d+)*\s*[:.)-]|"
    r"[a-e]\s*,|"
    r"(?:doc|position|rang|rank|groupe|group|facteur|factor|systeme|system|"
    r"etape|step|probleme|problem|exemple|example|cas|case)\s+[a-z0-9]"
    r")",
    re.IGNORECASE,
)
_BAD_PHRASE_RE = re.compile(
    r"\b(?:"
    r"imaginons|supposons|considerons|ne posez pas|message pour|"
    r"calcul detaille|resultats?|sortie finale|idee principale|point cle|"
    r"question cruciale|le but du jeu|c'est le|on compare|on fixe|si rel|"
    r"meme rmse|beaucoup plus eleve|liste ordonnee|classement ordonnee|"
    r"impact critique|au-dela de|construire le profil|mettre a jour|trouver les"
    r")\b",
    re.IGNORECASE,
)
_INCOMPLETE_PAREN_RE = re.compile(r"\([^)]*$|\bid\s*$", re.IGNORECASE)


class ConceptCandidate(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    bloom_level: BloomLevel = "understand"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)
    extraction_method: str = "llm"


class ConceptCandidateBatch(BaseModel):
    concepts: list[ConceptCandidate] = Field(default_factory=list)


@dataclass(slots=True)
class _ConceptAccumulator:
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    bloom_level: str = "understand"
    importance: float = 0.5
    source_file_ids: set[str] = field(default_factory=set)
    source_section_ids: set[str] = field(default_factory=set)
    source_chunk_ids: set[str] = field(default_factory=set)
    source_part_titles: set[str] = field(default_factory=set)
    extraction_methods: set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return normalize_concept_key(self.canonical_name)

    @property
    def alias_keys(self) -> set[str]:
        return {normalize_concept_key(value) for value in [self.canonical_name, *self.aliases] if value}


class ConceptInventoryService:
    """Builds and resolves the canonical concept inventory for a conversation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def rebuild_concepts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> list[CourseConceptRecord]:
        await self.ensure_schema(session)
        chunks = await self._load_chunks(session, conversation_id)
        candidates: list[ConceptCandidate] = []
        try:
            candidates = await self._llm_candidates(chunks, llm_options=llm_options)
        except Exception:  # noqa: BLE001
            logger.exception("LLM concept extraction failed; using deterministic fallback")
        if not candidates:
            candidates = self._fallback_candidates(chunks)

        existing = await self._load_all_concepts(session, conversation_id)
        concepts = self._merge_candidates(conversation_id, chunks, candidates)
        self._preserve_existing_canonical_names(concepts, existing)
        return await self._persist_concepts(session, conversation_id, concepts)

    async def load_concepts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[CourseConceptRecord]:
        await self.ensure_schema(session)
        result = await session.execute(
            select(CourseConceptRecord)
            .where(CourseConceptRecord.conversation_id == conversation_id)
            .order_by(CourseConceptRecord.importance.desc(), CourseConceptRecord.canonical_name)
        )
        return [
            concept
            for concept in result.scalars().all()
            if _active_course_concept(concept)
            if _valid_learning_concept_name(
                concept.canonical_name,
                f"{concept.description} {' '.join(concept.aliases or [])}",
            )
        ]

    async def resolve_names(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        names: list[str],
    ) -> dict[str, CourseConceptRecord | None]:
        concepts = await self.load_concepts(session, conversation_id)
        return {name: resolve_concept(name, concepts) for name in names}

    async def ensure_schema(self, session: AsyncSession) -> None:
        connection = await session.connection()

        def create_table(sync_connection) -> None:  # noqa: ANN001
            CourseConceptRecord.__table__.create(sync_connection, checkfirst=True)
            KnowledgeCheckRecord.__table__.create(sync_connection, checkfirst=True)
            KnowledgeAttemptRecord.__table__.create(sync_connection, checkfirst=True)

        await connection.run_sync(create_table)

    async def _load_chunks(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[SearchChunkRecord]:
        result = await session.execute(
            select(SearchChunkRecord)
            .where(SearchChunkRecord.conversation_id == conversation_id)
            .order_by(SearchChunkRecord.document_id, SearchChunkRecord.chunk_index)
        )
        return list(result.scalars().all())

    async def _load_all_concepts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[CourseConceptRecord]:
        result = await session.execute(
            select(CourseConceptRecord)
            .where(CourseConceptRecord.conversation_id == conversation_id)
            .order_by(CourseConceptRecord.importance.desc(), CourseConceptRecord.canonical_name)
        )
        return list(result.scalars().all())

    async def _persist_concepts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        concepts: list[_ConceptAccumulator],
    ) -> list[CourseConceptRecord]:
        existing = await self._load_all_concepts(session, conversation_id)
        existing_by_id = {concept.id: concept for concept in existing}
        reference_counts = await self._assessment_reference_counts(session, conversation_id)
        desired_records = [self._to_record(conversation_id, concept) for concept in concepts]
        desired_ids = {record.id for record in desired_records}
        persisted: list[CourseConceptRecord] = []

        for desired in desired_records:
            current = existing_by_id.get(desired.id)
            if current is None:
                session.add(desired)
                persisted.append(desired)
            else:
                _copy_concept_record(current, desired)
                persisted.append(current)

        now = datetime.now(timezone.utc)
        for current in existing:
            if current.id in desired_ids:
                continue
            if reference_counts.get(current.id, 0) > 0:
                current.source_file_ids = []
                current.source_section_ids = []
                current.source_chunk_ids = []
                current.importance = 0.0
                current.concept_metadata = {
                    **dict(current.concept_metadata or {}),
                    "inactive": True,
                    "inactive_reason": "no_current_course_sources",
                }
                current.updated_at = now
            else:
                await session.delete(current)

        await session.flush()
        return persisted

    async def _assessment_reference_counts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> dict[uuid.UUID, int]:
        counts: dict[uuid.UUID, int] = {}
        checks = await session.execute(
            select(KnowledgeCheckRecord.concept_id, func.count())
            .where(KnowledgeCheckRecord.conversation_id == conversation_id)
            .group_by(KnowledgeCheckRecord.concept_id)
        )
        for concept_id, count in checks.all():
            counts[concept_id] = counts.get(concept_id, 0) + int(count or 0)

        attempts = await session.execute(
            select(KnowledgeAttemptRecord.concept_id, func.count())
            .where(KnowledgeAttemptRecord.conversation_id == conversation_id)
            .group_by(KnowledgeAttemptRecord.concept_id)
        )
        for concept_id, count in attempts.all():
            counts[concept_id] = counts.get(concept_id, 0) + int(count or 0)
        return counts

    async def _llm_candidates(
        self,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> list[ConceptCandidate]:
        if not chunks:
            return []

        last_error: Exception | None = None
        for label, client in self._llm_clients(llm_options):
            try:
                response = await client.chat_structured(
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _format_chunks_for_llm(chunks)},
                    ],
                    schema=ConceptCandidateBatch,
                    options={"temperature": 0.1, "num_predict": 1800, "max_tokens": 1800},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "concept extraction with %s model %s failed; trying fallback if available",
                    label,
                    client.model,
                    exc_info=True,
                )
                continue

            candidates = [
                item.model_copy(update={"aliases": _clean_aliases(item.canonical_name, item.aliases)})
                for item in response.concepts
                if _valid_learning_concept_name(item.canonical_name, _chunk_text_for_candidate(item, chunks))
            ][:_MAX_CONCEPTS]
            if candidates:
                return candidates
            logger.warning(
                "concept extraction with %s model %s returned no valid learning concepts",
                label,
                client.model,
            )

        if last_error is not None:
            raise last_error
        return []

    def _llm_clients(self, llm_options: dict[str, Any] | None) -> list[tuple[str, OllamaClient]]:
        raw_llm = llm_options.get("llm") if isinstance(llm_options, dict) else None
        primary = build_llm_client_kwargs(
            default_base_url=self._settings.ollama_host,
            default_model=self._settings.ollama_chat_model,
            options=raw_llm if isinstance(raw_llm, dict) else None,
        )
        clients = [
            (
                "configured",
                OllamaClient(
                    str(primary["base_url"]),
                    str(primary["model"]),
                    provider=str(primary["provider"]),
                    api_key=primary["api_key"],
                ),
            )
        ]
        if not (
            primary["provider"] == "ollama"
            and primary["base_url"] == self._settings.ollama_host
            and primary["model"] == _LOCAL_FALLBACK_MODEL
        ):
            clients.append(
                (
                    "local-fallback",
                    OllamaClient(
                        self._settings.ollama_host,
                        _LOCAL_FALLBACK_MODEL,
                        provider="ollama",
                    ),
                )
            )
        return clients

    def _fallback_candidates(self, chunks: list[SearchChunkRecord]) -> list[ConceptCandidate]:
        out: list[ConceptCandidate] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            metadata = chunk.chunk_metadata or {}
            values: list[tuple[str, str, float, str]] = []
            section_title = str(metadata.get("section_title") or "").strip()
            section_summary = str(metadata.get("section_summary") or "").strip()
            raw_concepts = metadata.get("key_concepts") or []
            if isinstance(raw_concepts, list):
                for item in raw_concepts:
                    text = str(item)
                    if _same_concept_label(text, section_title):
                        continue
                    values.append((text, _concept_description(text, section_summary, chunk.text), 0.7, "metadata_key_concept"))
            for match in _BOLD_TERM_RE.finditer(chunk.text):
                term = match.group(1).strip()
                values.append((term, _concept_description(term, section_summary, chunk.text), 0.72, "bold_term"))
            for line in chunk.text.splitlines():
                match = _DEFINITION_RE.match(line.strip())
                if match:
                    values.append((match.group(1), match.group(2)[:300], 0.82, "definition"))

            for name, description, importance, method in values:
                clean_name = _clean_name(name)
                candidate_aliases = [_acronym(clean_name)]
                if (
                    section_title
                    and _same_concept_label(clean_name, _acronym(section_title))
                    and _valid_learning_concept_name(section_title, chunk.text)
                ):
                    candidate_aliases.append(clean_name)
                    clean_name = _clean_name(section_title)
                if not _valid_learning_concept_name(clean_name, chunk.text):
                    continue
                key = (chunk.id, normalize_concept_key(clean_name))
                if key in seen:
                    continue
                seen.add(key)
                aliases = _clean_aliases(clean_name, candidate_aliases)
                out.append(
                    ConceptCandidate(
                        canonical_name=clean_name,
                        aliases=aliases,
                        description=description,
                        bloom_level="understand",
                        importance=importance,
                        source_chunk_ids=[chunk.id],
                        extraction_method=method,
                    )
                )
        return out

    def _merge_candidates(
        self,
        conversation_id: uuid.UUID,
        chunks: list[SearchChunkRecord],
        candidates: list[ConceptCandidate],
    ) -> list[_ConceptAccumulator]:
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        accumulators: list[_ConceptAccumulator] = []

        for candidate in candidates:
            name = _clean_name(candidate.canonical_name)
            candidate_text = " ".join(
                chunks_by_id[str(chunk_id)].text
                for chunk_id in candidate.source_chunk_ids
                if str(chunk_id) in chunks_by_id
            )
            if not _valid_learning_concept_name(name, candidate_text):
                continue
            aliases = _clean_aliases(name, [*candidate.aliases, _acronym(name)])
            concept = _ConceptAccumulator(
                canonical_name=name,
                aliases=aliases,
                description=str(candidate.description or "").strip()[:700],
                bloom_level=_coerce_bloom(candidate.bloom_level),
                importance=max(0.0, min(1.0, float(candidate.importance))),
                extraction_methods={
                    str(
                        getattr(candidate, "extraction_method", "")
                        or ("llm" if len(candidate.source_chunk_ids) != 1 else "fallback")
                    )
                },
            )
            for chunk_id in candidate.source_chunk_ids:
                chunk = chunks_by_id.get(str(chunk_id))
                if chunk is None:
                    continue
                concept.source_chunk_ids.add(chunk.id)
                concept.source_section_ids.add(str(chunk.section_id))
                concept.source_file_ids.add(chunk.source_file_id)
                part_title = _course_part_title(chunk)
                if part_title:
                    concept.source_part_titles.add(part_title)
            if not concept.source_chunk_ids and chunks:
                chunk = chunks[0]
                concept.source_chunk_ids.add(chunk.id)
                concept.source_section_ids.add(str(chunk.section_id))
                concept.source_file_ids.add(chunk.source_file_id)
                part_title = _course_part_title(chunk)
                if part_title:
                    concept.source_part_titles.add(part_title)

            existing = _find_merge_target(concept, accumulators)
            if existing is None:
                accumulators.append(concept)
            else:
                _merge_into(existing, concept)

        return sorted(
            accumulators,
            key=lambda item: (-item.importance, item.canonical_name.casefold()),
        )[:_MAX_CONCEPTS]

    @staticmethod
    def _preserve_existing_canonical_names(
        concepts: list[_ConceptAccumulator],
        existing: list[CourseConceptRecord],
    ) -> None:
        for concept in concepts:
            matched = resolve_concept(concept.canonical_name, existing)
            if matched is None:
                for alias in concept.aliases:
                    matched = resolve_concept(alias, existing)
                    if matched is not None:
                        break
            if matched is None:
                continue
            old_name = concept.canonical_name
            concept.canonical_name = matched.canonical_name
            concept.aliases = _dedupe_strings([
                *list(matched.aliases or []),
                old_name,
                *concept.aliases,
            ])

    @staticmethod
    def _to_record(
        conversation_id: uuid.UUID,
        concept: _ConceptAccumulator,
    ) -> CourseConceptRecord:
        now = datetime.now(timezone.utc)
        return CourseConceptRecord(
            id=stable_concept_id(conversation_id, concept.canonical_name),
            conversation_id=conversation_id,
            canonical_key=concept.key,
            canonical_name=concept.canonical_name,
            aliases=_dedupe_strings(concept.aliases),
            description=concept.description,
            bloom_level=concept.bloom_level,
            importance=concept.importance,
            source_file_ids=sorted(concept.source_file_ids),
            source_section_ids=sorted(concept.source_section_ids),
            source_chunk_ids=sorted(concept.source_chunk_ids),
            concept_metadata={
                "extraction_methods": sorted(concept.extraction_methods),
                "course_parts": _course_parts_for_concept(concept),
            },
            created_at=now,
            updated_at=now,
        )


def resolve_concept(name: str, concepts: list[CourseConceptRecord]) -> CourseConceptRecord | None:
    key = normalize_concept_key(name)
    if not key:
        return None

    alias_index: dict[str, CourseConceptRecord] = {}
    for concept in concepts:
        for label in [concept.canonical_name, *(concept.aliases or [])]:
            alias_key = normalize_concept_key(label)
            if alias_key:
                alias_index.setdefault(alias_key, concept)
    if key in alias_index:
        return alias_index[key]

    best: tuple[float, CourseConceptRecord] | None = None
    for alias_key, concept in alias_index.items():
        if len(key) >= 4 and len(alias_key) >= 4 and (key in alias_key or alias_key in key):
            score = min(len(key), len(alias_key)) / max(len(key), len(alias_key))
        else:
            score = SequenceMatcher(a=key, b=alias_key).ratio()
        if score >= 0.92 and (best is None or score > best[0]):
            best = (score, concept)
    return best[1] if best else None


def _active_course_concept(concept: CourseConceptRecord) -> bool:
    metadata = concept.concept_metadata or {}
    return not bool(metadata.get("inactive")) and bool(concept.source_chunk_ids)


def _copy_concept_record(target: CourseConceptRecord, source: CourseConceptRecord) -> None:
    metadata = dict(source.concept_metadata or {})
    metadata.pop("inactive", None)
    metadata.pop("inactive_reason", None)
    target.canonical_key = source.canonical_key
    target.canonical_name = source.canonical_name
    target.aliases = list(source.aliases or [])
    target.description = source.description
    target.bloom_level = source.bloom_level
    target.importance = source.importance
    target.source_file_ids = list(source.source_file_ids or [])
    target.source_section_ids = list(source.source_section_ids or [])
    target.source_chunk_ids = list(source.source_chunk_ids or [])
    target.concept_metadata = metadata
    target.updated_at = source.updated_at


def stable_concept_id(conversation_id: uuid.UUID | str, canonical_name: str) -> uuid.UUID:
    key = normalize_concept_key(canonical_name)
    seed = f"concept:{conversation_id}:{key}"
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:{seed}:{sha1(seed.encode()).hexdigest()[:12]}")


def normalize_concept_key(value: str) -> str:
    return " ".join(_WORD_RE.findall(_ascii_fold(value).casefold()))


def _ascii_fold(value: str) -> str:
    return unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")


def _find_merge_target(
    concept: _ConceptAccumulator,
    accumulators: list[_ConceptAccumulator],
) -> _ConceptAccumulator | None:
    keys = concept.alias_keys
    for existing in accumulators:
        if keys & existing.alias_keys:
            return existing
    return None


def _merge_into(target: _ConceptAccumulator, source: _ConceptAccumulator) -> None:
    if _same_concept_label(target.canonical_name, _acronym(source.canonical_name)):
        target.canonical_name = source.canonical_name
        target.description = source.description or target.description
        target.bloom_level = source.bloom_level
    elif _same_concept_label(source.canonical_name, _acronym(target.canonical_name)):
        pass
    elif source.importance > target.importance:
        target.canonical_name = _prefer_longer_course_name(target.canonical_name, source.canonical_name)
        target.description = source.description or target.description
        target.bloom_level = source.bloom_level
    target.aliases = _dedupe_strings([*target.aliases, source.canonical_name, *source.aliases])
    target.importance = max(target.importance, source.importance)
    target.source_file_ids.update(source.source_file_ids)
    target.source_section_ids.update(source.source_section_ids)
    target.source_chunk_ids.update(source.source_chunk_ids)
    target.source_part_titles.update(source.source_part_titles)
    target.extraction_methods.update(source.extraction_methods)


def _prefer_longer_course_name(left: str, right: str) -> str:
    if left.isupper() and len(right) > len(left):
        return right
    if right.isupper() and len(left) > len(right):
        return left
    return right if len(right) > len(left) else left


def _format_chunks_for_llm(chunks: list[SearchChunkRecord]) -> str:
    parts: list[str] = []
    for chunk in chunks[:48]:
        metadata = chunk.chunk_metadata or {}
        section_title = str(metadata.get("section_title") or "")
        raw_concepts = [
            str(item)
            for item in (metadata.get("key_concepts") or [])[:12]
            if not _same_concept_label(str(item), section_title)
        ]
        concepts = ", ".join(raw_concepts)
        text = " ".join(chunk.text.split())
        if len(text) > 1400:
            text = text[:1400].rsplit(" ", 1)[0].strip()
        parts.append(
            "\n".join(
                [
                    f"chunk_id: {chunk.id}",
                    f"source: {chunk.source_filename}",
                    f"course_part: {metadata.get('heading_path') or ''}",
                    f"candidate_terms_under_part: {concepts}",
                    "text:",
                    text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _clean_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -*#:;")
    if ">" in text:
        parts = [part.strip() for part in text.split(">") if part.strip()]
        text = parts[-1] if parts else text
    text = re.sub(
        r"^(?:rappel|focus technique|focus|exemple num[ée]rique|exemple|"
        r"cas d['’]usage|point cl[ée]|id[ée]e principale)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text[:120]


def _clean_aliases(canonical_name: str, aliases: list[str]) -> list[str]:
    canonical_key = normalize_concept_key(canonical_name)
    out: list[str] = []
    for alias in aliases:
        text = _clean_name(alias)
        key = normalize_concept_key(text)
        if not key or key == canonical_key or len(text) > 80:
            continue
        out.append(text)
    return _dedupe_strings(out)[:12]


def _valid_concept_name(value: str) -> bool:
    text = _clean_name(value)
    key = normalize_concept_key(text)
    if not 2 <= len(text) <= 120 or not key:
        return False
    if _NOISE_RE.search(text):
        return False
    return True


def _valid_learning_concept_name(value: str, context: str = "") -> bool:
    text = _clean_name(value)
    ascii_text = _ascii_fold(text)
    key = normalize_concept_key(text)
    words = key.split()
    if not _valid_concept_name(text):
        return False
    if (
        _BAD_NAME_CHARS_RE.search(text)
        or _BAD_MARKDOWN_RE.search(text)
        or _LIST_OR_EXAMPLE_RE.search(ascii_text)
        or _BAD_PHRASE_RE.search(ascii_text)
        or _INCOMPLETE_PAREN_RE.search(text)
        or "%" in text
        or "=" in text
    ):
        return False
    if _SECTION_LABEL_RE.sub("", text).strip() != text:
        text = _SECTION_LABEL_RE.sub("", text).strip()
        ascii_text = _ascii_fold(text)
        key = normalize_concept_key(text)
        words = key.split()
    if not text or _FRAGMENT_END_RE.search(text):
        return False
    if len(words) == 1:
        token = words[0]
        if len(token) < 3:
            return False
        raw = str(value).strip()
        compact = re.sub(r"[^A-Za-z0-9]", "", raw)
        acronym_like = bool(compact) and any(ch.isupper() for ch in compact[1:]) and 2 <= len(compact) <= 10
        title_word = raw[:1].isupper() and not raw.isupper()
        mentioned_in_context = bool(token and token in normalize_concept_key(context))
        if not (raw.isupper() and 2 <= len(raw) <= 10 or acronym_like or (title_word and mentioned_in_context)):
            return False
    if len(words) > 8:
        return False
    if _looks_like_sentence_fragment(text):
        title_phrase = 2 <= len(words) <= 4 and any(word[:1].isupper() for word in text.split())
        if not title_phrase:
            return False
    is_acronym = str(value).strip().isupper() and 2 <= len(str(value).strip()) <= 10
    if is_acronym:
        return True
    content_words = [word for word in words if len(word) > 2]
    if not content_words:
        return False
    if len(words) <= 4:
        return True
    if len(words) <= 8 and any(word[:1].isupper() for word in text.split()):
        return True
    return bool(context and key in normalize_concept_key(context))


def _looks_like_sentence_fragment(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    if any(ch in text for ch in ".?!;"):
        return True
    words = text.split()
    if len(words) >= 6 and text[:1].islower():
        return True
    stop_start = {
        "and", "or", "but", "because", "with", "without", "for", "to",
        "et", "ou", "avec", "sans", "pour", "dans", "de", "des", "du",
    }
    return words[0].casefold() in stop_start if words else True


def _same_concept_label(left: str, right: str) -> bool:
    return bool(left and right and normalize_concept_key(left) == normalize_concept_key(right))


def _should_promote_section_title(
    section_title: str,
    text: str,
    raw_concepts: object,
) -> bool:
    return False


def _concept_description(term: str, section_summary: str, chunk_text: str) -> str:
    for line in chunk_text.splitlines():
        if normalize_concept_key(term) and normalize_concept_key(term) in normalize_concept_key(line):
            text = " ".join(line.split())
            if len(text) > 24:
                return text[:300]
    return (section_summary or "Key learning concept from the uploaded course part.")[:300]


def _chunk_text_for_candidate(
    candidate: ConceptCandidate,
    chunks: list[SearchChunkRecord],
) -> str:
    by_id = {chunk.id: chunk for chunk in chunks}
    text = " ".join(
        by_id[str(chunk_id)].text for chunk_id in candidate.source_chunk_ids if str(chunk_id) in by_id
    )
    if text:
        return text
    return " ".join(chunk.text[:700] for chunk in chunks[:8])


def _course_parts_for_concept(concept: _ConceptAccumulator) -> list[dict[str, str]]:
    sections = sorted(concept.source_section_ids)
    titles = sorted(concept.source_part_titles)
    out: list[dict[str, str]] = []
    for index, section_id in enumerate(sections):
        out.append(
            {
                "section_id": section_id,
                "title": titles[index] if index < len(titles) else "",
            }
        )
    return out


def _course_part_title(chunk: SearchChunkRecord) -> str:
    metadata = chunk.chunk_metadata or {}
    heading = str(metadata.get("heading_path") or "").strip()
    if heading:
        return heading
    title = str(metadata.get("section_title") or "").strip()
    return title


def _acronym(value: str) -> str:
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z]+", value) if word[:1].isalpha()]
    if len(words) < 2:
        return ""
    acronym = "".join(word[0].upper() for word in words)
    return acronym if 2 <= len(acronym) <= 8 else ""


def _coerce_bloom(value: str) -> str:
    text = str(value or "understand").strip().lower()
    return text if text in _BLOOM_LEVELS else "understand"


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _clean_name(value)
        key = normalize_concept_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


_SYSTEM_PROMPT = """You extract a stable course concept inventory for TeacherLM.

Think like a teacher building a progress checklist. The uploaded course is
organized into parts/sections, and each part contains learnable concepts.
Return only the concepts a student must understand to say they understood that
part of the course.

Rules:
- Use the exact chunk_id values for source_chunk_ids.
- Merge synonyms and abbreviations into aliases, e.g. SVD and singular value decomposition.
- canonical_name should be a short noun phrase students would recognize.
- The course can be about any domain. Do not require machine-learning,
  recommender-system, mathematical, or technical vocabulary.
- Do NOT return course parts, slide titles, agenda labels, week/session names,
  file titles, incomplete phrases, ordinary words, examples, or administrative text.
- Do NOT return equations, variables, vector symbols, percentages, numbered
  steps, positions/ranks, example users/items/docs, groups, or calculation rows.
- A good concept can be tested with a question such as "explain X", "apply X",
  or "compare X with Y". If that would be meaningless, exclude it.
- bloom_level must be one of remember, understand, apply, analyze.
- importance is 0.0 to 1.0 based on how central the concept is.
- Return only JSON matching the schema."""


_service: ConceptInventoryService | None = None


def get_concept_inventory_service() -> ConceptInventoryService:
    global _service
    if _service is None:
        _service = ConceptInventoryService()
    return _service

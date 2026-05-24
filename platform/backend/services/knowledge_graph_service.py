from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs

from config import Settings, get_settings
from db.models import (
    CourseConceptRecord,
    CourseDocumentRecord,
    CourseGraphRebuildRecord,
    CourseKnowledgeEdgeRecord,
    CourseKnowledgeNodeRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    CourseSectionRecord,
    KnowledgeAttemptRecord,
    KnowledgeCheckRecord,
    SearchChunkRecord,
)
from schemas.knowledge_graph import (
    KnowledgeGraphEdgeRead,
    KnowledgeGraphNodeRead,
    KnowledgeGraphRead,
    RemediationPath,
    RemediationStep,
)
from services.concept_inventory_service import get_concept_inventory_service
from services.learning_map_service import get_learning_map_service, normalize_learning_key
from services.learner_tracker import get_learner_tracker


logger = logging.getLogger(__name__)

NodeType = Literal[
    "course",
    "file",
    "section",
    "chunk",
    "phase",
    "objective",
    "concept",
    "skill",
    "procedure",
    "formula",
    "example",
    "misconception",
    "assessment",
]
EdgeType = Literal[
    "part_of",
    "teaches",
    "requires",
    "prerequisite_of",
    "supports",
    "explains",
    "applies",
    "example_of",
    "formula_for",
    "contrasts_with",
    "causes",
    "solves",
    "assessed_by",
    "remediates",
]

NODE_TYPES: set[str] = set(NodeType.__args__)  # type: ignore[attr-defined]
EDGE_TYPES: set[str] = set(EdgeType.__args__)  # type: ignore[attr-defined]
PREREQUISITE_RELATIONS = {"requires", "prerequisite_of"}
_LOCAL_FALLBACK_MODEL = "gemma4:e2b"


class _GraphNodeCandidate(BaseModel):
    node_type: str
    label: str
    description: str = ""
    ref_name: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)


class _GraphEdgeCandidate(BaseModel):
    source_label: str
    target_label: str
    relation_type: str
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)


class _GraphCandidateBatch(BaseModel):
    nodes: list[_GraphNodeCandidate] = Field(default_factory=list)
    edges: list[_GraphEdgeCandidate] = Field(default_factory=list)


@dataclass(slots=True)
class _NodeDraft:
    conversation_id: uuid.UUID
    node_type: str
    key: str
    label: str
    description: str = ""
    ref_id: str | None = None
    source_chunk_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> uuid.UUID:
        return stable_node_id(self.conversation_id, self.node_type, self.key)


@dataclass(slots=True)
class _EdgeDraft:
    conversation_id: uuid.UUID
    source_node_id: uuid.UUID
    target_node_id: uuid.UUID
    relation_type: str
    confidence: float = 0.6
    source_chunk_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> uuid.UUID:
        return stable_edge_id(
            self.conversation_id,
            self.source_node_id,
            self.target_node_id,
            self.relation_type,
        )


class KnowledgeGraphService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def ensure_schema(self, session: AsyncSession) -> None:
        connection = await session.connection()

        def create_tables(sync_connection) -> None:  # noqa: ANN001
            CourseKnowledgeNodeRecord.__table__.create(sync_connection, checkfirst=True)
            CourseKnowledgeEdgeRecord.__table__.create(sync_connection, checkfirst=True)
            CourseGraphRebuildRecord.__table__.create(sync_connection, checkfirst=True)

        await connection.run_sync(create_tables)

    async def rebuild_graph(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> KnowledgeGraphRead:
        await self.ensure_schema(session)
        now = datetime.now(timezone.utc)
        inputs = await _load_graph_inputs(session, conversation_id)
        node_drafts, edge_drafts = self._fallback_graph(conversation_id, inputs)

        llm_used = False
        if use_llm and inputs["chunks"] and inputs["concepts"]:
            try:
                llm_nodes, llm_edges = await self._llm_graph(conversation_id, inputs, llm_options=llm_options)
                node_drafts, edge_drafts = _merge_graph_drafts(node_drafts, edge_drafts, llm_nodes, llm_edges)
                llm_used = True
            except Exception:  # noqa: BLE001
                logger.exception("knowledge graph LLM extraction failed; using fallback graph")

        nodes = await self._persist_nodes(session, conversation_id, node_drafts, now)
        edges = await self._persist_edges(session, conversation_id, edge_drafts, now)
        session.add(
            CourseGraphRebuildRecord(
                conversation_id=conversation_id,
                status="completed",
                node_count=len(nodes),
                edge_count=len(edges),
                rebuild_metadata={"llm_used": llm_used},
                created_at=now,
            )
        )
        await session.flush()
        return await self.get_graph(session, conversation_id)

    async def get_graph(self, session: AsyncSession, conversation_id: uuid.UUID) -> KnowledgeGraphRead:
        await self.ensure_schema(session)
        nodes_result = await session.execute(
            select(CourseKnowledgeNodeRecord)
            .where(CourseKnowledgeNodeRecord.conversation_id == conversation_id, CourseKnowledgeNodeRecord.active.is_(True))
            .order_by(CourseKnowledgeNodeRecord.node_type, CourseKnowledgeNodeRecord.label)
        )
        nodes = list(nodes_result.scalars().all())
        edges_result = await session.execute(
            select(CourseKnowledgeEdgeRecord)
            .where(CourseKnowledgeEdgeRecord.conversation_id == conversation_id, CourseKnowledgeEdgeRecord.active.is_(True))
            .order_by(CourseKnowledgeEdgeRecord.relation_type)
        )
        edges = list(edges_result.scalars().all())
        return KnowledgeGraphRead(
            conversation_id=conversation_id,
            nodes=[_node_read(node) for node in nodes],
            edges=[_edge_read(edge) for edge in edges],
            node_count=len(nodes),
            edge_count=len(edges),
        )

    async def remediation_for_concept(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        concept_id: uuid.UUID,
        *,
        max_steps: int = 4,
    ) -> RemediationPath | None:
        await self.ensure_schema(session)
        graph = await self._load_active_graph_records(session, conversation_id)
        nodes, edges = graph
        target = _concept_node(nodes, concept_id)
        if target is None:
            return None
        concept_by_node = _concept_nodes_by_id(nodes)
        state = await get_learner_tracker().load_state(session, conversation_id)
        progress = {item.concept_id: item.mastery for item in state.concept_progress}
        prereq_ids = _prerequisite_node_ids(target.id, edges, depth=2)
        steps: list[RemediationStep] = []
        for node_id in prereq_ids:
            node = concept_by_node.get(node_id)
            if node is None:
                continue
            mastery = progress.get(str(node.ref_id), 0.0)
            if mastery >= 0.7:
                continue
            steps.append(
                RemediationStep(
                    concept_id=uuid.UUID(str(node.ref_id)) if node.ref_id else None,
                    concept_name=node.label,
                    mastery=mastery,
                    reason=f"{node.label} is a prerequisite for {target.label}.",
                    source_chunk_ids=list(node.source_chunk_ids or []),
                )
            )
            if len(steps) >= max_steps:
                break
        if not steps:
            return None
        return RemediationPath(
            target_concept_id=concept_id,
            target_concept_name=target.label,
            steps=steps,
        )

    async def remediation_for_wrong_answer(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        concept_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        path = await self.remediation_for_concept(session, conversation_id, concept_id)
        if path is None:
            return []
        return [path.model_dump(mode="json")]

    async def concept_prerequisites(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        concept_ids: list[str],
    ) -> dict[str, list[str]]:
        nodes, edges = await self._load_active_graph_records(session, conversation_id)
        by_ref = {str(node.ref_id): node for node in nodes if node.node_type == "concept" and node.ref_id}
        out: dict[str, list[str]] = {}
        for concept_id in concept_ids:
            node = by_ref.get(str(concept_id))
            if node is None:
                out[str(concept_id)] = []
                continue
            prereqs = []
            for prereq_node_id in _prerequisite_node_ids(node.id, edges, depth=1):
                prereq = next((item for item in nodes if item.id == prereq_node_id), None)
                if prereq and prereq.ref_id:
                    prereqs.append(str(prereq.ref_id))
            out[str(concept_id)] = _dedupe(prereqs)
        return out

    async def graph_hints_for_lesson(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        concept_ids: list[str],
    ) -> dict[str, Any]:
        nodes, edges = await self._load_active_graph_records(session, conversation_id)
        by_ref = {str(node.ref_id): node for node in nodes if node.node_type == "concept" and node.ref_id}
        concept_nodes = [by_ref[item] for item in concept_ids if item in by_ref]
        prereq_node_ids: list[uuid.UUID] = []
        next_node_ids: list[uuid.UUID] = []
        related_example_ids: list[uuid.UUID] = []
        for node in concept_nodes:
            prereq_node_ids.extend(_prerequisite_node_ids(node.id, edges, depth=1))
            next_node_ids.extend(
                edge.target_node_id
                for edge in edges
                if edge.source_node_id == node.id and edge.relation_type == "prerequisite_of"
            )
            related_example_ids.extend(
                edge.source_node_id
                for edge in edges
                if edge.target_node_id == node.id and edge.relation_type in {"example_of", "formula_for"}
            )
        node_by_id = {node.id: node for node in nodes}
        return {
            "prerequisite_concept_ids": [
                str(node.ref_id)
                for raw_id in _dedupe(str(item) for item in prereq_node_ids)
                if (node := node_by_id.get(uuid.UUID(raw_id))) is not None and node.ref_id
            ],
            "next_concept_ids": [
                str(node.ref_id)
                for raw_id in _dedupe(str(item) for item in next_node_ids)
                if (node := node_by_id.get(uuid.UUID(raw_id))) is not None and node.ref_id
            ],
            "related_example_ids": [str(item) for item in _dedupe(str(item) for item in related_example_ids)],
            "prerequisites": [
                node_by_id[uuid.UUID(raw_id)].label
                for raw_id in _dedupe(str(item) for item in prereq_node_ids)
                if uuid.UUID(raw_id) in node_by_id
            ],
            "next": [
                node_by_id[uuid.UUID(raw_id)].label
                for raw_id in _dedupe(str(item) for item in next_node_ids)
                if uuid.UUID(raw_id) in node_by_id
            ],
            "related_examples": [
                _example_hint(node_by_id[uuid.UUID(raw_id)])
                for raw_id in _dedupe(str(item) for item in related_example_ids)
                if uuid.UUID(raw_id) in node_by_id
            ],
        }

    async def graph_related_chunk_ids(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID | str,
        chunk_ids: list[str],
        *,
        limit: int = 4,
    ) -> list[str]:
        nodes, edges = await self._load_active_graph_records(session, uuid.UUID(str(conversation_id)))
        wanted = set(chunk_ids)
        chunk_node_ids = {node.id for node in nodes if node.node_type == "chunk" and node.ref_id in wanted}
        related_node_ids = set(chunk_node_ids)
        for edge in edges:
            if edge.source_node_id in chunk_node_ids:
                related_node_ids.add(edge.target_node_id)
            if edge.target_node_id in chunk_node_ids:
                related_node_ids.add(edge.source_node_id)
        node_by_id = {node.id: node for node in nodes}
        related_chunks: list[str] = []
        for node_id in related_node_ids:
            node = node_by_id.get(node_id)
            if node is None:
                continue
            if node.node_type == "chunk" and node.ref_id:
                related_chunks.append(node.ref_id)
            related_chunks.extend(str(item) for item in node.source_chunk_ids or [])
        return [item for item in _dedupe(related_chunks) if item not in wanted][:limit]

    async def _load_active_graph_records(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> tuple[list[CourseKnowledgeNodeRecord], list[CourseKnowledgeEdgeRecord]]:
        await self.ensure_schema(session)
        nodes = list(
            (
                await session.execute(
                    select(CourseKnowledgeNodeRecord).where(
                        CourseKnowledgeNodeRecord.conversation_id == conversation_id,
                        CourseKnowledgeNodeRecord.active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        edges = list(
            (
                await session.execute(
                    select(CourseKnowledgeEdgeRecord).where(
                        CourseKnowledgeEdgeRecord.conversation_id == conversation_id,
                        CourseKnowledgeEdgeRecord.active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        return nodes, edges

    async def _llm_graph(
        self,
        conversation_id: uuid.UUID,
        inputs: dict[str, list[Any]],
        *,
        llm_options: dict[str, Any] | None,
    ) -> tuple[list[_NodeDraft], list[_EdgeDraft]]:
        concepts: list[CourseConceptRecord] = inputs["concepts"]
        chunks: list[SearchChunkRecord] = inputs["chunks"]
        concept_lines = "\n".join(
            f"- {concept.canonical_name}: {concept.description[:160]}"
            for concept in concepts[:80]
        )
        source_text = "\n\n".join(
            f"[{chunk.id}] {' '.join(chunk.text.split())[:700]}"
            for chunk in chunks[:40]
        )
        user_prompt = (
            f"Canonical course concepts:\n{concept_lines}\n\n"
            f"Course source chunks:\n{source_text}"
        )
        last_error: Exception | None = None
        for label, client in self._llm_clients(llm_options):
            try:
                batch = await client.chat_structured(
                    messages=[
                        {"role": "system", "content": _GRAPH_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=_GraphCandidateBatch,
                    options={"temperature": 0.05, "num_predict": 1800, "max_tokens": 1800},
                )
                return self._candidate_graph(conversation_id, batch, inputs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("knowledge graph extraction with %s model %s failed", label, client.model, exc_info=True)
        if last_error is not None:
            raise last_error
        return [], []

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
                    OllamaClient(self._settings.ollama_host, _LOCAL_FALLBACK_MODEL, provider="ollama"),
                )
            )
        return clients

    def _candidate_graph(
        self,
        conversation_id: uuid.UUID,
        batch: _GraphCandidateBatch,
        inputs: dict[str, list[Any]],
    ) -> tuple[list[_NodeDraft], list[_EdgeDraft]]:
        concepts: list[CourseConceptRecord] = inputs["concepts"]
        concept_by_label = {_norm(concept.canonical_name): concept for concept in concepts}
        for concept in concepts:
            for alias in concept.aliases or []:
                concept_by_label.setdefault(_norm(alias), concept)

        nodes: dict[tuple[str, str], _NodeDraft] = {}
        for candidate in batch.nodes:
            node_type = candidate.node_type.strip().lower()
            if node_type not in {"skill", "procedure", "formula", "example", "misconception"}:
                continue
            label = _clean_label(candidate.label)
            if not label:
                continue
            key = normalize_learning_key(f"{node_type}:{label}")
            nodes[(node_type, key)] = _NodeDraft(
                conversation_id=conversation_id,
                node_type=node_type,
                key=key,
                label=label,
                description=_clean_label(candidate.description)[:800],
                source_chunk_ids=set(str(item) for item in candidate.source_chunk_ids),
                metadata={"extraction": "llm", "ref_name": candidate.ref_name},
            )

        existing_nodes = self._fallback_graph(conversation_id, inputs)[0]
        all_nodes = {(_norm(node.label), node.node_type): node for node in existing_nodes}
        for node in nodes.values():
            all_nodes[(_norm(node.label), node.node_type)] = node
        label_index = {_norm(node.label): node for node in all_nodes.values()}
        for concept in concepts:
            concept_node = next(
                (node for node in existing_nodes if node.node_type == "concept" and node.ref_id == str(concept.id)),
                None,
            )
            if concept_node:
                label_index[_norm(concept.canonical_name)] = concept_node

        edges: list[_EdgeDraft] = []
        for candidate in batch.edges:
            relation = candidate.relation_type.strip().lower()
            if relation not in EDGE_TYPES:
                continue
            source = label_index.get(_norm(candidate.source_label))
            target = label_index.get(_norm(candidate.target_label))
            if source is None:
                source_concept = concept_by_label.get(_norm(candidate.source_label))
                source = next((node for node in existing_nodes if source_concept and node.ref_id == str(source_concept.id)), None)
            if target is None:
                target_concept = concept_by_label.get(_norm(candidate.target_label))
                target = next((node for node in existing_nodes if target_concept and node.ref_id == str(target_concept.id)), None)
            if source is None or target is None or source.id == target.id:
                continue
            edges.append(
                _EdgeDraft(
                    conversation_id=conversation_id,
                    source_node_id=source.id,
                    target_node_id=target.id,
                    relation_type=relation,
                    confidence=float(candidate.confidence),
                    source_chunk_ids=set(str(item) for item in candidate.source_chunk_ids),
                    metadata={"extraction": "llm"},
                )
            )
        return list(nodes.values()), edges

    def _fallback_graph(
        self,
        conversation_id: uuid.UUID,
        inputs: dict[str, list[Any]],
    ) -> tuple[list[_NodeDraft], list[_EdgeDraft]]:
        nodes: dict[tuple[str, str], _NodeDraft] = {}
        edges: list[_EdgeDraft] = []

        course = _add_node(nodes, conversation_id, "course", "course", "Course", metadata={"source": "fallback"})
        documents: list[CourseDocumentRecord] = inputs["documents"]
        sections: list[CourseSectionRecord] = inputs["sections"]
        chunks: list[SearchChunkRecord] = inputs["chunks"]
        concepts: list[CourseConceptRecord] = inputs["concepts"]
        phases: list[CourseLearningPhaseRecord] = inputs["phases"]
        objectives: list[CourseLearningObjectiveRecord] = inputs["objectives"]

        file_nodes: dict[str, _NodeDraft] = {}
        for doc in documents:
            node = _add_node(
                nodes,
                conversation_id,
                "file",
                str(doc.id),
                doc.source_filename,
                ref_id=str(doc.id),
                metadata={"source_file_id": doc.source_file_id, "source": "fallback"},
            )
            file_nodes[str(doc.id)] = node
            edges.append(_edge(conversation_id, node, course, "part_of", confidence=1.0))

        section_nodes: dict[str, _NodeDraft] = {}
        for section in sections:
            node = _add_node(
                nodes,
                conversation_id,
                "section",
                str(section.id),
                section.title,
                description=section.summary,
                ref_id=str(section.id),
                metadata={"source": "fallback", "heading_path": section.heading_path},
            )
            section_nodes[str(section.id)] = node
            parent = section_nodes.get(str(section.parent_section_id)) if section.parent_section_id else file_nodes.get(str(section.document_id))
            if parent:
                edges.append(_edge(conversation_id, node, parent, "part_of", confidence=1.0))

        chunk_nodes: dict[str, _NodeDraft] = {}
        for chunk in chunks:
            label = " / ".join(str(item) for item in (chunk.heading_path or [])[-2:]) or f"Chunk {chunk.chunk_index + 1}"
            node = _add_node(
                nodes,
                conversation_id,
                "chunk",
                chunk.id,
                label,
                description=" ".join(chunk.text.split())[:300],
                ref_id=chunk.id,
                source_chunk_ids=[chunk.id],
                metadata={"source": "fallback", "source_filename": chunk.source_filename},
            )
            chunk_nodes[chunk.id] = node
            parent = section_nodes.get(str(chunk.section_id))
            if parent:
                edges.append(_edge(conversation_id, node, parent, "part_of", confidence=1.0))

        phase_nodes: dict[str, _NodeDraft] = {}
        for phase in phases:
            node = _add_node(
                nodes,
                conversation_id,
                "phase",
                str(phase.id),
                phase.title,
                description=phase.summary,
                ref_id=str(phase.id),
                source_chunk_ids=phase.source_chunk_ids,
                metadata={"source": "fallback", "order_index": phase.order_index},
            )
            phase_nodes[str(phase.id)] = node
            edges.append(_edge(conversation_id, node, course, "part_of", confidence=1.0))

        objective_nodes: dict[str, _NodeDraft] = {}
        for objective in objectives:
            node = _add_node(
                nodes,
                conversation_id,
                "objective",
                str(objective.id),
                objective.objective_text,
                ref_id=str(objective.id),
                source_chunk_ids=objective.source_chunk_ids,
                metadata={"source": "fallback", "order_index": objective.order_index, "bloom_level": objective.bloom_level},
            )
            objective_nodes[str(objective.id)] = node
            parent = phase_nodes.get(str(objective.phase_id))
            if parent:
                edges.append(_edge(conversation_id, node, parent, "part_of", confidence=1.0))

        concept_nodes: dict[str, _NodeDraft] = {}
        for concept in concepts:
            if (concept.concept_metadata or {}).get("inactive"):
                continue
            node = _add_node(
                nodes,
                conversation_id,
                "concept",
                str(concept.id),
                concept.canonical_name,
                description=concept.description,
                ref_id=str(concept.id),
                source_chunk_ids=concept.source_chunk_ids,
                metadata={"source": "fallback", "aliases": concept.aliases, "bloom_level": concept.bloom_level},
            )
            concept_nodes[str(concept.id)] = node
            for chunk_id in concept.source_chunk_ids or []:
                if chunk_id in chunk_nodes:
                    edges.append(_edge(conversation_id, node, chunk_nodes[chunk_id], "supports", confidence=0.9))

        for objective in objectives:
            objective_node = objective_nodes.get(str(objective.id))
            if objective_node is None:
                continue
            previous_concept: _NodeDraft | None = None
            for raw_id in objective.concept_ids or []:
                concept_node = concept_nodes.get(str(raw_id))
                if concept_node is None:
                    continue
                edges.append(_edge(conversation_id, objective_node, concept_node, "teaches", confidence=0.9))
                if previous_concept is not None:
                    edges.append(_edge(conversation_id, concept_node, previous_concept, "requires", confidence=0.45))
                previous_concept = concept_node

        return list(nodes.values()), _dedupe_edges(edges)

    async def _persist_nodes(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        drafts: list[_NodeDraft],
        now: datetime,
    ) -> list[CourseKnowledgeNodeRecord]:
        existing = {
            node.id: node
            for node in (
                await session.execute(
                    select(CourseKnowledgeNodeRecord).where(CourseKnowledgeNodeRecord.conversation_id == conversation_id)
                )
            ).scalars().all()
        }
        desired_ids = {draft.id for draft in drafts}
        records: list[CourseKnowledgeNodeRecord] = []
        for draft in drafts:
            record = existing.get(draft.id)
            if record is None:
                record = CourseKnowledgeNodeRecord(
                    id=draft.id,
                    conversation_id=conversation_id,
                    node_type=draft.node_type,
                    node_key=draft.key,
                    label=draft.label,
                    description=draft.description,
                    ref_id=draft.ref_id,
                    source_chunk_ids=sorted(draft.source_chunk_ids),
                    node_metadata=draft.metadata,
                    active=True,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
            else:
                record.node_type = draft.node_type
                record.node_key = draft.key
                record.label = draft.label
                record.description = draft.description
                record.ref_id = draft.ref_id
                record.source_chunk_ids = sorted(draft.source_chunk_ids)
                record.node_metadata = draft.metadata
                record.active = True
                record.updated_at = now
            records.append(record)
        for node_id, record in existing.items():
            if node_id not in desired_ids:
                record.active = False
                record.updated_at = now
        await session.flush()
        return records

    async def _persist_edges(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        drafts: list[_EdgeDraft],
        now: datetime,
    ) -> list[CourseKnowledgeEdgeRecord]:
        existing = {
            edge.id: edge
            for edge in (
                await session.execute(
                    select(CourseKnowledgeEdgeRecord).where(CourseKnowledgeEdgeRecord.conversation_id == conversation_id)
                )
            ).scalars().all()
        }
        desired_ids = {draft.id for draft in drafts}
        records: list[CourseKnowledgeEdgeRecord] = []
        for draft in drafts:
            record = existing.get(draft.id)
            if record is None:
                record = CourseKnowledgeEdgeRecord(
                    id=draft.id,
                    conversation_id=conversation_id,
                    source_node_id=draft.source_node_id,
                    target_node_id=draft.target_node_id,
                    relation_type=draft.relation_type,
                    confidence=draft.confidence,
                    source_chunk_ids=sorted(draft.source_chunk_ids),
                    edge_metadata=draft.metadata,
                    active=True,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
            else:
                record.source_node_id = draft.source_node_id
                record.target_node_id = draft.target_node_id
                record.relation_type = draft.relation_type
                record.confidence = draft.confidence
                record.source_chunk_ids = sorted(draft.source_chunk_ids)
                record.edge_metadata = draft.metadata
                record.active = True
                record.updated_at = now
            records.append(record)
        for edge_id, record in existing.items():
            if edge_id not in desired_ids:
                record.active = False
                record.updated_at = now
        await session.flush()
        return records


async def _load_graph_inputs(session: AsyncSession, conversation_id: uuid.UUID) -> dict[str, list[Any]]:
    phases, objectives = await get_learning_map_service().load_map(session, conversation_id)
    concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
    documents = list((await session.execute(select(CourseDocumentRecord).where(CourseDocumentRecord.conversation_id == conversation_id))).scalars().all())
    sections = list((await session.execute(select(CourseSectionRecord).where(CourseSectionRecord.conversation_id == conversation_id).order_by(CourseSectionRecord.order_index))).scalars().all())
    chunks = list((await session.execute(select(SearchChunkRecord).where(SearchChunkRecord.conversation_id == conversation_id).order_by(SearchChunkRecord.chunk_index))).scalars().all())
    return {
        "documents": documents,
        "sections": sections,
        "chunks": chunks,
        "concepts": concepts,
        "phases": phases,
        "objectives": objectives,
    }


def stable_node_id(conversation_id: uuid.UUID | str, node_type: str, key: str) -> uuid.UUID:
    seed = f"knowledge-node:{conversation_id}:{node_type}:{normalize_learning_key(key)}"
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:{seed}:{sha1(seed.encode()).hexdigest()[:12]}")


def stable_edge_id(
    conversation_id: uuid.UUID | str,
    source_node_id: uuid.UUID | str,
    target_node_id: uuid.UUID | str,
    relation_type: str,
) -> uuid.UUID:
    seed = f"knowledge-edge:{conversation_id}:{source_node_id}:{target_node_id}:{relation_type}"
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:{seed}:{sha1(seed.encode()).hexdigest()[:12]}")


def _add_node(
    nodes: dict[tuple[str, str], _NodeDraft],
    conversation_id: uuid.UUID,
    node_type: str,
    key: str,
    label: str,
    *,
    description: str = "",
    ref_id: str | None = None,
    source_chunk_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> _NodeDraft:
    clean_key = normalize_learning_key(f"{node_type}:{key}")
    map_key = (node_type, clean_key)
    current = nodes.get(map_key)
    if current is None:
        current = _NodeDraft(
            conversation_id=conversation_id,
            node_type=node_type,
            key=clean_key,
            label=_clean_label(label)[:512] or node_type.title(),
            description=_clean_label(description)[:1000],
            ref_id=ref_id,
            source_chunk_ids=set(str(item) for item in source_chunk_ids or []),
            metadata=metadata or {},
        )
        nodes[map_key] = current
    else:
        current.source_chunk_ids.update(str(item) for item in source_chunk_ids or [])
        current.metadata = {**current.metadata, **(metadata or {})}
    return current


def _edge(
    conversation_id: uuid.UUID,
    source: _NodeDraft,
    target: _NodeDraft,
    relation: str,
    *,
    confidence: float = 0.6,
    source_chunk_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> _EdgeDraft:
    chunks = set(source_chunk_ids or [])
    chunks.update(source.source_chunk_ids)
    chunks.update(target.source_chunk_ids)
    return _EdgeDraft(
        conversation_id=conversation_id,
        source_node_id=source.id,
        target_node_id=target.id,
        relation_type=relation,
        confidence=confidence,
        source_chunk_ids=chunks,
        metadata=metadata or {"source": "fallback"},
    )


def _merge_graph_drafts(
    base_nodes: list[_NodeDraft],
    base_edges: list[_EdgeDraft],
    extra_nodes: list[_NodeDraft],
    extra_edges: list[_EdgeDraft],
) -> tuple[list[_NodeDraft], list[_EdgeDraft]]:
    nodes = {node.id: node for node in base_nodes}
    for node in extra_nodes:
        if node.id in nodes:
            nodes[node.id].source_chunk_ids.update(node.source_chunk_ids)
            nodes[node.id].metadata = {**nodes[node.id].metadata, **node.metadata}
        else:
            nodes[node.id] = node
    return list(nodes.values()), _dedupe_edges([*base_edges, *extra_edges])


def _dedupe_edges(edges: list[_EdgeDraft]) -> list[_EdgeDraft]:
    by_id: dict[uuid.UUID, _EdgeDraft] = {}
    for edge in edges:
        current = by_id.get(edge.id)
        if current is None:
            by_id[edge.id] = edge
        else:
            current.confidence = max(current.confidence, edge.confidence)
            current.source_chunk_ids.update(edge.source_chunk_ids)
            current.metadata = {**current.metadata, **edge.metadata}
    return list(by_id.values())


def _node_read(node: CourseKnowledgeNodeRecord) -> KnowledgeGraphNodeRead:
    return KnowledgeGraphNodeRead(
        id=node.id,
        node_type=node.node_type,
        label=node.label,
        description=node.description,
        ref_id=node.ref_id,
        source_chunk_ids=list(node.source_chunk_ids or []),
        metadata=dict(node.node_metadata or {}),
    )


def _edge_read(edge: CourseKnowledgeEdgeRecord) -> KnowledgeGraphEdgeRead:
    return KnowledgeGraphEdgeRead(
        id=edge.id,
        source_node_id=edge.source_node_id,
        target_node_id=edge.target_node_id,
        relation_type=edge.relation_type,
        confidence=edge.confidence,
        source_chunk_ids=list(edge.source_chunk_ids or []),
        metadata=dict(edge.edge_metadata or {}),
    )


def _concept_node(nodes: list[CourseKnowledgeNodeRecord], concept_id: uuid.UUID) -> CourseKnowledgeNodeRecord | None:
    return next(
        (node for node in nodes if node.node_type == "concept" and str(node.ref_id) == str(concept_id)),
        None,
    )


def _concept_nodes_by_id(nodes: list[CourseKnowledgeNodeRecord]) -> dict[uuid.UUID, CourseKnowledgeNodeRecord]:
    return {node.id: node for node in nodes if node.node_type == "concept" and node.ref_id}


def _prerequisite_node_ids(
    target_node_id: uuid.UUID,
    edges: list[CourseKnowledgeEdgeRecord],
    *,
    depth: int,
) -> list[uuid.UUID]:
    result: list[uuid.UUID] = []
    frontier = [target_node_id]
    seen = {target_node_id}
    for _ in range(depth):
        next_frontier: list[uuid.UUID] = []
        for node_id in frontier:
            for edge in edges:
                prereq_id: uuid.UUID | None = None
                if edge.relation_type == "requires" and edge.source_node_id == node_id:
                    prereq_id = edge.target_node_id
                elif edge.relation_type == "prerequisite_of" and edge.target_node_id == node_id:
                    prereq_id = edge.source_node_id
                if prereq_id is None or prereq_id in seen:
                    continue
                seen.add(prereq_id)
                result.append(prereq_id)
                next_frontier.append(prereq_id)
        frontier = next_frontier
        if not frontier:
            break
    return result


def _clean_label(text: str) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split())


def _example_hint(node: CourseKnowledgeNodeRecord) -> str:
    description = _clean_label(node.description)
    if description:
        return f"{node.label}: {description}"
    return node.label


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_GRAPH_SYSTEM_PROMPT = """You extract a domain-general course knowledge graph from uploaded course material.
Return structured JSON only.

Create only meaningful teachable nodes and relationships. Prefer canonical concepts already listed by the platform.
Allowed node types: skill, procedure, formula, example, misconception.
Allowed relation types: requires, prerequisite_of, explains, applies, example_of, formula_for, contrasts_with, causes, solves, remediates.

Rules:
- Do not output titles, table fragments, raw variables, percentages, HTML, or incomplete phrases as concepts.
- Use source_chunk_ids exactly from the provided chunks when possible.
- Use requires/prerequisite_of only when one idea truly needs another.
- Link examples and formulas to the concept they teach.
- Include misconceptions only when the course material suggests a common confusion or limitation.
"""


_service: KnowledgeGraphService | None = None


def get_knowledge_graph_service() -> KnowledgeGraphService:
    global _service
    if _service is None:
        _service = KnowledgeGraphService()
    return _service

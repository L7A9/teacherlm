from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from teacherlm_core.schemas.chunk import Chunk

from local_api.db import get_store, utc_now


NODE_TYPES = {
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
}
EDGE_TYPES = {
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
}


@dataclass(slots=True)
class NodeDraft:
    conversation_id: str
    node_type: str
    key: str
    label: str
    description: str = ""
    ref_id: str | None = None
    source_chunk_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_node_id(self.conversation_id, self.node_type, self.key)


@dataclass(slots=True)
class EdgeDraft:
    conversation_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    confidence: float = 0.6
    source_chunk_ids: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_edge_id(
            self.conversation_id,
            self.source_node_id,
            self.target_node_id,
            self.relation_type,
        )


class KnowledgeGraphService:
    def rebuild_graph(self, conversation_id: str) -> dict[str, Any]:
        nodes, edges = self._fallback_graph(conversation_id)
        self._persist(conversation_id, nodes, edges)
        return self.get_graph(conversation_id)

    def get_graph(self, conversation_id: str) -> dict[str, Any]:
        nodes = [_node_read(row) for row in get_store().query(
            """
            SELECT * FROM knowledge_graph_nodes
            WHERE conversation_id = ? AND active = 1
            ORDER BY node_type ASC, label ASC
            """,
            (conversation_id,),
        )]
        edges = [_edge_read(row) for row in get_store().query(
            """
            SELECT * FROM knowledge_graph_edges
            WHERE conversation_id = ? AND active = 1
            ORDER BY relation_type ASC
            """,
            (conversation_id,),
        )]
        return {
            "conversation_id": conversation_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def graph_relevant_chunks(
        self,
        conversation_id: str,
        query: str,
        *,
        limit: int = 12,
        source_file_ids: list[str] | None = None,
    ) -> list[Chunk]:
        terms = _important_query_terms(query)
        if not terms:
            return []
        graph = self.get_graph(conversation_id)
        nodes = graph["nodes"]
        edges = graph["edges"]
        if not nodes:
            graph = self.rebuild_graph(conversation_id)
            nodes = graph["nodes"]
            edges = graph["edges"]
        node_scores = _score_graph_nodes(nodes, terms)
        if not node_scores:
            return []
        chunk_ids = _graph_chunk_ids(node_scores, nodes, edges, limit=max(limit * 2, 16))
        if not chunk_ids:
            return []
        chunks = get_store().list_chunks(conversation_id, source_file_ids=source_file_ids)
        by_id = {row["id"]: row for row in chunks}
        out: list[Chunk] = []
        for index, chunk_id in enumerate(chunk_ids):
            row = by_id.get(chunk_id)
            if row is None or _is_low_information_text(row.get("text", "")):
                continue
            metadata = dict(row.get("metadata", {}))
            metadata.update({"retrieval_via": "knowledge_graph"})
            out.append(
                Chunk(
                    text=row["text"],
                    source=row["source_filename"],
                    score=max(0.65, 1.0 - index * 0.03),
                    chunk_id=row["id"],
                    metadata=metadata,
                )
            )
            if len(out) >= limit:
                break
        return out

    def graph_related_chunk_ids(
        self,
        conversation_id: str,
        chunk_ids: list[str],
        *,
        limit: int = 4,
    ) -> list[str]:
        graph = self.get_graph(conversation_id)
        nodes = graph["nodes"]
        edges = graph["edges"]
        wanted = set(chunk_ids)
        chunk_node_ids = {
            node["id"]
            for node in nodes
            if node["node_type"] == "chunk" and node.get("ref_id") in wanted
        }
        related_node_ids = set(chunk_node_ids)
        for edge in edges:
            if edge["source_node_id"] in chunk_node_ids:
                related_node_ids.add(edge["target_node_id"])
            if edge["target_node_id"] in chunk_node_ids:
                related_node_ids.add(edge["source_node_id"])
        node_by_id = {node["id"]: node for node in nodes}
        related_chunks: list[str] = []
        for node_id in related_node_ids:
            node = node_by_id.get(node_id)
            if node is None:
                continue
            if node["node_type"] == "chunk" and node.get("ref_id"):
                related_chunks.append(str(node["ref_id"]))
            related_chunks.extend(str(item) for item in node.get("source_chunk_ids") or [])
        return [item for item in _dedupe(related_chunks) if item not in wanted][:limit]

    def mindmap_context_chunks(
        self,
        conversation_id: str,
        *,
        source_file_ids: list[str] | None = None,
    ) -> list[Chunk]:
        """Return the complete source-scoped graph for a fresh mind-map rebuild."""
        graph = self.get_graph(conversation_id)
        if not graph["nodes"]:
            graph = self.rebuild_graph(conversation_id)

        chunk_rows = get_store().list_chunks(conversation_id, source_file_ids=source_file_ids or None)
        if not chunk_rows:
            return []
        allowed_chunk_ids = {str(row["id"]) for row in chunk_rows}
        allowed_file_ids = {str(row["source_file_id"]) for row in chunk_rows if row.get("source_file_id")}
        scoped_nodes = [
            node
            for node in graph["nodes"]
            if _node_is_in_source_scope(node, allowed_chunk_ids, allowed_file_ids, bool(source_file_ids))
        ]
        scoped_node_ids = {str(node["id"]) for node in scoped_nodes}
        scoped_edges = [
            edge
            for edge in graph["edges"]
            if edge["source_node_id"] in scoped_node_ids and edge["target_node_id"] in scoped_node_ids
        ]
        scoped_nodes.sort(key=lambda node: (str(node.get("node_type") or ""), str(node.get("label") or "").casefold()))
        scoped_edges.sort(key=lambda edge: (str(edge.get("relation_type") or ""), str(edge.get("id") or "")))

        node_payload = [
            {
                "id": str(node["id"]),
                "label": str(node.get("label") or ""),
                "node_type": str(node.get("node_type") or "concept"),
                "description": str(node.get("description") or ""),
                "ref_id": node.get("ref_id"),
                "source_chunk_ids": [
                    str(chunk_id)
                    for chunk_id in node.get("source_chunk_ids") or []
                    if str(chunk_id) in allowed_chunk_ids
                ],
                "metadata": dict(node.get("metadata") or {}),
            }
            for node in scoped_nodes
            if str(node.get("label") or "").strip()
        ]
        node_by_id = {node["id"]: node for node in node_payload}
        edge_payload = [
            {
                "source_node_id": str(edge["source_node_id"]),
                "target_node_id": str(edge["target_node_id"]),
                "source_label": node_by_id[str(edge["source_node_id"])]["label"],
                "target_label": node_by_id[str(edge["target_node_id"])]["label"],
                "relation_type": str(edge.get("relation_type") or "supports"),
                "confidence": float(edge.get("confidence") or 0.0),
                "source_chunk_ids": [
                    str(chunk_id)
                    for chunk_id in edge.get("source_chunk_ids") or []
                    if str(chunk_id) in allowed_chunk_ids
                ],
                "metadata": dict(edge.get("metadata") or {}),
            }
            for edge in scoped_edges
            if str(edge["source_node_id"]) in node_by_id and str(edge["target_node_id"]) in node_by_id
        ]
        if not node_payload:
            return []

        lines = ["Complete knowledge graph for the selected source files:"]
        lines.extend(
            f"- [{node['node_type']}] {node['label']}"
            + (f": {node['description']}" if node["description"] else "")
            for node in node_payload
        )
        if edge_payload:
            lines.append("Knowledge graph relationships:")
            lines.extend(
                f"- {edge['source_label']} --{edge['relation_type']}--> {edge['target_label']}"
                for edge in edge_payload
            )
        scope_key = ",".join(sorted(allowed_file_ids)) or "all"
        scope_hash = hashlib.sha1(scope_key.encode("utf-8")).hexdigest()[:12]
        return [
            Chunk(
                text="\n".join(lines),
                source="knowledge_graph",
                score=1.0,
                chunk_id=f"mindmap-graph:{conversation_id}:{scope_hash}",
                metadata={
                    "context_type": "mindmap_graph_context",
                    "retrieval_via": "knowledge_graph",
                    "retrieval_mode": "graph_search",
                    "graph_search_strategy": "complete_source_scoped_course_graph",
                    "graph_complete": True,
                    "graph_nodes": node_payload,
                    "graph_edges": edge_payload,
                    "graph_node_count": len(node_payload),
                    "graph_edge_count": len(edge_payload),
                    "source_file_ids": sorted(allowed_file_ids),
                    "source_chunk_ids": sorted(allowed_chunk_ids),
                },
            )
        ]

    def _fallback_graph(self, conversation_id: str) -> tuple[list[NodeDraft], list[EdgeDraft]]:
        nodes: dict[tuple[str, str], NodeDraft] = {}
        edges: list[EdgeDraft] = []

        course = _add_node(nodes, conversation_id, "course", "course", "Course", metadata={"source": "fallback"})
        files = get_store().list_files(conversation_id)
        documents = get_store().query(
            "SELECT * FROM course_documents WHERE conversation_id = ?",
            (conversation_id,),
        )
        sections = get_store().query(
            "SELECT * FROM course_sections WHERE conversation_id = ? ORDER BY order_index ASC",
            (conversation_id,),
        )
        chunks = get_store().list_chunks(conversation_id)
        concepts = get_store().query(
            "SELECT * FROM concept_inventory WHERE conversation_id = ? ORDER BY name ASC",
            (conversation_id,),
        )
        phases = get_store().query(
            "SELECT * FROM learning_phases WHERE conversation_id = ? ORDER BY order_index ASC",
            (conversation_id,),
        )
        objectives = get_store().query(
            "SELECT * FROM learning_objectives WHERE conversation_id = ?",
            (conversation_id,),
        )

        file_by_source: dict[str, NodeDraft] = {}
        for file_row in files:
            node = _add_node(
                nodes,
                conversation_id,
                "file",
                file_row["id"],
                file_row["filename"],
                ref_id=file_row["id"],
                metadata={"source_file_id": file_row["id"], "source": "fallback"},
            )
            file_by_source[file_row["id"]] = node
            edges.append(_edge(conversation_id, node, course, "part_of", confidence=1.0))

        doc_nodes: dict[str, NodeDraft] = {}
        for doc in documents:
            parent = file_by_source.get(doc["source_file_id"], course)
            node = _add_node(
                nodes,
                conversation_id,
                "section",
                f"document:{doc['id']}",
                doc["title"],
                ref_id=doc["id"],
                metadata={"source_file_id": doc["source_file_id"], "source": "fallback"},
            )
            doc_nodes[doc["id"]] = node
            edges.append(_edge(conversation_id, node, parent, "part_of", confidence=1.0))

        section_nodes: dict[str, NodeDraft] = {}
        for section in sections:
            heading_path = _json(section.get("heading_path_json"), [])
            title = heading_path[-1] if heading_path else "Section"
            node = _add_node(
                nodes,
                conversation_id,
                "section",
                section["id"],
                title,
                description=section.get("summary") or "",
                ref_id=section["id"],
                metadata={
                    "source": "fallback",
                    "heading_path": heading_path,
                    "source_file_id": section.get("source_file_id"),
                },
            )
            section_nodes[section["id"]] = node
            parent = doc_nodes.get(section["document_id"]) or file_by_source.get(section["source_file_id"]) or course
            edges.append(_edge(conversation_id, node, parent, "part_of", confidence=1.0))

        chunk_nodes: dict[str, NodeDraft] = {}
        concept_chunk_ids: dict[str, set[str]] = {}
        concept_file_ids: dict[str, set[str]] = {}
        concept_section_ids: dict[str, set[str]] = {}
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            label = _chunk_label(chunk)
            node = _add_node(
                nodes,
                conversation_id,
                "chunk",
                chunk["id"],
                label,
                description=" ".join(str(chunk.get("text") or "").split())[:300],
                ref_id=chunk["id"],
                source_chunk_ids=[chunk["id"]],
                metadata={
                    "source": "fallback",
                    "source_filename": chunk["source_filename"],
                    "source_file_id": chunk["source_file_id"],
                },
            )
            chunk_nodes[chunk["id"]] = node
            parent = section_nodes.get(str(chunk.get("section_id"))) or file_by_source.get(chunk["source_file_id"]) or course
            edges.append(_edge(conversation_id, node, parent, "part_of", confidence=1.0))

            for concept in metadata.get("key_concepts") or []:
                key = _learning_key(str(concept))
                concept_chunk_ids.setdefault(key, set()).add(chunk["id"])
                concept_file_ids.setdefault(key, set()).add(chunk["source_file_id"])
                if chunk.get("section_id"):
                    concept_section_ids.setdefault(key, set()).add(str(chunk["section_id"]))

            for formula in _extract_formula_labels(chunk["text"]):
                formula_node = _add_node(
                    nodes,
                    conversation_id,
                    "formula",
                    f"{chunk['id']}:{formula}",
                    formula[:120],
                    description=formula[:500],
                    source_chunk_ids=[chunk["id"]],
                    metadata={"source": "fallback"},
                )
                edges.append(_edge(conversation_id, formula_node, node, "formula_for", confidence=0.75))

            if re.search(r"\bexample\b|\bfor example\b|\bexemple\b", chunk["text"], flags=re.IGNORECASE):
                example_node = _add_node(
                    nodes,
                    conversation_id,
                    "example",
                    chunk["id"],
                    f"Example in {label}",
                    description=" ".join(chunk["text"].split())[:500],
                    source_chunk_ids=[chunk["id"]],
                    metadata={"source": "fallback"},
                )
                edges.append(_edge(conversation_id, example_node, node, "example_of", confidence=0.7))

        concept_nodes: dict[str, NodeDraft] = {}
        for concept in concepts:
            name = str(concept["name"])
            key = _learning_key(name)
            source_chunk_ids = concept_chunk_ids.get(key, set())
            metadata = _json(concept.get("metadata_json"), {})
            aliases = _json(concept.get("aliases_json"), [])
            node = _add_node(
                nodes,
                conversation_id,
                "concept",
                key,
                name,
                description=str(metadata.get("description") or ""),
                ref_id=concept["id"],
                source_chunk_ids=sorted(source_chunk_ids),
                metadata={
                    "source": "fallback",
                    "aliases": aliases,
                    "source_file_ids": sorted(concept_file_ids.get(key, set())),
                    "source_section_ids": sorted(concept_section_ids.get(key, set())),
                },
            )
            concept_nodes[key] = node
            for chunk_id in source_chunk_ids:
                chunk_node = chunk_nodes.get(chunk_id)
                if chunk_node:
                    edges.append(_edge(conversation_id, node, chunk_node, "supports", confidence=0.9))

        previous_concept: NodeDraft | None = None
        for key in sorted(concept_nodes):
            current = concept_nodes[key]
            if previous_concept is not None:
                edges.append(_edge(conversation_id, current, previous_concept, "requires", confidence=0.45))
            previous_concept = current

        phase_nodes: dict[str, NodeDraft] = {}
        for phase in phases:
            node = _add_node(
                nodes,
                conversation_id,
                "phase",
                phase["id"],
                phase["title"],
                ref_id=phase["id"],
                metadata=_json(phase.get("metadata_json"), {}),
            )
            phase_nodes[phase["id"]] = node
            edges.append(_edge(conversation_id, node, course, "part_of", confidence=1.0))

        for objective in objectives:
            metadata = _json(objective.get("metadata_json"), {})
            node = _add_node(
                nodes,
                conversation_id,
                "objective",
                objective["id"],
                objective["objective_text"],
                ref_id=objective["id"],
                source_chunk_ids=[str(item) for item in metadata.get("source_chunk_ids", [])],
                metadata=metadata,
            )
            parent = phase_nodes.get(str(objective.get("phase_id"))) or course
            edges.append(_edge(conversation_id, node, parent, "part_of", confidence=1.0))
            for concept_id in metadata.get("concept_ids", []):
                concept_node = next((item for item in concept_nodes.values() if item.ref_id == str(concept_id)), None)
                if concept_node:
                    edges.append(_edge(conversation_id, node, concept_node, "teaches", confidence=0.9))

        return list(nodes.values()), _dedupe_edges(edges)

    def _persist(self, conversation_id: str, nodes: list[NodeDraft], edges: list[EdgeDraft]) -> None:
        now = utc_now()
        with get_store()._locked_conn() as conn:  # noqa: SLF001 - central store transaction helper.
            conn.execute("DELETE FROM knowledge_graph_edges WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM knowledge_graph_nodes WHERE conversation_id = ?", (conversation_id,))
            for node in nodes:
                conn.execute(
                    """
                    INSERT INTO knowledge_graph_nodes
                      (id, conversation_id, label, node_type, node_key, description, ref_id,
                       source_chunk_ids_json, active, created_at, updated_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        node.id,
                        conversation_id,
                        node.label,
                        node.node_type,
                        node.key,
                        node.description,
                        node.ref_id,
                        json.dumps(sorted(node.source_chunk_ids)),
                        now,
                        now,
                        json.dumps(node.metadata),
                    ),
                )
            for edge in edges:
                conn.execute(
                    """
                    INSERT INTO knowledge_graph_edges
                      (id, conversation_id, source_node_id, target_node_id, edge_type, relation_type,
                       confidence, source_chunk_ids_json, active, created_at, updated_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        edge.id,
                        conversation_id,
                        edge.source_node_id,
                        edge.target_node_id,
                        edge.relation_type,
                        edge.relation_type,
                        edge.confidence,
                        json.dumps(sorted(edge.source_chunk_ids)),
                        now,
                        now,
                        json.dumps(edge.metadata),
                    ),
                )
            conn.commit()


def stable_node_id(conversation_id: str, node_type: str, key: str) -> str:
    seed = f"knowledge-node:{conversation_id}:{node_type}:{_learning_key(key)}"
    return f"node_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:28]}"


def stable_edge_id(conversation_id: str, source_node_id: str, target_node_id: str, relation_type: str) -> str:
    seed = f"knowledge-edge:{conversation_id}:{source_node_id}:{target_node_id}:{relation_type}"
    return f"edge_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:28]}"


def _add_node(
    nodes: dict[tuple[str, str], NodeDraft],
    conversation_id: str,
    node_type: str,
    key: str,
    label: str,
    *,
    description: str = "",
    ref_id: str | None = None,
    source_chunk_ids: list[str] | set[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> NodeDraft:
    clean_type = node_type if node_type in NODE_TYPES else "concept"
    clean_key = _learning_key(f"{clean_type}:{key}")
    map_key = (clean_type, clean_key)
    current = nodes.get(map_key)
    if current is None:
        current = NodeDraft(
            conversation_id=conversation_id,
            node_type=clean_type,
            key=clean_key,
            label=_clean_label(label)[:512] or clean_type.title(),
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
    conversation_id: str,
    source: NodeDraft,
    target: NodeDraft,
    relation: str,
    *,
    confidence: float = 0.6,
    source_chunk_ids: list[str] | set[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EdgeDraft:
    clean_relation = relation if relation in EDGE_TYPES else "supports"
    chunks = set(str(item) for item in source_chunk_ids or [])
    chunks.update(source.source_chunk_ids)
    chunks.update(target.source_chunk_ids)
    return EdgeDraft(
        conversation_id=conversation_id,
        source_node_id=source.id,
        target_node_id=target.id,
        relation_type=clean_relation,
        confidence=confidence,
        source_chunk_ids=chunks,
        metadata=metadata or {"source": "fallback"},
    )


def _node_read(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _json(row.get("metadata_json"), {})
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "node_type": row["node_type"],
        "node_key": row.get("node_key") or "",
        "label": row["label"],
        "description": row.get("description") or "",
        "ref_id": row.get("ref_id"),
        "source_chunk_ids": _json(row.get("source_chunk_ids_json"), []),
        "metadata": metadata,
        "active": bool(row.get("active", 1)),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _edge_read(row: dict[str, Any]) -> dict[str, Any]:
    relation_type = row.get("relation_type") or row.get("edge_type") or "supports"
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "source_node_id": row["source_node_id"],
        "target_node_id": row["target_node_id"],
        "edge_type": relation_type,
        "relation_type": relation_type,
        "confidence": float(row.get("confidence") or 0.6),
        "source_chunk_ids": _json(row.get("source_chunk_ids_json"), []),
        "metadata": _json(row.get("metadata_json"), {}),
        "active": bool(row.get("active", 1)),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _score_graph_nodes(nodes: list[dict[str, Any]], terms: set[str]) -> list[tuple[float, dict[str, Any]]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for node in nodes:
        metadata = node.get("metadata") or {}
        aliases = metadata.get("aliases") if isinstance(metadata, dict) else []
        haystack = " ".join(
            [
                str(node.get("label") or ""),
                str(node.get("description") or ""),
                " ".join(str(item) for item in aliases or []),
            ]
        ).casefold()
        if not haystack.strip():
            continue
        exact_hits = sum(1 for term in terms if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", haystack))
        fuzzy_hits = sum(1 for term in terms if len(term) >= 5 and term in haystack)
        score = float(exact_hits * 3 + fuzzy_hits)
        if node.get("node_type") in {"concept", "objective", "skill", "procedure", "formula", "example"}:
            score += 0.5
        if score > 0:
            scored.append((score, node))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _graph_chunk_ids(
    node_scores: list[tuple[float, dict[str, Any]]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    node_by_id = {node["id"]: node for node in nodes}
    selected_node_ids = {node["id"] for _score, node in node_scores[:8]}
    ids: list[str] = []

    def add_from_node(node: dict[str, Any] | None) -> None:
        if node is None:
            return
        if node.get("node_type") == "chunk" and node.get("ref_id"):
            ids.append(str(node["ref_id"]))
        ids.extend(str(item) for item in node.get("source_chunk_ids") or [])

    for _score, node in node_scores[:8]:
        add_from_node(node)

    for edge in edges:
        source_id = edge["source_node_id"]
        target_id = edge["target_node_id"]
        touches_selected = source_id in selected_node_ids or target_id in selected_node_ids
        if not touches_selected:
            continue
        ids.extend(str(item) for item in edge.get("source_chunk_ids") or [])
        if source_id in selected_node_ids:
            add_from_node(node_by_id.get(target_id))
        if target_id in selected_node_ids:
            add_from_node(node_by_id.get(source_id))
    return _dedupe(ids)[:limit]


def _dedupe_edges(edges: list[EdgeDraft]) -> list[EdgeDraft]:
    by_id: dict[str, EdgeDraft] = {}
    for edge in edges:
        current = by_id.get(edge.id)
        if current is None:
            by_id[edge.id] = edge
        else:
            current.confidence = max(current.confidence, edge.confidence)
            current.source_chunk_ids.update(edge.source_chunk_ids)
            current.metadata = {**current.metadata, **edge.metadata}
    return list(by_id.values())


def _important_query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[\w\u0600-\u06ff][\w\u0600-\u06ff+/#.-]*", query):
        term = raw.casefold().strip("._-")
        if len(term) < 3 or term in _GRAPH_QUERY_STOPWORDS:
            continue
        terms.add(term)
    return terms


def _chunk_label(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    heading = metadata.get("heading_path_list") or metadata.get("heading_path") or []
    if isinstance(heading, str):
        parts = [part.strip() for part in heading.split(">") if part.strip()]
    else:
        parts = [str(item).strip() for item in heading if str(item).strip()]
    return " / ".join(parts[-2:]) or f"Chunk {int(chunk.get('chunk_index') or 0) + 1}"


def _extract_formula_labels(text: str) -> list[str]:
    blocks = re.findall(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", text, flags=re.DOTALL)
    inline = re.findall(r"(?m)^\s*([A-Za-z0-9_(),+\-*/^=\s]{3,}=.+)$", text)
    labels = [" ".join(part for part in match if part).strip() for match in blocks]
    labels.extend(item.strip() for item in inline)
    return [item for item in _dedupe(labels) if item][:4]


def _is_low_information_text(text: str) -> bool:
    compact = " ".join(str(text or "").split())
    if len(compact) < 18:
        return True
    alpha = sum(1 for char in compact if char.isalpha())
    return alpha < 8 and len(compact) < 80


def _node_is_in_source_scope(
    node: dict[str, Any],
    allowed_chunk_ids: set[str],
    allowed_file_ids: set[str],
    source_filter_active: bool,
) -> bool:
    if not source_filter_active:
        return True
    if node.get("node_type") == "course":
        return True
    metadata = node.get("metadata") or {}
    node_chunk_ids = {str(item) for item in node.get("source_chunk_ids") or []}
    if node_chunk_ids & allowed_chunk_ids:
        return True
    if node.get("node_type") == "chunk" and str(node.get("ref_id") or "") in allowed_chunk_ids:
        return True
    if node.get("node_type") == "file" and str(node.get("ref_id") or "") in allowed_file_ids:
        return True
    raw_metadata_file_ids = metadata.get("source_file_ids") or []
    if isinstance(raw_metadata_file_ids, str):
        raw_metadata_file_ids = [raw_metadata_file_ids]
    metadata_file_ids = {str(item) for item in raw_metadata_file_ids if item}
    if metadata.get("source_file_id"):
        metadata_file_ids.add(str(metadata["source_file_id"]))
    return bool(metadata_file_ids & allowed_file_ids)


def _clean_label(text: str) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split())


def _learning_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-") or "item"


def _dedupe(values: list[str] | set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _json(value: object, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return value if value is not None else default


_GRAPH_QUERY_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "can",
    "course",
    "cours",
    "define",
    "describe",
    "document",
    "documents",
    "explain",
    "file",
    "files",
    "for",
    "from",
    "give",
    "how",
    "lesson",
    "material",
    "materials",
    "me",
    "please",
    "show",
    "summarize",
    "teach",
    "tell",
    "the",
    "this",
    "what",
    "why",
    "with",
    "you",
    "quel",
    "quelle",
    "est",
    "dans",
    "pour",
    "explique",
    "expliquer",
}


_knowledge_graph_service: KnowledgeGraphService | None = None


def get_knowledge_graph_service() -> KnowledgeGraphService:
    global _knowledge_graph_service
    if _knowledge_graph_service is None:
        _knowledge_graph_service = KnowledgeGraphService()
    return _knowledge_graph_service

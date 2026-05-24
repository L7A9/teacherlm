from __future__ import annotations

import sys
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db.models import (  # noqa: E402
    CourseConceptRecord,
    CourseDocumentRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    CourseSectionRecord,
    SearchChunkRecord,
)
from services.concept_inventory_service import stable_concept_id  # noqa: E402
from services.knowledge_graph_service import (  # noqa: E402
    KnowledgeGraphService,
    _GraphCandidateBatch,
    _GraphEdgeCandidate,
    _GraphNodeCandidate,
    _prerequisite_node_ids,
    stable_node_id,
)
from services.learning_map_service import stable_objective_id, stable_phase_id  # noqa: E402


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SECTION_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class KnowledgeGraphServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_stable_node_ids_are_deterministic(self) -> None:
        first = stable_node_id(CONV_ID, "concept", "Photosynthesis")
        second = stable_node_id(CONV_ID, "concept", "photosynthesis")

        self.assertEqual(first, second)

    def test_fallback_graph_links_learning_map_and_chunks(self) -> None:
        service = KnowledgeGraphService()
        phase = _phase("Plant Energy")
        concept = _concept("Photosynthesis")
        objective = _objective(phase, "Explain photosynthesis", [concept])

        nodes, edges = service._fallback_graph(  # noqa: SLF001
            CONV_ID,
            {
                "documents": [_document()],
                "sections": [_section()],
                "chunks": [_chunk()],
                "concepts": [concept],
                "phases": [phase],
                "objectives": [objective],
            },
        )

        node_types = {node.node_type for node in nodes}
        edge_types = {edge.relation_type for edge in edges}
        self.assertIn("concept", node_types)
        self.assertIn("objective", node_types)
        self.assertIn("chunk", node_types)
        self.assertIn("teaches", edge_types)
        self.assertIn("supports", edge_types)
        self.assertIn("part_of", edge_types)

    def test_prerequisite_trace_supports_requires_and_prerequisite_of(self) -> None:
        service = KnowledgeGraphService()
        phase = _phase("Plant Energy")
        first = _concept("Light")
        second = _concept("Photosynthesis")
        objective = _objective(phase, "Explain photosynthesis", [first, second])
        nodes, edges = service._fallback_graph(  # noqa: SLF001
            CONV_ID,
            {
                "documents": [],
                "sections": [],
                "chunks": [],
                "concepts": [first, second],
                "phases": [phase],
                "objectives": [objective],
            },
        )
        target = next(node for node in nodes if node.ref_id == str(second.id))

        prereqs = _prerequisite_node_ids(
            target.id,
            [
                SimpleNamespace(
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    relation_type=edge.relation_type,
                )
                for edge in edges
            ],
            depth=1,
        )

        self.assertTrue(prereqs)

    async def test_llm_graph_uses_settings_model_then_gemma_fallback(self) -> None:
        attempted_models: list[str] = []

        class FakeOllamaClient:
            def __init__(
                self,
                base_url: str,
                model: str,
                *,
                provider: str = "ollama",
                api_key: str | None = None,
            ) -> None:
                self.base_url = base_url
                self.model = model
                self.provider = provider
                self.api_key = api_key

            async def chat_structured(self, **_: object) -> _GraphCandidateBatch:
                attempted_models.append(self.model)
                if self.model == "settings-model":
                    raise RuntimeError("settings model unavailable")
                return _GraphCandidateBatch(
                    nodes=[
                        _GraphNodeCandidate(
                            node_type="example",
                            label="Plant example",
                            source_chunk_ids=["chunk-1"],
                        )
                    ],
                    edges=[
                        _GraphEdgeCandidate(
                            source_label="Plant example",
                            target_label="Photosynthesis",
                            relation_type="example_of",
                            source_chunk_ids=["chunk-1"],
                        )
                    ],
                )

        service = KnowledgeGraphService(
            settings=SimpleNamespace(
                ollama_host="http://backend-ollama:11434",
                ollama_chat_model="backend-default",
            )
        )

        with patch("services.knowledge_graph_service.OllamaClient", FakeOllamaClient):
            nodes, edges = await service._llm_graph(  # noqa: SLF001
                CONV_ID,
                {
                    "documents": [],
                    "sections": [],
                    "chunks": [_chunk()],
                    "concepts": [_concept("Photosynthesis")],
                    "phases": [],
                    "objectives": [],
                },
                llm_options={
                    "llm": {
                        "enabled": True,
                        "provider": "ollama",
                        "model": "settings-model",
                        "base_url": "http://settings-ollama:11434",
                    }
                },
            )

        self.assertEqual(attempted_models, ["settings-model", "gemma4:e2b"])
        self.assertEqual(nodes[0].node_type, "example")
        self.assertEqual(edges[0].relation_type, "example_of")


def _document() -> CourseDocumentRecord:
    return CourseDocumentRecord(
        id=DOC_ID,
        conversation_id=CONV_ID,
        uploaded_file_id=uuid.uuid4(),
        source_file_id="uploads/lecture.pdf",
        source_filename="lecture.pdf",
        title="Lecture",
        text_hash="abc",
        course_metadata={},
    )


def _section() -> CourseSectionRecord:
    return CourseSectionRecord(
        id=SECTION_ID,
        conversation_id=CONV_ID,
        document_id=DOC_ID,
        parent_section_id=None,
        level=1,
        title="Photosynthesis",
        heading_path=["Photosynthesis"],
        order_index=0,
        text="Photosynthesis turns light into chemical energy.",
        summary="Photosynthesis basics.",
        key_concepts=[],
        equations=[],
        tables=[],
        timeline_events=[],
        section_metadata={},
    )


def _chunk() -> SearchChunkRecord:
    return SearchChunkRecord(
        id="chunk-1",
        conversation_id=CONV_ID,
        document_id=DOC_ID,
        section_id=SECTION_ID,
        source_filename="lecture.pdf",
        source_file_id="uploads/lecture.pdf",
        text="Photosynthesis uses light energy.",
        chunk_index=0,
        token_count=10,
        heading_path=["Photosynthesis"],
        chunk_metadata={},
    )


def _phase(title: str) -> CourseLearningPhaseRecord:
    return CourseLearningPhaseRecord(
        id=stable_phase_id(CONV_ID, title),
        conversation_id=CONV_ID,
        phase_key=title.casefold(),
        title=title,
        summary="",
        order_index=0,
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        phase_metadata={},
    )


def _objective(
    phase: CourseLearningPhaseRecord,
    text: str,
    concepts: list[CourseConceptRecord],
) -> CourseLearningObjectiveRecord:
    return CourseLearningObjectiveRecord(
        id=stable_objective_id(CONV_ID, phase.phase_key, text),
        conversation_id=CONV_ID,
        phase_id=phase.id,
        objective_key=text.casefold(),
        objective_text=text,
        bloom_level="understand",
        order_index=0,
        concept_ids=[str(concept.id) for concept in concepts],
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        objective_metadata={},
    )


def _concept(name: str) -> CourseConceptRecord:
    return CourseConceptRecord(
        id=stable_concept_id(CONV_ID, name),
        conversation_id=CONV_ID,
        canonical_key=name.casefold(),
        canonical_name=name,
        aliases=[],
        description=f"{name} description.",
        bloom_level="understand",
        importance=0.8,
        source_file_ids=[],
        source_section_ids=[str(SECTION_ID)],
        source_chunk_ids=["chunk-1"],
        concept_metadata={},
    )


if __name__ == "__main__":
    unittest.main()

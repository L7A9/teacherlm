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

from db.models import SearchChunkRecord  # noqa: E402
from services.concept_inventory_service import (  # noqa: E402
    ConceptCandidate,
    ConceptCandidateBatch,
    ConceptInventoryService,
    _ConceptAccumulator,
    _active_course_concept,
    _valid_learning_concept_name,
    normalize_concept_key,
    resolve_concept,
)


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SECTION_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class ConceptInventoryServiceTests(unittest.TestCase):
    def test_fallback_extracts_learning_concepts_but_not_course_part_titles(self) -> None:
        service = ConceptInventoryService()
        chunk = _chunk(
            "SVD: singular value decomposition for matrix factorization.",
            metadata={
                "section_title": "Plan de la séance",
                "section_summary": "Collaborative filtering method.",
                "key_concepts": ["Plan de la séance", "SVD", "Latent factors"],
            },
        )

        candidates = service._fallback_candidates([chunk])
        names = {item.canonical_name for item in candidates}

        self.assertNotIn("Plan de la séance", names)
        self.assertIn("SVD", names)
        self.assertIn("Latent factors", names)

    def test_aliases_merge_abbreviation_with_expanded_name(self) -> None:
        service = ConceptInventoryService()
        chunk = _chunk(
            "Singular Value Decomposition explains SVD.",
            metadata={
                "section_title": "Singular Value Decomposition",
                "key_concepts": ["SVD"],
            },
        )

        concepts = service._merge_candidates(CONV_ID, [chunk], service._fallback_candidates([chunk]))
        names = [concept.canonical_name for concept in concepts]

        self.assertEqual(len([name for name in names if normalize_concept_key(name) in {"svd", "singular value decomposition"}]), 1)
        concept = resolve_concept("SVD", [service._to_record(CONV_ID, item) for item in concepts])
        self.assertIsNotNone(concept)
        self.assertEqual(concept.canonical_name, "Singular Value Decomposition")

    def test_fallback_rejects_fragments_and_generic_titles(self) -> None:
        service = ConceptInventoryService()
        chunk = _chunk(
            "This part introduces evaluation with precision, recall, and NDCG.",
            metadata={
                "section_title": "Introduction",
                "section_summary": "Evaluation overview.",
                "key_concepts": ["Introduction", "and", "Evaluation metrics"],
            },
        )

        names = {item.canonical_name for item in service._fallback_candidates([chunk])}

        self.assertNotIn("Introduction", names)
        self.assertNotIn("and", names)
        self.assertIn("Evaluation metrics", names)

    def test_quality_gate_rejects_slide_trash_examples(self) -> None:
        bad = [
            "$\\vec{h}_t$",
            "$W_h, W_x, W_y$",
            "i = 2 (Note 3)",
            "3%",
            "1. Construire le profil de l’utilisateur",
            "4.3.4 Étape 3 : Calculer le nDCG",
            "Doc 1",
            "Position 10 ($i = 10$)",
            "Système Alpha",
            "Advanced Python (id",
            "Hypothèse de linéarité stricte</mark",
            "beaucoup plus élevé",
            "Elle permet de récupérer tout type de données",
            "La forme générale est",
            "L'attribut pour préciser cette orientation est Android",
            "a. en XML",
            "Ref",
        ]
        for name in bad:
            with self.subTest(name=name):
                self.assertFalse(_valid_learning_concept_name(name, "recommendation metric model"))

    def test_quality_gate_keeps_real_course_concepts(self) -> None:
        good = [
            "Filtrage Collaboratif",
            "Factorisation de Matrices",
            "TF-IDF",
            "nDCG",
            "A/B Testing",
            "Démarrage à froid",
            "Corrélation de Pearson",
            "Auto-encodeurs pour le Filtrage Collaboratif",
        ]
        for name in good:
            with self.subTest(name=name):
                self.assertTrue(_valid_learning_concept_name(name, "recommender systems evaluation model"))

    def test_quality_gate_keeps_non_ml_course_concepts(self) -> None:
        good = [
            "Photosynthesis",
            "Due Process",
            "Past Tense",
            "Derivative",
            "For Loop",
        ]
        for name in good:
            with self.subTest(name=name):
                self.assertTrue(_valid_learning_concept_name(name, f"{name} is explained in the course."))


class ConceptInventoryServiceAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_rebuild_uses_llm_candidates_before_fallback(self) -> None:
        service = _LLMFirstService()
        session = _FakeSession()

        records = await service.rebuild_concepts(session, CONV_ID)

        self.assertEqual([record.canonical_name for record in records], ["Collaborative Filtering"])
        self.assertFalse(service.fallback_called)

    async def test_persist_inactivates_referenced_retired_concepts(self) -> None:
        service = _ReferenceAwareService(reference_count=1)
        old = service.existing[0]
        session = _FakeSession()

        records = await service._persist_concepts(session, CONV_ID, [])

        self.assertEqual(records, [])
        self.assertEqual(session.deleted, [])
        self.assertEqual(old.source_chunk_ids, [])
        self.assertTrue(old.concept_metadata["inactive"])
        self.assertFalse(_active_course_concept(old))

    async def test_persist_deletes_unreferenced_retired_concepts(self) -> None:
        service = _ReferenceAwareService(reference_count=0)
        old = service.existing[0]
        session = _FakeSession()

        records = await service._persist_concepts(session, CONV_ID, [])

        self.assertEqual(records, [])
        self.assertEqual(session.deleted, [old])

    async def test_persist_updates_existing_stable_concept_in_place(self) -> None:
        service = _ReferenceAwareService(reference_count=1, name="Collaborative Filtering")
        old = service.existing[0]
        session = _FakeSession()
        incoming = _ConceptAccumulator(
            canonical_name="Collaborative Filtering",
            aliases=["CF"],
            description="Updated description.",
            importance=0.9,
            source_chunk_ids={"chunk-new"},
            source_section_ids={str(SECTION_ID)},
            source_file_ids={"uploads/new.pdf"},
        )

        records = await service._persist_concepts(session, CONV_ID, [incoming])

        self.assertEqual(records, [old])
        self.assertEqual(session.deleted, [])
        self.assertEqual(old.description, "Updated description.")
        self.assertEqual(old.source_chunk_ids, ["chunk-new"])
        self.assertFalse(old.concept_metadata.get("inactive", False))

    async def test_llm_candidates_use_settings_model_first(self) -> None:
        chunk = _chunk(
            "Collaborative filtering is a recommendation method based on user-item interactions.",
            metadata={
                "section_title": "Recommendation Systems",
                "section_summary": "Collaborative filtering method.",
                "key_concepts": ["Collaborative Filtering"],
            },
        )
        calls: list[tuple[str, str, str, str | None]] = []

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
                calls.append((base_url, model, provider, api_key))

            async def chat_structured(self, **_: object) -> ConceptCandidateBatch:
                return _concept_batch(chunk.id)

        service = ConceptInventoryService(
            settings=SimpleNamespace(
                ollama_host="http://backend-ollama:11434",
                ollama_chat_model="backend-default",
            )
        )

        with patch("services.concept_inventory_service.OllamaClient", FakeOllamaClient):
            candidates = await service._llm_candidates(
                [chunk],
                llm_options={
                    "llm": {
                        "enabled": True,
                        "provider": "openai_compatible",
                        "model": "settings-model",
                        "base_url": "https://settings.example/v1",
                        "api_key": "secret",
                    }
                },
            )

        self.assertEqual(candidates[0].canonical_name, "Collaborative Filtering")
        self.assertEqual(calls[0], ("https://settings.example/v1", "settings-model", "openai_compatible", "secret"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], ("http://backend-ollama:11434", "gemma4:e2b", "ollama", None))

    async def test_llm_candidates_fall_back_to_gemma_when_settings_model_fails(self) -> None:
        chunk = _chunk(
            "Collaborative filtering is a recommendation method based on user-item interactions.",
            metadata={
                "section_title": "Recommendation Systems",
                "section_summary": "Collaborative filtering method.",
                "key_concepts": ["Collaborative Filtering"],
            },
        )
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

            async def chat_structured(self, **_: object) -> ConceptCandidateBatch:
                attempted_models.append(self.model)
                if self.model == "settings-model":
                    raise RuntimeError("settings model unavailable")
                return _concept_batch(chunk.id)

        service = ConceptInventoryService(
            settings=SimpleNamespace(
                ollama_host="http://backend-ollama:11434",
                ollama_chat_model="backend-default",
            )
        )

        with patch("services.concept_inventory_service.OllamaClient", FakeOllamaClient):
            candidates = await service._llm_candidates(
                [chunk],
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
        self.assertEqual(candidates[0].canonical_name, "Collaborative Filtering")


def _chunk(text: str, metadata: dict[str, object]) -> SearchChunkRecord:
    return SearchChunkRecord(
        id=str(uuid.uuid4()),
        conversation_id=CONV_ID,
        document_id=DOC_ID,
        section_id=SECTION_ID,
        source_filename="lecture.pdf",
        source_file_id="uploads/lecture.pdf",
        text=text,
        chunk_index=0,
        token_count=20,
        heading_path=["Course", str(metadata.get("section_title") or "Section")],
        chunk_metadata=metadata,
    )


class _FakeSession:
    def __init__(self) -> None:
        self.records: list[object] = []
        self.deleted: list[object] = []

    async def execute(self, _statement: object) -> None:
        return None

    def add_all(self, records: list[object]) -> None:
        self.records.extend(records)

    async def flush(self) -> None:
        return None

    async def delete(self, record: object) -> None:
        self.deleted.append(record)


class _LLMFirstService(ConceptInventoryService):
    def __init__(self) -> None:
        super().__init__()
        self.fallback_called = False
        self.chunk = _chunk(
            "Collaborative filtering is a recommendation method based on user-item interactions.",
            metadata={
                "section_title": "Recommendation Systems",
                "section_summary": "Collaborative filtering method.",
                "key_concepts": ["Fallback Trash"],
            },
        )

    async def ensure_schema(self, session: object) -> None:
        return None

    async def _load_chunks(self, session: object, conversation_id: uuid.UUID) -> list[SearchChunkRecord]:
        return [self.chunk]

    async def _llm_candidates(
        self,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, object] | None = None,
    ):  # noqa: ANN201
        return [
            ConceptCandidate(
                canonical_name="Collaborative Filtering",
                description="Recommendation method based on user-item interactions.",
                source_chunk_ids=[self.chunk.id],
            )
        ]

    def _fallback_candidates(self, chunks: list[SearchChunkRecord]):  # noqa: ANN201
        self.fallback_called = True
        return []

    async def load_concepts(self, session: object, conversation_id: uuid.UUID) -> list[object]:
        return []

    async def _load_all_concepts(self, session: object, conversation_id: uuid.UUID) -> list[object]:
        return []

    async def _persist_concepts(self, session: object, conversation_id: uuid.UUID, concepts: list[object]):  # noqa: ANN201
        records = [self._to_record(conversation_id, item) for item in concepts]
        session.add_all(records)
        await session.flush()
        return records


class _ReferenceAwareService(ConceptInventoryService):
    def __init__(self, *, reference_count: int, name: str = "Retired Concept") -> None:
        super().__init__()
        self.reference_count = reference_count
        self.existing = [
            self._to_record(
                CONV_ID,
                _ConceptAccumulator(
                    canonical_name=name,
                    description="Old description.",
                    source_chunk_ids={"chunk-old"},
                    source_section_ids={str(SECTION_ID)},
                    source_file_ids={"uploads/old.pdf"},
                ),
            )
        ]

    async def _load_all_concepts(self, session: object, conversation_id: uuid.UUID):  # noqa: ANN201
        return self.existing

    async def _assessment_reference_counts(self, session: object, conversation_id: uuid.UUID):  # noqa: ANN201
        return {self.existing[0].id: self.reference_count} if self.reference_count else {}


def _concept_batch(chunk_id: str) -> ConceptCandidateBatch:
    return ConceptCandidateBatch(
        concepts=[
            ConceptCandidate(
                canonical_name="Collaborative Filtering",
                aliases=["CF"],
                description="Recommendation method based on user-item interactions.",
                source_chunk_ids=[chunk_id],
            )
        ]
    )


if __name__ == "__main__":
    unittest.main()

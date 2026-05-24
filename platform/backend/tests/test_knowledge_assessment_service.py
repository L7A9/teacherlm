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
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    SearchChunkRecord,
)
from services.concept_inventory_service import stable_concept_id  # noqa: E402
from services.knowledge_assessment_service import (  # noqa: E402
    KnowledgeAssessmentService,
    _GeneratedCheck,
    _heuristic_short_answer_score,
)
from services.learning_map_service import stable_objective_id, stable_phase_id  # noqa: E402
from schemas.knowledge_check import KnowledgeCheckResult  # noqa: E402


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SECTION_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class KnowledgeAssessmentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_objective_grading_accepts_mcq_option_label(self) -> None:
        service = KnowledgeAssessmentService()
        concept = _concept("Matrix Factorization")
        service._generate_check = _raise_generation_error  # type: ignore[method-assign]
        check = await service._build_check(
            CONV_ID,
            concept,
            [
                _chunk("Matrix Factorization learns latent factors for recommendations."),
                _chunk("Collaborative filtering compares users and items."),
                _chunk("Content-based filtering uses item attributes."),
            ],
            "mcq",
            [
                concept,
                _concept("Collaborative Filtering"),
                _concept("Content-Based Filtering"),
            ],
        )
        correct_option = check.options[check.answer_key["correct_index"]]

        score, is_correct, _feedback = await service._grade_check(check, concept, correct_option)

        self.assertEqual(score, 1.0)
        self.assertTrue(is_correct)

    async def test_short_answer_heuristic_scores_course_terms(self) -> None:
        concept = _concept(
            "Singular Value Decomposition",
            aliases=["SVD"],
            description="Matrix factorization method with latent factors.",
        )
        service = KnowledgeAssessmentService()
        service._generate_check = _raise_generation_error  # type: ignore[method-assign]
        check = await service._build_check(
            CONV_ID,
            concept,
            [_chunk("SVD is a matrix factorization method using latent factors.")],
            "short_answer",
            [concept],
        )

        score = _heuristic_short_answer_score(
            "SVD decomposes a matrix into latent factors for recommendation.",
            concept,
            check,
        )

        self.assertGreaterEqual(score, 0.6)

    def test_quiz_concept_resolution_uses_aliases(self) -> None:
        service = KnowledgeAssessmentService()
        concept = _concept("Singular Value Decomposition", aliases=["SVD"])

        resolved = service._resolve_quiz_concept(
            type("Question", (), {"concept_id": None, "concept": "SVD"})(),
            [concept],
        )

        self.assertEqual(resolved, concept)

    def test_result_model_preserves_question_index(self) -> None:
        result = KnowledgeCheckResult(
            check_id=uuid.uuid4(),
            concept_id=uuid.uuid4(),
            concept_name="Matrix Factorization",
            question_index=2,
            score=1.0,
            is_correct=True,
            feedback="Correct.",
            evidence_strength="medium",
            mastery_delta=0.16,
        )

        self.assertEqual(result.model_dump()["question_index"], 2)

    async def test_select_concepts_by_objective_id(self) -> None:
        service = KnowledgeAssessmentService()
        concept = _concept("Photosynthesis")
        phase = _phase("Plant Energy")
        objective = _objective(phase, "Explain photosynthesis", [concept])

        selected = await service._select_concepts(
            session=None,  # type: ignore[arg-type]
            conversation_id=CONV_ID,
            concepts=[concept],
            phases=[phase],
            objectives=[objective],
            concept_id=None,
            phase_id=None,
            objective_id=objective.id,
            count=1,
        )

        self.assertEqual(selected, [concept])

    async def test_select_concepts_by_phase_id(self) -> None:
        service = KnowledgeAssessmentService()
        concept = _concept("Due Process")
        phase = _phase("Legal Foundations")
        objective = _objective(phase, "Explain due process", [concept])

        selected = await service._select_concepts(
            session=None,  # type: ignore[arg-type]
            conversation_id=CONV_ID,
            concepts=[concept],
            phases=[phase],
            objectives=[objective],
            concept_id=None,
            phase_id=phase.id,
            objective_id=None,
            count=1,
        )

        self.assertEqual(selected, [concept])

    async def test_check_generation_falls_back_to_gemma_after_settings_model_fails(self) -> None:
        concept = _concept("Matrix Factorization")
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

            async def chat_structured(self, **_: object) -> _GeneratedCheck:
                attempted_models.append(self.model)
                if self.model == "settings-model":
                    raise RuntimeError("settings model unavailable")
                return _GeneratedCheck(
                    question_type="short_answer",
                    prompt="Explain Matrix Factorization using latent factors.",
                    rubric="Matrix Factorization learns latent factors.",
                )

        service = KnowledgeAssessmentService(
            settings=SimpleNamespace(
                ollama_host="http://backend-ollama:11434",
                ollama_chat_model="backend-default",
            )
        )

        with patch("services.knowledge_assessment_service.OllamaClient", FakeOllamaClient):
            generated = await service._generate_check(
                concept=concept,
                source_chunks=[_chunk("Matrix Factorization learns latent factors.")],
                question_type="short_answer",
                alternatives=[],
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
        self.assertEqual(generated.prompt, "Explain Matrix Factorization using latent factors.")


def _concept(
    name: str,
    *,
    aliases: list[str] | None = None,
    description: str = "Course concept description.",
) -> CourseConceptRecord:
    return CourseConceptRecord(
        id=stable_concept_id(CONV_ID, name),
        conversation_id=CONV_ID,
        canonical_key=name.casefold(),
        canonical_name=name,
        aliases=aliases or [],
        description=description,
        bloom_level="understand",
        importance=0.8,
        source_file_ids=["uploads/lecture.pdf"],
        source_section_ids=[str(SECTION_ID)],
        source_chunk_ids=["chunk-1"],
        concept_metadata={},
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


def _chunk(text: str) -> SearchChunkRecord:
    return SearchChunkRecord(
        id="chunk-1",
        conversation_id=CONV_ID,
        document_id=DOC_ID,
        section_id=SECTION_ID,
        source_filename="lecture.pdf",
        source_file_id="uploads/lecture.pdf",
        text=text,
        chunk_index=0,
        token_count=20,
        heading_path=["Course", "Section"],
        chunk_metadata={"section_title": "Matrix Factorization"},
    )


async def _raise_generation_error(**_: object) -> object:
    raise RuntimeError("skip LLM in unit test")


if __name__ == "__main__":
    unittest.main()

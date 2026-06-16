from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


GENERATORS_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
for path in (GENERATORS_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "minio" not in sys.modules:
    minio_stub = types.ModuleType("minio")

    class Minio:  # noqa: D101
        pass

    minio_stub.Minio = Minio
    sys.modules["minio"] = minio_stub

if "fastembed" not in sys.modules:
    fastembed_stub = types.ModuleType("fastembed")

    class TextEmbedding:  # noqa: D101
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            pass

        def embed(self, texts):  # noqa: ANN001
            return [[0.0] for _ in texts]

    fastembed_stub.TextEmbedding = TextEmbedding
    sys.modules["fastembed"] = fastembed_stub

from quiz_gen.pipeline import _resolve_allowed_kinds, _top_up_with_grounded_questions  # noqa: E402
from quiz_gen.schemas import ConceptCard, ExtractedConcepts, MCQ, QuestionSlot, QuizPlan, TrueFalse  # noqa: E402
from quiz_gen.services.difficulty_adapter import plan_question_mix  # noqa: E402
from quiz_gen.services.quality_validator import is_valid  # noqa: E402
from teacherlm_core.schemas.chunk import Chunk  # noqa: E402
from teacherlm_core.schemas.learner_state import LearnerState  # noqa: E402


class QuizTypeOptionsTests(unittest.TestCase):
    def test_frontend_aliases_resolve_to_one_strict_quiz_kind(self) -> None:
        self.assertEqual(_resolve_allowed_kinds({"question_types": "multiple_choice"}), ["mcq"])
        self.assertEqual(_resolve_allowed_kinds({"question_types": "true_false"}), ["true_false"])
        self.assertEqual(
            _resolve_allowed_kinds({"question_types": ["true_false", "multiple_choice"]}),
            ["true_false"],
        )

    def test_fill_and_short_answer_options_fall_back_to_mcq(self) -> None:
        self.assertEqual(_resolve_allowed_kinds({"question_types": "short_answer"}), ["mcq"])
        self.assertEqual(_resolve_allowed_kinds({"question_types": "fill_blank"}), ["mcq"])

    def test_planner_respects_selected_mcq_or_true_false_only(self) -> None:
        extracted = _extracted_concepts()
        learner = LearnerState(conversation_id="conversation")

        mcq_plan = plan_question_mix(
            learner,
            extracted,
            4,
            seed=1,
            allowed_kinds=["mcq"],
        )
        self.assertTrue(mcq_plan.slots)
        self.assertEqual({slot.kind for slot in mcq_plan.slots}, {"mcq"})

        tf_plan = plan_question_mix(
            learner,
            extracted,
            4,
            seed=1,
            allowed_kinds=["true_false"],
        )
        self.assertTrue(tf_plan.slots)
        self.assertEqual({slot.kind for slot in tf_plan.slots}, {"true_false"})

    def test_default_planner_never_uses_fill_blank(self) -> None:
        plan = plan_question_mix(
            LearnerState(conversation_id="conversation"),
            _extracted_concepts(),
            8,
            seed=1,
        )

        self.assertNotIn("fill_blank", {slot.kind for slot in plan.slots})

    def test_mcq_validation_requires_at_least_four_choices(self) -> None:
        question = MCQ.model_construct(
            type="mcq",
            bloom_level="remember",
            question="Which concept is covered?",
            options=["Photosynthesis", "Respiration", "Diffusion"],
            correct_index=0,
            explanation="The source covers photosynthesis.",
            concept="Photosynthesis",
            source_chunk_id="chunk-1",
        )

        self.assertFalse(is_valid(question))

    def test_top_up_adds_grounded_non_generic_questions(self) -> None:
        chunks = _chunks()
        plan = QuizPlan(
            slots=[
                QuestionSlot(concept="Photosynthesis", bloom_level="remember", kind="mcq", slot_kind="coverage"),
                QuestionSlot(concept="Cellular respiration", bloom_level="understand", kind="mcq", slot_kind="coverage"),
            ],
            total=2,
            counts={"struggling": 0, "coverage": 2, "stretch": 0},
        )

        questions = _top_up_with_grounded_questions(
            questions=[],
            plan=plan,
            concept_to_chunk_ids={
                "Photosynthesis": ["chunk-1"],
                "Cellular respiration": ["chunk-2"],
            },
            chunks=chunks,
            target_count=2,
        )

        self.assertEqual(len(questions), 2)
        for question in questions:
            self.assertTrue(is_valid(question, {chunk.chunk_id: chunk for chunk in chunks}))
            self.assertNotIn("course material", question.question.lower())
            self.assertNotIn("retrieved section", question.question.lower())

    def test_top_up_falls_back_to_true_false_without_mcq_distractors(self) -> None:
        chunks = [_chunks()[0]]
        plan = QuizPlan(
            slots=[
                QuestionSlot(concept="Photosynthesis", bloom_level="remember", kind="mcq", slot_kind="coverage"),
            ],
            total=1,
            counts={"struggling": 0, "coverage": 1, "stretch": 0},
        )

        questions = _top_up_with_grounded_questions(
            questions=[],
            plan=plan,
            concept_to_chunk_ids={"Photosynthesis": ["chunk-1"]},
            chunks=chunks,
            target_count=1,
        )

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].type, "true_false")
        self.assertTrue(is_valid(questions[0], {"chunk-1": chunks[0]}))

    def test_mcq_validation_rejects_answer_shown_in_question(self) -> None:
        question = MCQ(
            bloom_level="understand",
            question="What role does photosynthesis play in plants?",
            options=[
                "Photosynthesis",
                "Cellular respiration",
                "Osmosis",
                "Diffusion",
            ],
            correct_index=0,
            explanation="The source describes photosynthesis.",
            concept="Photosynthesis",
            source_chunk_id="chunk-1",
        )

        self.assertFalse(is_valid(question))

    def test_mcq_validation_rejects_generic_source_questions(self) -> None:
        question = MCQ(
            bloom_level="remember",
            question="Which concept is directly supported by the retrieved section Photosynthesis?",
            options=["Photosynthesis", "Cellular respiration", "Osmosis", "Diffusion"],
            correct_index=0,
            explanation="The retrieved source section supports Photosynthesis.",
            concept="Photosynthesis",
            source_chunk_id="chunk-1",
        )

        self.assertFalse(is_valid(question))

    def test_true_false_validation_rejects_generic_source_questions(self) -> None:
        question = TrueFalse(
            bloom_level="remember",
            question="The course material covers Photosynthesis in the section Photosynthesis.",
            answer=True,
            explanation="Yes. The retrieved source section is Photosynthesis.",
            concept="Photosynthesis",
            source_chunk_id="chunk-1",
        )

        self.assertFalse(is_valid(question))

    def test_course_style_mcq_can_pass_quality_validation(self) -> None:
        question = MCQ(
            bloom_level="understand",
            question="In plants, what is the main transformation carried out during this process?",
            options=[
                "Light energy is converted into chemical energy.",
                "Glucose is broken down to release stored energy.",
                "Water diffuses across a selectively permeable membrane.",
                "Particles move from high to low concentration.",
            ],
            correct_index=0,
            explanation="The source states that plants convert light energy into chemical energy.",
            concept="Photosynthesis",
            source_chunk_id="chunk-1",
        )

        self.assertTrue(is_valid(question))


def _extracted_concepts() -> ExtractedConcepts:
    return ExtractedConcepts(
        remember=[
            ConceptCard(name="Photosynthesis", bloom_level="remember", source_chunk_ids=["chunk-1"]),
            ConceptCard(name="Cellular respiration", bloom_level="remember", source_chunk_ids=["chunk-2"]),
            ConceptCard(name="Diffusion", bloom_level="remember", source_chunk_ids=["chunk-3"]),
            ConceptCard(name="Osmosis", bloom_level="remember", source_chunk_ids=["chunk-4"]),
        ]
    )


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            text="Photosynthesis converts light energy into chemical energy in plants.",
            source="biology.pdf",
            score=1.0,
            chunk_id="chunk-1",
            metadata={"section_title": "Photosynthesis", "key_concepts": ["Photosynthesis"]},
        ),
        Chunk(
            text="Cellular respiration releases energy from glucose.",
            source="biology.pdf",
            score=0.9,
            chunk_id="chunk-2",
            metadata={"section_title": "Cellular respiration", "key_concepts": ["Cellular respiration"]},
        ),
        Chunk(
            text="Diffusion moves particles from high concentration to low concentration.",
            source="biology.pdf",
            score=0.8,
            chunk_id="chunk-3",
            metadata={"section_title": "Diffusion", "key_concepts": ["Diffusion"]},
        ),
        Chunk(
            text="Osmosis is the diffusion of water across a membrane.",
            source="biology.pdf",
            score=0.7,
            chunk_id="chunk-4",
            metadata={"section_title": "Osmosis", "key_concepts": ["Osmosis"]},
        ),
    ]


if __name__ == "__main__":
    unittest.main()

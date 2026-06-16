from __future__ import annotations

import asyncio
import uuid
import unittest
from unittest.mock import patch

from db.models import (
    CourseBuilderChapterRecord,
    CourseConceptRecord,
    CourseDocumentRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    CourseSectionRecord,
    SearchChunkRecord,
)
from services.coursebuilder_service import (
    CourseBuilderContextPack,
    CourseBuilderService,
    PASS_SCORE,
    _CourseOutline,
    _OutlineChapter,
    _OutlineLesson,
    _QuizQuestionCandidate,
    _coursebuilder_chapter_locked,
    _fallback_outline,
    _fallback_lesson_blocks,
    _fallback_questions,
    _format_context_pack,
    _chapter_query,
    _has_rich_teaching_material,
    _is_thin_lesson_content,
    _intake_chapter_pool,
    _intake_lesson_pool,
    _looks_like_structure_only_chunk,
    _lesson_supported_chunks,
    _lesson_query,
    _reserved_lesson_chunk_ids,
    _repair_damaged_latex,
    _selected_index,
    _stable_id,
    _title_supported_chunks,
    _usable_lessons,
    _valid_quiz_question_rows,
    _valid_source_chunk_ids,
    extract_source_structure,
    select_representative_chunks,
)
from services.coursebuilder_validation import insufficient_source_message
from services.coursebuilder_rag import _looks_like_navigation_chunk


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class CourseBuilderServiceTests(unittest.TestCase):
    def test_stable_ids_are_deterministic(self) -> None:
        first = _stable_id(CONV_ID, "chapter:0:Foundations")
        second = _stable_id(CONV_ID, " chapter:0:foundations ")

        self.assertEqual(first, second)

    def test_pass_threshold_is_seventy_percent(self) -> None:
        self.assertEqual(PASS_SCORE, 0.7)

    def test_coursebuilder_chapter_locking_can_be_disabled_by_boolean(self) -> None:
        self.assertFalse(_coursebuilder_chapter_locked(1, prior_completed=False, lock_chapters=False))
        self.assertTrue(_coursebuilder_chapter_locked(1, prior_completed=False, lock_chapters=True))
        self.assertFalse(_coursebuilder_chapter_locked(1, prior_completed=True, lock_chapters=True))

    def test_fallback_outline_uses_source_hierarchy_as_ordered_chapters(self) -> None:
        chunks = [
            _chunk("chunk-1", ["Foundations", "Definition"], "Foundational explanation."),
            _chunk("chunk-2", ["Applications", "Example"], "Applied example."),
        ]

        outline = _fallback_outline(chunks)

        self.assertEqual([chapter.title for chapter in outline.chapters], ["Foundations", "Applications"])
        self.assertEqual(outline.chapters[0].lessons[0].title, "Definition")

    def test_extracts_table_of_contents_into_chapters_and_subchapters(self) -> None:
        chunks = [
            _chunk(
                "toc-1",
                ["Contents"],
                "\n".join(
                    [
                        "Contents",
                        "ONE MOROCCAN ORIGINS 1",
                        "Human origins 1",
                        "The geography of Morocco 4",
                        "TWO ISLAMIC MOROCCO 19",
                        "The Arab conquests 19",
                        "The Idrisids 27",
                    ]
                ),
            )
        ]

        chapters = extract_source_structure(chunks)

        self.assertEqual([chapter.title for chapter in chapters], ["MOROCCAN ORIGINS", "ISLAMIC MOROCCO"])
        self.assertEqual(
            [lesson.title for lesson in chapters[0].lessons],
            ["Human origins", "The geography of Morocco"],
        )

    def test_extract_source_structure_prefers_normalized_intake_units(self) -> None:
        chunks = [
            _chunk(
                "w1",
                ["rs course", "Plan de la seance"],
                "Source plan item: Introduction au Filtrage Collaboratif",
                metadata={
                    "course_unit_index": 1,
                    "course_unit_title": "Semaine 1 : Fondements",
                    "course_unit_role": "primary",
                    "subchapter_titles": ["Information overload", "Definition and objectives"],
                },
            ),
            _chunk(
                "w2",
                ["rs course", "Plan de la seance"],
                "Source plan item: Matrix factorization",
                metadata={
                    "course_unit_index": 2,
                    "course_unit_title": "Semaine 2 : Collaborative Filtering",
                    "course_unit_role": "primary",
                    "subchapter_titles": ["User-based CF", "Matrix factorization"],
                },
            ),
            _chunk(
                "guide",
                ["Guide", "Introduction"],
                "Guide table of contents.",
                metadata={
                    "course_unit_index": 3,
                    "course_unit_title": "Guide Complet d'Evaluation",
                    "course_unit_role": "supplemental",
                    "subchapter_titles": ["Introduction"],
                },
            ),
        ]

        chapters = extract_source_structure(chunks)

        self.assertEqual(
            [chapter.title for chapter in chapters],
            ["Semaine 1 : Fondements", "Semaine 2 : Collaborative Filtering"],
        )
        self.assertEqual(
            [lesson.title for lesson in chapters[1].lessons],
            ["User-based CF", "Matrix factorization"],
        )

    def test_extract_source_structure_filters_noisy_intake_plan_titles(self) -> None:
        chunks = [
            _chunk(
                "plan",
                ["Lecture", "Semaine 1", "Plan de la seance"],
                "Source plan item: Le probleme de la surcharge informationnelle",
                metadata={
                    "course_unit_index": 1,
                    "course_unit_title": "Semaine 1 : Fondements",
                    "course_unit_role": "primary",
                    "subchapter_titles": [
                        "Le probleme de la surcharge informationnelle",
                        "Definition et objectifs d'un systeme de recommandation",
                        "Le Problème",
                        "Judson Meinhart | Behavioral Finance, Millennial",
                        "PEOPLE WHO BOUGHT",
                        "jars and",
                        "% sales, right table has",
                    ],
                },
            )
        ]

        chapters = extract_source_structure(chunks)

        self.assertEqual([chapter.title for chapter in chapters], ["Semaine 1 : Fondements"])
        self.assertEqual(
            [lesson.title for lesson in chapters[0].lessons],
            [
                "Le probleme de la surcharge informationnelle",
                "Definition et objectifs d'un systeme de recommandation",
            ],
        )

    def test_extract_source_structure_merges_near_duplicate_intake_subchapters(self) -> None:
        chunks = [
            _chunk(
                "plan",
                ["Lecture", "Semaine 1", "Plan de la seance"],
                "Source plan item: Definition and Core Objective of Recommendation Systems",
                metadata={
                    "course_unit_index": 1,
                    "course_unit_title": "Recommendation Systems",
                    "course_unit_role": "primary",
                    "subchapter_titles": [
                        "Definition and Core Objective of Recommendation Systems",
                        "Definition and Core Purpose of Recommendation Systems",
                        "Collaborative Filtering Basics",
                    ],
                },
            )
        ]

        chapters = extract_source_structure(chunks)

        self.assertEqual([chapter.title for chapter in chapters], ["Recommendation Systems"])
        self.assertEqual(
            [lesson.title for lesson in chapters[0].lessons],
            [
                "Definition and Core Objective of Recommendation Systems",
                "Collaborative Filtering Basics",
            ],
        )
        self.assertIn(
            "Definition and Core Purpose of Recommendation Systems",
            chapters[0].lessons[0].source_queries,
        )

    def test_usable_lessons_merges_near_duplicate_generated_titles(self) -> None:
        chapter = _OutlineChapter(
            title="Recommendation Systems",
            lessons=[
                _OutlineLesson(
                    title="Definition and Core Objective of Recommendation Systems",
                    learning_objectives=["Understand the objective."],
                    source_queries=["objective query"],
                ),
                _OutlineLesson(
                    title="Definition and Core Purpose of Recommendation Systems",
                    learning_objectives=["Understand the purpose."],
                    source_queries=["purpose query"],
                ),
                _OutlineLesson(title="Collaborative Filtering Basics"),
            ],
        )

        lessons = _usable_lessons(chapter, [])

        self.assertEqual(
            [lesson.title for lesson in lessons],
            [
                "Definition and Core Objective of Recommendation Systems",
                "Collaborative Filtering Basics",
            ],
        )
        self.assertIn("purpose query", lessons[0].source_queries)
        self.assertIn("Understand the purpose.", lessons[0].learning_objectives)

    def test_lesson_retrieval_pool_prefers_normalized_subchapter_scope(self) -> None:
        chapter = _OutlineChapter(title="Semaine 2 : Collaborative Filtering")
        lesson = _OutlineLesson(title="Matrix factorization")
        matrix = _chunk(
            "matrix",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Matrix factorization"],
            "SVD learns latent factors.",
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
                "subchapter_title": "Matrix factorization",
            },
        )
        neighbor = _chunk(
            "neighbor",
            ["rs course", "Semaine 2 : Collaborative Filtering", "User-based CF"],
            "Neighbors compare similar users.",
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
                "subchapter_title": "User-based CF",
            },
        )
        other = _chunk(
            "other",
            ["rs course", "Semaine 1 : Foundations"],
            "Introductory material.",
            metadata={
                "course_unit_title": "Semaine 1 : Foundations",
                "course_unit_role": "primary",
            },
        )

        chapter_pool = _intake_chapter_pool([matrix, neighbor, other], chapter)
        lesson_pool = _intake_lesson_pool(chapter_pool, lesson)

        self.assertEqual({chunk.id for chunk in chapter_pool}, {"matrix", "neighbor"})
        self.assertEqual([chunk.id for chunk in lesson_pool], ["matrix", "neighbor"])

    def test_lesson_pool_keeps_unit_source_material_after_plan_marker_match(self) -> None:
        lesson = _OutlineLesson(title="Les Deux Grandes Familles du CF")
        plan = _chunk(
            "plan",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Les Deux Grandes Familles du CF"],
            "Source plan item: Les Deux Grandes Familles du CF",
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
                "subchapter_title": "Les Deux Grandes Familles du CF",
            },
        )
        source = _chunk(
            "source",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Source material"],
            (
                "Memory-based collaborative filtering uses observed user-item interactions directly to make "
                "recommendations. It compares users or items with similarity measures, then transfers preferences "
                "from the nearest neighbours to predict what a target user may like. Model-based collaborative "
                "filtering follows a different strategy because it learns a compact model from the rating matrix. "
                "The model can be a factorization, a clustering method, or another learned representation that "
                "captures latent preference patterns and generalizes beyond the exact examples already observed."
            ),
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
                "subchapter_titles": ["Les Deux Grandes Familles du CF"],
            },
        )

        lesson_pool = _intake_lesson_pool([plan, source], lesson)

        self.assertEqual([chunk.id for chunk in lesson_pool], ["plan", "source"])
        self.assertTrue(_looks_like_structure_only_chunk(plan))
        self.assertFalse(_looks_like_structure_only_chunk(source))

    def test_chapter_pool_keeps_untagged_source_sections_inside_normalized_unit(self) -> None:
        doc_id = uuid.uuid4()
        chapter = _OutlineChapter(title="Semaine 1 : Fondements")
        plan = _chunk(
            "plan",
            ["course", "Semaine 1 : Fondements", "Intro"],
            "Source plan item: Intro",
            document_id=doc_id,
            index=0,
            metadata={
                "course_unit_index": 1,
                "course_unit_title": "Semaine 1 : Fondements",
                "course_unit_role": "primary",
                "subchapter_title": "Intro",
            },
        )
        source_marker = _chunk(
            "source-marker",
            ["course", "Semaine 1 : Fondements", "Source material"],
            "Semaine 1 source material.",
            document_id=doc_id,
            index=1,
            metadata={
                "course_unit_index": 1,
                "course_unit_title": "Semaine 1 : Fondements",
                "course_unit_role": "primary",
                "subchapter_titles": ["Intro"],
            },
        )
        body = _chunk(
            "body",
            ["Real Body Heading"],
            "Surcharge informationnelle source explanation with enough teaching material.",
            document_id=doc_id,
            index=2,
            metadata={},
        )
        next_unit = _chunk(
            "next",
            ["course", "Semaine 2 : Filtering"],
            "Source plan item: Filtering",
            document_id=doc_id,
            index=3,
            metadata={
                "course_unit_index": 2,
                "course_unit_title": "Semaine 2 : Filtering",
                "course_unit_role": "primary",
                "subchapter_title": "Filtering",
            },
        )

        chapter_pool = _intake_chapter_pool([plan, source_marker, body, next_unit], chapter)

        self.assertEqual([chunk.id for chunk in chapter_pool], ["plan", "source-marker", "body"])

    def test_rich_lesson_retrieval_expands_plan_marker_to_source_material(self) -> None:
        class _Rag:
            def __init__(self, expanded: list[SearchChunkRecord]) -> None:
                self.calls = 0
                self.expanded = expanded

            async def retrieve_lesson_chunks(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
                self.calls += 1
                return self.expanded

        chapter = _OutlineChapter(title="Semaine 2 : Collaborative Filtering")
        lesson = _OutlineLesson(title="Les Deux Grandes Familles du CF")
        plan = _chunk(
            "plan",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Les Deux Grandes Familles du CF"],
            "Source plan item: Les Deux Grandes Familles du CF",
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
                "subchapter_title": "Les Deux Grandes Familles du CF",
            },
        )
        memory = _chunk(
            "memory",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Source material"],
            (
                "Memory-based collaborative filtering is built around direct comparisons in the observed ratings "
                "or interaction matrix. A user-based method searches for users with similar historical behaviour, "
                "then recommends items that those neighbours appreciated. An item-based method compares items "
                "instead, so the recommendation can be derived from objects that receive similar patterns of "
                "ratings. These methods remain close to the data and are often intuitive because the explanation "
                "can point back to neighbours or similar items."
            ),
            metadata={"course_unit_title": "Semaine 2 : Collaborative Filtering", "course_unit_role": "primary"},
        )
        model = _chunk(
            "model",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Source material"],
            (
                "Model-based collaborative filtering learns a representation from the rating data before making "
                "predictions. Matrix factorization is a common example because it represents users and items with "
                "latent factors, then combines those factors to estimate missing ratings. This learned model can "
                "generalize from sparse observations, reduce noise, and capture hidden preference dimensions that "
                "are not visible from a single neighbour comparison. The two families therefore differ in how they "
                "turn past interactions into future recommendations."
            ),
            metadata={"course_unit_title": "Semaine 2 : Collaborative Filtering", "course_unit_role": "primary"},
        )
        rag = _Rag([memory, model])
        service = CourseBuilderService()
        service._rag = rag  # type: ignore[assignment]

        selected, retrieval_count = asyncio.run(
            service._ensure_rich_lesson_chunks(
                None,  # type: ignore[arg-type]
                CONV_ID,
                chapter,
                lesson,
                "Les Deux Grandes Familles du CF",
                current_chunks=[],
                retrieved_chunks=[plan],
                fallback_chunks=[plan, memory, model],
                used_chunk_ids=set(),
            )
        )

        self.assertEqual(retrieval_count, 1)
        self.assertEqual([chunk.id for chunk in selected], ["memory", "model"])
        self.assertTrue(_has_rich_teaching_material(selected))

    def test_rich_lesson_retrieval_reuses_unit_source_material_for_later_subchapter(self) -> None:
        chapter = _OutlineChapter(title="Semaine 2 : Collaborative Filtering")
        lesson = _OutlineLesson(title="Les Deux Grandes Familles du CF")
        source = _chunk(
            "unit-source",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Source material"],
            (
                "Memory-based collaborative filtering uses observed user-item interactions directly to make "
                "recommendations. It compares users or items with similarity measures, then transfers preferences "
                "from the nearest neighbours to predict what a target user may like. Model-based collaborative "
                "filtering follows a different strategy because it learns a compact model from the rating matrix. "
                "The model can be a factorization, a clustering method, or another learned representation that "
                "captures latent preference patterns and generalizes beyond the exact examples already observed. "
                "The important teaching contrast is that memory-based methods stay close to the original examples, "
                "while model-based methods compress those examples into parameters that can fill missing ratings. "
                "This contrast helps explain why recommender systems can behave differently when data is sparse, "
                "when users are new, or when the catalog has many items with few direct interactions. A student "
                "should also notice that the two families answer different questions: memory-based approaches ask "
                "which existing users or items are similar enough to borrow evidence from, while model-based "
                "approaches ask which learned representation can summarize the evidence and support prediction. "
                "Keeping both views together gives the lesson enough context to compare intuition, scalability, "
                "and generalization without leaving the uploaded source material."
            ),
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
                "subchapter_titles": ["Les Deux Grandes Familles du CF", "Matrix factorization"],
            },
        )
        service = CourseBuilderService()

        selected, retrieval_count = asyncio.run(
            service._ensure_rich_lesson_chunks(
                None,  # type: ignore[arg-type]
                CONV_ID,
                chapter,
                lesson,
                "Les Deux Grandes Familles du CF",
                current_chunks=[],
                retrieved_chunks=[source],
                fallback_chunks=[source],
                used_chunk_ids={"unit-source"},
            )
        )

        self.assertEqual(retrieval_count, 0)
        self.assertEqual([chunk.id for chunk in selected], ["unit-source"])
        self.assertTrue(_has_rich_teaching_material(selected))

    def test_lesson_supported_chunks_rejects_chapter_only_support(self) -> None:
        lesson = _OutlineLesson(
            title="Matrix factorization",
            learning_objectives=["Understand latent factors."],
            source_queries=["SVD"],
        )
        broad = _chunk(
            "broad",
            ["Semaine 2 : Collaborative Filtering"],
            "Collaborative filtering recommends items from patterns in user behavior.",
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
            },
        )

        selected = _lesson_supported_chunks([broad], lesson, used_chunk_ids=set())

        self.assertEqual(selected, [])

    def test_lesson_supported_chunks_prefers_unused_lesson_evidence(self) -> None:
        lesson = _OutlineLesson(title="Matrix factorization", source_queries=["SVD"])
        repeated = _chunk(
            "repeated",
            ["Semaine 2 : Collaborative Filtering", "Matrix factorization"],
            "Matrix factorization introduces SVD for latent factors.",
        )
        fresh = _chunk(
            "fresh",
            ["Semaine 2 : Collaborative Filtering", "Matrix factorization"],
            "SVD decomposes the ratings matrix into latent user and item factors.",
        )

        selected = _lesson_supported_chunks(
            [repeated, fresh],
            lesson,
            used_chunk_ids={"repeated"},
        )

        self.assertEqual([chunk.id for chunk in selected], ["fresh"])

    def test_lesson_supported_chunks_reuses_unit_source_metadata(self) -> None:
        lesson = _OutlineLesson(title="Les Deux Grandes Familles du CF")
        source = _chunk(
            "source",
            ["rs course", "Semaine 2 : Collaborative Filtering", "Source material"],
            "Memory-based methods compare neighbours, while model-based methods learn compact latent patterns.",
            metadata={
                "course_unit_title": "Semaine 2 : Collaborative Filtering",
                "course_unit_role": "primary",
                "subchapter_titles": ["Les Deux Grandes Familles du CF"],
            },
        )

        selected = _lesson_supported_chunks(
            [source],
            lesson,
            used_chunk_ids={"source"},
        )

        self.assertEqual([chunk.id for chunk in selected], ["source"])

    def test_only_exact_subchapter_chunks_are_reserved_between_lessons(self) -> None:
        exact = _chunk(
            "exact",
            ["course", "Semaine 1", "Intro"],
            "Exact lesson source.",
            metadata={"subchapter_title": "Intro"},
        )
        broad = _chunk(
            "broad",
            ["course", "Body"],
            "Broad unit source material.",
            metadata={},
        )

        self.assertEqual(_reserved_lesson_chunk_ids([exact, broad]), {"exact"})

    def test_lesson_supported_chunks_rejects_only_reused_lesson_evidence(self) -> None:
        lesson = _OutlineLesson(title="Matrix factorization", source_queries=["SVD"])
        repeated = _chunk(
            "repeated",
            ["Semaine 2 : Collaborative Filtering", "Matrix factorization"],
            "Matrix factorization introduces SVD for latent factors.",
        )

        selected = _lesson_supported_chunks(
            [repeated],
            lesson,
            used_chunk_ids={"repeated"},
        )

        self.assertEqual(selected, [])

    def test_fallback_quiz_is_grounded_in_chunk_ids(self) -> None:
        chunk = _chunk("chunk-1", ["Foundations"], "A supported statement from the uploaded file.")
        chapter = CourseBuilderChapterRecord(
            id=_stable_id(CONV_ID, "chapter"),
            course_id=_stable_id(CONV_ID, "course"),
            conversation_id=CONV_ID,
            title="Foundations",
            order_index=0,
        )

        questions = _fallback_questions(chapter, [chunk])

        self.assertEqual(questions[0].correct_index, 0)
        self.assertEqual(questions[0].source_chunk_ids, ["chunk-1"])

    def test_selected_index_accepts_index_label_or_option_text(self) -> None:
        options = ["Alpha", "Beta", "Gamma"]

        self.assertEqual(_selected_index(1, options), 1)
        self.assertEqual(_selected_index("B", options), 1)
        self.assertEqual(_selected_index("Gamma", options), 2)
        self.assertIsNone(_selected_index("", options))

    def test_representative_chunks_are_balanced_across_documents(self) -> None:
        doc_a = uuid.uuid4()
        doc_b = uuid.uuid4()
        chunks = [
            *[_chunk(f"a-{idx}", ["Doc A", f"Section {idx}"], f"A text {idx}", document_id=doc_a, index=idx) for idx in range(6)],
            *[_chunk(f"b-{idx}", ["Doc B", f"Section {idx}"], f"B text {idx}", document_id=doc_b, index=idx) for idx in range(6)],
        ]

        selected = select_representative_chunks(chunks, max_chunks=4)

        self.assertEqual(len(selected), 4)
        self.assertEqual({chunk.document_id for chunk in selected}, {doc_a, doc_b})
        self.assertNotEqual([chunk.id for chunk in selected], [chunk.id for chunk in chunks[:4]])

    def test_context_pack_format_contains_summary_plan_concepts_and_catalog(self) -> None:
        phase_id = uuid.uuid4()
        objective_id = uuid.uuid4()
        chunk = _chunk("chunk-1", ["Foundations"], "Matrix factorization source text.")
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[_document("Course.pdf", "Recommender Systems")],
            sections=[],
            phases=[_phase(phase_id, "Foundations", "Start here.")],
            objectives=[_objective(objective_id, phase_id, "Explain matrix factorization.")],
            concepts=[_concept("Matrix factorization")],
            representative_chunks=[chunk],
        )

        rendered = _format_context_pack(context)

        self.assertIn("Rich course summary", rendered)
        self.assertIn("Extracted source course structure", rendered)
        self.assertIn("phase_id=", rendered)
        self.assertIn("Matrix factorization", rendered)
        self.assertIn("chunk_id=chunk-1", rendered)

    def test_outline_without_source_skeleton_uses_deterministic_fallback_not_llm(self) -> None:
        service = CourseBuilderService()
        called = False

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            nonlocal called
            called = True
            return _CourseOutline(
                title="Generated Course",
                chapters=[
                    _OutlineChapter(
                        title="Foundations",
                        lessons=[_OutlineLesson(title="Intro")],
                    )
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[],
            sections=[],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[_chunk("chunk-1", ["Foundations"], "Representative source text.")],
        )

        outline = asyncio.run(service._outline(CONV_ID, [_chunk("chunk-1", ["Foundations"], "Representative source text.")], context, llm_options=None))

        self.assertFalse(called)
        self.assertEqual(outline.chapters[0].title, "Foundations")

    def test_outline_prefers_extracted_source_structure_when_it_is_richer(self) -> None:
        service = CourseBuilderService()
        source_structure = [
            _OutlineChapter(
                title="Moroccan Origins",
                lessons=[
                    _OutlineLesson(title="Human origins"),
                    _OutlineLesson(title="The geography of Morocco"),
                ],
            ),
            _OutlineChapter(
                title="Islamic Morocco",
                lessons=[_OutlineLesson(title="The Arab conquests")],
            ),
        ]

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            return _CourseOutline(
                title="Generated Course",
                chapters=[
                    _OutlineChapter(
                        title="Generic Foundations",
                        lessons=[_OutlineLesson(title="Overview")],
                    )
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[],
            sections=[],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
            source_structure=source_structure,
        )

        outline = asyncio.run(service._outline(CONV_ID, [], context, llm_options=None))

        self.assertEqual([chapter.title for chapter in outline.chapters], ["Moroccan Origins", "Islamic Morocco"])
        self.assertEqual(outline.chapters[0].lessons[1].title, "The geography of Morocco")

    def test_outline_uses_parser_markdown_plan_before_course_generation(self) -> None:
        service = CourseBuilderService()
        captured: dict[str, str] = {}

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            captured["user"] = messages[1]["content"]
            return _CourseOutline(
                title="Morocco from Empire to Independence",
                chapters=[
                    _OutlineChapter(
                        title="Moroccan Origins",
                        lessons=[
                            _OutlineLesson(title="Human Origins"),
                            _OutlineLesson(title="Geography of Morocco"),
                        ],
                    )
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[],
            sections=[],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
            markdown_planning_context=(
                "### Markdown source 1: morocco.pdf\n"
                "```markdown\n"
                "Contents\n"
                "ONE MOROCCAN ORIGINS 1\n"
                "Human Origins 1\n"
                "Geography of Morocco 4\n"
                "```"
            ),
            markdown_source_count=1,
            markdown_raw_chars=120,
        )

        outline = asyncio.run(service._outline(CONV_ID, [], context, llm_options=None))

        self.assertIn("Parser markdown files", captured["user"])
        self.assertIn("ONE MOROCCAN ORIGINS", captured["user"])
        self.assertEqual(outline.chapters[0].title, "Moroccan Origins")
        self.assertEqual([lesson.title for lesson in outline.chapters[0].lessons], ["Human Origins", "Geography of Morocco"])

    def test_outline_keeps_normalized_primary_units_when_markdown_contains_guide_toc(self) -> None:
        service = CourseBuilderService()
        captured: dict[str, str] = {}
        source_structure = [
            _OutlineChapter(
                title="Semaine 1 : Fondements",
                lessons=[_OutlineLesson(title="Definition et objectifs")],
            ),
            _OutlineChapter(
                title="Semaine 2 : Collaborative Filtering",
                lessons=[_OutlineLesson(title="Matrix factorization")],
            ),
        ]

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            captured["user"] = messages[1]["content"]
            return _CourseOutline(
                title="Guide Evaluation",
                chapters=[
                    _OutlineChapter(
                        title="Introduction : Au-dela de la Notation",
                        lessons=[_OutlineLesson(title="Le Probleme Fondamental")],
                    )
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[
                _document(
                    "rs_course.pdf",
                    "rs course",
                    metadata={"intake_normalized": True, "primary_unit_count": 2},
                )
            ],
            sections=[
                _section(
                    "Definition et objectifs",
                    ["rs course", "Semaine 1 : Fondements", "Definition et objectifs"],
                    metadata={
                        "course_unit_title": "Semaine 1 : Fondements",
                        "course_unit_role": "primary",
                    },
                )
            ],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
            source_structure=source_structure,
            markdown_planning_context=(
                "### Markdown source 1: rs_course.pdf\n"
                "```markdown\n"
                "Table des matieres\n"
                "1 Introduction : Au-dela de la Notation 4\n"
                "1.1 Le Probleme Fondamental 4\n"
                "2 Pourquoi la RMSE Ne Suffit Pas ? 5\n"
                "```"
            ),
            markdown_source_count=1,
            markdown_raw_chars=160,
        )

        outline = asyncio.run(service._outline(CONV_ID, [], context, llm_options=None))

        self.assertIn("Semaine 1 : Fondements", captured["user"])
        self.assertNotIn("- chapter 1: Introduction : Au-dela de la Notation", captured["user"])
        self.assertEqual(
            [chapter.title for chapter in outline.chapters],
            ["Semaine 1 : Fondements", "Semaine 2 : Collaborative Filtering"],
        )
        self.assertEqual(outline.chapters[1].lessons[0].title, "Matrix factorization")

    def test_outline_localizes_primary_unit_titles_when_language_selected(self) -> None:
        service = CourseBuilderService()
        captured: dict[str, str] = {}
        source_structure = [
            _OutlineChapter(
                title="Semaine 1 : Fondements",
                lessons=[_OutlineLesson(title="Definition et objectifs")],
            ),
            _OutlineChapter(
                title="Semaine 2 : Collaborative Filtering",
                lessons=[_OutlineLesson(title="Matrix factorization")],
            ),
        ]

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return _CourseOutline(
                title="Evaluation Guide",
                language="en-us",
                chapters=[
                    _OutlineChapter(
                        title="Week 1: Foundations",
                        lessons=[_OutlineLesson(title="Definition and objectives")],
                    ),
                    _OutlineChapter(
                        title="Week 2: Collaborative Filtering",
                        lessons=[_OutlineLesson(title="Matrix factorization")],
                    ),
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[
                _document(
                    "rs_course.pdf",
                    "rs course",
                    metadata={"intake_normalized": True, "primary_unit_count": 2},
                )
            ],
            sections=[
                _section(
                    "Definition et objectifs",
                    ["rs course", "Semaine 1 : Fondements", "Definition et objectifs"],
                    metadata={
                        "course_unit_title": "Semaine 1 : Fondements",
                        "course_unit_role": "primary",
                    },
                )
            ],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
            source_structure=source_structure,
            markdown_planning_context=(
                "### Markdown source 1: rs_course.pdf\n"
                "```markdown\n"
                "Table des matieres\n"
                "1 Semaine 1 : Fondements 4\n"
                "1.1 Definition et objectifs 4\n"
                "2 Semaine 2 : Collaborative Filtering 5\n"
                "```"
            ),
            markdown_source_count=1,
            markdown_raw_chars=160,
        )

        outline = asyncio.run(service._outline(CONV_ID, [], context, llm_options={"language": "en-us"}))

        self.assertIn("English (US)", captured["system"])
        self.assertIn("selected settings language is English (US)", captured["user"])
        self.assertEqual(
            [chapter.title for chapter in outline.chapters],
            ["Week 1: Foundations", "Week 2: Collaborative Filtering"],
        )
        self.assertEqual(outline.chapters[0].lessons[0].title, "Definition and objectives")
        self.assertIn("Semaine 1 : Fondements", outline.chapters[0].source_queries)
        self.assertIn("Definition et objectifs", outline.chapters[0].lessons[0].source_queries)

    def test_outline_repairs_untranslated_chapter_title_when_lessons_are_localized(self) -> None:
        service = CourseBuilderService()
        captured: dict[str, list[str] | str] = {"schemas": []}
        source_structure = [
            _OutlineChapter(
                title="Semaine 1 : Fondements",
                lessons=[_OutlineLesson(title="Definition et objectifs")],
            ),
            _OutlineChapter(
                title="Introduction a la RMSE",
                lessons=[_OutlineLesson(title="Erreur quadratique moyenne")],
            ),
        ]

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            captured["schemas"].append(schema.__name__)  # type: ignore[union-attr]
            if schema.__name__ == "_LocalizedChapterTitleBatch":
                captured["repair_prompt"] = messages[1]["content"]
                return schema(chapters=[{"index": 0, "title": "Week 1: Foundations"}])
            return _CourseOutline(
                title="Evaluation Guide",
                language="en-us",
                chapters=[
                    _OutlineChapter(
                        title="Semaine 1 : Fondements",
                        lessons=[_OutlineLesson(title="Definition and objectives")],
                    ),
                    _OutlineChapter(
                        title="Introduction to RMSE",
                        lessons=[_OutlineLesson(title="Root mean squared error")],
                    )
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[
                _document(
                    "rs_course.pdf",
                    "rs course",
                    metadata={"intake_normalized": True, "primary_unit_count": 1},
                )
            ],
            sections=[],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
            source_structure=source_structure,
            markdown_planning_context=(
                "### Markdown source 1: rs_course.pdf\n"
                "```markdown\n"
                "Table des matieres\n"
                "1 Semaine 1 : Fondements 4\n"
                "1.1 Definition et objectifs 4\n"
                "2 Introduction a la RMSE 8\n"
                "2.1 Erreur quadratique moyenne 8\n"
                "```"
            ),
            markdown_source_count=1,
            markdown_raw_chars=120,
        )

        outline = asyncio.run(service._outline(CONV_ID, [], context, llm_options={"language": "en-us"}))

        self.assertEqual(captured["schemas"], ["_CourseOutline", "_LocalizedChapterTitleBatch"])
        self.assertIn("Semaine 1 : Fondements", str(captured["repair_prompt"]))
        self.assertEqual(outline.chapters[0].title, "Week 1: Foundations")
        self.assertEqual(outline.chapters[0].lessons[0].title, "Definition and objectives")
        self.assertIn("Semaine 1 : Fondements", outline.chapters[0].source_queries)
        self.assertEqual(outline.chapters[1].title, "Introduction to RMSE")
        self.assertEqual(outline.chapters[1].lessons[0].title, "Root mean squared error")

    def test_source_outline_titles_are_localized_without_markdown_when_language_selected(self) -> None:
        service = CourseBuilderService()
        captured: dict[str, str] = {}
        source_structure = [
            _OutlineChapter(
                title="Semaine 1 : Fondements",
                lessons=[_OutlineLesson(title="Definition et objectifs")],
            )
        ]

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return _CourseOutline(
                title="Recommender Systems",
                language="en-us",
                chapters=[
                    _OutlineChapter(
                        title="Week 1: Foundations",
                        lessons=[_OutlineLesson(title="Definition and objectives")],
                    )
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[],
            sections=[],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
            source_structure=source_structure,
        )

        outline = asyncio.run(service._outline(CONV_ID, [], context, llm_options={"language": "en-us"}))

        self.assertIn("CourseBuilder outline for display", captured["system"])
        self.assertIn("Target language: English (US)", captured["user"])
        self.assertEqual(outline.chapters[0].title, "Week 1: Foundations")
        self.assertEqual(outline.chapters[0].lessons[0].title, "Definition and objectives")
        self.assertIn("Semaine 1 : Fondements", outline.chapters[0].source_queries)
        self.assertIn("Definition et objectifs", outline.chapters[0].lessons[0].source_queries)

    def test_fallback_outline_titles_are_localized_when_language_selected(self) -> None:
        service = CourseBuilderService()
        captured: dict[str, str] = {}
        chunks = [
            _chunk(
                "fallback-1",
                ["Semaine 1 : Fondements"],
                "Definition et objectifs du cours.",
            )
        ]

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return _CourseOutline(
                title="Foundations",
                language="en-us",
                chapters=[
                    _OutlineChapter(
                        title="Week 1: Foundations",
                        lessons=[_OutlineLesson(title="Definition and objectives")],
                    )
                ],
            )

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[],
            sections=[],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
        )

        with patch("services.coursebuilder_service.extract_source_structure", return_value=[]):
            outline = asyncio.run(service._outline(CONV_ID, chunks, context, llm_options={"language": "en-us"}))

        self.assertIn("CourseBuilder outline for display", captured["system"])
        self.assertIn("Target language: English (US)", captured["user"])
        self.assertEqual(outline.language, "en-us")
        self.assertEqual(outline.chapters[0].title, "Week 1: Foundations")
        self.assertEqual(outline.chapters[0].lessons[0].title, "Definition and objectives")
        self.assertIn("Semaine 1 : Fondements", outline.chapters[0].source_queries)
        self.assertIn("Semaine 1 : Fondements", outline.chapters[0].lessons[0].source_queries)

    def test_markdown_planning_falls_back_to_markdown_toc_when_llm_returns_prose(self) -> None:
        service = CourseBuilderService()

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            raise RuntimeError("model returned prose")

        service._structured = fake_structured  # type: ignore[method-assign]
        context = CourseBuilderContextPack(
            rich_summary="Rich summary text.",
            documents=[_document("morocco.pdf", "Morocco from Empire to Independence")],
            sections=[],
            phases=[],
            objectives=[],
            concepts=[],
            representative_chunks=[],
            markdown_planning_context=(
                "# Extracted heading index\n"
                "- # MOROCCO\n"
                "- ## FROM EMPIRE TO INDEPENDENCE\n\n"
                "# Likely table of contents lines\n"
                "# MOROCCO\n"
                "## FROM EMPIRE TO INDEPENDENCE\n"
                "Contents\n"
                "ONE MOROCCAN ORIGINS 1\n"
                "Human Origins 1\n"
                "Geography of Morocco 4\n"
                "TWO ISLAMIC MOROCCO 19\n"
                "Origins of Islam 19\n"
                "The Idrisids 27\n"
                "<table>\n"
                "<tr>\n"
                "<td>Morocco, Israel and the Arabs</td>\n"
                "<td>180</td>\n"
                "</tr>\n"
                "</table>\n"
                "## Maps\n"
                "1. Morocco 2\n"
            ),
            markdown_source_count=1,
            markdown_raw_chars=180,
        )

        outline = asyncio.run(service._outline(CONV_ID, [], context, llm_options=None))

        self.assertEqual([chapter.title for chapter in outline.chapters], ["MOROCCAN ORIGINS", "ISLAMIC MOROCCO"])
        self.assertEqual(outline.title, "Morocco: From Empire To Independence")
        self.assertEqual(outline.chapters[1].lessons[1].title, "The Idrisids")
        self.assertIn("Morocco, Israel and the Arabs", [lesson.title for lesson in outline.chapters[1].lessons])

    def test_coursebuilder_rag_filters_navigation_chunks(self) -> None:
        toc = _chunk(
            "toc",
            ["Contents"],
            "Human origins 3 The geography of Morocco 4 Carthaginian Morocco 6 Roman Morocco 10",
        )
        body = _chunk(
            "body",
            ["MOROCCAN ORIGINS", "Human origins"],
            "Human occupation in Morocco can be studied through archaeology and early settlement evidence.",
        )

        self.assertTrue(_looks_like_navigation_chunk(toc))
        self.assertFalse(_looks_like_navigation_chunk(body))

    def test_retrieval_queries_include_plan_hints(self) -> None:
        phase_id = uuid.uuid4()
        objective_id = uuid.uuid4()
        context = CourseBuilderContextPack(
            rich_summary="",
            documents=[],
            sections=[],
            phases=[_phase(phase_id, "Phase title", "Phase summary")],
            objectives=[_objective(objective_id, phase_id, "Apply latent factors.")],
            concepts=[],
            representative_chunks=[],
        )
        chapter = _OutlineChapter(
            title="Latent Factors",
            description="Chapter description.",
            phase_id=str(phase_id),
            objective_ids=[str(objective_id)],
            source_queries=["factor query"],
        )
        lesson = _OutlineLesson(
            title="SVD",
            learning_objectives=["Understand SVD."],
            source_queries=["svd query"],
        )

        self.assertIn("Apply latent factors", _chapter_query(chapter, context))
        self.assertIn("factor query", _lesson_query(chapter, lesson, context))
        self.assertIn("svd query", _lesson_query(chapter, lesson, context))

    def test_valid_source_chunk_ids_filters_invalid_llm_citations(self) -> None:
        chunks = [_chunk("valid-1", ["Foundations"], "Source text.")]

        self.assertEqual(_valid_source_chunk_ids(["missing", "valid-1", "valid-1"], chunks), ["valid-1"])

    def test_subchapter_title_filter_keeps_matching_lesson_chunks(self) -> None:
        human = _chunk("human", ["MOROCCAN ORIGINS", "Human origins"], "Human occupation evidence.")
        geography = _chunk("geo", ["MOROCCAN ORIGINS", "The geography of Morocco"], "Mountains and coastlines.")

        filtered = _title_supported_chunks([human, geography], "Human origins")

        self.assertEqual([chunk.id for chunk in filtered], ["human"])

    def test_lesson_content_without_retrieved_chunks_is_insufficient_source(self) -> None:
        service = CourseBuilderService()
        chapter = CourseBuilderChapterRecord(
            id=_stable_id(CONV_ID, "chapter"),
            course_id=_stable_id(CONV_ID, "course"),
            conversation_id=CONV_ID,
            title="Foundations",
            order_index=0,
        )

        content = asyncio.run(
            service._lesson_content(
                None,  # type: ignore[arg-type]
                CONV_ID,
                chapter,
                _OutlineLesson(title="Missing subchapter"),
                [],
                fallback_chunks=[],
                llm_options=None,
            )
        )

        self.assertEqual(content.support_status, "insufficient_source_material")
        self.assertEqual(content.blocks[0].block_type, "warning")
        self.assertEqual(content.blocks[0].content, insufficient_source_message())

    def test_lesson_content_prompt_requests_scientific_math_and_tables(self) -> None:
        service = CourseBuilderService()
        captured: dict[str, str] = {}
        chapter = CourseBuilderChapterRecord(
            id=_stable_id(CONV_ID, "chapter"),
            course_id=_stable_id(CONV_ID, "course"),
            conversation_id=CONV_ID,
            title="Evaluation Metrics",
            order_index=0,
        )

        async def fake_structured(messages, schema, *, llm_options):  # noqa: ANN001
            captured["system"] = messages[0]["content"]
            return schema(blocks=[])

        service._structured = fake_structured  # type: ignore[method-assign]

        asyncio.run(
            service._lesson_content_from_chunks(
                chapter,
                _OutlineLesson(title="RMSE"),
                [
                    _chunk(
                        "metric",
                        ["Evaluation", "RMSE"],
                        "RMSE = sqrt(1/n sum_i (y_i - yhat_i)^2). The table compares RMSE and MAE.",
                    )
                ],
                llm_options=None,
            )
        )

        self.assertIn("LaTeX math delimiters", captured["system"])
        self.assertIn("$$...$$", captured["system"])
        self.assertIn("Markdown tables", captured["system"])
        self.assertIn("data_json={columns, rows}", captured["system"])

    def test_repairs_json_damaged_latex_formula_commands(self) -> None:
        damaged = "RMSE = oot{2}{\x0crac{1}{N} ext{sum}_{u,i} ( hat{r}_{u,i} - r_{u,i})^2} ext{ (1)}"

        repaired = _repair_damaged_latex(damaged)

        self.assertEqual(
            repaired,
            r"RMSE = \sqrt{\frac{1}{N} \sum_{u,i} ( \hat{r}_{u,i} - r_{u,i})^2} \text{ (1)}",
        )

    def test_title_only_lesson_block_content_is_too_thin(self) -> None:
        self.assertTrue(_is_thin_lesson_content("explanation", "SVD", "SVD"))
        self.assertTrue(
            _is_thin_lesson_content(
                "explanation",
                "Source plan item: Les Deux Grandes Familles du CF",
                "Les Deux Grandes Familles du CF",
            )
        )
        self.assertTrue(
            _is_thin_lesson_content(
                "explanation",
                "SVD explains latent factors.",
                "SVD",
            )
        )
        self.assertFalse(
            _is_thin_lesson_content(
                "explanation",
                (
                    "SVD explains a ratings matrix by representing users and items through latent factors. "
                    "The source connects this representation to recommendation because missing preferences can "
                    "be estimated from patterns shared across similar users and similar items. "
                    "This matters in the lesson because it turns sparse observed ratings into a structured model "
                    "that can support prediction while remaining grounded in the matrix factorization view. "
                    "The learner should understand this representation before moving to evaluation or tuning. "
                    "A rich lesson block should connect the algebraic operation to the teaching goal: the matrix "
                    "is not decomposed for its own sake, but to expose preference dimensions that can be reused "
                    "when the platform has to recommend items with incomplete ratings. That connection gives the "
                    "student enough context to distinguish model-based collaborative filtering from simpler "
                    "neighbour comparisons."
                ),
                "SVD",
            )
        )

    def test_fallback_lesson_blocks_do_not_use_plan_marker_as_content(self) -> None:
        blocks = _fallback_lesson_blocks(
            "Les Deux Grandes Familles du CF",
            [
                _chunk(
                    "plan",
                    ["rs course", "Plan de la seance"],
                    "Source plan item: Les Deux Grandes Familles du CF",
                )
            ],
        )

        self.assertEqual(blocks[0].block_type, "warning")
        self.assertEqual(blocks[0].content, insufficient_source_message())

    def test_fallback_lesson_blocks_use_rich_source_paragraphs(self) -> None:
        chunks = [
            _chunk(
                "chunk-1",
                ["Latent Factors", "SVD"],
                (
                    "SVD decomposes the user-item ratings matrix into lower-dimensional user and item factors. "
                    "Those factors summarize hidden preference patterns that are not written directly in the "
                    "original table. In a recommendation setting, the model compares these factors to estimate "
                    "ratings that a user has not provided yet. The decomposition is useful because the source "
                    "matrix is usually sparse and direct comparison between users can miss indirect structure."
                ),
            ),
            _chunk(
                "chunk-2",
                ["Latent Factors", "Prediction"],
                (
                    "For example, two users can receive similar recommendations when their latent vectors point "
                    "toward the same item factors. The predicted score comes from combining the user factors and "
                    "item factors, so the method links observed ratings to unseen preferences. This also explains "
                    "why dimensionality reduction can reduce noise while preserving the main preference signals."
                ),
            ),
        ]

        blocks = _fallback_lesson_blocks("SVD", chunks)

        self.assertGreaterEqual(len(blocks[0].content.split()), 55)
        self.assertIn("\n\n", blocks[0].content)
        self.assertNotEqual(blocks[0].content.strip(), "SVD")
        self.assertEqual(blocks[0].source_chunk_ids, ["chunk-1", "chunk-2"])

    def test_valid_quiz_rows_require_valid_source_citations(self) -> None:
        chunk = _chunk("valid-1", ["Foundations"], "A supported statement from the source.")
        rows = _valid_quiz_question_rows(
            [
                _QuizQuestionCandidate(
                    prompt="Unsupported?",
                    options=["A", "B"],
                    source_chunk_ids=[],
                ),
                _QuizQuestionCandidate(
                    prompt="Supported?",
                    options=["A", "B", "C", "D"],
                    correct_index=2,
                    source_chunk_ids=["valid-1"],
                ),
            ],
            [chunk],
            CourseBuilderService()._rag,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0].prompt, "Supported?")
        self.assertEqual(rows[0][2], 2)


def _chunk(
    chunk_id: str,
    heading_path: list[str],
    text: str,
    *,
    document_id: uuid.UUID | None = None,
    section_id: uuid.UUID | None = None,
    index: int = 0,
    metadata: dict[str, object] | None = None,
) -> SearchChunkRecord:
    return SearchChunkRecord(
        id=chunk_id,
        conversation_id=CONV_ID,
        document_id=document_id or uuid.uuid4(),
        section_id=section_id or uuid.uuid4(),
        source_filename="course.pdf",
        source_file_id="course.pdf",
        text=text,
        chunk_index=index,
        token_count=16,
        heading_path=heading_path,
        chunk_metadata=metadata or {},
    )


def _document(
    source_filename: str,
    title: str,
    *,
    metadata: dict[str, object] | None = None,
) -> CourseDocumentRecord:
    return CourseDocumentRecord(
        id=uuid.uuid4(),
        conversation_id=CONV_ID,
        uploaded_file_id=uuid.uuid4(),
        source_file_id=source_filename,
        source_filename=source_filename,
        title=title,
        text_hash="hash",
        course_metadata=metadata or {},
    )


def _section(
    title: str,
    heading_path: list[str],
    *,
    metadata: dict[str, object] | None = None,
) -> CourseSectionRecord:
    return CourseSectionRecord(
        id=uuid.uuid4(),
        conversation_id=CONV_ID,
        document_id=uuid.uuid4(),
        level=len(heading_path),
        title=title,
        heading_path=heading_path,
        order_index=0,
        text=f"{title} source text.",
        summary=f"{title} summary.",
        key_concepts=[],
        equations=[],
        tables=[],
        timeline_events=[],
        section_metadata=metadata or {},
    )


def _phase(phase_id: uuid.UUID, title: str, summary: str) -> CourseLearningPhaseRecord:
    return CourseLearningPhaseRecord(
        id=phase_id,
        conversation_id=CONV_ID,
        phase_key=title.lower(),
        title=title,
        summary=summary,
        order_index=0,
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        phase_metadata={},
    )


def _objective(
    objective_id: uuid.UUID,
    phase_id: uuid.UUID,
    objective_text: str,
) -> CourseLearningObjectiveRecord:
    return CourseLearningObjectiveRecord(
        id=objective_id,
        conversation_id=CONV_ID,
        phase_id=phase_id,
        objective_key=objective_text.lower(),
        objective_text=objective_text,
        bloom_level="understand",
        order_index=0,
        concept_ids=[],
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        objective_metadata={},
    )


def _concept(name: str) -> CourseConceptRecord:
    return CourseConceptRecord(
        id=uuid.uuid4(),
        conversation_id=CONV_ID,
        canonical_key=name.lower(),
        canonical_name=name,
        aliases=[],
        description=f"{name} description.",
        bloom_level="understand",
        importance=0.8,
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        concept_metadata={},
    )

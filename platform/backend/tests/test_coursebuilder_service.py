from __future__ import annotations

import asyncio
import uuid
import unittest

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
    _fallback_questions,
    _format_context_pack,
    _chapter_query,
    _intake_chapter_pool,
    _intake_lesson_pool,
    _lesson_supported_chunks,
    _lesson_query,
    _selected_index,
    _stable_id,
    _title_supported_chunks,
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
        self.assertEqual([chunk.id for chunk in lesson_pool], ["matrix"])

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

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacherlm_core"))
sys.path.insert(0, str(ROOT / "local_api"))


def _chunk(chunk_id: str, text: str, *, unit: str = "", lesson: str = "") -> dict:
    return {
        "id": chunk_id,
        "source_file_id": "file-1",
        "source_filename": "course.md",
        "chunk_index": int(chunk_id.rsplit("-", 1)[-1]),
        "text": text,
        "metadata": {
            "heading_path": " > ".join(item for item in (unit, lesson) if item),
            "heading_path_list": [item for item in (unit, lesson) if item],
            "course_unit_title": unit,
            "subchapter_title": lesson,
            "course_unit_role": "primary",
        },
    }


def test_intake_keeps_primary_units_and_marks_supplemental_material() -> None:
    from local_api.services.course_intake import normalize_course_intake

    markdown = """Week 1: Linear Algebra
Plan de la séance: 1. Vectors 2. Matrix operations
Vectors are quantities with magnitude and direction.
Week 2: Optimization
Plan: 1. Gradients 2. Convexity
Gradients guide iterative search.
References
The references supplement the weekly teaching units and do not define another chapter.
"""
    intake = normalize_course_intake(
        raw_markdown=markdown,
        cleaned_markdown=markdown,
        source_filename="course.md",
    )

    assert [(unit.title, unit.role) for unit in intake.units] == [
        ("Week 1: Linear Algebra", "primary"),
        ("Week 2: Optimization", "primary"),
        ("References", "supplemental"),
    ]
    assert [item.title for item in intake.units[0].subchapters] == ["Vectors", "Matrix operations"]
    assert "## Week 1: Linear Algebra" in intake.markdown


def test_intake_keeps_all_top_level_plan_items_across_nested_bullets() -> None:
    from local_api.services.course_intake import normalize_course_intake

    markdown = """Week 3: Applied Methods
# Plan de la séance

1. **Foundations:** Core definitions.
2. **First method:**
    * Supporting variant A.
    * Supporting variant B.
3. **Second method:** A separate planned topic.
4. **Evaluation and deployment.**

# Foundations
Teaching content begins here.
"""
    intake = normalize_course_intake(
        raw_markdown=markdown,
        cleaned_markdown=markdown,
        source_filename="course.md",
    )

    assert [item.title for item in intake.units[0].subchapters] == [
        "Foundations: Core definitions",
        "First method",
        "Second method: A separate planned topic",
        "Evaluation and deployment",
    ]


def test_markdown_session_plans_are_primary_and_ignore_supplementary_toc() -> None:
    from local_api.services.coursebuilder_structure import extract_source_structure

    markdown = """### Markdown source 1: lecture_1.pdf
```markdown
General Subject
Week 1: Foundations
# Plan de la séance
1. Core concepts
2. Standard method
    * Supporting detail
3. Evaluation
# Core concepts
Content.
```

### Markdown source 2: lecture_2.pdf
```markdown
General Subject
Week 2: Applications
# Plan de la séance
1. Worked examples
2. Deployment
# Worked examples
Content.
```

### Markdown source 3: guide.pdf
```markdown
Table of Contents
Chapter 1: Supplemental details .... 1
Reference tables .... 2
```
"""
    structure = extract_source_structure(chunks=[], sections=[], documents=[], markdown=markdown)

    assert structure is not None
    assert structure.title == "General Subject"
    assert [chapter.title for chapter in structure.chapters] == ["Week 1: Foundations", "Week 2: Applications"]
    assert [lesson.title for lesson in structure.chapters[0].lessons] == [
        "Core concepts",
        "Standard method",
        "Evaluation",
    ]


def test_markdown_plan_normalizes_subchapter_alias_before_validation() -> None:
    import local_api.services.coursebuilder as coursebuilder
    from local_api.services.coursebuilder_structure import SourceChapter, SourceLesson, SourceStructure

    payload = {
        "title": "General Subject",
        "chapters": [
            {
                "title": "Week 1: Foundations",
                "subchapters": ["Core concepts", {"name": "Evaluation"}],
            }
        ],
    }
    structure = SourceStructure(
        title="General Subject",
        origin="markdown_toc",
        chapters=[
            SourceChapter(
                title="Week 1: Foundations",
                lessons=[SourceLesson("Core concepts"), SourceLesson("Evaluation")],
            )
        ],
    )

    coursebuilder._normalize_markdown_outline_payload(payload, structure)
    outline = coursebuilder.CourseOutline.model_validate(payload)

    assert [lesson.title for lesson in outline.chapters[0].lessons] == ["Core concepts", "Evaluation"]

    compact_payload = {
        "title": "General Subject",
        "chapters": [{"title": "Week 1: Foundations", "lessons": [{"title": "Core concepts"}, "Evaluation"]}],
    }
    coursebuilder._normalize_markdown_plan_payload(compact_payload, structure)
    compact_plan = coursebuilder.MarkdownCoursePlan.model_validate(compact_payload)
    assert compact_plan.chapters[0].subchapters == ["Core concepts", "Evaluation"]


def test_evidence_binding_keeps_single_unit_files_in_their_source_chapters() -> None:
    import local_api.services.coursebuilder as coursebuilder

    outline = coursebuilder.CourseOutline(
        title="General Subject",
        chapters=[
            coursebuilder.OutlineChapter(
                title="Week 1: Foundations",
                lessons=[coursebuilder.OutlineLesson(title="Core concepts")],
            ),
            coursebuilder.OutlineChapter(
                title="Week 2: Evaluation",
                lessons=[coursebuilder.OutlineLesson(title="Evaluation metrics")],
            ),
        ],
    )
    chunks = [
        {
            **_chunk("week-one-0", "Foundational definitions and core concepts."),
            "source_file_id": "file-one",
            "source_filename": "first_source.pdf",
            "metadata": {
                "course_unit_title": "Week 1: Foundations",
                "course_unit_role": "primary",
                "heading_path": "Core concepts",
            },
        },
        {
            **_chunk("week-one-1", "Additional foundational evidence."),
            "source_file_id": "file-one",
            "source_filename": "first_source.pdf",
            "metadata": {"heading_path": "Foundations"},
        },
        {
            **_chunk("week-two-0", "Evaluation metrics compare system quality."),
            "source_file_id": "file-two",
            "source_filename": "second_source.pdf",
            "metadata": {
                "course_unit_title": "Week 2: Evaluation",
                "course_unit_role": "primary",
                "heading_path": "Evaluation metrics",
            },
        },
        {
            **_chunk("guide-0", "A guide to evaluation metrics and deployment checks."),
            "source_file_id": "guide",
            "source_filename": "student_guide.pdf",
            "metadata": {"heading_path": "Evaluation guidance"},
        },
    ]

    bound = coursebuilder._validate_plan_with_graph(outline, chunks, None, preserve_structure=True)

    assert set(bound.chapters[0].source_chunk_ids) == {"week-one-0", "week-one-1"}
    assert set(bound.chapters[1].source_chunk_ids) == {"week-two-0", "guide-0"}


def test_evidence_binding_splits_one_source_across_matching_subchapters() -> None:
    import local_api.services.coursebuilder as coursebuilder

    outline = coursebuilder.CourseOutline(
        title="General Subject",
        chapters=[
            coursebuilder.OutlineChapter(
                title="Week 1: Methods",
                lessons=[
                    coursebuilder.OutlineLesson(title="Foundations"),
                    coursebuilder.OutlineLesson(title="Evaluation metrics"),
                    coursebuilder.OutlineLesson(title="Deployment"),
                ],
            )
        ],
    )
    chunks = [
        {
            **_chunk("shared-0", "Foundational concepts define the method."),
            "source_file_id": "shared-file",
            "source_filename": "course.pdf",
            "metadata": {"section_title": "Foundations", "heading_path": "Foundations"},
        },
        {
            **_chunk("shared-1", "Evaluation metrics measure ranking quality."),
            "source_file_id": "shared-file",
            "source_filename": "course.pdf",
            "metadata": {"section_title": "Evaluation metrics", "heading_path": "Evaluation metrics"},
        },
        {
            **_chunk("shared-2", "Deployment moves the system into production."),
            "source_file_id": "shared-file",
            "source_filename": "course.pdf",
            "metadata": {"section_title": "Deployment", "heading_path": "Deployment"},
        },
    ]

    bound = coursebuilder._validate_plan_with_graph(outline, chunks, None, preserve_structure=True)

    assert [lesson.source_chunk_ids for lesson in bound.chapters[0].lessons] == [
        ["shared-0"],
        ["shared-1"],
        ["shared-2"],
    ]


def test_preserved_plan_rebinds_stale_first_lesson_content_and_keeps_fallbacks_distinct() -> None:
    import local_api.services.coursebuilder as coursebuilder

    unit = "Week 1: Core methods"
    lessons = ["Alpha foundations", "Beta procedure", "Gamma evaluation"]
    markers = [
        _chunk(f"marker-{index}", f"Source plan item: {title}", unit=unit, lesson=title)
        for index, title in enumerate(lessons)
    ]
    teaching_chunks = []
    for index, title in enumerate(lessons, start=3):
        topic = title.split()[0]
        chunk = _chunk(
            f"content-{index}",
            " ".join(
                [
                    f"{topic} has its own mechanism, constraints, worked explanation, and supported conclusion."
                ]
                * 28
            ),
            unit=unit,
        )
        chunk["metadata"]["section_title"] = title
        chunk["metadata"]["heading_path"] = f"{unit} > {title}"
        teaching_chunks.append(chunk)
    chunks = [*markers, *teaching_chunks]
    stale_ids = [chunk["id"] for chunk in chunks]
    outline = coursebuilder.CourseOutline(
        title="Methods",
        chapters=[
            coursebuilder.OutlineChapter(
                title=unit,
                source_chunk_ids=stale_ids,
                lessons=[
                    coursebuilder.OutlineLesson(
                        title=title,
                        source_chunk_ids=stale_ids if index == 0 else [],
                    )
                    for index, title in enumerate(lessons)
                ],
            )
        ],
    )

    bound = coursebuilder._validate_plan_with_graph(
        outline,
        chunks,
        graph=None,
        preserve_structure=True,
    )

    assert [
        [source_id for source_id in lesson.source_chunk_ids if source_id.startswith("content-")]
        for lesson in bound.chapters[0].lessons
    ] == [["content-3"], ["content-4"], ["content-5"]]

    course = coursebuilder._build_course_from_outline_fallback(
        "conversation-1",
        chunks,
        "fingerprint",
        bound,
        "build-1",
        None,
    )
    contents = [
        "\n".join(block["content"] for block in lesson["blocks"])
        for lesson in course["chapters"][0]["lessons"]
    ]
    assert len(set(contents)) == len(lessons)
    assert all(topic in content for topic, content in zip(("Alpha", "Beta", "Gamma"), contents, strict=True))


def test_teaching_bullets_and_named_architectures_are_not_lost_as_plan_structure() -> None:
    import local_api.services.coursebuilder as coursebuilder

    challenge = _chunk(
        "chunk-0",
        "\n".join(
            f"* Challenge {index} explains a supported limitation and its practical consequence"
            for index in range(10)
        ),
    )
    challenge["metadata"]["section_title"] = "Challenges and limitations of Deep Learning"
    challenge["metadata"]["heading_path"] = "Challenges and limitations of Deep Learning"
    cover = _chunk("chunk-1", "Course title Instructor Programme", unit="Source material")
    cover["metadata"]["section_title"] = "Source material"
    architecture = _chunk(
        "chunk-2",
        "Auto-encoders learn a compressed representation before reconstructing the supported input features. " * 20,
    )
    architecture["metadata"]["section_title"] = "Auto-encoders for collaborative filtering"
    architecture["metadata"]["heading_path"] = "Auto-encoders for collaborative filtering"
    broad = coursebuilder.OutlineLesson(title="Deep Learning for recommendation")
    overview = coursebuilder.OutlineLesson(title="Overview of other neural architectures")
    by_id = {architecture["id"]: architecture}

    assert not coursebuilder._looks_like_structure_only_chunk(challenge)
    assert coursebuilder._looks_like_structure_only_chunk(cover)
    assert coursebuilder._chunk_lesson_affinity(architecture, overview, by_id, set()) > (
        coursebuilder._chunk_lesson_affinity(architecture, broad, by_id, set())
    )


def test_source_structure_outline_preserves_exact_titles_order_and_counts() -> None:
    import local_api.services.coursebuilder as coursebuilder
    from local_api.services.coursebuilder_structure import SourceChapter, SourceLesson, SourceStructure

    rich = " ".join(["Matrices encode linear transformations and preserve the stated algebraic relationships."] * 30)
    chunks = [
        _chunk("chunk-0", rich, unit="Week 2: Matrices", lesson="Matrix products"),
        _chunk("chunk-1", rich, unit="Week 1: Vectors", lesson="Vector spaces"),
        _chunk("chunk-2", rich, unit="Week 1: Vectors", lesson="Linear independence"),
    ]
    structure = SourceStructure(
        title="Linear Algebra",
        origin="intake_metadata",
        chapters=[
            SourceChapter(
                title="Week 1: Vectors",
                lessons=[SourceLesson("Vector spaces"), SourceLesson("Linear independence")],
            ),
            SourceChapter(title="Week 2: Matrices", lessons=[SourceLesson("Matrix products")]),
        ],
    )

    outline = coursebuilder._outline_from_source_structure(structure, chunks)
    validated = coursebuilder._validate_plan_with_graph(
        outline,
        chunks,
        graph=None,
        preserve_structure=True,
    )

    assert [chapter.title for chapter in validated.chapters] == ["Week 1: Vectors", "Week 2: Matrices"]
    assert [lesson.title for lesson in validated.chapters[0].lessons] == ["Vector spaces", "Linear independence"]
    assert [lesson.title for lesson in validated.chapters[1].lessons] == ["Matrix products"]
    assert all(not lesson.title.startswith("Introduction to") for chapter in validated.chapters for lesson in chapter.lessons)
    covered = {source_id for chapter in validated.chapters for lesson in chapter.lessons for source_id in lesson.source_chunk_ids}
    assert covered == {"chunk-0", "chunk-1", "chunk-2"}

    changed = outline.model_copy(deep=True)
    changed.chapters[0].lessons[0].title = "A rewritten lesson title"
    assert coursebuilder._same_source_skeleton(outline, outline.model_copy(deep=True))
    assert not coursebuilder._same_source_skeleton(outline, changed)


def test_block_pipeline_retries_then_falls_back_without_losing_chapter(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    rich = " ".join(
        [
            "A vector space is closed under addition and scalar multiplication, with each axiom defining a required relationship."
        ]
        * 35
    )
    chunks = [_chunk("chunk-0", rich, unit="Vectors", lesson="Vector spaces")]
    outline = coursebuilder.OutlineChapter(
        title="Vectors",
        source_chunk_ids=["chunk-0"],
        lessons=[
            coursebuilder.OutlineLesson(
                title="Vector spaces",
                source_chunk_ids=["chunk-0"],
                source_queries=["vector space axioms"],
            )
        ],
    )

    class Retrieval:
        async def retrieve_for(self, **_kwargs):
            return []

    calls = {"block": 0}

    async def fake_complete(_provider, _messages, *, json_schema, **_kwargs):
        title = json_schema.get("title")
        if title == "LessonBlockPlanBatch":
            return json.dumps(
                {"blocks": [{"block_type": "definition", "title": "Core explanation", "source_query": "vector space axioms"}]}
            )
        if title == "DraftBlock":
            calls["block"] += 1
            return json.dumps(
                {
                    "block_type": "definition",
                    "title": "Core explanation",
                    "content": "This response is deliberately too short to satisfy the teaching quality threshold.",
                    "source_chunk_ids": ["chunk-0"],
                    "source_query": "vector space axioms",
                }
            )
        raise RuntimeError("quiz model unavailable")

    monkeypatch.setattr(coursebuilder, "get_retrieval_service", lambda: Retrieval())
    monkeypatch.setattr(coursebuilder, "complete_text", fake_complete)
    chapter = asyncio.run(
        coursebuilder._build_chapter_with_quality(
            SimpleNamespace(),
            "conversation-1",
            0,
            outline,
            chunks,
            "",
        )
    )

    assert calls["block"] == 2
    assert chapter["generation_metadata"]["block_retry_count"] == 1
    assert chapter["generation_metadata"]["fallback_block_count"] == 1
    assert chapter["lessons"][0]["blocks"][0]["validation_status"] == "supported"
    assert chapter["lessons"][0]["blocks"][0]["source_chunk_ids"] == ["chunk-0"]
    for block in chapter["lessons"][0]["blocks"]:
        assert block["validation_status"] in {"supported", "insufficient_source_material"}
        assert block["citations"] or block["validation_status"] == "insufficient_source_material"


def test_platform_style_provider_wrappers_are_normalized_into_local_blocks(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    rich = " ".join(
        ["Matrix factorization represents users and items with grounded latent factors."] * 35
    )
    chunks = [_chunk("chunk-0", rich, unit="Collaborative filtering", lesson="Model-based CF")]
    lesson = coursebuilder.OutlineLesson(title="Model-based CF")
    chapter = coursebuilder.OutlineChapter(title="Collaborative filtering", lessons=[lesson])

    responses = iter(
        [
            {
                "lesson_title": "Model-based CF",
                "teaching_blocks": [
                    {
                        "type": "explanation",
                        "heading": "Latent-factor mechanism",
                        "content_focus": "SVD latent user and item factors",
                    }
                ],
            },
            {
                "lesson_block": {
                    "type": "explanation",
                    "title": "Latent-factor mechanism",
                    "content": {
                        "meaning": " ".join(["The source defines latent user and item factors."] * 35),
                        "mechanism": " ".join(["Their interaction reconstructs supported preferences."] * 30),
                    },
                    "citations": ["chunk-0"],
                }
            },
        ]
    )

    async def wrapped_provider(*_args, **_kwargs):
        return json.dumps(next(responses))

    monkeypatch.setattr(coursebuilder, "complete_text", wrapped_provider)
    plans = asyncio.run(
        coursebuilder._plan_lesson_blocks_with_llm(
            SimpleNamespace(),
            chapter,
            lesson,
            chunks,
            "",
        )
    )
    block = asyncio.run(
        coursebuilder._generate_lesson_block(
            SimpleNamespace(),
            chapter,
            lesson,
            plans[0],
            chunks,
        )
    )

    assert plans[0].block_type == "markdown"
    assert plans[0].source_query == "SVD latent user and item factors"
    assert block.block_type == "markdown"
    assert block.source_chunk_ids == ["chunk-0"]
    assert "**Meaning:**" in block.content
    assert "**Mechanism:**" in block.content
    root_text = coursebuilder._normalize_generated_block_payload(
        {"text": "A provider may return lesson prose directly at the root level.", "citations": ["chunk-0"]},
        plans[0],
    )
    assert root_text["content"].startswith("A provider may return")
    assert root_text["source_chunk_ids"] == ["chunk-0"]


def test_standard_build_generates_one_model_backed_block_per_lesson(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    text = " ".join(
        ["Gradient descent updates parameters using the supported negative gradient mechanism."] * 40
    )
    chunks = [_chunk("chunk-0", text, unit="Optimization", lesson="Gradient descent")]
    chunks[0]["metadata"]["equations"] = [r"nDCG@K = \frac{DCG@K}{IDCG@K}"]
    outline = coursebuilder.OutlineChapter(
        title="Optimization",
        source_chunk_ids=["chunk-0"],
        lessons=[
            coursebuilder.OutlineLesson(
                title="Gradient descent",
                source_chunk_ids=["chunk-0"],
                source_queries=["negative gradient update"],
            )
        ],
    )

    class Retrieval:
        async def retrieve_for(self, **_kwargs):
            return [SimpleNamespace(chunk_id="chunk-0")]

    async def one_pass_content(_provider, _messages, *, json_schema, **_kwargs):
        if json_schema.get("title") == "DraftBlock":
            return json.dumps(
                {
                    "text": " ".join(
                        ["The negative gradient gives the supported direction for reducing the objective."] * 35
                    ),
                    "citations": ["chunk-0"],
                }
            )
        raise RuntimeError("quiz generation disabled")

    async def unexpected_plan(*_args, **_kwargs):
        raise AssertionError("standard builds must not invoke the multi-block planner")

    monkeypatch.setattr(coursebuilder, "get_retrieval_service", lambda: Retrieval())
    monkeypatch.setattr(coursebuilder, "complete_text", one_pass_content)
    monkeypatch.setattr(coursebuilder, "_plan_lesson_blocks_with_llm", unexpected_plan)
    chapter = asyncio.run(
        coursebuilder._build_chapter_with_quality(
            SimpleNamespace(),
            "conversation-1",
            0,
            outline,
            chunks,
            "",
            detailed_blocks=False,
        )
    )

    blocks = chapter["lessons"][0]["blocks"]
    assert blocks[0]["validation_status"] == "supported"
    assert blocks[0]["source_chunk_ids"] == ["chunk-0"]
    assert "negative gradient" in blocks[0]["content"].casefold()
    assert len(blocks) == 1
    assert chapter["generation_metadata"]["fallback_block_count"] == 0


def test_unsupported_lesson_does_not_borrow_neighbor_content(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    plan_marker = _chunk(
        "chunk-0",
        "Source plan item: Ethics and bias in education",
        unit="Evaluation",
        lesson="Ethics and bias in education",
    )
    metric = _chunk(
        "chunk-1",
        " ".join(["nDCG discounts ranking gain by logarithmic position."] * 35),
        unit="Evaluation",
    )
    metric["metadata"]["section_title"] = "Ranking metrics"
    chapter = coursebuilder.OutlineChapter(
        title="Evaluation",
        source_chunk_ids=["chunk-0", "chunk-1"],
        lessons=[
            coursebuilder.OutlineLesson(
                title="Ethics and bias in education",
                source_chunk_ids=["chunk-0"],
            )
        ],
    )

    class Retrieval:
        async def retrieve_for(self, **_kwargs):
            return [SimpleNamespace(chunk_id="chunk-0"), SimpleNamespace(chunk_id="chunk-1")]

    monkeypatch.setattr(coursebuilder, "get_retrieval_service", lambda: Retrieval())
    selected, _count = asyncio.run(
        coursebuilder._retrieve_lesson_evidence(
            "conversation-1",
            chapter,
            chapter.lessons[0],
            [plan_marker, metric],
        )
    )
    fallback, status = coursebuilder._fallback_generated_block(
        coursebuilder.LessonBlockPlan(
            title="Ethics and bias in education",
            source_query="ethics bias education",
        ),
        [plan_marker],
    )

    assert selected == []
    assert status == "insufficient_source_material"
    assert fallback.source_chunk_ids == []
    assert "source plan item" not in fallback.content.casefold()


def test_planned_teaching_evidence_wins_over_broad_retrieval_hits(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    planned = _chunk(
        "chunk-0",
        " ".join(
            [
                "The user-item matrix stores explicit and implicit interactions as the data model for recommendation."
            ]
            * 10
        ),
        unit="Recommendation foundations",
        lesson="Data modeling",
    )
    neighbor = _chunk(
        "chunk-1",
        " ".join(["Popularity recommends the same frequently selected resources to every learner."] * 20),
        unit="Recommendation foundations",
        lesson="Recommendation taxonomy",
    )
    chapter = coursebuilder.OutlineChapter(
        title="Recommendation foundations",
        source_chunk_ids=["chunk-0", "chunk-1"],
        lessons=[
            coursebuilder.OutlineLesson(
                title="Data modeling",
                source_chunk_ids=["chunk-0"],
                source_queries=[
                    "Recommendation foundations",
                    "user-item matrix interaction data",
                ],
            )
        ],
    )

    class Retrieval:
        async def retrieve_for(self, **_kwargs):
            return [SimpleNamespace(chunk_id="chunk-1")]

    monkeypatch.setattr(coursebuilder, "get_retrieval_service", lambda: Retrieval())
    selected, _count = asyncio.run(
        coursebuilder._retrieve_lesson_evidence(
            "conversation-1",
            chapter,
            chapter.lessons[0],
            [planned, neighbor],
        )
    )

    assert [chunk["id"] for chunk in selected] == ["chunk-0"]


def test_fallback_focuses_on_the_lesson_inside_a_mixed_source_chunk() -> None:
    import local_api.services.coursebuilder as coursebuilder

    mixed = _chunk(
        "chunk-0",
        " ".join(
            [
                "IDCG is the ideal discounted cumulative gain used to normalize a ranking metric.",
                "The metric discounts relevance according to an item's position in a ranked list.",
                "Ethical recommendation requires examining popularity bias in the training interactions.",
                "Popularity bias can amplify already visible items and reduce exposure for minority interests.",
                "Educational recommenders must also protect learner autonomy and avoid reinforcing a filter bubble.",
                "A diversity objective broadens exposure while the interface should explain why an item was recommended.",
            ]
            * 8
        ),
        unit="Evaluation",
    )
    fallback, status = coursebuilder._fallback_generated_block(
        coursebuilder.LessonBlockPlan(
            title="Ethics and bias in education",
            source_query="ethics popularity bias learner autonomy filter bubble",
        ),
        [mixed],
    )

    assert status == "supported"
    assert "popularity bias" in fallback.content.casefold()
    assert "learner autonomy" in fallback.content.casefold()
    assert "idcg" not in fallback.content.casefold()


def test_quiz_keeps_valid_questions_and_tops_up_only_rejected_rows(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    text = " ".join(["Gradient descent follows the negative gradient under the stated learning-rate conditions."] * 20)
    chunks = [_chunk("chunk-0", text)]

    async def partial_quiz(_provider, _messages, **_kwargs):
        return json.dumps(
            {
                "questions": [
                    {
                        "prompt": "Which statement correctly describes the supported gradient descent update?",
                        "options": ["Negative gradient", "Positive gradient", "No gradient", "Random direction"],
                        "correct_index": 0,
                        "explanation": "The source specifies movement along the negative gradient.",
                        "source_chunk_id": "chunk-0",
                    },
                    {
                        "prompt": "Which unsupported source should be trusted?",
                        "options": ["A", "B", "C", "D"],
                        "correct_index": 0,
                        "explanation": "This citation is outside the allowed evidence set.",
                        "source_chunk_id": "unknown",
                    },
                ]
            }
        )

    monkeypatch.setattr(coursebuilder, "complete_text", partial_quiz)
    quiz = asyncio.run(coursebuilder._build_quiz_with_llm(SimpleNamespace(), "Optimization", chunks, 4, "chapter"))

    assert len(quiz["questions"]) == 4
    assert quiz["questions"][0]["prompt"].startswith("Which statement correctly")
    assert quiz["generation_metadata"]["fallback_question_count"] == 3
    assert all(question["source_chunk_ids"] == ["chunk-0"] for question in quiz["questions"])


def test_generated_block_rejects_unknown_citations(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunks = [_chunk("chunk-0", " ".join(["Evidence defines the supported mathematical relationship."] * 40))]

    async def invalid_block(_provider, _messages, **_kwargs):
        return json.dumps(
            {
                "block_type": "definition",
                "title": "Definition",
                "content": " ".join(["A detailed explanation is present, but its citation is deliberately invalid."] * 20),
                "source_chunk_ids": ["unknown"],
                "source_query": "definition",
            }
        )

    monkeypatch.setattr(coursebuilder, "complete_text", invalid_block)
    with pytest.raises(ValueError, match="outside its retrieval scope"):
        asyncio.run(
            coursebuilder._generate_lesson_block(
                SimpleNamespace(),
                coursebuilder.OutlineChapter(
                    title="Chapter",
                    source_chunk_ids=["chunk-0"],
                    lessons=[coursebuilder.OutlineLesson(title="Lesson", source_chunk_ids=["chunk-0"])],
                ),
                coursebuilder.OutlineLesson(title="Lesson", source_chunk_ids=["chunk-0"]),
                coursebuilder.LessonBlockPlan(block_type="definition", title="Definition", source_query="definition"),
                chunks,
            )
        )


def test_markdown_toc_merges_duplicate_chapters_and_preserves_order() -> None:
    from local_api.services.coursebuilder_structure import extract_source_structure

    markdown = """### Markdown source 1: book.md
```markdown
Contents
Chapter 1: Vector Spaces ........ 1
Definitions ........ 2
Subspaces ........ 4
Chapter 2: Matrices ........ 8
Matrix products ........ 9
Chapter 1: Vector Spaces ........ 12
Applications ........ 13
```
"""
    structure = extract_source_structure(chunks=[], sections=[], documents=[], markdown=markdown)

    assert structure is not None
    assert structure.origin == "markdown_toc"
    assert [chapter.title for chapter in structure.chapters] == ["Vector Spaces", "Matrices"]
    assert [lesson.title for lesson in structure.chapters[0].lessons] == [
        "Definitions",
        "Subspaces",
        "Applications",
    ]


def test_representative_context_is_balanced_across_documents() -> None:
    from local_api.services.coursebuilder_structure import select_representative_chunks

    chunks = [
        {
            "id": f"{file_id}-{index}",
            "source_file_id": file_id,
            "chunk_index": index,
            "text": f"{file_id} evidence {index}",
        }
        for file_id in ("primary-a", "primary-b", "supplemental")
        for index in range(7)
    ]
    selected = select_representative_chunks(chunks, per_file=3)

    assert len(selected) == 9
    assert {
        file_id: sum(chunk["source_file_id"] == file_id for chunk in selected)
        for file_id in ("primary-a", "primary-b", "supplemental")
    } == {"primary-a": 3, "primary-b": 3, "supplemental": 3}


def test_navigation_and_plan_marker_chunks_are_not_teaching_evidence() -> None:
    import local_api.services.coursebuilder as coursebuilder

    assert coursebuilder._looks_like_structure_only_chunk(
        _chunk("chunk-0", "Previous page Next page Back to top Download PDF")
    )
    assert coursebuilder._looks_like_structure_only_chunk(
        _chunk("chunk-1", "Source plan item: Matrix factorization")
    )
    assert not coursebuilder._looks_like_structure_only_chunk(
        _chunk("chunk-2", " ".join(["Matrix factorization explains latent user-item relationships."] * 30))
    )


def test_equation_blocks_must_match_retrieved_source_math(monkeypatch) -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunk = _chunk("chunk-0", "The force relationship is $$F = ma$$ and each symbol is defined in the source.")
    chunk["metadata"]["equations"] = ["F = ma"]

    async def generated(_provider, _messages, **_kwargs):
        return json.dumps(
            {
                "block_type": "equation",
                "title": "Force law",
                "content": "$$F = ma$$\n\nThe source relates force, mass, and acceleration through this expression.",
                "source_chunk_ids": ["chunk-0"],
            }
        )

    monkeypatch.setattr(coursebuilder, "complete_text", generated)
    block = asyncio.run(
        coursebuilder._generate_lesson_block(
            SimpleNamespace(),
            coursebuilder.OutlineChapter(
                title="Mechanics",
                source_chunk_ids=["chunk-0"],
                lessons=[coursebuilder.OutlineLesson(title="Force", source_chunk_ids=["chunk-0"])],
            ),
            coursebuilder.OutlineLesson(title="Force", source_chunk_ids=["chunk-0"]),
            coursebuilder.LessonBlockPlan(block_type="equation", title="Force law", source_query="force equation"),
            [chunk],
        )
    )
    assert block.source_chunk_ids == ["chunk-0"]


def test_table_blocks_reject_values_not_present_in_retrieved_rows() -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunk = _chunk("chunk-0", "| Metric | Value |\n| --- | --- |\n| Precision | 0.82 |")
    plan = coursebuilder.LessonBlockPlan(block_type="table", title="Metrics", source_query="metric values")
    unsupported = coursebuilder.DraftBlock(
        block_type="table",
        title="Metrics",
        content="| Metric | Value |\n| --- | --- |\n| Recall | 99.0 |",
        source_chunk_ids=["chunk-0"],
    )

    with pytest.raises(ValueError, match="not supported"):
        coursebuilder._validate_generated_block(unsupported, plan, [chunk], "Evaluation")


def test_deterministic_special_blocks_do_not_duplicate_generated_equations() -> None:
    import local_api.services.coursebuilder as coursebuilder

    chunk = _chunk("chunk-0", "The source defines $$F = ma$$ for the stated variables.")
    generated = [{"block_type": "equation", "content": "$$F = ma$$", "source_chunk_ids": ["chunk-0"]}]

    special = coursebuilder._special_blocks(
        "Force",
        [chunk],
        start_index=1,
        existing_blocks=generated,
    )

    assert all(block["block_type"] != "equation" for block in special)


def test_parser_markup_becomes_structured_content_without_repeated_prose() -> None:
    import local_api.services.coursebuilder as coursebuilder
    from local_api.services.ingestion import _clean_text

    source = """The source introduces a general comparison method and explains why ordered results matter.
It then supplies the values used by the method in a structured example.
<table><tr><th>Item</th><th>Score</th></tr><tr><td>Alpha</td><td>5</td></tr><tr><td>Beta</td><td>100</td></tr></table>
<page_number>8</page_number>
$$S = A / B$$
$$R = S / M$$
The interpretation connects the computed result to the decision described by the source."""
    cleaned = _clean_text(source)

    assert "<table" not in cleaned
    assert "page_number" not in cleaned
    assert "| Item | Score |" in cleaned

    blocks = coursebuilder._lesson_blocks("General comparison", [_chunk("chunk-0", source)])
    assert [block["block_type"] for block in blocks] == ["markdown", "table", "equation"]
    assert blocks[1]["data_json"]["headers"] == ["Item", "Score"]
    assert blocks[1]["data_json"]["rows"] == [["Alpha", "5"], ["Beta", "100"]]
    assert blocks[2]["content"].count("$$") == 4
    assert all("page_number" not in block["content"] for block in blocks)
    assert coursebuilder._timeline_events(source) == []


def test_redundant_fallback_blocks_are_removed_but_distinct_examples_remain() -> None:
    import local_api.services.coursebuilder as coursebuilder

    explanation = (
        "The source defines a general process and states its operating conditions. "
        "It also explains how the result should be interpreted in later decisions."
    )
    blocks = [
        {"block_type": "markdown", "content": explanation},
        {"block_type": "example", "content": explanation},
        {"block_type": "summary", "content": "The result should be interpreted in later decisions."},
        {
            "block_type": "example",
            "content": "A separate worked case applies the process to concrete values supplied by the source.",
        },
    ]

    deduped = coursebuilder._dedupe_lesson_blocks(blocks)
    assert [block["block_type"] for block in deduped] == ["markdown", "example"]

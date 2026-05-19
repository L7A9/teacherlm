from __future__ import annotations

import json
from pathlib import Path


def test_practical_eval_catalog_has_required_student_question_mix() -> None:
    path = Path(__file__).resolve().parents[1] / "evals" / "practical_student_questions.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) >= 30
    categories = {case["category"] for case in cases}
    assert {
        "general_overview",
        "vague_confused",
        "definition",
        "formula",
        "comparison",
        "exam_prep",
        "quiz",
        "mindmap",
        "out_of_scope",
    }.issubset(categories)

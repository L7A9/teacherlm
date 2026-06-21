from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any


_UNIT_RE = re.compile(
    r"\b(?P<label>semaine|week|lecture|chapter|chapitre|module|unit|unite|unité)\s*"
    r"(?:n[°o.]?\s*)?"
    r"(?P<number>\d{1,2}|[ivxlcdm]{1,8}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    r"\s*(?:[:.\-–]\s*)?(?P<title>.{0,180})",
    re.IGNORECASE,
)
_SUPPLEMENTAL_RE = re.compile(
    r"\b(student\s+guide|study\s+guide|syllabus|appendix|annex|bibliograph(?:y|ie)|references?|"
    r"recommended\s+readings?|lectures?\s+recommandees?|table\s+des\s+matieres)\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(plan\s+de\s+la\s+seance|agenda|outline|course\s+outline|contents|"
    r"table\s+of\s+contents|table\s+des\s+matieres)\b",
    re.IGNORECASE,
)
_NUMBERED_ITEM_RE = re.compile(
    r"(?:^|\s)(?P<number>\d{1,2})\s*[.)]?\s*(?P<title>.+?)"
    r"(?=(?:\s+\d{1,2}\s*[.)]?\s*(?=[A-ZÀ-ÖØ-Þ0-9]))|$)",
    re.DOTALL,
)
_PAGE_COUNTER_RE = re.compile(r"\b\d{1,3}\s*/\s*\d{1,3}\b")
_NOISE_TITLE_RE = re.compile(
    r"^(?:plan\s+de\s+la\s+seance|agenda|outline|contents?|table\s+des\s+matieres|"
    r"references?|bibliograph(?:y|ie)|source\s+material)$",
    re.IGNORECASE,
)
_FORMULA_RE = re.compile(r"[=∈≈Σ∑√^]|\\[a-zA-Z]+|\|[A-Z]\|")

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


@dataclass(slots=True)
class NormalizedSubchapter:
    title: str
    order_index: int
    confidence: float = 0.75


@dataclass(slots=True)
class NormalizedCourseUnit:
    title: str
    role: str
    order_index: int
    source_line_index: int
    source_end_line_index: int
    unit_number: int | None = None
    confidence: float = 0.8
    subchapters: list[NormalizedSubchapter] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedCourseIntake:
    markdown: str
    units: list[NormalizedCourseUnit]
    normalized: bool
    metadata: dict[str, Any]


class CourseIntakeNormalizer:
    """Turn noisy parser markdown into explicit, source-grounded course units."""

    def normalize(
        self,
        *,
        raw_markdown: str,
        cleaned_markdown: str,
        source_filename: str,
    ) -> NormalizedCourseIntake:
        markdown = str(cleaned_markdown or raw_markdown or "")
        lines = markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        units = _detect_units(lines, source_filename)
        if not units:
            return NormalizedCourseIntake(
                markdown=markdown,
                units=[],
                normalized=False,
                metadata={
                    "intake_normalized": False,
                    "primary_unit_count": 0,
                    "supplemental_unit_count": 0,
                    "intake_reason": "no_course_units_detected",
                },
            )

        units = _attach_subchapters(units, lines)
        metadata = {
            "intake_normalized": True,
            "primary_unit_count": sum(unit.role == "primary" for unit in units),
            "supplemental_unit_count": sum(unit.role == "supplemental" for unit in units),
            "course_units": [_unit_metadata(unit) for unit in units],
        }
        return NormalizedCourseIntake(
            markdown=_render_normalized_markdown(lines, units, source_filename),
            units=units,
            normalized=True,
            metadata=metadata,
        )


def section_intake_metadata(
    heading_path: list[str],
    section_title: str,
    intake_metadata: dict[str, Any],
) -> dict[str, Any]:
    units = intake_metadata.get("course_units") or []
    if not isinstance(units, list):
        return {}
    by_title = {
        _title_key(str(unit.get("course_unit_title") or "")): unit
        for unit in units
        if isinstance(unit, dict) and unit.get("course_unit_title")
    }
    matched: dict[str, Any] | None = None
    for title in heading_path:
        matched = by_title.get(_title_key(title))
        if matched is not None:
            break
    if matched is None:
        return {}

    subchapters = [
        _clean_title(str(item))
        for item in matched.get("subchapter_titles") or []
        if _clean_title(str(item))
    ]
    section_key = _title_key(section_title)
    exact_subchapter = next((title for title in subchapters if _title_key(title) == section_key), "")
    metadata = {
        "course_unit_index": matched.get("course_unit_index"),
        "course_unit_title": matched.get("course_unit_title"),
        "course_unit_role": matched.get("course_unit_role") or "primary",
        "subchapter_titles": subchapters,
        "intake_confidence": matched.get("intake_confidence", 0.0),
    }
    if exact_subchapter:
        metadata["subchapter_title"] = exact_subchapter
    return metadata


def _detect_units(lines: list[str], source_filename: str) -> list[NormalizedCourseUnit]:
    candidates: list[NormalizedCourseUnit] = []
    seen_numbers: set[int] = set()
    seen_titles: set[str] = set()
    supplemental_started = False
    for index, raw_line in enumerate(lines):
        line = _compact(raw_line)
        if not line:
            continue
        if supplemental_started:
            continue
        unit = _primary_unit_from_line(line, index)
        if unit is not None:
            key = _title_key(unit.title)
            if unit.unit_number in seen_numbers or key in seen_titles:
                continue
            if unit.unit_number is not None:
                seen_numbers.add(unit.unit_number)
            seen_titles.add(key)
            candidates.append(unit)
            continue
        if candidates and not supplemental_started and _SUPPLEMENTAL_RE.search(_strip_accents(line)):
            supplemental_started = True
            candidates.append(
                NormalizedCourseUnit(
                    title=_clean_title(line) or "Supplemental Material",
                    role="supplemental",
                    order_index=len(candidates),
                    source_line_index=index,
                    source_end_line_index=len(lines),
                    confidence=0.65,
                )
            )

    if not candidates and _SUPPLEMENTAL_RE.search(_strip_accents(PurePath(source_filename).stem)):
        candidates.append(
            NormalizedCourseUnit(
                title=PurePath(source_filename).stem.replace("_", " ").strip() or "Supplemental Material",
                role="supplemental",
                order_index=0,
                source_line_index=0,
                source_end_line_index=len(lines),
                confidence=0.55,
            )
        )

    candidates.sort(key=lambda unit: unit.source_line_index)
    for index, unit in enumerate(candidates):
        unit.order_index = index
        unit.source_end_line_index = (
            candidates[index + 1].source_line_index if index + 1 < len(candidates) else len(lines)
        )
    return candidates


def _primary_unit_from_line(line: str, line_index: int) -> NormalizedCourseUnit | None:
    folded_line = _strip_accents(line)
    match = _UNIT_RE.search(folded_line)
    if not match:
        return None
    if match.end("label") < len(folded_line) and folded_line[match.end("label")].isalnum():
        return None
    if match.end("number") < len(folded_line) and folded_line[match.end("number")].isalnum():
        return None
    number = _number_value(match.group("number"))
    if number is None:
        return None
    title = _clean_unit_title(line[match.start():])
    if not _valid_title(title):
        title = f"{match.group('label').title()} {number}"
    return NormalizedCourseUnit(
        title=title,
        role="primary",
        order_index=0,
        source_line_index=line_index,
        source_end_line_index=line_index + 1,
        unit_number=number,
        confidence=0.9,
    )


def _attach_subchapters(
    units: list[NormalizedCourseUnit],
    lines: list[str],
) -> list[NormalizedCourseUnit]:
    for unit in units:
        unit_lines = lines[unit.source_line_index:unit.source_end_line_index]
        titles = _plan_titles(unit_lines) or _heading_like_titles(unit_lines)
        unit.subchapters = [
            NormalizedSubchapter(title=title, order_index=index)
            for index, title in enumerate(_dedupe_titles(titles)[:14])
        ]
        if unit.role == "primary" and not unit.subchapters:
            unit.subchapters = [NormalizedSubchapter(unit.title, 0, confidence=0.45)]
    return units


def _plan_titles(lines: list[str]) -> list[str]:
    titles: list[str] = []
    for index, raw_line in enumerate(lines[:120]):
        line = _compact(raw_line)
        if not _PLAN_RE.search(_strip_accents(line)):
            continue
        titles.extend(_numbered_titles(line))
        for extra in lines[index + 1:index + 80]:
            raw_extra = str(extra or "")
            compact = _compact(extra)
            if not compact:
                continue
            if titles and re.match(r"^#{1,6}\s+\S+", raw_extra.strip()):
                break
            if _primary_unit_from_line(compact, index) is not None:
                break
            found = _numbered_titles(compact)
            if found:
                titles.extend(found)
        if titles:
            break
    return titles


def _heading_like_titles(lines: list[str]) -> list[str]:
    titles: list[str] = []
    for raw_line in lines[:180]:
        line = _compact(raw_line)
        heading = re.match(r"^#{2,6}\s+(.+)$", line)
        if heading and _valid_plan_title(heading.group(1)):
            titles.append(_clean_subchapter_title(heading.group(1)))
            continue
        if not line or _PLAN_RE.search(_strip_accents(line)) or _primary_unit_from_line(line, 0):
            continue
        titles.extend(_numbered_titles(line))
        if len(titles) >= 14:
            break
    return titles


def _numbered_titles(text: str) -> list[str]:
    return [
        title
        for match in _NUMBERED_ITEM_RE.finditer(_compact(text))
        if (title := _clean_subchapter_title(match.group("title"))) and _valid_plan_title(title)
    ]


def _render_normalized_markdown(
    lines: list[str],
    units: list[NormalizedCourseUnit],
    source_filename: str,
) -> str:
    root = PurePath(source_filename).stem.replace("_", " ").strip() or source_filename
    out = [f"# {root}", ""]
    for unit in units:
        out.extend([f"## {unit.title}", ""])
        for subchapter in unit.subchapters:
            out.extend([f"### {subchapter.title}", f"Source plan item: {subchapter.title}", ""])
        out.extend(["### Source material", ""])
        source = "\n".join(lines[unit.source_line_index:unit.source_end_line_index]).strip()
        if source:
            out.extend([source, ""])
    return "\n".join(out).strip() + "\n"


def _unit_metadata(unit: NormalizedCourseUnit) -> dict[str, Any]:
    return {
        "course_unit_index": unit.unit_number if unit.unit_number is not None else unit.order_index + 1,
        "course_unit_title": unit.title,
        "course_unit_role": unit.role,
        "source_line_index": unit.source_line_index,
        "source_end_line_index": unit.source_end_line_index,
        "subchapter_titles": [subchapter.title for subchapter in unit.subchapters],
        "intake_confidence": unit.confidence,
    }


def _clean_unit_title(value: str) -> str:
    text = _PAGE_COUNTER_RE.sub(" ", value)
    text = re.sub(
        r"\b(?:prof(?:essor)?|dr\.?|university|université|école|school|master)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return _clean_title(text)


def _clean_subchapter_title(value: str) -> str:
    text = re.sub(r"^[\-*•▶✓\s]+", "", str(value or "").strip())
    text = re.sub(r"\s+\d{1,4}$", "", text)
    return _clean_title(text)


def _clean_title(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip(" -*•▶✓:;,.#")[:180]


def _valid_title(value: str) -> bool:
    text = _clean_title(value)
    folded = _strip_accents(text).casefold()
    if not 4 <= len(text) <= 150 or _NOISE_TITLE_RE.fullmatch(folded):
        return False
    if _FORMULA_RE.search(text) and len(text.split()) <= 8:
        return False
    if re.fullmatch(r"\d{1,4}|[ivxlcdm]+|page\s+\d+|slide\s+\d+", folded):
        return False
    return sum(character.isalpha() for character in text) >= 3


def _valid_plan_title(value: str) -> bool:
    text = _clean_title(value)
    folded = _strip_accents(text).casefold().strip(" :-")
    if not _valid_title(text) or "|" in text or "%" in text:
        return False
    if folded in {"problem", "the problem", "probleme", "le probleme"}:
        return False
    return not re.search(r"\b(and|et|or|ou|with|avec)$", folded)


def _dedupe_titles(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        title = _clean_title(value)
        key = _title_key(title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(title)
    return out


def _number_value(value: str) -> int | None:
    text = _strip_accents(value).casefold()
    if text.isdigit():
        return int(text)
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    return _roman_to_int(text)


def _roman_to_int(value: str) -> int | None:
    if not value or not re.fullmatch(r"[ivxlcdm]+", value):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for character in reversed(value):
        current = values[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total or None


def _title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _strip_accents(value).casefold())


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


_normalizer: CourseIntakeNormalizer | None = None


def get_course_intake_normalizer() -> CourseIntakeNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = CourseIntakeNormalizer()
    return _normalizer


def normalize_course_intake(
    *,
    raw_markdown: str,
    cleaned_markdown: str,
    source_filename: str,
) -> NormalizedCourseIntake:
    return get_course_intake_normalizer().normalize(
        raw_markdown=raw_markdown,
        cleaned_markdown=cleaned_markdown,
        source_filename=source_filename,
    )

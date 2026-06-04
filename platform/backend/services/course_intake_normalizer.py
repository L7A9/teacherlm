from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


_UNIT_RE = re.compile(
    r"\b(?P<label>semaine|week|lecture|chapter|module|unit|unite|unit[eé])\s*"
    r"(?:n[°o.]?\s*)?"
    r"(?P<number>\d{1,2}|[ivxlcdm]{1,8}|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    r"\s*(?:[:.\-–]\s*)?(?P<title>.{0,180})",
    re.IGNORECASE,
)
_SUPPLEMENTAL_RE = re.compile(
    r"\b(guide|student\s+guide|guide\s+for\s+students|syllabus|appendix|annex|"
    r"bibliography|references?|lectures?\s+recommended|table\s+des\s+mati[eè]res)\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(plan\s+de\s+la\s+s\s*(?:e|é)ance|agenda|outline|course\s+outline|contents|"
    r"table\s+of\s+contents|table\s+des\s+mati[eè]res)\b",
    re.IGNORECASE,
)
_NUMBERED_ITEM_RE = re.compile(
    r"(?:^|\s)(?P<number>\d{1,2})\s*[\.)]?\s*(?P<title>.+?)(?=(?:\s+\d{1,2}\s*[\.)]?\s*(?=[A-ZÀ-ÖØ-Þ0-9]))|$)",
    re.DOTALL,
)
_TRAILING_CUTOFF_RE = re.compile(
    r"\b(?:pr\.?|prof(?:essor)?|dr\.?|[eé]cole|school|master|october|november|december|"
    r"janvier|fevrier|f[eé]vrier|mars|avril|mai|juin|juillet|aout|ao[uû]t|septembre|"
    r"octobre|novembre|d[eé]cembre)\b",
    re.IGNORECASE,
)
_PAGE_COUNTER_RE = re.compile(r"\b\d{1,3}\s*/\s*\d{1,3}\b")
_NOISE_TITLE_RE = re.compile(
    r"^(?:plan\s+de\s+la\s+s[eé]ance|agenda|outline|contents?|table\s+des\s+mati[eè]res|"
    r"references?|bibliography|source\s+material)$",
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
    """Repairs messy uploaded course files into explicit source-grounded units."""

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
            metadata = {
                "intake_normalized": False,
                "primary_unit_count": 0,
                "supplemental_unit_count": 0,
                "intake_reason": "no_course_units_detected",
            }
            return NormalizedCourseIntake(markdown=markdown, units=[], normalized=False, metadata=metadata)

        units = _attach_subchapters(units, lines)
        normalized_markdown = _render_normalized_markdown(lines, units, source_filename)
        metadata = {
            "intake_normalized": True,
            "primary_unit_count": sum(1 for unit in units if unit.role == "primary"),
            "supplemental_unit_count": sum(1 for unit in units if unit.role == "supplemental"),
            "course_units": [_unit_metadata(unit) for unit in units],
        }
        return NormalizedCourseIntake(
            markdown=normalized_markdown,
            units=units,
            normalized=True,
            metadata=metadata,
        )


def _detect_units(lines: list[str], source_filename: str) -> list[NormalizedCourseUnit]:
    candidates: list[NormalizedCourseUnit] = []
    seen_primary_numbers: set[int] = set()
    seen_primary_titles: set[str] = set()
    supplemental_started = False

    for index, raw_line in enumerate(lines):
        line = _compact(raw_line)
        if not line:
            continue
        unit = _primary_unit_from_line(line, index)
        if unit is not None:
            if _title_needs_continuation(unit.title) and index + 1 < len(lines):
                unit.title = _clean_unit_title(f"{unit.title} {_compact(lines[index + 1])}")
            key = _title_key(unit.title)
            if unit.unit_number in seen_primary_numbers or key in seen_primary_titles:
                continue
            seen_primary_numbers.add(unit.unit_number or -1)
            seen_primary_titles.add(key)
            candidates.append(unit)
            continue

        if candidates and not supplemental_started and _looks_supplemental_boundary(line, source_filename):
            supplemental_started = True
            candidates.append(
                NormalizedCourseUnit(
                    title=_clean_title(line)[:140] or "Supplemental Material",
                    role="supplemental",
                    order_index=len(candidates),
                    source_line_index=index,
                    source_end_line_index=len(lines),
                    confidence=0.65,
                )
            )

    if not candidates and _looks_supplemental_filename(source_filename):
        candidates.append(
            NormalizedCourseUnit(
                title=PurePosixPath(source_filename).stem.replace("_", " ").strip() or "Supplemental Material",
                role="supplemental",
                order_index=0,
                source_line_index=0,
                source_end_line_index=len(lines),
                confidence=0.55,
            )
        )

    candidates.sort(key=lambda unit: unit.source_line_index)
    for idx, unit in enumerate(candidates):
        unit.order_index = idx
        unit.source_end_line_index = (
            candidates[idx + 1].source_line_index if idx + 1 < len(candidates) else len(lines)
        )
    return candidates


def _primary_unit_from_line(line: str, line_index: int) -> NormalizedCourseUnit | None:
    match = _UNIT_RE.search(_strip_accents(line))
    if not match:
        return None
    number = _number_value(match.group("number"))
    if number is None:
        return None
    original_tail = line[match.start() :]
    title = _clean_unit_title(original_tail)
    if not _valid_title(title):
        label = match.group("label").title()
        title = f"{label} {number}"
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
        unit_lines = lines[unit.source_line_index : unit.source_end_line_index]
        titles = _plan_titles(unit_lines)
        if not titles:
            titles = _heading_like_titles(unit_lines)
        unit.subchapters = [
            NormalizedSubchapter(title=title, order_index=index)
            for index, title in enumerate(_dedupe_titles(titles)[:12])
        ]
        if unit.role == "primary" and not unit.subchapters:
            unit.subchapters = [NormalizedSubchapter(title=unit.title, order_index=0, confidence=0.45)]
    return units


def _plan_titles(lines: list[str]) -> list[str]:
    titles: list[str] = []
    for index, raw_line in enumerate(lines[:120]):
        line = _compact(raw_line)
        if not _PLAN_RE.search(_strip_accents(line)):
            continue

        titles.extend(_numbered_titles(line))
        for extra in lines[index + 1 : index + 40]:
            extra_line = _compact(extra)
            if not extra_line:
                if titles:
                    break
                continue
            if _primary_unit_from_line(extra_line, index) is not None:
                break
            line_titles = _numbered_titles(extra_line)
            if not line_titles:
                if titles:
                    break
                continue
            titles.extend(line_titles)
            if len(titles) >= 12:
                break
        if titles:
            return titles
    return titles


def _heading_like_titles(lines: list[str]) -> list[str]:
    titles: list[str] = []
    for raw_line in lines[:180]:
        line = _compact(raw_line)
        if not line or _PLAN_RE.search(_strip_accents(line)):
            continue
        if _primary_unit_from_line(line, 0) is not None:
            continue
        numbered = _numbered_titles(line)
        if numbered:
            titles.extend(numbered)
            continue
        words = line.split()
        if 2 <= len(words) <= 12 and _valid_plan_title(line):
            letters = [ch for ch in line if ch.isalpha()]
            upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
            title_like = sum(1 for word in words if word[:1].isupper()) >= max(2, len(words) // 2)
            if title_like or upper_ratio > 0.55:
                titles.append(_clean_subchapter_title(line))
        if len(titles) >= 12:
            break
    return titles


def _numbered_titles(text: str) -> list[str]:
    normalized = _compact(text)
    titles: list[str] = []
    for match in _NUMBERED_ITEM_RE.finditer(normalized):
        title = _clean_subchapter_title(match.group("title"))
        if _valid_plan_title(title):
            titles.append(title)
    return titles


def _render_normalized_markdown(
    lines: list[str],
    units: list[NormalizedCourseUnit],
    source_filename: str,
) -> str:
    root_title = PurePosixPath(source_filename).stem.replace("_", " ").strip() or source_filename
    out: list[str] = [f"# {root_title}", ""]
    for unit in units:
        out.extend(["## " + unit.title, ""])
        for subchapter in unit.subchapters:
            out.extend([f"### {subchapter.title}", f"Source plan item: {subchapter.title}", ""])
        out.extend(["### Source material", ""])
        source = "\n".join(lines[unit.source_line_index : unit.source_end_line_index]).strip()
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
    cutoff = _TRAILING_CUTOFF_RE.search(_strip_accents(text))
    if cutoff:
        text = text[: cutoff.start()]
    return _clean_title(text)


def _clean_subchapter_title(value: str) -> str:
    text = re.sub(r"^[\-*•▶✓\s]+", "", str(value or "").strip())
    text = re.sub(r"\s+", " ", text)
    if ":" in text:
        before, after = text.split(":", 1)
        if len(before.split()) >= 3 or len(after) > 80:
            text = before
    text = re.sub(r"\s+\d{1,4}$", "", text)
    return _clean_title(text)


def _clean_title(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" -*•▶✓:;,.#")
    return text[:180]


def _valid_title(value: str) -> bool:
    text = _clean_title(value)
    lower = _strip_accents(text).lower()
    if not text or len(text) < 4 or len(text) > 150:
        return False
    if _NOISE_TITLE_RE.fullmatch(lower):
        return False
    if _FORMULA_RE.search(text) and len(text.split()) <= 8:
        return False
    if re.search(r"\b(pr\.?|prof(?:essor)?|dr\.?|ecole|school|master|enseignant)\b", lower):
        return False
    if re.fullmatch(
        r"(?:october|november|december|janvier|fevrier|mars|avril|mai|juin|juillet|aout|"
        r"septembre|octobre|novembre|decembre)(?:\s+\d{1,2},?\s+\d{4}|\s+\d{4})?",
        lower,
    ):
        return False
    if re.fullmatch(r"\d{1,4}|[ivxlcdm]+|page\s+\d{1,4}|slide\s+\d{1,4}", lower):
        return False
    useful = sum(1 for ch in text if ch.isalpha())
    return useful >= 3


def _valid_plan_title(value: str) -> bool:
    text = _clean_title(value)
    lower = _strip_accents(text).lower().strip(" :-")
    if not _valid_title(text):
        return False
    if "|" in text or "%" in text:
        return False
    if lower in {"le probleme", "probleme", "the problem", "problem"}:
        return False
    if re.search(r"\b(behavioral finance|bought|jars?|millennial|right table|left table|sales)\b", lower):
        return False
    if re.search(r"\b(and|et|or|ou|has|avec)$", lower):
        return False
    return True


def _title_needs_continuation(title: str) -> bool:
    lower = _strip_accents(title).lower().strip()
    return bool(
        re.search(r"\b(?:aux|des|de|du|dans|et|with|and|of|the|to)$", lower)
        or lower.endswith(("systemes", "syst emes", "modeles", "mod eles"))
    )


def _looks_supplemental_boundary(line: str, source_filename: str) -> bool:
    lower = _strip_accents(line).lower()
    if _primary_unit_from_line(line, 0) is not None:
        return False
    return bool(_SUPPLEMENTAL_RE.search(lower)) and (
        lower.startswith(("guide", "appendix", "annex", "syllabus", "references", "bibliography"))
        or _looks_supplemental_filename(source_filename)
    )


def _looks_supplemental_filename(source_filename: str) -> bool:
    return bool(_SUPPLEMENTAL_RE.search(_strip_accents(source_filename).lower()))


def _number_value(value: str) -> int | None:
    raw = _strip_accents(str(value or "").strip().lower())
    if raw.isdigit():
        number = int(raw)
        return number if 0 < number <= 50 else None
    if raw in _NUMBER_WORDS:
        return _NUMBER_WORDS[raw]
    return _roman_value(raw)


def _roman_value(value: str) -> int | None:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    if not value or any(ch not in values for ch in value):
        return None
    total = 0
    previous = 0
    for ch in reversed(value):
        current = values[ch]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total if 0 < total <= 50 else None


def _strip_accents(value: str) -> str:
    replacements = {
        "´": "",
        "`": "",
        "ˆ": "",
        "¸": "",
        "œ": "oe",
        "Œ": "OE",
    }
    text = str(value or "")
    for source, target in replacements.items():
        text = text.replace(source, target)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _strip_accents(value).casefold())


def _dedupe_titles(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        title = _clean_subchapter_title(value)
        key = _title_key(title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(title)
    return out


_normalizer: CourseIntakeNormalizer | None = None


def get_course_intake_normalizer() -> CourseIntakeNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = CourseIntakeNormalizer()
    return _normalizer

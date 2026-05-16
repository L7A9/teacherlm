from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import PurePosixPath


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^\s*((?:\d+\.)+\d*|[IVXLC]+\.)\s+(.{3,140})$")
_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_PAGE_COUNTER = re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$")
_MONTH_DATE_LINE = re.compile(
    r"^\s*(?:"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre"
    r")\s+\d{1,2},?\s+\d{4}\s*$",
    re.IGNORECASE,
)
_DISPLAY_EQUATION = re.compile(r"(\$\$.*?\$\$|\\\[.*?\\\])", re.DOTALL)
_INLINE_EQUATION = re.compile(r"(?<!\$)\$[^$\n]{2,160}\$(?!\$)|\\\([^)]{2,160}\\\)")
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre"
    r")\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_BOLD_TERM = re.compile(r"\*\*([^*\n]{2,80})\*\*")
_DEFINITION_LINE = re.compile(r"^\s*(?:[-*]\s*)?([A-Z0-9][^:]{2,80})\s*:\s+(.{10,})$")


@dataclass(slots=True)
class CourseTable:
    rows: list[str]
    order_index: int


@dataclass(slots=True)
class CourseSection:
    id: uuid.UUID
    parent_id: uuid.UUID | None
    level: int
    title: str
    heading_path: list[str]
    order_index: int
    text: str
    summary: str
    key_concepts: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    tables: list[CourseTable] = field(default_factory=list)
    timeline_events: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


@dataclass(slots=True)
class CourseDocument:
    id: uuid.UUID
    title: str
    source_filename: str
    sections: list[CourseSection]
    metadata: dict[str, object] = field(default_factory=dict)


class CourseStructureExtractor:
    """Build a section-level course model from cleaned parser markdown.

    The extractor is deterministic and intentionally local-first: it relies on
    heading/table/equation heuristics rather than an LLM so ingestion can be
    measured and re-run without network calls beyond parsing.
    """

    def extract(
        self,
        markdown: str,
        *,
        conversation_id: uuid.UUID | str,
        source_file_id: str,
        source_filename: str,
    ) -> CourseDocument:
        title = self._document_title(markdown, source_filename)
        document_id = _stable_uuid(f"document:{conversation_id}:{source_file_id}")

        sections: list[CourseSection] = []
        heading_stack: list[tuple[int, str, uuid.UUID]] = []
        current_title = title
        current_level = 1
        current_lines: list[str] = []
        section_order = 0

        def flush() -> None:
            nonlocal section_order, current_lines
            text = "\n".join(current_lines).strip()
            current_lines = []
            if not text:
                return
            path = [title for _, title, _ in heading_stack] or [current_title]
            parent_id = heading_stack[-2][2] if len(heading_stack) >= 2 else None
            section_id = _stable_uuid(
                f"section:{conversation_id}:{source_file_id}:{section_order}:{' > '.join(path)}"
            )
            section = CourseSection(
                id=section_id,
                parent_id=parent_id,
                level=current_level,
                title=current_title,
                heading_path=path,
                order_index=section_order,
                text=text,
                summary=_summarize(text),
                key_concepts=_extract_key_concepts(text, current_title),
                equations=_extract_equations(text),
                tables=_extract_tables(text),
                timeline_events=_extract_timeline_events(text),
            )
            sections.append(section)
            section_order += 1

        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()
            heading = _parse_heading(line)
            if heading:
                flush()
                level, heading_title = heading
                heading_stack = [
                    item for item in heading_stack if item[0] < level
                ]
                provisional_id = _stable_uuid(
                    f"section-heading:{conversation_id}:{source_file_id}:{section_order}:{heading_title}"
                )
                heading_stack.append((level, heading_title, provisional_id))
                current_title = heading_title
                current_level = level
                continue
            if _is_noise_line(line):
                continue
            current_lines.append(line)

        flush()

        if not sections and markdown.strip():
            text = markdown.strip()
            sections.append(
                CourseSection(
                    id=_stable_uuid(f"section:{conversation_id}:{source_file_id}:0:root"),
                    parent_id=None,
                    level=1,
                    title=title,
                    heading_path=[title],
                    order_index=0,
                    text=text,
                    summary=_summarize(text),
                    key_concepts=_extract_key_concepts(text, title),
                    equations=_extract_equations(text),
                    tables=_extract_tables(text),
                    timeline_events=_extract_timeline_events(text),
                )
            )

        return CourseDocument(
            id=document_id,
            title=title,
            source_filename=source_filename,
            sections=sections,
            metadata={
                "extractor": "course-structure-v1",
                "source_file_id": source_file_id,
                "section_count": len(sections),
            },
        )

    @staticmethod
    def _document_title(markdown: str, source_filename: str) -> str:
        for line in markdown.splitlines():
            heading = _parse_heading(line)
            if heading:
                return heading[1]
        return PurePosixPath(source_filename).stem.replace("_", " ").strip() or source_filename


def _parse_heading(line: str) -> tuple[int, str] | None:
    if _is_noise_line(line):
        return None
    markdown = _MARKDOWN_HEADING.match(line)
    if markdown:
        return len(markdown.group(1)), _clean_title(markdown.group(2))
    numbered = _NUMBERED_HEADING.match(line)
    if numbered:
        return max(1, numbered.group(1).count(".")), _clean_title(numbered.group(2))
    if _looks_like_plain_heading(line):
        return 2, _clean_title(line)
    return None


def _looks_like_plain_heading(line: str) -> bool:
    value = line.strip()
    if "|" in value or "$" in value or "\\" in value:
        return False
    if _TABLE_ROW.match(value) or _TABLE_DIVIDER.match(value):
        return False
    if not value or value.endswith((".", ":", ";", ",", "!", "?")):
        return False
    words = value.split()
    if not 2 <= len(words) <= 12:
        return False
    letters = [ch for ch in value if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    title_like = sum(1 for word in words if word[:1].isupper()) >= max(2, len(words) // 2)
    short_label = len(words) <= 8 and value[:1].isupper() and not any(ch.isdigit() for ch in value)
    return upper_ratio > 0.55 or title_like or short_label


def _is_noise_line(line: str) -> bool:
    value = line.strip()
    if not value:
        return False
    return bool(_PAGE_COUNTER.fullmatch(value) or _MONTH_DATE_LINE.fullmatch(value))


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip(" -*\t#"))


def _summarize(text: str, max_chars: int = 700) -> str:
    compact = " ".join(text.split())
    sentences = _SENTENCE_SPLIT.split(compact)
    summary = " ".join(sentences[:2]).strip() if sentences else compact
    if len(summary) > max_chars:
        return summary[:max_chars].rsplit(" ", 1)[0].strip()
    return summary


def _extract_key_concepts(text: str, title: str) -> list[str]:
    concepts: list[str] = [title]
    for match in _BOLD_TERM.finditer(text):
        concepts.append(match.group(1).strip())
    for line in text.splitlines():
        definition = _DEFINITION_LINE.match(line)
        if definition:
            concepts.append(definition.group(1).strip(" -*"))
    return _dedupe([c for c in concepts if 2 <= len(c) <= 90])[:20]


def _extract_equations(text: str) -> list[str]:
    equations = [m.group(0).strip() for m in _DISPLAY_EQUATION.finditer(text)]
    equations.extend(m.group(0).strip() for m in _INLINE_EQUATION.finditer(text))
    return _dedupe(equations)[:30]


def _extract_tables(text: str) -> list[CourseTable]:
    tables: list[CourseTable] = []
    current: list[str] = []
    for line in text.splitlines():
        if _TABLE_ROW.match(line) or _TABLE_DIVIDER.match(line):
            current.append(line.strip())
            continue
        if current:
            if len(current) >= 2:
                tables.append(CourseTable(rows=current, order_index=len(tables)))
            current = []
    if len(current) >= 2:
        tables.append(CourseTable(rows=current, order_index=len(tables)))
    return tables[:20]


def _extract_timeline_events(text: str) -> list[str]:
    events: list[str] = []
    for line in text.splitlines():
        compact = " ".join(line.split())
        if len(compact) > 240:
            compact = compact[:240].rsplit(" ", 1)[0].strip()
        if compact and _DATE_RE.search(compact):
            events.append(compact)
    return _dedupe(events)[:30]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = " ".join(value.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(value)
    return out


def _stable_uuid(seed: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:{seed}:{sha1(seed.encode()).hexdigest()[:12]}")


_extractor: CourseStructureExtractor | None = None


def get_course_structure_extractor() -> CourseStructureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = CourseStructureExtractor()
    return _extractor

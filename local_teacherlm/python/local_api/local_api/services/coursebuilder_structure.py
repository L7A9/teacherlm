from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MAX_MARKDOWN_PLANNING_CHARS = 240_000
MAX_LESSONS_PER_CHAPTER = 14


@dataclass(slots=True)
class SourceLesson:
    title: str
    source_queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SourceChapter:
    title: str
    description: str = ""
    source_queries: list[str] = field(default_factory=list)
    lessons: list[SourceLesson] = field(default_factory=list)


@dataclass(slots=True)
class SourceStructure:
    title: str
    origin: str
    chapters: list[SourceChapter]


@dataclass(slots=True)
class MarkdownPlanningContext:
    text: str
    source_count: int
    raw_chars: int


def load_markdown_planning_context(files: list[dict[str, Any]], data_dir: Path) -> MarkdownPlanningContext:
    parts: list[str] = []
    raw_chars = 0
    source_count = 0
    remaining = MAX_MARKDOWN_PLANNING_CHARS
    ordered = sorted(files, key=lambda item: (item.get("created_at", ""), item.get("id", "")))
    for index, file in enumerate(ordered[:10], start=1):
        if remaining <= 0:
            break
        file_id = str(file.get("id") or "")
        candidates = [
            data_dir / "objects" / "parsed" / f"{file_id}.md",
            data_dir / "objects" / "cleaned" / f"{file_id}.txt",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            continue
        try:
            markdown = path.read_text(encoding="utf-8")
        except OSError:
            continue
        raw_chars += len(markdown)
        source_count += 1
        excerpt = _markdown_planning_view(markdown, max_chars=max(2_000, remaining - 200))
        block = (
            f"### Markdown source {index}: {file.get('filename', path.name)}\n"
            f"```markdown\n{excerpt}\n```"
        )
        if len(block) > remaining:
            block = block[:remaining].rsplit("\n", 1)[0].strip()
        parts.append(block)
        remaining -= len(block) + 2
    return MarkdownPlanningContext(text="\n\n".join(parts), source_count=source_count, raw_chars=raw_chars)


def select_representative_chunks(chunks: list[dict[str, Any]], *, per_file: int = 12) -> list[dict[str, Any]]:
    if per_file <= 0:
        return []
    by_file: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_file.setdefault(str(chunk.get("source_file_id") or chunk.get("source_filename") or "source"), []).append(chunk)
    selected: list[dict[str, Any]] = []
    for rows in by_file.values():
        rows = sorted(rows, key=lambda item: int(item.get("chunk_index") or 0))
        if len(rows) <= per_file:
            selected.extend(rows)
            continue
        if per_file == 1:
            selected.append(rows[0])
            continue
        indexes = {round(index * (len(rows) - 1) / (per_file - 1)) for index in range(per_file)}
        selected.extend(rows[index] for index in sorted(indexes))
    return selected


def extract_source_structure(
    *,
    chunks: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    markdown: str,
) -> SourceStructure | None:
    candidates = [
        ("markdown_toc", _from_markdown(markdown)),
        ("intake_metadata", _from_intake_metadata(chunks, documents)),
        ("section_headings", _from_paths([(item.get("heading_path") or [], item.get("text") or "") for item in sections])),
        ("chunk_headings", _from_paths([(_heading_path(chunk), chunk.get("text") or "") for chunk in chunks])),
    ]
    for origin, chapters in candidates:
        if structure_score(chapters) >= 3:
            return SourceStructure(
                title=_course_title(documents, markdown, chapters),
                origin=origin,
                chapters=chapters,
            )
    return None


def structure_score(chapters: list[SourceChapter]) -> int:
    if not chapters:
        return 0
    lesson_count = sum(len(chapter.lessons) for chapter in chapters)
    return len(chapters) * 2 + lesson_count + sum(len(chapter.lessons) > 1 for chapter in chapters)


def _from_intake_metadata(
    chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[SourceChapter]:
    rows: list[tuple[int, dict[str, Any], str]] = []
    for document in documents:
        metadata = document.get("metadata") or {}
        for unit in metadata.get("course_units") or []:
            if isinstance(unit, dict) and unit.get("course_unit_title"):
                rows.append((_safe_int(unit.get("course_unit_index"), len(rows) + 1), unit, document.get("title") or ""))
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if metadata.get("course_unit_title"):
            rows.append((_safe_int(metadata.get("course_unit_index"), int(chunk.get("chunk_index") or 0) + 1), metadata, chunk.get("text") or ""))
    if not rows:
        return []
    roles = {str(metadata.get("course_unit_role") or "primary") for _, metadata, _ in rows}
    wanted_role = "primary" if "primary" in roles else "supplemental"
    units: dict[str, dict[str, Any]] = {}
    for order, metadata, text in rows:
        if str(metadata.get("course_unit_role") or "primary") != wanted_role:
            continue
        title = _clean_title(str(metadata.get("course_unit_title") or ""))
        if not _valid_title(title):
            continue
        key = f"{order}:{_title_key(title)}"
        unit = units.setdefault(key, {"order": order, "title": title, "description": _first_sentence(text), "lessons": []})
        unit["order"] = min(unit["order"], order)
        lesson_values = [metadata.get("subchapter_title"), *(metadata.get("subchapter_titles") or [])]
        for value in lesson_values:
            _append_lesson(unit["lessons"], title, str(value or ""))
    return [
        SourceChapter(
            title=unit["title"],
            description=unit["description"],
            source_queries=[unit["title"]],
            lessons=unit["lessons"] or [SourceLesson(title=unit["title"], source_queries=[unit["title"]])],
        )
        for unit in sorted(units.values(), key=lambda item: (item["order"], item["title"]))
    ]


def _from_markdown(markdown: str) -> list[SourceChapter]:
    raw_sources = re.findall(r"```markdown\n([\s\S]*?)\n```", str(markdown or "")) or [str(markdown or "")]
    best: list[SourceChapter] = []
    combined: list[SourceChapter] = []
    session_plans: list[SourceChapter] = []
    for source in raw_sources:
        session_plan = _from_session_plan(source)
        if session_plan is not None:
            existing_session = next(
                (item for item in session_plans if _title_key(item.title) == _title_key(session_plan.title)),
                None,
            )
            if existing_session is None:
                session_plans.append(session_plan)
            else:
                for lesson in session_plan.lessons:
                    _append_lesson(existing_session.lessons, existing_session.title, lesson.title)
        toc = _toc_lines(source)
        toc_structure = _from_toc_lines(toc)
        heading_structure = _from_markdown_headings(source)
        candidate = _merge_duplicate_chapters(max((toc_structure, heading_structure), key=structure_score))
        if structure_score(candidate) > structure_score(best):
            best = candidate
        for chapter in candidate:
            existing = next((item for item in combined if _title_key(item.title) == _title_key(chapter.title)), None)
            if existing is None:
                combined.append(chapter)
                continue
            for lesson in chapter.lessons:
                _append_lesson(existing.lessons, existing.title, lesson.title)
    if session_plans:
        return sorted(session_plans, key=lambda chapter: (_session_plan_order(chapter.title), _title_key(chapter.title)))
    primary = [chapter for chapter in combined if not _looks_supplemental_title(chapter.title)]
    combined = primary or combined
    return combined if structure_score(combined) >= structure_score(best) else best


def _from_session_plan(markdown: str) -> SourceChapter | None:
    """Extract one explicit course unit and its top-level agenda items from parser Markdown."""
    lines = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    unit_title = _session_unit_title(lines)
    if not unit_title:
        return None
    best_titles: list[str] = []
    for index, raw_line in enumerate(lines[:240]):
        compact = re.sub(r"\s+", " ", raw_line.strip())
        folded = _fold(re.sub(r"^#{1,6}\s+", "", compact))
        if not re.search(
            r"\b(plan de la seance|agenda|outline|course outline|contents|table of contents|table des matieres)\b",
            folded,
        ):
            continue
        titles = _top_level_numbered_titles(compact)
        for extra in lines[index + 1:index + 100]:
            if re.match(r"^#{1,6}\s+\S+", extra.strip()):
                break
            titles.extend(_top_level_numbered_titles(extra))
        titles = _dedupe(_clean_structure_title(title) for title in titles if _valid_title(title))
        if len(titles) > len(best_titles):
            best_titles = titles
    if not best_titles:
        return None
    return SourceChapter(
        title=unit_title,
        source_queries=[unit_title],
        lessons=[SourceLesson(title=title, source_queries=[unit_title, title]) for title in best_titles[:MAX_LESSONS_PER_CHAPTER]],
    )


def _session_unit_title(lines: list[str]) -> str:
    candidates: list[str] = []
    unit_re = re.compile(
        r"^(?:semaine|week|lecture|chapter|chapitre|module|unit|unite|unité)\s+"
        r"(?:n[°o.]?\s*)?(?:\d{1,2}|[ivxlcdm]{1,8})\b.*$",
        re.IGNORECASE,
    )
    for index, raw_line in enumerate(lines[:160]):
        line = re.sub(r"^#{1,6}\s+", "", raw_line.strip())
        line = _clean_structure_title(line)
        if not unit_re.match(line):
            continue
        candidate = line
        if index + 1 < len(lines):
            next_raw = lines[index + 1]
            next_line = _clean_structure_title(next_raw)
            if (
                next_raw.strip()
                and not next_raw.lstrip().startswith("#")
                and not re.match(r"^(?:prof|pr\.|dr\.|master|ecole|école|university|universite|université)\b", next_line, re.I)
                and not re.search(r"\b\d{4}\b", next_line)
                and len(next_line) <= 70
            ):
                candidate = _clean_structure_title(f"{candidate} {next_line}")
        if _valid_title(candidate):
            candidates.append(candidate)
    return max(candidates, key=len, default="")


def _top_level_numbered_titles(value: str) -> list[str]:
    match = re.match(r"^\s*(?:#{1,6}\s*)?(?:\d{1,2})\s*[.)]\s*(.+?)\s*$", str(value or ""))
    if match:
        return [match.group(1)]
    return [
        match.group(1)
        for match in re.finditer(
            r"(?:^|\s)(?:\d{1,2})\s*[.)]\s*(.+?)(?=\s+\d{1,2}\s*[.)]\s+|$)",
            str(value or ""),
        )
    ]


def _session_plan_order(title: str) -> int:
    match = re.match(
        r"^(?:semaine|week|lecture|chapter|chapitre|module|unit|unite|unité)\s+"
        r"(?:n[°o.]?\s*)?(\d{1,3})\b",
        _fold(title),
    )
    return int(match.group(1)) if match else 10_000


def _merge_duplicate_chapters(chapters: list[SourceChapter]) -> list[SourceChapter]:
    merged: list[SourceChapter] = []
    by_title: dict[str, SourceChapter] = {}
    for chapter in chapters:
        key = _title_key(chapter.title)
        existing = by_title.get(key)
        if existing is None:
            existing = SourceChapter(
                title=chapter.title,
                description=chapter.description,
                source_queries=list(chapter.source_queries),
                lessons=[],
            )
            by_title[key] = existing
            merged.append(existing)
        else:
            existing.source_queries = _dedupe([*existing.source_queries, *chapter.source_queries])
            existing.description = existing.description or chapter.description
        for lesson in chapter.lessons:
            _append_lesson(existing.lessons, existing.title, lesson.title)
    return merged


def _from_markdown_headings(markdown: str) -> list[SourceChapter]:
    paths: list[tuple[list[str], str]] = []
    stack: list[str] = []
    current_path: list[str] = []
    current_text: list[str] = []
    for line in str(markdown or "").splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            if current_path:
                paths.append((current_path, " ".join(current_text)))
            level = len(match.group(1))
            stack = stack[:level - 1] + [_clean_title(match.group(2))]
            current_path = list(stack)
            current_text = []
        elif current_path:
            current_text.append(line)
    if current_path:
        paths.append((current_path, " ".join(current_text)))
    return _from_paths(paths)


def _from_paths(items: list[tuple[list[str], str]]) -> list[SourceChapter]:
    cleaned: list[tuple[list[str], str]] = []
    for path, text in items:
        safe_path = [_clean_title(str(part)) for part in path]
        safe_path = [part for part in safe_path if _valid_title(part)]
        if safe_path:
            cleaned.append((safe_path, text))
    if not cleaned:
        return []
    chapter_level = _chapter_level([path for path, _ in cleaned])
    lesson_level = chapter_level + 1
    chapters: dict[str, SourceChapter] = {}
    for path, text in cleaned:
        if len(path) <= chapter_level:
            continue
        chapter_title = path[chapter_level]
        key = _title_key(chapter_title)
        chapter = chapters.setdefault(
            key,
            SourceChapter(
                title=chapter_title,
                description=_first_sentence(text),
                source_queries=[chapter_title],
            ),
        )
        if len(path) > lesson_level:
            _append_lesson(chapter.lessons, chapter_title, path[lesson_level])
    for chapter in chapters.values():
        if not chapter.lessons:
            chapter.lessons.append(SourceLesson(title=chapter.title, source_queries=[chapter.title]))
    return list(chapters.values())


def _toc_lines(markdown: str) -> list[str]:
    selected: list[str] = []
    in_contents = False
    for raw in str(markdown or "").splitlines()[:1_200]:
        line = re.sub(r"\s+", " ", raw.strip()).strip(" .\t")
        folded = _fold(line)
        if folded in {"contents", "table of contents", "table des matieres"}:
            in_contents = True
            continue
        if in_contents and line:
            selected.append(line)
        elif _toc_chapter_title(line) or _toc_lesson_title(line):
            selected.append(line)
    return selected[:500]


def _from_toc_lines(lines: list[str]) -> list[SourceChapter]:
    chapters: list[SourceChapter] = []
    current: SourceChapter | None = None
    for line in lines:
        chapter_title = _toc_chapter_title(line)
        if chapter_title:
            current = SourceChapter(title=chapter_title, source_queries=[chapter_title])
            chapters.append(current)
            continue
        lesson_title = _toc_lesson_title(line)
        if current is not None and lesson_title:
            _append_lesson(current.lessons, current.title, lesson_title)
    for chapter in chapters:
        if not chapter.lessons:
            chapter.lessons.append(SourceLesson(title=chapter.title, source_queries=[chapter.title]))
    return chapters


def _toc_chapter_title(line: str) -> str:
    number = r"(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    patterns = [
        rf"^(?:chapter|chapitre|unit|unite|unité|week|semaine|module)\s+{number}[:.\-\s]+(.+?)(?:\s+\d{{1,4}})?$",
        rf"^(?:chapter\s+)?{number}\s+(.+?)\s+\d{{1,4}}$",
    ]
    for pattern in patterns:
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if match:
            return _clean_structure_title(match.group(1))
    return ""


def _toc_lesson_title(line: str) -> str:
    match = re.match(r"^(.+?)\s+\.{0,}\s*\d{1,4}$", line)
    if not match or _toc_chapter_title(line):
        return ""
    title = _clean_structure_title(match.group(1))
    return title if _valid_title(title) else ""


def _append_lesson(lessons: list[SourceLesson], chapter_title: str, raw_title: str) -> None:
    title = _clean_structure_title(raw_title)
    if not _valid_title(title):
        return
    for lesson in lessons:
        if _titles_equivalent(lesson.title, title, chapter_title):
            lesson.source_queries = _dedupe([*lesson.source_queries, title, chapter_title])
            return
    if len(lessons) < MAX_LESSONS_PER_CHAPTER:
        lessons.append(SourceLesson(title=title, source_queries=[chapter_title, title]))


def _titles_equivalent(left: str, right: str, context: str) -> bool:
    if _title_key(left) == _title_key(right):
        return True
    context_terms = set(_terms(context))
    left_terms = [term for term in _terms(left) if term not in context_terms] or _terms(left)
    right_terms = [term for term in _terms(right) if term not in context_terms] or _terms(right)
    if not left_terms or not right_terms:
        return False
    left_set, right_set = set(left_terms), set(right_terms)
    shared = len(left_set & right_set)
    minimum = min(len(left_set), len(right_set))
    return minimum >= 2 and shared / minimum >= 0.8 and SequenceMatcher(None, " ".join(left_terms), " ".join(right_terms)).ratio() >= 0.82


def _chapter_level(paths: list[list[str]]) -> int:
    nested = [path for path in paths if len(path) > 1]
    if not nested:
        return 0
    root_counts = Counter(_title_key(path[0]) for path in nested)
    dominant, count = root_counts.most_common(1)[0]
    second_titles = {_title_key(path[1]) for path in nested if _valid_title(path[1])}
    dominant_title = next((path[0] for path in nested if _title_key(path[0]) == dominant), "")
    if count / len(nested) >= 0.75 and len(second_titles) >= 2:
        explicit_root_unit = re.match(
            r"^(?:chapter|chapitre|unit|unite|week|semaine|module)\s+(?:\d+|[ivxlcdm]+)\b",
            _fold(dominant_title),
        )
        explicit_child_units = sum(
            bool(re.match(
                r"^(?:chapter|chapitre|unit|unite|week|semaine|module)\s+(?:\d+|[ivxlcdm]+)\b",
                _fold(path[1]),
            ))
            for path in nested
        )
        document_root = any(term in _fold(dominant_title) for term in ("course", "book", "textbook", ".pdf", ".doc"))
        if not explicit_root_unit and (document_root or explicit_child_units >= 2):
            return 1
    return 0


def _looks_supplemental_title(value: str) -> bool:
    folded = _fold(value)
    return any(
        term in folded
        for term in (
            "appendix", "annex", "bibliography", "references", "recommended reading",
            "student guide", "syllabus", "table of contents", "table des matieres",
        )
    )


def _markdown_planning_view(markdown: str, *, max_chars: int) -> str:
    text = str(markdown or "").strip()
    if len(text) <= max_chars:
        return text
    headings = [line.strip() for line in text.splitlines() if re.match(r"^#{1,6}\s+\S+", line.strip())]
    toc = _toc_lines(text)
    view = "\n".join([
        "# Extracted heading index",
        *headings[:500],
        "",
        "# Likely table of contents lines",
        *toc[:500],
        "",
        "# Opening markdown excerpt",
        text[: min(max_chars // 2, 30_000)],
    ])
    return view[:max_chars].rsplit("\n", 1)[0].strip()


def _course_title(documents: list[dict[str, Any]], markdown: str, chapters: list[SourceChapter]) -> str:
    shared_markdown_title = _shared_markdown_title(markdown)
    if shared_markdown_title:
        return shared_markdown_title
    document_title = next((_clean_title(str(item.get("title") or "")) for item in documents if _valid_title(str(item.get("title") or ""))), "")
    if document_title:
        return document_title
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    if match and _valid_title(match.group(1)):
        return _clean_title(match.group(1))
    return chapters[0].title if chapters else "Generated course"


def _shared_markdown_title(markdown: str) -> str:
    raw_sources = re.findall(r"```markdown\n([\s\S]*?)\n```", str(markdown or "")) or [str(markdown or "")]
    candidates: list[tuple[int, str]] = []
    for source_index, source in enumerate(raw_sources):
        seen_in_source: set[str] = set()
        for raw_line in source.splitlines()[:60]:
            title = _clean_structure_title(re.sub(r"^#{1,6}\s+", "", raw_line.strip()))
            folded = _fold(title)
            if not _valid_title(title) or not (5 <= len(title) <= 140):
                continue
            if re.match(r"^(?:semaine|week|lecture|chapter|chapitre|module|unit|unite)\s+\d+\b", folded):
                continue
            if any(
                term in folded
                for term in (
                    ".pdf", "source material", "source plan item", "plan de la seance", "agenda",
                    "outline", "professor", "universite", "university", "ecole normale", "master ",
                )
            ):
                continue
            if re.match(r"^(?:pr|prof|professor|dr)[.]?\s+", folded):
                continue
            key = _title_key(title)
            if key not in seen_in_source:
                candidates.append((source_index, title))
                seen_in_source.add(key)
    counts = Counter(_title_key(title) for _, title in candidates)
    minimum = 2 if len(raw_sources) > 1 else 1
    for _, title in candidates:
        if counts[_title_key(title)] >= minimum:
            return title
    return ""


def _heading_path(chunk: dict[str, Any]) -> list[str]:
    metadata = chunk.get("metadata") or {}
    path = metadata.get("heading_path_list") or []
    if isinstance(path, list):
        return [str(item) for item in path]
    return [item.strip() for item in str(metadata.get("heading_path") or "").split(">") if item.strip()]


def _clean_structure_title(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^[\-*•]+\s*", "", text)
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"\s+\d{1,4}$", "", text)
    text = re.sub(r"^(?:chapter|chapitre|part|section)\s+(?:\d+|[ivxlcdm]+)[:.\-\s]+", "", text, flags=re.IGNORECASE)
    return _clean_title(text)


def _valid_title(value: str) -> bool:
    text = _clean_title(value)
    folded = _fold(text)
    if len(text) < 3 or len(text) > 140 or re.fullmatch(r"(?:\d+|[ivxlcdm]+|page\s+\d+|slide\s+\d+)", folded):
        return False
    if re.search(r"\.(?:pdf|docx?|pptx?|html?)\b", folded) or any(token in text for token in ("</", "<td", "<tr")):
        return False
    return folded not in {
        "contents", "table of contents", "table des matieres", "copyright", "bibliography",
        "references", "index", "notes", "course", "course outline", "document", "lesson", "section",
        "source material", "extracted heading index", "likely table of contents lines", "opening markdown excerpt",
    }


def _clean_title(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip(" -*•:;,.#")[:180]


def _first_sentence(value: str) -> str:
    compact = " ".join(str(value or "").split())
    return re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)[0][:500] if compact else ""


def _terms(value: str) -> list[str]:
    stopwords = {"the", "and", "for", "from", "with", "chapter", "course", "lesson", "section", "de", "des", "du", "la", "le", "les", "et"}
    return [term for term in re.findall(r"[a-z0-9]+", _fold(value)) if len(term) >= 3 and term not in stopwords and not term.isdigit()]


def _title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold(value))


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _fold(value).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def decode_course_rows(rows: list[dict[str, Any]], *, metadata_key: str, heading_key: str | None = None) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop(metadata_key) or "{}")
        except (json.JSONDecodeError, TypeError):
            item["metadata"] = {}
        if heading_key:
            try:
                item["heading_path"] = json.loads(item.pop(heading_key) or "[]")
            except (json.JSONDecodeError, TypeError):
                item["heading_path"] = []
        decoded.append(item)
    return decoded

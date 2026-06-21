from __future__ import annotations

import html
import re
from typing import Any


_PAGE_ARTIFACT_RE = re.compile(
    r"<\s*(?:page_number|page_header|page_footer|page_break)\b[^>]*>[\s\S]*?"
    r"<\s*/\s*(?:page_number|page_header|page_footer|page_break)\s*>",
    re.IGNORECASE,
)
_SELF_CLOSING_PAGE_ARTIFACT_RE = re.compile(
    r"<\s*(?:page_number|page_header|page_footer|page_break)\b[^>]*/\s*>",
    re.IGNORECASE,
)
_HTML_TABLE_RE = re.compile(r"<table\b[^>]*>[\s\S]*?</table\s*>", re.IGNORECASE)
_HTML_ROW_RE = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr\s*>", re.IGNORECASE)
_HTML_CELL_RE = re.compile(r"<t([hd])\b[^>]*>([\s\S]*?)</t\1\s*>", re.IGNORECASE)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def normalize_extracted_markup(value: str) -> str:
    """Convert parser-only markup into safe Markdown before it reaches retrieval or UI."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _PAGE_ARTIFACT_RE.sub("\n", text)
    text = _SELF_CLOSING_PAGE_ARTIFACT_RE.sub("\n", text)
    text = _HTML_TABLE_RE.sub(lambda match: f"\n\n{_html_table_to_markdown(match.group(0))}\n\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_tables_for_prose(value: str) -> str:
    """Remove structured rows while keeping surrounding teaching prose."""
    normalized = normalize_extracted_markup(value)
    lines = normalized.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if index + 1 < len(lines) and _is_markdown_row(lines[index]) and _is_separator_row(lines[index + 1]):
            index += 2
            while index < len(lines) and _is_markdown_row(lines[index]):
                index += 1
            continue
        kept.append(lines[index])
        index += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def extract_markdown_tables(value: str) -> list[dict[str, Any]]:
    """Return normalized table data from either HTML tables or Markdown tables."""
    text = normalize_extracted_markup(value)
    lines = text.splitlines()
    tables: list[dict[str, Any]] = []
    index = 0
    while index + 1 < len(lines):
        if not (_is_markdown_row(lines[index]) and _is_separator_row(lines[index + 1])):
            index += 1
            continue
        block = [lines[index], lines[index + 1]]
        index += 2
        while index < len(lines) and _is_markdown_row(lines[index]):
            block.append(lines[index])
            index += 1
        headers = _table_cells(block[0])
        rows = [_pad_row(_table_cells(line), len(headers)) for line in block[2:]]
        if headers and rows:
            tables.append(
                {
                    "caption": "Source table",
                    "headers": headers,
                    "rows": rows,
                    "markdown": "\n".join(block),
                }
            )
    return tables


def _html_table_to_markdown(value: str) -> str:
    rows: list[list[str]] = []
    header_flags: list[list[bool]] = []
    for row_match in _HTML_ROW_RE.finditer(value):
        cells: list[str] = []
        flags: list[bool] = []
        for cell_match in _HTML_CELL_RE.finditer(row_match.group(1)):
            cells.append(_clean_html_cell(cell_match.group(2)))
            flags.append(cell_match.group(1).casefold() == "h")
        if cells:
            rows.append(cells)
            header_flags.append(flags)
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    rows = [_pad_row(row, width) for row in rows]
    if not all(header_flags[0]):
        rows.insert(0, [f"Column {index + 1}" for index in range(width)])
    header, body = rows[0], rows[1:]
    return "\n".join(
        [
            _markdown_row(header),
            _markdown_row(["---"] * width),
            *(_markdown_row(row) for row in body),
        ]
    )


def _clean_html_cell(value: str) -> str:
    text = _HTML_BREAK_RE.sub(" ", value)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def _markdown_row(cells: list[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _is_markdown_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", line))


def _table_cells(line: str) -> list[str]:
    value = line.strip().strip("|")
    cells = re.split(r"(?<!\\)\|", value)
    return [cell.strip().replace(r"\|", "|") for cell in cells]


def _pad_row(row: list[str], width: int) -> list[str]:
    return [*row[:width], *([""] * max(0, width - len(row)))]

from __future__ import annotations

import re
from dataclasses import dataclass


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_PAGE_COUNTER_RE = re.compile(r"\b\d{1,3}\s*/\s*\d{1,3}\b")
_DOT_LEADER_RE = re.compile(r"\.{5,}\s*\d{1,4}\s*$")
_DANGLING_IMAGE_FILE_RE = re.compile(r"\bpage_\d+_[\w_]+\.(?:png|jpg|jpeg|webp)\b", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_REPEATED_DOTS_RE = re.compile(r"(?:\s*\.\s*){5,}")
_MONTH_DATE_RE = re.compile(
    r"\b(?:"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre"
    r")\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_PROFESSOR_FOOTER_RE = re.compile(
    r"\bPr\.\s*Abdelaaziz\s+Hessane\b.*?(?:\d{4})?\s*(?:\d{1,3}\s*/\s*\d{1,3})?",
    re.IGNORECASE,
)
_COURSE_FOOTER_FRAGMENT_RE = re.compile(
    r"\b(?:École Normale Supérieure|Ecole Normale Superieure|Systèmes de Recommandation et Blockchain|"
    r"Syst[eè]mes Intelligents pour l[’']?[ÉE]ducation)\b",
    re.IGNORECASE,
)
_RESIDUAL_FOOTER_RE = re.compile(
    r"(?:#\s*)?\bde\s+Mekn[eè]s\b|"
    r"\(\s*,?\s*(?:de\s+Mekn[eè]s|Intelligents pour l[’']?[ÉE]ducation)\s*\)|"
    r"Master\s+\"?\s*\"?",
    re.IGNORECASE,
)
_EMPTY_PARENS_RE = re.compile(r"\(\s*[,;|]?\s*\)")

_NOISE_PHRASES = (
    "navigation controls",
    "navigation icons",
    "footer icons",
    "footer navigation",
    "page number icon",
    "presentation controls",
    "slide navigation",
    "seal or stamp",
    "stamp/seal",
    "logo",
)


@dataclass(slots=True)
class CleaningStats:
    original_lines: int = 0
    kept_lines: int = 0
    removed_lines: int = 0


class DocumentCleaningService:
    """Removes parser and slide boilerplate before text is embedded.

    The cleaner is intentionally conservative around educational content:
    equations, markdown tables, HTML tables, bullet lists, and headings are kept.
    It targets artifacts that repeatedly polluted the retrieval benchmark.
    """

    def clean_markdown(self, markdown: str) -> str:
        cleaned, _stats = self.clean_markdown_with_stats(markdown)
        return cleaned

    def clean_markdown_with_stats(self, markdown: str) -> tuple[str, CleaningStats]:
        stats = CleaningStats()
        if not markdown or not markdown.strip():
            return "", stats

        text = markdown.replace("\r\n", "\n").replace("\r", "\n")
        text = _HTML_COMMENT_RE.sub(" ", text)
        text = _MARKDOWN_IMAGE_RE.sub(" ", text)
        text = _HTML_IMAGE_RE.sub(" ", text)

        out: list[str] = []
        blank_pending = False
        previous_normalized = ""

        for raw_line in text.splitlines():
            stats.original_lines += 1
            line = self._clean_line(raw_line)
            if not line:
                blank_pending = bool(out)
                continue

            normalized = " ".join(line.lower().split())
            if self._should_drop_line(line, normalized):
                stats.removed_lines += 1
                blank_pending = bool(out)
                continue

            if normalized == previous_normalized:
                stats.removed_lines += 1
                continue

            if blank_pending and out and out[-1] != "":
                out.append("")
            out.append(line)
            previous_normalized = normalized
            blank_pending = False
            stats.kept_lines += 1

        return "\n".join(out).strip() + ("\n" if out else ""), stats

    def _clean_line(self, raw_line: str) -> str:
        line = raw_line.strip()
        if not line:
            return ""

        line = _PROFESSOR_FOOTER_RE.sub(" ", line)
        line = _COURSE_FOOTER_FRAGMENT_RE.sub(" ", line)
        line = _RESIDUAL_FOOTER_RE.sub(" ", line)
        line = _EMPTY_PARENS_RE.sub(" ", line)
        line = _MONTH_DATE_RE.sub(" ", line)
        line = _PAGE_COUNTER_RE.sub(" ", line)
        line = _DANGLING_IMAGE_FILE_RE.sub(" ", line)
        line = _REPEATED_DOTS_RE.sub(" ", line)
        line = _MULTISPACE_RE.sub(" ", line)
        return line.strip(" |•\t")

    def _should_drop_line(self, line: str, normalized: str) -> bool:
        if not normalized:
            return True
        if any(phrase in normalized for phrase in _NOISE_PHRASES):
            return True
        if _DOT_LEADER_RE.search(line):
            return True
        if _PAGE_COUNTER_RE.fullmatch(line):
            return True
        if _MONTH_DATE_RE.fullmatch(line):
            return True
        if _DANGLING_IMAGE_FILE_RE.search(line):
            return True
        if self._is_mostly_punctuation(line):
            return True
        return False

    @staticmethod
    def _is_mostly_punctuation(line: str) -> bool:
        if len(line) < 8:
            return False
        useful = sum(1 for ch in line if ch.isalnum())
        return useful / len(line) < 0.25


_cleaner: DocumentCleaningService | None = None


def get_document_cleaner() -> DocumentCleaningService:
    global _cleaner
    if _cleaner is None:
        _cleaner = DocumentCleaningService()
    return _cleaner

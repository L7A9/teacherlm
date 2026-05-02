from __future__ import annotations

import re
from pathlib import PurePosixPath

from teacherlm_core.schemas.chunk import Chunk

from ..schemas import MindMap, MindMapNode


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•▶]|\d+[.)])\s+(.+?)\s*$")
_BOLD_LABEL_RE = re.compile(r"^\s*\*\*(.{4,80}?)\*\*\s*:?\s*$")
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_HTML_RE = re.compile(r"<[^>]+>")
_LATEX_RE = re.compile(r"\${1,2}.*?\${1,2}")

_MAX_LABEL_CHARS = 80
_MIN_BRANCHES = 3

_GENERIC_TITLES = {
    "introduction",
    "conclusion",
    "overview",
    "resume",
    "résumé",
    "course",
    "cours",
    "lecture",
    "chapitre",
    "chapter",
}

_NOISE_PHRASES = {
    "navigation",
    "footer",
    "toolbar",
    "logo",
    "seal",
    "presentation controls",
}


def build_from_chunks(chunks: list[Chunk], *, max_nodes: int) -> MindMap | None:
    """Build a course outline directly from parsed Markdown structure.

    LlamaCloud Markdown usually preserves headings and many slide bullets.
    Using that structure first gives broad, faithful coverage across subjects
    without forcing a domain-specific template.
    """
    if not chunks:
        return None

    central_topic = _infer_central_topic(chunks)
    branches: list[MindMapNode] = []
    stack: list[tuple[int, MindMapNode]] = []
    seen: set[str] = set()
    node_budget = max(12, max_nodes)
    node_count = 1

    def add_node(depth: int, label: str) -> MindMapNode | None:
        nonlocal node_count
        if node_count >= node_budget:
            return None
        label = _clean_label(label)
        key = _norm(label)
        if not label or key in seen or key == _norm(central_topic):
            return None
        seen.add(key)

        node = MindMapNode(text=label, children=[])
        node_count += 1

        if depth <= 1 or not stack:
            branches.append(node)
            stack.clear()
            stack.append((1, node))
            return node

        while stack and stack[-1][0] >= depth:
            stack.pop()
        if not stack:
            branches.append(node)
            stack.append((1, node))
            return node

        stack[-1][1].children.append(node)
        stack.append((depth, node))
        return node

    for chunk in chunks:
        for raw_line in chunk.text.splitlines():
            line = raw_line.strip()
            if not line or _is_noise(line):
                continue

            heading = _HEADING_RE.match(line)
            if heading:
                level = len(heading.group(1))
                label = heading.group(2)
                if level == 1 and not branches and _similar(label, central_topic):
                    continue
                add_node(max(1, level - 1), label)
                continue

            bold = _BOLD_LABEL_RE.match(line)
            if bold:
                add_node(2 if stack else 1, bold.group(1))
                continue

            bullet = _BULLET_RE.match(line)
            if bullet and stack and node_count < node_budget:
                label = _clean_label(bullet.group(1))
                if _useful_leaf(label):
                    parent = stack[-1][1]
                    if len(parent.children) < 5:
                        key = f"{_norm(parent.text)}::{_norm(label)}"
                        if key not in seen:
                            seen.add(key)
                            parent.children.append(MindMapNode(text=label))
                            node_count += 1

    branches = _prune_empty_and_tiny(branches)
    if len(branches) < _MIN_BRANCHES:
        return None
    return MindMap(central_topic=central_topic, branches=branches[:10])


def _infer_central_topic(chunks: list[Chunk]) -> str:
    for chunk in chunks[:12]:
        for line in chunk.text.splitlines():
            match = _HEADING_RE.match(line.strip())
            if not match:
                continue
            label = _clean_label(match.group(2))
            if label and _norm(label) not in _GENERIC_TITLES:
                return label[:60]

    first_source = PurePosixPath(chunks[0].source).stem
    return first_source.replace("_", " ").replace("-", " ").strip()[:60] or "Course Map"


def _clean_label(text: str) -> str:
    text = _IMAGE_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = _LATEX_RE.sub("", text)
    text = re.sub(r"[*_`>#]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -:;•▶")
    if len(text) > _MAX_LABEL_CHARS:
        text = text[: _MAX_LABEL_CHARS - 1].rstrip() + "…"
    return text


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _similar(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


def _is_noise(line: str) -> bool:
    lowered = line.casefold()
    return any(phrase in lowered for phrase in _NOISE_PHRASES)


def _useful_leaf(label: str) -> bool:
    if len(label) < 4:
        return False
    lowered = label.casefold()
    if any(phrase in lowered for phrase in _NOISE_PHRASES):
        return False
    if lowered.startswith(("http://", "https://")):
        return False
    return True


def _prune_empty_and_tiny(branches: list[MindMapNode]) -> list[MindMapNode]:
    out: list[MindMapNode] = []
    for branch in branches:
        if branch.children:
            out.append(branch)
    return out or branches

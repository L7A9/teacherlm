from ..schemas import MindMap, MindMapNode


def _bullet(node: MindMapNode, depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    lines.append(f"{indent}- {node.text.strip()}")
    for child in node.children:
        _bullet(child, depth + 1, lines)


def compile(mindmap: MindMap) -> str:
    """Compile a MindMap into Markmap-compatible markdown.

    Markmap expects a single H1 root and nested bullet lists for branches.
    Indentation is two spaces per level; the bullet character is `-`.
    """
    lines: list[str] = [f"# {mindmap.central_topic.strip()}", ""]
    for branch in mindmap.branches:
        _bullet(branch, depth=0, lines=lines)
    return "\n".join(lines) + "\n"

from ..schemas import MindMap, MindMapNode


def escape(text: str) -> str:
    """Strip characters that break Mermaid mindmap parsing."""
    return (
        text.replace("(", "[")
        .replace(")", "]")
        .replace('"', "'")
        .strip()
    )


def append_node(node: MindMapNode, depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    lines.append(f"{indent}{escape(node.text)}")
    for child in node.children:
        append_node(child, depth + 1, lines)


def compile(mindmap: MindMap) -> str:
    lines = ["mindmap"]
    lines.append(f"  root(({escape(mindmap.central_topic)}))")
    for branch in mindmap.branches:
        append_node(branch, depth=2, lines=lines)
    return "\n".join(lines)

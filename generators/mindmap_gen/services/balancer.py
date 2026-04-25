from ..schemas import MindMap, MindMapNode

_MAX_DIRECT_CHILDREN = 7


def _count_nodes(node: MindMapNode) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def _total_nodes(mm: MindMap) -> int:
    return 1 + sum(_count_nodes(b) for b in mm.branches)


def _all_leaves_with_path(
    node: MindMapNode,
    depth: int,
    path: list[MindMapNode],
) -> list[tuple[int, MindMapNode, MindMapNode]]:
    """Return (depth, parent, leaf) tuples for every leaf in the subtree."""
    out: list[tuple[int, MindMapNode, MindMapNode]] = []
    for child in node.children:
        if not child.children:
            out.append((depth + 1, node, child))
        else:
            out.extend(_all_leaves_with_path(child, depth + 1, path + [node]))
    return out


def _collect_leaves(mm: MindMap) -> list[tuple[int, MindMapNode, MindMapNode]]:
    leaves: list[tuple[int, MindMapNode, MindMapNode]] = []
    for branch in mm.branches:
        leaves.extend(_all_leaves_with_path(branch, depth=1, path=[branch]))
    return leaves


def _promote_singletons(node: MindMapNode) -> MindMapNode:
    """If a node has exactly one child, collapse: keep parent label,
    take the grandchildren as new children. Recurse first."""
    node.children = [_promote_singletons(c) for c in node.children]
    if len(node.children) == 1 and node.children[0].children:
        only = node.children[0]
        node.children = only.children
    return node


def _cap_children(node: MindMapNode, limit: int = _MAX_DIRECT_CHILDREN) -> None:
    if len(node.children) > limit:
        node.children = node.children[:limit]
    for c in node.children:
        _cap_children(c, limit)


def _trim_one_leaf(mm: MindMap) -> bool:
    """Remove the lowest-priority leaf in the tree. Returns True if trimmed."""
    leaves = _collect_leaves(mm)
    if not leaves:
        return False
    # Lowest priority = deepest first, then longest label, then last position.
    leaves.sort(key=lambda t: (-t[0], -len(t[2].text)))
    _, parent, leaf = leaves[0]
    try:
        parent.children.remove(leaf)
        return True
    except ValueError:
        return False


def balance(mindmap: MindMap, max_nodes: int) -> MindMap:
    """Promote single-child chains, cap fan-out, trim to max_nodes."""
    for branch in mindmap.branches:
        _promote_singletons(branch)

    for branch in mindmap.branches:
        _cap_children(branch)
    if len(mindmap.branches) > _MAX_DIRECT_CHILDREN:
        mindmap.branches = mindmap.branches[:_MAX_DIRECT_CHILDREN]

    while _total_nodes(mindmap) > max_nodes:
        if not _trim_one_leaf(mindmap):
            break

    return mindmap

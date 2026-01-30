import random

import torch
from lxml import html

from wubba.const import (
    KEEP_TAGS,
    NUM_SEMANTIC_GROUPS,
    SEMANTIC_EQUIVALENTS,
    SEMANTIC_GROUPS,
    TAG_TO_ROLE,
    TAG_TO_SEMANTIC_GROUP,
    TagRole,
)


def create_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Returns float mask where 1.0=valid, 0.0=padding based on first channel."""
    return (tensor[..., 0] != 0).float()


def is_tree_equal(tree1: html.HtmlElement, tree2: html.HtmlElement) -> bool:
    """Recursively checks structural equality of two HTML trees."""
    if tree1.tag != tree2.tag:
        return False

    if len(tree1) != len(tree2):
        return False

    return all(is_tree_equal(child1, child2) for child1, child2 in zip(tree1, tree2, strict=True))


def compute_subtree_hash(element: html.HtmlElement, max_depth: int = 5) -> int:
    """Returns O(n) structure hash for subtree deduplication."""
    if max_depth == 0 or len(element) == 0:
        return hash(element.tag)

    child_hashes = tuple(sorted(compute_subtree_hash(child, max_depth - 1) for child in element))
    return hash((element.tag, len(element), child_hashes))


def clean_tree(tree: html.HtmlElement) -> html.HtmlElement:
    """Removes non-KEEP_TAGS and duplicate adjacent subtrees in-place."""
    keep_tags_set = set(KEEP_TAGS)

    def _clean_recursive(element: html.HtmlElement):
        # Iterate over a copy of children, as we modify the list
        for child in list(element):
            if child.tag not in keep_tags_set:
                element.remove(child)
            else:
                _clean_recursive(child)

        if len(element) > 1:
            i = 0
            while i < len(element) - 1:
                if is_tree_equal(element[i], element[i + 1]):
                    element.remove(element[i + 1])
                else:
                    i += 1

    _clean_recursive(tree)
    return tree


def normalize_tree(
    tree: html.HtmlElement,
    max_siblings: int = 10,
    max_depth: int = 10,
    collapse_consecutive: bool = True,
    aggressive_dedup: bool = True,
) -> html.HtmlElement:
    """Normalizes tree: removes non-kept tags, deduplicates, limits siblings/depth."""
    keep_tags_set = set(KEEP_TAGS)

    def _normalize_recursive(element: html.HtmlElement, current_depth: int):
        for child in list(element):
            if child.tag not in keep_tags_set:
                element.remove(child)
            else:
                _normalize_recursive(child, current_depth + 1)

        if current_depth >= max_depth:
            for child in list(element):
                element.remove(child)
            return

        if collapse_consecutive and len(element) > 1:
            i = 0
            while i < len(element) - 1:
                if (
                    element[i].tag == element[i + 1].tag
                    and len(element[i]) == 0
                    and len(element[i + 1]) == 0
                ):
                    # Both are leaf nodes with same tag, remove the second
                    element.remove(element[i + 1])
                else:
                    i += 1

        if aggressive_dedup and len(element) > 1:
            seen_hashes: set = set()
            to_remove = []
            for child in element:
                child_hash = compute_subtree_hash(child)
                if child_hash in seen_hashes:
                    to_remove.append(child)
                else:
                    seen_hashes.add(child_hash)
            for child in to_remove:
                element.remove(child)

        children = list(element)
        if len(children) > max_siblings:
            keep_first = max_siblings // 2
            keep_last = max_siblings - keep_first
            to_keep = set(children[:keep_first] + children[-keep_last:])
            for child in children:
                if child not in to_keep:
                    element.remove(child)

    _normalize_recursive(tree, 0)
    return tree


def get_max_depth(tree: html.HtmlElement) -> int:
    """Returns maximum depth (root=0)."""
    if not len(tree):
        return 0

    return 1 + max(get_max_depth(child) for child in tree)


def get_max_position(tree: html.HtmlElement) -> int:
    """Returns maximum sibling position (0-indexed)."""
    max_pos = 0
    for element in tree.iter():
        num_children = len(element)
        if num_children > 1:
            max_pos = max(max_pos, num_children - 1)
    return max_pos


def random_node(tree: html.HtmlElement) -> html.HtmlElement:
    """Returns a random non-root node. Raises IndexError if none exist."""
    nodes = list(tree.iter())[1:]
    if not nodes:
        raise IndexError("Tree contains only a root node.")

    return random.choice(nodes)


def tree_node_mask(tree: html.HtmlElement) -> html.HtmlElement:
    """Removes a random node, avoiding high-level sections. In-place."""
    if not len(tree):
        return tree

    try:
        node_to_remove = random_node(tree)

        if node_to_remove.getparent() == tree and len(node_to_remove):
            node_to_remove = random_node(node_to_remove)

        parent = node_to_remove.getparent()
        if parent is not None:
            parent.remove(node_to_remove)
    except IndexError:
        pass

    return tree


def tree_child_mask(tree: html.HtmlElement) -> html.HtmlElement:
    """Removes all children of a random non-root node. In-place."""
    potential_parents = [node for node in tree.iter() if node != tree and len(node) > 0]

    if not potential_parents:
        return tree

    parent_to_modify = random.choice(potential_parents)
    for child in list(parent_to_modify):
        parent_to_modify.remove(child)

    return tree


def tree_node_rotate(tree: html.HtmlElement) -> html.HtmlElement:
    """Moves a random leaf node up one level. In-place."""
    rotatable_nodes = [
        node
        for node in tree.iter()
        if not len(node) and node.getparent() is not None and node.getparent() != tree
    ]

    if not rotatable_nodes:
        return tree

    node_to_rotate = random.choice(rotatable_nodes)
    parent = node_to_rotate.getparent()
    if parent is None:
        return tree
    grandparent = parent.getparent()
    if grandparent is None:
        return tree
    grandparent.append(node_to_rotate)

    return tree


def compute_subtree_depth(element: html.HtmlElement) -> int:
    """Returns distance to deepest leaf (0 for leaves)."""
    if len(element) == 0:
        return 0
    return 1 + max(compute_subtree_depth(child) for child in element)


def _precompute_subtree_depths(
    element: html.HtmlElement,
) -> dict[html.HtmlElement, int]:
    """Returns subtree depths for all nodes in O(n) time."""
    depths: dict[html.HtmlElement, int] = {}

    def _compute(node: html.HtmlElement) -> int:
        if len(node) == 0:
            depths[node] = 0
            return 0
        max_child_depth = max(_compute(child) for child in node)
        depths[node] = max_child_depth + 1
        return depths[node]

    _compute(element)
    return depths


def tree_to_list(tree: html.HtmlElement) -> list[tuple[str, int, int]]:
    """Returns list of (tag, depth, position) tuples via pre-order traversal."""
    result = []

    def _traverse(element: html.HtmlElement, depth: int):
        for i, child in enumerate(element):
            result.append((child.tag, depth + 1, i))
            _traverse(child, depth + 1)

    _traverse(tree, 0)
    return result


def tree_to_rich_list(
    tree: html.HtmlElement,
) -> list[tuple[str, int, int, int, int, int, int, str, int, int]]:
    """Returns 10-dimensional feature tuples for each node via pre-order traversal."""
    result = []
    subtree_depths = _precompute_subtree_depths(tree)

    def _traverse(element: html.HtmlElement, depth: int, parent_tag: str):
        num_siblings = len(element)
        for i, child in enumerate(element):
            tag = child.tag
            semantic_group = TAG_TO_SEMANTIC_GROUP.get(tag, NUM_SEMANTIC_GROUPS)
            num_children = len(child)
            is_leaf = 1 if num_children == 0 else 0
            tag_role = TAG_TO_ROLE.get(tag, TagRole.LEAF)
            child_subtree_depth = subtree_depths.get(child, 0)

            result.append(
                (
                    tag,
                    semantic_group,
                    depth + 1,
                    i,
                    num_children,
                    num_siblings,
                    is_leaf,
                    parent_tag,
                    tag_role,
                    child_subtree_depth,
                )
            )
            _traverse(child, depth + 1, tag)

    _traverse(tree, 0, tree.tag)
    return result


def tree_subtree_shuffle(tree: html.HtmlElement) -> html.HtmlElement:
    """Shuffles children order of a random node. In-place."""
    nodes_with_children = [node for node in tree.iter() if len(node) >= 2]

    if not nodes_with_children:
        return tree

    node_to_shuffle = random.choice(nodes_with_children)
    children = list(node_to_shuffle)
    random.shuffle(children)

    for child in list(node_to_shuffle):
        node_to_shuffle.remove(child)
    for child in children:
        node_to_shuffle.append(child)

    return tree


def tree_subtree_sample(tree: html.HtmlElement, sample_ratio: float = 0.7) -> html.HtmlElement:
    """Randomly keeps a subset of children at each level. In-place."""

    def _sample_recursive(element: html.HtmlElement):
        for child in list(element):
            _sample_recursive(child)

        children = list(element)
        if len(children) > 1:
            num_to_keep = max(1, int(len(children) * sample_ratio))
            if num_to_keep < len(children):
                children_to_keep = set(random.sample(children, num_to_keep))
                for child in list(element):
                    if child not in children_to_keep:
                        element.remove(child)

    _sample_recursive(tree)
    return tree


def tree_depth_truncate(tree: html.HtmlElement, max_depth: int = 6) -> html.HtmlElement:
    """Removes all nodes beyond max_depth. In-place."""

    def _truncate_recursive(element: html.HtmlElement, current_depth: int):
        if current_depth >= max_depth:
            for child in list(element):
                element.remove(child)
        else:
            for child in list(element):
                _truncate_recursive(child, current_depth + 1)

    _truncate_recursive(tree, 0)
    return tree


def tree_sibling_drop(tree: html.HtmlElement, drop_prob: float = 0.3) -> html.HtmlElement:
    """Randomly drops siblings with given probability. In-place."""

    def _drop_recursive(element: html.HtmlElement):
        for child in list(element):
            _drop_recursive(child)

        children = list(element)
        if len(children) > 1:
            for child in children[:-1]:
                if random.random() < drop_prob:
                    element.remove(child)

    _drop_recursive(tree)
    return tree


def tree_semantic_replace(tree: html.HtmlElement, replace_prob: float = 0.2) -> html.HtmlElement:
    """Replaces tags with semantic equivalents (div/section, strong/b, etc.). In-place."""
    for node in tree.iter():
        if node.tag in SEMANTIC_EQUIVALENTS and random.random() < replace_prob:
            equivalents = SEMANTIC_EQUIVALENTS[node.tag]
            node.tag = random.choice(equivalents)

    return tree


def tree_skeleton_extract(
    tree: html.HtmlElement,
    keep_groups: set[str] | None = None,
) -> html.HtmlElement:
    """Keeps only structural tags (containers, navigation, lists, tables). In-place."""
    if keep_groups is None:
        keep_groups = {"container", "navigation", "list", "table", "layout"}

    keep_tags: set[str] = set()
    for group in keep_groups:
        if group in SEMANTIC_GROUPS:
            keep_tags.update(SEMANTIC_GROUPS[group])

    def _extract_recursive(element: html.HtmlElement):
        for child in list(element):
            if child.tag not in keep_tags:
                element.remove(child)
            else:
                _extract_recursive(child)

    _extract_recursive(tree)
    return tree

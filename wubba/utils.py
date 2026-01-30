import random
from typing import List, Tuple

import torch
from lxml import html

from wubba.const import KEEP_TAGS


def create_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Creates a boolean mask from a tensor of token IDs.

    The mask is true for tokens with a non-zero ID (i.e., not padding).

    Args:
        tensor: An input tensor of shape (..., 3) where the first channel
            is the token ID.

    Returns:
        A float tensor mask where 1.0 indicates a valid token and 0.0
        indicates a padding token.
    """
    return (tensor[..., 0] != 0).float()


def is_tree_equal(tree1: html.HtmlElement, tree2: html.HtmlElement) -> bool:
    """Compares two lxml HTML trees for structural equality.

    This function recursively checks if two trees have the same tag and
    the same number of children, then compares their children.

    Args:
        tree1: The first HTML tree.
        tree2: The second HTML tree.

    Returns:
        True if the trees are structurally identical, False otherwise.
    """
    if tree1.tag != tree2.tag:
        return False

    if len(tree1) != len(tree2):
        return False

    return all(is_tree_equal(child1, child2) for child1, child2 in zip(tree1, tree2))


def clean_tree(tree: html.HtmlElement) -> html.HtmlElement:
    """Removes unwanted tags and duplicate adjacent subtrees from an HTML tree.

    This function modifies the tree in-place.

    Args:
        tree: The HTML tree to clean.

    Returns:
        The cleaned HTML tree.
    """
    # Use a set for faster lookups
    keep_tags_set = set(KEEP_TAGS)

    def _clean_recursive(element: html.HtmlElement):
        # Iterate over a copy of children, as we modify the list
        for child in list(element):
            if child.tag not in keep_tags_set:
                element.remove(child)
            else:
                _clean_recursive(child)

        # Remove duplicate adjacent subtrees after cleaning children
        if len(element) > 1:
            i = 0
            while i < len(element) - 1:
                if is_tree_equal(element[i], element[i + 1]):
                    element.remove(element[i + 1])
                else:
                    i += 1

    _clean_recursive(tree)
    return tree


def get_max_depth(tree: html.HtmlElement) -> int:
    """Calculates the maximum depth of an HTML tree.

    The root node is at depth 0.

    Args:
        tree: The HTML tree to analyze.

    Returns:
        The maximum depth of the tree.
    """
    if not len(tree):
        return 0

    return 1 + max(get_max_depth(child) for child in tree)


def get_max_position(tree: html.HtmlElement) -> int:
    """Calculates the maximum sibling position in an HTML tree.

    The position is the zero-based index of a node among its siblings.

    Args:
        tree: The HTML tree to analyze.

    Returns:
        The maximum position index found in the tree.
    """
    max_pos = 0
    for element in tree.iter():
        num_children = len(element)
        if num_children > 1:
            max_pos = max(max_pos, num_children - 1)
    return max_pos


def random_node(tree: html.HtmlElement) -> html.HtmlElement:
    """Selects a random non-root node from an HTML tree.

    Args:
        tree: The HTML tree.

    Returns:
        A randomly selected node.

    Raises:
        IndexError: If the tree has no non-root nodes.
    """
    # list(tree.iter()) includes the root, so we skip it.
    nodes = list(tree.iter())[1:]
    if not nodes:
        raise IndexError("Tree contains only a root node.")

    return random.choice(nodes)


def tree_node_mask(tree: html.HtmlElement) -> html.HtmlElement:
    """Removes a single random node from an HTML tree.

    This function modifies the tree in-place. It tries to avoid removing
    large, high-level sections by descending into a child of the root if
    one is selected initially.

    Args:
        tree: The HTML tree to modify.

    Returns:
        The modified HTML tree.
    """
    if not len(tree):
        return tree

    try:
        node_to_remove = random_node(tree)

        # Avoid removing a direct child of the root if it has children.
        # This prevents deleting a large part of the tree.
        if node_to_remove.getparent() == tree and len(node_to_remove):
            node_to_remove = random_node(node_to_remove)

        node_to_remove.getparent().remove(node_to_remove)
    except IndexError:
        # No removable nodes found.
        pass

    return tree


def tree_child_mask(tree: html.HtmlElement) -> html.HtmlElement:
    """Removes all children of a randomly selected non-root node.

    This function modifies the tree in-place.

    Args:
        tree: The HTML tree to modify.

    Returns:
        The modified HTML tree.
    """
    # Find all non-root nodes that have children.
    potential_parents = [node for node in tree.iter() if node != tree and len(node) > 0]

    if not potential_parents:
        return tree

    # Select a random parent and remove all its children.
    parent_to_modify = random.choice(potential_parents)
    for child in list(parent_to_modify):
        parent_to_modify.remove(child)

    return tree


def tree_node_rotate(tree: html.HtmlElement) -> html.HtmlElement:
    """Moves a random leaf node to be a sibling of its parent.

    This function modifies the tree in-place. It selects a "grandchild"
    leaf node and moves it up one level in the hierarchy.

    Args:
        tree: The HTML tree to modify.

    Returns:
        The modified HTML tree.
    """
    # Find all leaf nodes that are not direct children of the root.
    rotatable_nodes = [
        node
        for node in tree.iter()
        if not len(node) and node.getparent() is not None and node.getparent() != tree
    ]

    if not rotatable_nodes:
        return tree

    node_to_rotate = random.choice(rotatable_nodes)
    parent = node_to_rotate.getparent()
    grandparent = parent.getparent()

    # Move node to be a child of the grandparent.
    grandparent.append(node_to_rotate)

    return tree


def tree_to_list(tree: html.HtmlElement) -> List[Tuple[str, int, int]]:
    """Converts an HTML tree to a list of node features.

    Each node is represented by a tuple containing its tag, depth, and
    position among its siblings. The conversion uses a pre-order traversal.

    Args:
        tree: The HTML tree to convert.

    Returns:
        A list of (tag, depth, position) tuples.
    """
    result = []

    def _traverse(element: html.HtmlElement, depth: int):
        for i, child in enumerate(element):
            result.append((child.tag, depth + 1, i))
            _traverse(child, depth + 1)

    # Start traversal from the root's children
    _traverse(tree, 0)
    return result

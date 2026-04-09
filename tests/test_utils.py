import torch
from lxml import html

from wubba.utils import clean_tree, create_mask, normalize_tree


def test_create_mask_uses_first_feature_channel_for_padding() -> None:
    tensor = torch.tensor(
        [
            [[1, 2], [0, 3]],
            [[4, 5], [0, 0]],
        ]
    )

    mask = create_mask(tensor)

    assert torch.equal(mask, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_clean_tree_removes_non_keep_tags_and_adjacent_duplicates() -> None:
    tree = html.fromstring(
        """
        <body>
          <custom-widget></custom-widget>
          <div><p></p></div>
          <div><p></p></div>
        </body>
        """
    )

    cleaned = clean_tree(tree)

    children = list(cleaned)
    assert [child.tag for child in children] == ["div"]
    assert children[0][0].tag == "p"


def test_normalize_tree_limits_depth_and_siblings() -> None:
    tree = html.fromstring(
        """
        <body>
          <div id="a"><p><span></span></p></div>
          <section id="b"></section>
          <article id="c"></article>
          <nav id="d"></nav>
          <table id="e"></table>
        </body>
        """
    )

    normalized = normalize_tree(tree, max_siblings=3, max_depth=2)

    children = list(normalized)
    assert len(children) == 3
    assert children[0].get("id") == "a"
    assert children[-1].get("id") == "e"
    assert len(children[0][0]) == 0

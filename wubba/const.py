from enum import Enum


class AugmentationType(Enum):
    ORIGINAL = "original"
    NODE_MASK = "node_mask"
    CHILD_MASK = "child_mask"
    NODE_ROTATE = "node_rotate"


KEEP_TAGS = [
    "body",
    "form",
    "div",
    "h1",
    "h2",
    "article",
    "header",
    "footer",
    "section",
    "nav",
    "aside",
    "main",
    "iframe",
    "center",
]

VOCAB = {tag: index + 1 for index, tag in enumerate(KEEP_TAGS)}
VOCAB["<pad>"] = 0  # Padding token

from enum import Enum, IntEnum


class AugmentationType(Enum):
    """HTML tree augmentation types."""

    ORIGINAL = "original"
    NODE_MASK = "node_mask"
    CHILD_MASK = "child_mask"
    NODE_ROTATE = "node_rotate"
    SUBTREE_SHUFFLE = "subtree_shuffle"
    SUBTREE_SAMPLE = "subtree_sample"
    DEPTH_TRUNCATE = "depth_truncate"
    SIBLING_DROP = "sibling_drop"
    SEMANTIC_REPLACE = "semantic_replace"
    SKELETON_EXTRACT = "skeleton_extract"


SEMANTIC_GROUPS: dict[str, list[str]] = {
    "container": ["body", "div", "section", "article", "main", "aside", "span"],
    "heading": ["h1", "h2", "h3", "h4", "h5", "h6"],
    "navigation": ["nav", "a", "menu", "menuitem"],
    "list": ["ul", "ol", "li", "dl", "dt", "dd"],
    "table": ["table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col"],
    "form": [
        "form",
        "input",
        "button",
        "select",
        "textarea",
        "label",
        "fieldset",
        "legend",
        "option",
        "optgroup",
    ],
    "media": [
        "img",
        "video",
        "audio",
        "picture",
        "figure",
        "figcaption",
        "canvas",
        "svg",
        "source",
        "track",
    ],
    "text": [
        "p",
        "strong",
        "em",
        "b",
        "i",
        "u",
        "s",
        "blockquote",
        "pre",
        "code",
        "cite",
        "q",
        "abbr",
        "mark",
        "small",
        "sub",
        "sup",
        "time",
        "address",
    ],
    "layout": ["header", "footer", "center", "iframe", "hr", "br", "wbr"],
    "semantic": ["details", "summary", "dialog", "template", "slot"],
    "embedded": ["object", "embed", "param", "noscript", "script", "style", "link", "meta"],
}

# Build unique tag list preserving first occurrence order
_seen: set[str] = set()
KEEP_TAGS: list[str] = []
for tags in SEMANTIC_GROUPS.values():
    for tag in tags:
        if tag not in _seen:
            _seen.add(tag)
            KEEP_TAGS.append(tag)

VOCAB: dict[str, int] = {tag: index + 1 for index, tag in enumerate(KEEP_TAGS)}
VOCAB["<pad>"] = 0
VOCAB["<unk>"] = len(KEEP_TAGS) + 1

SEMANTIC_GROUP_NAMES: list[str] = list(SEMANTIC_GROUPS.keys())
TAG_TO_SEMANTIC_GROUP: dict[str, int] = {}
for group_idx, (_group_name, tags) in enumerate(SEMANTIC_GROUPS.items()):
    for tag in tags:
        TAG_TO_SEMANTIC_GROUP[tag] = group_idx

NUM_SEMANTIC_GROUPS: int = len(SEMANTIC_GROUPS)


# Semantic equivalents: tags interchangeable without changing layout
SEMANTIC_EQUIVALENTS: dict[str, list[str]] = {
    # Container equivalents
    "div": ["section", "article"],
    "section": ["div", "article"],
    "article": ["div", "section"],
    # Text emphasis equivalents
    "strong": ["b"],
    "b": ["strong"],
    "em": ["i"],
    "i": ["em"],
    # List type equivalents
    "ul": ["ol"],
    "ol": ["ul"],
    # Heading level equivalents (adjacent levels only)
    "h1": ["h2"],
    "h2": ["h1", "h3"],
    "h3": ["h2", "h4"],
    "h4": ["h3", "h5"],
    "h5": ["h4", "h6"],
    "h6": ["h5"],
}


class TagRole(IntEnum):
    """Structural role of HTML tags in the DOM tree."""

    ROOT = 0
    STRUCTURAL = 1
    CONTAINER = 2
    LIST_CONTAINER = 3
    LEAF_CONTAINER = 4
    LEAF = 5


TAG_TO_ROLE: dict[str, int] = {
    # ROOT
    "body": TagRole.ROOT,
    # STRUCTURAL
    "nav": TagRole.STRUCTURAL,
    "header": TagRole.STRUCTURAL,
    "footer": TagRole.STRUCTURAL,
    "main": TagRole.STRUCTURAL,
    "aside": TagRole.STRUCTURAL,
    # CONTAINER
    "div": TagRole.CONTAINER,
    "section": TagRole.CONTAINER,
    "article": TagRole.CONTAINER,
    "span": TagRole.CONTAINER,
    # LIST_CONTAINER
    "ul": TagRole.LIST_CONTAINER,
    "ol": TagRole.LIST_CONTAINER,
    "dl": TagRole.LIST_CONTAINER,
    "table": TagRole.LIST_CONTAINER,
    "thead": TagRole.LIST_CONTAINER,
    "tbody": TagRole.LIST_CONTAINER,
    "tfoot": TagRole.LIST_CONTAINER,
    "menu": TagRole.LIST_CONTAINER,
    "form": TagRole.LIST_CONTAINER,
    "fieldset": TagRole.LIST_CONTAINER,
    "select": TagRole.LIST_CONTAINER,
    "optgroup": TagRole.LIST_CONTAINER,
    "colgroup": TagRole.LIST_CONTAINER,
    "figure": TagRole.LIST_CONTAINER,
    "picture": TagRole.LIST_CONTAINER,
    "details": TagRole.LIST_CONTAINER,
    # LEAF_CONTAINER
    "p": TagRole.LEAF_CONTAINER,
    "li": TagRole.LEAF_CONTAINER,
    "dt": TagRole.LEAF_CONTAINER,
    "dd": TagRole.LEAF_CONTAINER,
    "tr": TagRole.LEAF_CONTAINER,
    "th": TagRole.LEAF_CONTAINER,
    "td": TagRole.LEAF_CONTAINER,
    "caption": TagRole.LEAF_CONTAINER,
    "a": TagRole.LEAF_CONTAINER,
    "button": TagRole.LEAF_CONTAINER,
    "label": TagRole.LEAF_CONTAINER,
    "legend": TagRole.LEAF_CONTAINER,
    "option": TagRole.LEAF_CONTAINER,
    "menuitem": TagRole.LEAF_CONTAINER,
    "blockquote": TagRole.LEAF_CONTAINER,
    "pre": TagRole.LEAF_CONTAINER,
    "address": TagRole.LEAF_CONTAINER,
    "figcaption": TagRole.LEAF_CONTAINER,
    "summary": TagRole.LEAF_CONTAINER,
    "h1": TagRole.LEAF_CONTAINER,
    "h2": TagRole.LEAF_CONTAINER,
    "h3": TagRole.LEAF_CONTAINER,
    "h4": TagRole.LEAF_CONTAINER,
    "h5": TagRole.LEAF_CONTAINER,
    "h6": TagRole.LEAF_CONTAINER,
    # LEAF (default for most inline/empty elements)
    "img": TagRole.LEAF,
    "input": TagRole.LEAF,
    "textarea": TagRole.LEAF,
    "video": TagRole.LEAF,
    "audio": TagRole.LEAF,
    "canvas": TagRole.LEAF,
    "svg": TagRole.LEAF,
    "source": TagRole.LEAF,
    "track": TagRole.LEAF,
    "iframe": TagRole.LEAF,
    "embed": TagRole.LEAF,
    "object": TagRole.LEAF,
    "param": TagRole.LEAF,
    "col": TagRole.LEAF,
    "hr": TagRole.LEAF,
    "br": TagRole.LEAF,
    "wbr": TagRole.LEAF,
    "strong": TagRole.LEAF,
    "em": TagRole.LEAF,
    "b": TagRole.LEAF,
    "i": TagRole.LEAF,
    "u": TagRole.LEAF,
    "s": TagRole.LEAF,
    "code": TagRole.LEAF,
    "cite": TagRole.LEAF,
    "q": TagRole.LEAF,
    "abbr": TagRole.LEAF,
    "mark": TagRole.LEAF,
    "small": TagRole.LEAF,
    "sub": TagRole.LEAF,
    "sup": TagRole.LEAF,
    "time": TagRole.LEAF,
    "center": TagRole.LEAF,
    "dialog": TagRole.LEAF,
    "template": TagRole.LEAF,
    "slot": TagRole.LEAF,
    "noscript": TagRole.LEAF,
    "script": TagRole.LEAF,
    "style": TagRole.LEAF,
    "link": TagRole.LEAF,
    "meta": TagRole.LEAF,
}

NUM_TAG_ROLES: int = len(TagRole)


class AugmentationTier(Enum):
    """Augmentation impact level: MICRO (local), MESO (subtree), MACRO (global)."""

    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"


AUGMENTATION_TIERS: dict[AugmentationTier, list[AugmentationType]] = {
    AugmentationTier.MICRO: [
        AugmentationType.SEMANTIC_REPLACE,
    ],
    AugmentationTier.MESO: [
        AugmentationType.NODE_MASK,
        AugmentationType.CHILD_MASK,
        AugmentationType.NODE_ROTATE,
        AugmentationType.SUBTREE_SHUFFLE,
        AugmentationType.SIBLING_DROP,
    ],
    AugmentationTier.MACRO: [
        AugmentationType.SUBTREE_SAMPLE,
        AugmentationType.DEPTH_TRUNCATE,
        AugmentationType.SKELETON_EXTRACT,
    ],
}

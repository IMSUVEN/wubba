"""HTML data processing, augmentation, and DataModule."""

import copy
import random
from collections.abc import Callable

import lightning as L
import numpy as np
import pandas as pd
import torch
from lxml import html
from torch.utils.data import DataLoader, Dataset

from wubba.config import Config
from wubba.const import (
    AUGMENTATION_TIERS,
    SEMANTIC_GROUPS,
    TAG_TO_SEMANTIC_GROUP,
    VOCAB,
    AugmentationTier,
    AugmentationType,
)
from wubba.utils import (
    clean_tree,
    normalize_tree,
    tree_child_mask,
    tree_depth_truncate,
    tree_node_mask,
    tree_node_rotate,
    tree_semantic_replace,
    tree_sibling_drop,
    tree_skeleton_extract,
    tree_subtree_sample,
    tree_subtree_shuffle,
    tree_to_rich_list,
)

FEATURE_DIM_BASE = 10
FEATURE_DIM_EXTENDED = 15


class HTMLDataProcessor:
    """Converts HTML to tensor features with augmentation support."""

    def __init__(
        self,
        max_depth: int,
        max_position: int,
        max_sequence_length: int,
        max_children: int = 64,
        max_siblings: int = 64,
        max_subtree_depth: int = 32,
        aug_strong_prob: float = 0.8,
        aug_depth_truncate_max: int = 8,
        aug_sibling_drop_prob: float = 0.3,
        aug_subtree_sample_ratio: float = 0.7,
        aug_semantic_replace_prob: float = 0.2,
        aug_skeleton_keep_groups: list[str] | None = None,
        aug_tier_weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
        # Phase 2: Extended features
        use_extended_features: bool = False,
        # Phase 2: Contextual augmentation
        use_contextual_aug: bool = False,
        contextual_structure_prob: float = 0.3,
        contextual_semantic_prob: float = 0.2,
        # Phase 2: Tree Mixup
        use_tree_mixup: bool = False,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.3,
    ):
        self.max_depth = max_depth
        self.max_position = max_position
        self.max_sequence_length = max_sequence_length
        self.max_children = max_children
        self.max_siblings = max_siblings
        self.max_subtree_depth = max_subtree_depth
        self.vocab_map = VOCAB

        # Feature configuration
        self.use_extended_features = use_extended_features
        self.feature_dim = FEATURE_DIM_EXTENDED if use_extended_features else FEATURE_DIM_BASE

        # Augmentation parameters
        self.aug_strong_prob = aug_strong_prob
        self.aug_depth_truncate_max = aug_depth_truncate_max
        self.aug_sibling_drop_prob = aug_sibling_drop_prob
        self.aug_subtree_sample_ratio = aug_subtree_sample_ratio
        self.aug_semantic_replace_prob = aug_semantic_replace_prob
        self.aug_skeleton_keep_groups = set(
            aug_skeleton_keep_groups
            if aug_skeleton_keep_groups is not None
            else ["container", "navigation", "list", "table", "layout"]
        )
        self.aug_tier_weights = aug_tier_weights

        # Contextual augmentation
        self.use_contextual_aug = use_contextual_aug
        self.contextual_structure_prob = contextual_structure_prob
        self.contextual_semantic_prob = contextual_semantic_prob

        # Tree Mixup
        self.use_tree_mixup = use_tree_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        # Semantic group sets for contextual augmentation
        self._container_groups = {"container", "layout"}
        self._leaf_groups = {"text", "media", "form"}

    def _apply_augmentation(
        self, tree: html.HtmlElement, aug_type: AugmentationType
    ) -> html.HtmlElement:
        tree_copy = copy.deepcopy(tree)
        if aug_type == AugmentationType.NODE_MASK:
            return tree_node_mask(tree_copy)
        if aug_type == AugmentationType.CHILD_MASK:
            return tree_child_mask(tree_copy)
        if aug_type == AugmentationType.NODE_ROTATE:
            return tree_node_rotate(tree_copy)
        if aug_type == AugmentationType.SUBTREE_SHUFFLE:
            return tree_subtree_shuffle(tree_copy)
        if aug_type == AugmentationType.SUBTREE_SAMPLE:
            return tree_subtree_sample(tree_copy, self.aug_subtree_sample_ratio)
        if aug_type == AugmentationType.DEPTH_TRUNCATE:
            return tree_depth_truncate(tree_copy, self.aug_depth_truncate_max)
        if aug_type == AugmentationType.SIBLING_DROP:
            return tree_sibling_drop(tree_copy, self.aug_sibling_drop_prob)
        if aug_type == AugmentationType.SEMANTIC_REPLACE:
            return tree_semantic_replace(tree_copy, self.aug_semantic_replace_prob)
        if aug_type == AugmentationType.SKELETON_EXTRACT:
            return tree_skeleton_extract(tree_copy, self.aug_skeleton_keep_groups)
        return tree_copy  # ORIGINAL

    def create_augmented_pair(
        self, tree: html.HtmlElement
    ) -> tuple[html.HtmlElement, html.HtmlElement]:
        """Creates two augmented views for contrastive learning."""
        tiers = [AugmentationTier.MICRO, AugmentationTier.MESO, AugmentationTier.MACRO]

        def apply_tiered_augs(base_tree: html.HtmlElement, num_augs: int) -> html.HtmlElement:
            aug_tree = copy.deepcopy(base_tree)

            if self.use_contextual_aug:
                aug_tree = self._apply_contextual_augmentation(aug_tree)

            for _ in range(num_augs):
                tier = random.choices(tiers, weights=self.aug_tier_weights, k=1)[0]
                tier_augs = AUGMENTATION_TIERS[tier]

                if tier_augs:
                    aug_type = random.choice(tier_augs)
                    aug_tree = self._apply_augmentation(aug_tree, aug_type)

            return aug_tree

        if random.random() < self.aug_strong_prob:
            num_augs1 = random.randint(2, 3)
            num_augs2 = random.randint(1, 2)
        else:
            num_augs1 = random.randint(1, 2)
            num_augs2 = 1

        tree1 = apply_tiered_augs(tree, num_augs1)
        tree2 = apply_tiered_augs(tree, num_augs2)

        return tree1, tree2

    def _apply_contextual_augmentation(self, tree: html.HtmlElement) -> html.HtmlElement:
        """Applies node-type-aware augmentation (structural for containers, semantic for leaves)."""
        tree_copy = copy.deepcopy(tree)

        for node in tree_copy.iter():
            semantic_group_name = None

            # Get semantic group name
            for name, tags in SEMANTIC_GROUPS.items():
                if node.tag in tags:
                    semantic_group_name = name
                    break

            if semantic_group_name in self._container_groups:
                # Structural augmentation for containers
                if random.random() < self.contextual_structure_prob and len(node) >= 2:
                    # Shuffle children
                    children = list(node)
                    random.shuffle(children)
                    for child in list(node):
                        node.remove(child)
                    for child in children:
                        node.append(child)

            elif (
                semantic_group_name in self._leaf_groups
                and random.random() < self.contextual_semantic_prob
            ):
                # Semantic augmentation for leaf-like nodes
                from wubba.const import SEMANTIC_EQUIVALENTS

                if node.tag in SEMANTIC_EQUIVALENTS:
                    node.tag = random.choice(SEMANTIC_EQUIVALENTS[node.tag])

        return tree_copy

    def tree_to_features(self, tree: html.HtmlElement) -> list[list[int]]:
        """Converts HTML tree to feature sequence (10 or 15 dimensions)."""
        tree_list = tree_to_rich_list(tree)
        max_tree_depth = max((item[2] for item in tree_list), default=1) if tree_list else 1

        features = []
        for (
            tag,
            semantic_group,
            depth,
            position,
            num_children,
            sibling_count,
            is_leaf,
            parent_tag,
            tag_role,
            subtree_depth,
        ) in tree_list:
            # Get tag_id and parent_tag_id, use <unk> token for unknown tags
            tag_id = self.vocab_map.get(tag, self.vocab_map.get("<unk>", 1))
            parent_tag_id = self.vocab_map.get(parent_tag, self.vocab_map.get("<unk>", 1))

            feature = [
                tag_id,
                semantic_group,
                min(depth, self.max_depth - 1),
                min(position, self.max_position - 1),
                min(num_children, self.max_children - 1),
                min(sibling_count, self.max_siblings - 1),
                is_leaf,
                parent_tag_id,
                tag_role,
                min(subtree_depth, self.max_subtree_depth - 1),
            ]

            if self.use_extended_features:
                # Placeholders for text-based features (filled during HTML parsing)
                text_length_bucket = 0
                has_text = 0
                attribute_count = 0
                css_class_hash = 0
                path_depth_ratio = int((depth / max(max_tree_depth, 1)) * 15)

                feature.extend(
                    [
                        text_length_bucket,
                        has_text,
                        attribute_count,
                        css_class_hash,
                        path_depth_ratio,
                    ]
                )

            features.append(feature)

        return features

    def tree_to_features_extended(self, tree: html.HtmlElement) -> list[list[int]]:
        """Extracts extended 15-dimensional features from HTML elements."""
        from wubba.const import NUM_SEMANTIC_GROUPS, TAG_TO_ROLE, TagRole
        from wubba.utils import _precompute_subtree_depths

        features = []
        subtree_depths = _precompute_subtree_depths(tree)
        max_tree_depth = 0

        def get_depth(elem: html.HtmlElement, d: int = 0) -> int:
            max_d = d
            for child in elem:
                max_d = max(max_d, get_depth(child, d + 1))
            return max_d

        max_tree_depth = get_depth(tree)

        def traverse(element: html.HtmlElement, depth: int, parent_tag: str):
            num_siblings = len(element)
            for i, child in enumerate(element):
                tag = child.tag
                semantic_group = TAG_TO_SEMANTIC_GROUP.get(tag, NUM_SEMANTIC_GROUPS)
                num_children = len(child)
                is_leaf = 1 if num_children == 0 else 0
                tag_role = TAG_TO_ROLE.get(tag, TagRole.LEAF)
                child_subtree_depth = subtree_depths.get(child, 0)

                tag_id = self.vocab_map.get(tag, self.vocab_map.get("<unk>", 1))
                parent_tag_id = self.vocab_map.get(parent_tag, self.vocab_map.get("<unk>", 1))

                feature = [
                    tag_id,
                    semantic_group,
                    min(depth + 1, self.max_depth - 1),
                    min(i, self.max_position - 1),
                    min(num_children, self.max_children - 1),
                    min(num_siblings, self.max_siblings - 1),
                    is_leaf,
                    parent_tag_id,
                    tag_role,
                    min(child_subtree_depth, self.max_subtree_depth - 1),
                ]

                text = child.text or ""
                text_length = len(text.strip())
                text_length_bucket = min(7, text_length // 20)
                has_text = 1 if text_length > 0 else 0
                attribute_count = min(15, len(child.attrib))
                css_classes = child.get("class", "")
                css_class_hash = hash(css_classes) % 128 if css_classes else 0
                path_depth_ratio = int(((depth + 1) / max(max_tree_depth, 1)) * 15)

                feature.extend(
                    [
                        text_length_bucket,
                        has_text,
                        attribute_count,
                        css_class_hash,
                        path_depth_ratio,
                    ]
                )

                features.append(feature)
                traverse(child, depth + 1, tag)

        traverse(tree, 0, tree.tag)
        return features

    def features_to_tensor(self, features: list[list[int]]) -> torch.Tensor:
        """Pads/truncates features to fixed sequence length."""
        features = features[: self.max_sequence_length]
        padding_needed = self.max_sequence_length - len(features)
        if padding_needed > 0:
            features.extend([[0] * self.feature_dim] * padding_needed)

        return torch.tensor(features, dtype=torch.long)

    def apply_feature_mixup(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Mixes two feature tensors with Beta-sampled lambda."""
        if random.random() > self.mixup_prob:
            return features1, 1.0

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        # Probabilistic mixing for discrete features
        mask = torch.rand_like(features1.float()) < lam
        mixed = torch.where(mask, features1, features2)

        return mixed, lam

    def tree_to_tensor(self, tree: html.HtmlElement) -> torch.Tensor:
        features = self.tree_to_features(tree)
        return self.features_to_tensor(features)

    def html_to_tensor(self, html_content: str) -> torch.Tensor:
        try:
            tree = html.fromstring(html_content)
            return self.tree_to_tensor(tree)
        except (ValueError, html.etree.ParserError):
            return torch.zeros(self.max_sequence_length, self.feature_dim, dtype=torch.long)

    def html_clean_to_tensor(self, html_content: str) -> torch.Tensor:
        try:
            tree = html.fromstring(html_content)
            tree = clean_tree(tree)
            return self.tree_to_tensor(tree)
        except (ValueError, html.etree.ParserError):
            return torch.zeros(self.max_sequence_length, self.feature_dim, dtype=torch.long)

    def html_normalize_to_tensor(
        self,
        html_content: str,
        max_siblings: int = 10,
        max_depth: int = 10,
    ) -> torch.Tensor:
        """Converts HTML to tensor with aggressive normalization and deduplication."""
        try:
            tree = html.fromstring(html_content)
            tree = normalize_tree(
                tree,
                max_siblings=max_siblings,
                max_depth=max_depth,
                collapse_consecutive=True,
                aggressive_dedup=True,
            )
            return self.tree_to_tensor(tree)
        except (ValueError, html.etree.ParserError):
            return torch.zeros(self.max_sequence_length, self.feature_dim, dtype=torch.long)

    def html_to_tensor_pair(self, html_content: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Creates deterministic tensor pair (no augmentation) for validation."""
        tensor = self.html_clean_to_tensor(html_content)
        return tensor, tensor.clone()

    def augment_html_to_tensor_pair(self, html_content: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Creates two augmented tensor views for contrastive learning."""
        try:
            tree = html.fromstring(html_content)
        except (ValueError, html.etree.ParserError):
            tensor = torch.zeros(self.max_sequence_length, self.feature_dim, dtype=torch.long)
            return tensor, tensor

        tree1, tree2 = self.create_augmented_pair(tree)

        if self.use_extended_features:
            features1 = self.tree_to_features_extended(tree1)
            features2 = self.tree_to_features_extended(tree2)
        else:
            features1 = self.tree_to_features(tree1)
            features2 = self.tree_to_features(tree2)

        tensor1 = self.features_to_tensor(features1)
        tensor2 = self.features_to_tensor(features2)

        if self.use_tree_mixup and random.random() < self.mixup_prob:
            tensor1, _ = self.apply_feature_mixup(tensor1, tensor2)

        return tensor1, tensor2


class HTMLLayoutDataset(Dataset):
    """PyTorch Dataset for HTML documents."""

    def __init__(self, data: list[str], transform: Callable):
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.transform(self.data[idx])


class WubbaDataModule(L.LightningDataModule):
    """Lightning DataModule for HTML training data."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.data_processor = HTMLDataProcessor(
            max_depth=config.max_depth,
            max_position=config.max_position,
            max_sequence_length=config.max_sequence_length,
            max_children=config.max_children,
            max_siblings=config.max_siblings,
            max_subtree_depth=config.max_subtree_depth,
            aug_strong_prob=config.aug_strong_prob,
            aug_depth_truncate_max=config.aug_depth_truncate_max,
            aug_sibling_drop_prob=config.aug_sibling_drop_prob,
            aug_subtree_sample_ratio=config.aug_subtree_sample_ratio,
            aug_semantic_replace_prob=config.aug_semantic_replace_prob,
            aug_skeleton_keep_groups=config.aug_skeleton_keep_groups,
            aug_tier_weights=config.aug_tier_weights,
            use_extended_features=config.use_extended_features,
            use_contextual_aug=config.use_contextual_aug,
            contextual_structure_prob=config.contextual_structure_prob,
            contextual_semantic_prob=config.contextual_semantic_prob,
            use_tree_mixup=config.use_tree_mixup,
            mixup_alpha=config.mixup_alpha,
            mixup_prob=config.mixup_prob,
        )
        self.train_data_path = config.data_dir / "train_data.parquet"
        self.val_data_path = config.data_dir / "val_data.parquet"
        self.train_dataset: HTMLLayoutDataset | None = None
        self.val_dataset: HTMLLayoutDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            train_df = pd.read_parquet(self.train_data_path)
            self.train_dataset = HTMLLayoutDataset(
                data=train_df["html"].tolist(),
                transform=self.data_processor.augment_html_to_tensor_pair,
            )

            val_df = pd.read_parquet(self.val_data_path)
            self.val_dataset = HTMLLayoutDataset(
                data=val_df["html"].tolist(),
                transform=self.data_processor.html_to_tensor_pair,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            drop_last=False,
        )

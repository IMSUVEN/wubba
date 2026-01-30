import copy
import random
from typing import Callable, List, Tuple

import lightning as L
import pandas as pd
import torch
from lxml import html
from torch.utils.data import DataLoader, Dataset

from wubba.config import Config
from wubba.const import VOCAB, AugmentationType
from wubba.utils import (
    clean_tree,
    tree_child_mask,
    tree_node_mask,
    tree_node_rotate,
    tree_to_list,
)


class HTMLDataProcessor:
    """Processes HTML strings into tensors for model consumption.

    This class handles parsing HTML, applying augmentations, converting DOM
    trees to feature sequences, and padding/truncating them into tensors.

    Attributes:
        max_depth: The maximum depth to consider for a node.
        max_position: The maximum sibling position to consider.
        max_sequence_length: The fixed length for output sequences.
    """

    def __init__(
        self,
        max_depth: int,
        max_position: int,
        max_sequence_length: int,
    ):
        """Initializes the data processor.

        Args:
            max_depth: Maximum node depth.
            max_position: Maximum node position among siblings.
            max_sequence_length: The sequence length for the model.
        """
        self.max_depth = max_depth
        self.max_position = max_position
        self.max_sequence_length = max_sequence_length
        self.vocab_map = VOCAB

    def _apply_augmentation(
        self, tree: html.HtmlElement, aug_type: AugmentationType
    ) -> html.HtmlElement:
        """Applies a single augmentation to a deep copy of an HTML tree."""
        tree_copy = copy.deepcopy(tree)
        if aug_type == AugmentationType.NODE_MASK:
            return tree_node_mask(tree_copy)
        if aug_type == AugmentationType.CHILD_MASK:
            return tree_child_mask(tree_copy)
        if aug_type == AugmentationType.NODE_ROTATE:
            return tree_node_rotate(tree_copy)
        return tree_copy  # ORIGINAL

    def create_augmented_pair(
        self, tree: html.HtmlElement
    ) -> Tuple[html.HtmlElement, html.HtmlElement]:
        """Creates two augmented views of an HTML tree.

        This method applies a combination of weak and strong augmentations
        to generate two related but different versions of the input tree,
        suitable for contrastive learning.

        Args:
            tree: The original HTML tree.

        Returns:
            A tuple containing two augmented HTML trees.
        """
        weak_augs = [AugmentationType.ORIGINAL, AugmentationType.NODE_ROTATE]
        strong_augs = [AugmentationType.NODE_MASK, AugmentationType.CHILD_MASK]

        def apply_augs(base_tree, aug_list, count):
            aug_tree = copy.deepcopy(base_tree)
            selected = random.sample(aug_list, min(count, len(aug_list)))
            for aug in selected:
                aug_tree = self._apply_augmentation(aug_tree, aug)
            return aug_tree

        # Probabilistically choose between strong and weak augmentation strategies
        # to create pairs with varying difficulty.
        if random.random() < 0.8:  # 80% chance for strong augmentations
            # Both views are strongly augmented
            tree1 = apply_augs(tree, strong_augs, 2)
            tree2 = apply_augs(tree, strong_augs, 1)
        else:
            # One view is a mix, the other is weakly augmented
            tree1 = apply_augs(tree, weak_augs + strong_augs, 1)
            tree2 = apply_augs(tree, weak_augs, 1)

        return tree1, tree2

    def tree_to_features(self, tree: html.HtmlElement) -> List[List[int]]:
        """Converts an HTML tree into a sequence of node features.

        Args:
            tree: The HTML tree.

        Returns:
            A list of features, where each feature is a list of
            [tag_id, depth, position].
        """
        tree_list = tree_to_list(tree)
        features = []
        for tag, depth, position in tree_list:
            tag_id = self.vocab_map.get(tag, 0)  # Default to padding
            features.append(
                [
                    tag_id,
                    min(depth, self.max_depth - 1),
                    min(position, self.max_position - 1),
                ]
            )
        return features

    def features_to_tensor(self, features: List[List[int]]) -> torch.Tensor:
        """Converts a list of features to a padded/truncated tensor."""
        # Truncate if longer than max_sequence_length
        features = features[: self.max_sequence_length]

        # Pad if shorter
        padding_needed = self.max_sequence_length - len(features)
        if padding_needed > 0:
            features.extend([[0, 0, 0]] * padding_needed)

        return torch.tensor(features, dtype=torch.long)

    def tree_to_tensor(self, tree: html.HtmlElement) -> torch.Tensor:
        """Converts an HTML tree to a tensor."""
        features = self.tree_to_features(tree)
        return self.features_to_tensor(features)

    def html_to_tensor(self, html_content: str) -> torch.Tensor:
        """Converts a raw HTML string to a tensor."""
        try:
            tree = html.fromstring(html_content)
            return self.tree_to_tensor(tree)
        except (ValueError, html.etree.ParserError):
            # Handle empty or malformed HTML
            return torch.zeros(self.max_sequence_length, 3, dtype=torch.long)

    def html_clean_to_tensor(self, html_content: str) -> torch.Tensor:
        """Converts a raw HTML string to a tensor."""
        try:
            tree = html.fromstring(html_content)
            tree = clean_tree(tree)
            return self.tree_to_tensor(tree)
        except (ValueError, html.etree.ParserError):
            # Handle empty or malformed HTML
            return torch.zeros(self.max_sequence_length, 3, dtype=torch.long)

    def augment_html_to_tensor_pair(
        self, html_content: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts an HTML string to a pair of augmented tensors."""
        try:
            tree = html.fromstring(html_content)
        except (ValueError, html.etree.ParserError):
            # If HTML is bad, return two zero tensors
            tensor = torch.zeros(self.max_sequence_length, 3, dtype=torch.long)
            return tensor, tensor

        tree1, tree2 = self.create_augmented_pair(tree)
        tensor1 = self.features_to_tensor(self.tree_to_features(tree1))
        tensor2 = self.features_to_tensor(self.tree_to_features(tree2))
        return tensor1, tensor2


class HTMLLayoutDataset(Dataset):
    """A PyTorch Dataset for HTML documents.

    Args:
        data: A list of HTML strings.
        transform: A callable to be applied to each HTML string.
    """

    def __init__(self, data: List[str], transform: Callable):
        super().__init__()
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.transform(self.data[idx])


class WubbaDataModule(L.LightningDataModule):
    """LightningDataModule for loading and preparing HTML data.

    This module handles loading data from parquet files, setting up
    PyTorch Datasets, and creating DataLoaders for training and validation.

    Args:
        config: A Config object with data and training parameters.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.data_processor = HTMLDataProcessor(
            max_depth=config.max_depth,
            max_position=config.max_position,
            max_sequence_length=config.max_sequence_length,
        )
        self.train_data_path = config.data_dir / "train_data.parquet"
        self.val_data_path = config.data_dir / "val_data.parquet"
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        """Loads and splits the data. Called once per trainer."""
        if stage == "fit" or stage is None:
            train_df = pd.read_parquet(self.train_data_path)
            self.train_dataset = HTMLLayoutDataset(
                data=train_df["html"].tolist(),
                transform=self.data_processor.augment_html_to_tensor_pair,
            )

            val_df = pd.read_parquet(self.val_data_path)
            self.val_dataset = HTMLLayoutDataset(
                data=val_df["html"].tolist(),
                transform=self.data_processor.augment_html_to_tensor_pair,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
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
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            drop_last=True,
        )

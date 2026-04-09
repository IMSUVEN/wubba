import numpy as np
import pytest
import torch

from wubba.data import HTMLDataProcessor


def make_processor(**kwargs: object) -> HTMLDataProcessor:
    defaults = {
        "max_depth": 8,
        "max_position": 8,
        "max_sequence_length": 6,
    }
    defaults.update(kwargs)
    return HTMLDataProcessor(**defaults)


def test_html_to_tensor_returns_zero_tensor_for_invalid_html() -> None:
    processor = make_processor()

    tensor = processor.html_to_tensor("\x00")

    assert torch.equal(tensor, torch.zeros(6, 10, dtype=torch.long))


def test_html_to_tensor_pair_is_deterministic_for_validation() -> None:
    processor = make_processor()

    tensor1, tensor2 = processor.html_to_tensor_pair("<body><div><p></p></div></body>")

    assert torch.equal(tensor1, tensor2)


def test_tree_to_features_extended_adds_extra_feature_columns() -> None:
    processor = make_processor(use_extended_features=True)

    tensor1, tensor2 = processor.augment_html_to_tensor_pair(
        '<body><div class="hero"><p>Hello</p></div></body>'
    )

    assert tensor1.shape == (6, 15)
    assert tensor2.shape == (6, 15)


def test_apply_feature_mixup_can_fully_select_second_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = make_processor(use_tree_mixup=True, mixup_prob=1.0, mixup_alpha=0.2)
    features1 = torch.ones(4, 10, dtype=torch.long)
    features2 = torch.full((4, 10), 7, dtype=torch.long)

    monkeypatch.setattr(np.random, "beta", lambda a, b: 0.0)
    monkeypatch.setattr(torch, "rand_like", lambda tensor, *args, **kwargs: torch.ones_like(tensor))
    mixed, lam = processor.apply_feature_mixup(features1, features2)

    assert lam == 0.0
    assert torch.equal(mixed, features2)

import torch

from wubba.metrics import CollapseDetector, EmbeddingMetrics


def test_uniformity_returns_inf_for_single_embedding() -> None:
    z = torch.tensor([[1.0, 0.0, 0.0]])

    result = EmbeddingMetrics.uniformity(z)

    assert result == float("inf")


def test_compute_all_includes_alignment_when_second_view_is_provided() -> None:
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z2 = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    metrics = EmbeddingMetrics.compute_all(z1, z2)

    assert "alignment" in metrics
    assert 0.0 <= metrics["rank_ratio"] <= 1.0
    assert metrics["sim_max"] >= metrics["sim_min"]


def test_cosine_similarity_stats_handles_single_pair_without_warning_shape() -> None:
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    stats = EmbeddingMetrics.cosine_similarity_stats(z)

    assert stats == {
        "sim_min": 0.0,
        "sim_mean": 0.0,
        "sim_max": 0.0,
        "sim_std": 0.0,
    }


def test_collapse_detector_flags_collapsing_embeddings() -> None:
    detector = CollapseDetector(rank_threshold=0.6, std_threshold=0.05)
    embeddings = torch.ones(8, 4)

    status = detector.check(embeddings)

    assert status.is_collapsing is True
    assert status.min_std == 0.0
    assert status.warning is not None


def test_collapse_detector_tracks_declining_trend() -> None:
    detector = CollapseDetector(rank_threshold=0.1, std_threshold=0.0)
    detector.history = [
        {"rank_ratio": 0.9, "min_std": 0.5},
        {"rank_ratio": 0.6, "min_std": 0.3},
    ]
    status = detector.check(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
            ]
        )
    )

    assert status.trend == "declining"

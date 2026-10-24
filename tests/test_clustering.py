"""Tests for src/clustering.py — feature extraction, DBSCAN wrapper, and
the k-NN parameter diagnostic. All truth-boundary claims are tested
structurally (never asserting on particle_id/truth from within this
module, since it doesn't accept them as arguments at all).
"""
import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    extract_features,
    extract_features_scaled,
    cluster_tracks,
    knn_distance_diagnostic,
)


def _transformed_df(rows):
    """rows: list of (hit_id, x2, y2, z2)."""
    return pd.DataFrame(rows, columns=["hit_id", "x2", "y2", "z2"])


class TestExtractFeatures:
    def test_xy_representation_returns_two_columns(self):
        df = _transformed_df([(1, 0.1, 0.2, 5.0), (2, 0.3, 0.4, 6.0)])
        X = extract_features(df, representation="xy")
        assert X.shape == (2, 2)
        np.testing.assert_allclose(X, [[0.1, 0.2], [0.3, 0.4]])

    def test_xyz_representation_returns_three_columns(self):
        df = _transformed_df([(1, 0.1, 0.2, 5.0)])
        X = extract_features(df, representation="xyz")
        assert X.shape == (1, 3)
        np.testing.assert_allclose(X, [[0.1, 0.2, 5.0]])

    def test_rejects_unknown_representation(self):
        df = _transformed_df([(1, 0.1, 0.2, 5.0)])
        with pytest.raises(ValueError):
            extract_features(df, representation="not_a_real_option")

    def test_never_includes_particle_id_even_if_present_in_df(self):
        # Safety-boundary test: even if the caller's DataFrame happens to
        # have a particle_id column attached (e.g. accidentally merged in
        # upstream), extract_features must not pick it up, since it only
        # selects a fixed, explicit set of columns.
        df = _transformed_df([(1, 0.1, 0.2, 5.0)])
        df["particle_id"] = [999999]
        X = extract_features(df, representation="xyz")
        assert X.shape == (1, 3)  # NOT 4 -- particle_id excluded


class TestExtractFeaturesScaled:
    def test_output_has_zero_mean_unit_std_per_column(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "hit_id": np.arange(n),
            "x2": rng.normal(0, 1, n),
            "y2": rng.normal(0, 1, n),
            "z2": rng.normal(50, 20, n),  # different scale, like real z2
        })
        X_scaled, info = extract_features_scaled(df, representation="xyz")
        assert X_scaled.mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-9)
        assert X_scaled.std(axis=0) == pytest.approx([1, 1, 1], abs=1e-9)
        assert info["representation"] == "xyz"
        assert len(info["means"]) == 3
        assert len(info["stds"]) == 3

    def test_handles_zero_variance_column_without_crashing(self):
        df = _transformed_df([(1, 0.5, 0.5, 0.5), (2, 0.5, 0.5, 0.5)])  # all identical
        X_scaled, info = extract_features_scaled(df, representation="xyz")
        assert not np.isnan(X_scaled).any()
        assert not np.isinf(X_scaled).any()


class TestClusterTracks:
    def test_returns_expected_keys(self):
        X = np.array([[0, 0], [0.1, 0.1], [10, 10]])
        result = cluster_tracks(X, eps=0.5, min_samples=2)
        expected_keys = {"labels", "n_clusters", "n_noise", "n_hits",
                          "runtime_seconds", "eps", "min_samples"}
        assert expected_keys.issubset(result.keys())

    def test_two_tight_clusters_are_found(self):
        # Two obviously separated tight clusters -- a sanity check that
        # DBSCAN wiring itself works correctly, independent of any real
        # TrackML data.
        cluster_a = np.array([[0, 0], [0.01, 0.01], [0.02, 0], [0, 0.02]])
        cluster_b = np.array([[10, 10], [10.01, 10.01], [10.02, 10], [10, 10.02]])
        X = np.vstack([cluster_a, cluster_b])
        result = cluster_tracks(X, eps=0.1, min_samples=2)
        assert result["n_clusters"] == 2
        assert result["n_noise"] == 0

    def test_noise_label_is_minus_one(self):
        # An isolated point far from everything else should be noise (-1)
        cluster_a = np.array([[0, 0], [0.01, 0.01], [0.02, 0], [0, 0.02]])
        isolated = np.array([[1000, 1000]])
        X = np.vstack([cluster_a, isolated])
        result = cluster_tracks(X, eps=0.1, min_samples=2)
        assert result["labels"][-1] == -1
        assert result["n_noise"] == 1

    def test_runtime_is_recorded_and_nonnegative(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        result = cluster_tracks(X, eps=0.5, min_samples=3)
        assert result["runtime_seconds"] >= 0

    def test_deterministic_output_for_same_input(self):
        # DBSCAN itself is deterministic given fixed input/params (no
        # internal randomness) -- confirm our wrapper doesn't introduce any.
        X = np.random.default_rng(1).normal(size=(30, 2))
        result_a = cluster_tracks(X, eps=0.5, min_samples=3)
        result_b = cluster_tracks(X, eps=0.5, min_samples=3)
        np.testing.assert_array_equal(result_a["labels"], result_b["labels"])


class TestKnnDistanceDiagnostic:
    def test_output_length_matches_input(self):
        X = np.random.default_rng(0).normal(size=(40, 2))
        distances = knn_distance_diagnostic(X, k=3)
        assert len(distances) == 40

    def test_output_is_sorted_ascending(self):
        X = np.random.default_rng(0).normal(size=(40, 2))
        distances = knn_distance_diagnostic(X, k=3)
        assert (np.diff(distances) >= 0).all()

    def test_known_case_two_points(self):
        # Two points at known distance apart, k=1: each point's nearest
        # (only) neighbor is the other point, at the known distance.
        X = np.array([[0.0, 0.0], [3.0, 4.0]])  # distance = 5 (3-4-5 triangle)
        distances = knn_distance_diagnostic(X, k=1)
        np.testing.assert_allclose(distances, [5.0, 5.0])

    def test_does_not_count_self_as_neighbor(self):
        # A point exactly at the origin plus several points far away --
        # if self-counting were a bug, the "nearest neighbor" distance
        # would be 0 for every point (distance to itself).
        X = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]])
        distances = knn_distance_diagnostic(X, k=1)
        assert (distances > 0).all()

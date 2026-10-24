import numpy as np
import pandas as pd
import pytest

from src.m9_5_geometry_miss_audit import (
    add_geometry_features, cohort_counts, cohort_masks, deterministic_prefix_sample,
    summarize_geometry, wrapped_delta_phi,
)


def test_wrapped_azimuth_crossing_branch_cut_is_small_and_signed():
    source = np.deg2rad([179.0, -179.0])
    target = np.deg2rad([-179.0, 179.0])
    result = wrapped_delta_phi(source, target)
    assert result[0] == pytest.approx(np.deg2rad(2.0))
    assert result[1] == pytest.approx(np.deg2rad(-2.0))
    assert np.all(np.abs(result) <= np.pi)


def _mini_frames():
    hits = pd.DataFrame({
        "hit_id": [1, 2, 3, 4],
        "x": [np.cos(np.deg2rad(179)), 2*np.cos(np.deg2rad(-179)), 3, 4],
        "y": [np.sin(np.deg2rad(179)), 2*np.sin(np.deg2rad(-179)), 0, 0],
        "z": [10., 15., -2., 7.],
        "particle_id": [99, 88, 77, 66],  # ignored by coordinate features
    })
    traced = pd.DataFrame({
        "source_hit_id": [1, 1, 2, 3], "target_hit_id": [2, 3, 4, 4],
        "baseline_recovered": [True, False, False, False],
        "baseline_missed": [False, True, True, True],
        "per_bin_rank": [1., 2., np.nan, 14.],
        "target_bin_searched": [True, True, False, True],
        "query20_present": [True, True, False, True],
        "radial_step_pass": [True, False, True, True],
    })
    return hits, traced


def test_geometry_feature_calculation_uses_only_hit_coordinates_and_preserves_input():
    hits, traced = _mini_frames()
    hits_before, traced_before = hits.copy(deep=True), traced.copy(deep=True)
    result = add_geometry_features(hits, traced)
    assert result.loc[0, "delta_phi_rad"] == pytest.approx(np.deg2rad(2.0))
    assert result.loc[0, "delta_z_mm"] == pytest.approx(5.0)
    assert result.loc[0, "distance_transverse_mm"] > 0
    assert result.loc[0, "distance_xyz_mm"] > result.loc[0, "distance_transverse_mm"]
    pd.testing.assert_frame_equal(hits, hits_before)
    pd.testing.assert_frame_equal(traced, traced_before)
    assert "particle_id" not in result.columns


def test_cohort_membership_and_radial_overlap_are_explicit():
    hits, traced = _mini_frames()
    featured = add_geometry_features(hits, traced)
    counts = cohort_counts(featured)
    assert counts == {
        "recovered_m91": 1, "missed_rank_1_10": 1, "missed_rank_11_20": 1,
        "missed_rank_above_20": 0, "missed_target_bin_unsearched": 1,
        "missed_radial_filter": 1,
    }
    masks = cohort_masks(featured)
    assert masks["missed_rank_1_10"].iloc[1]
    assert masks["missed_radial_filter"].iloc[1]  # overlap by definition
    assert not masks["missed_rank_above_20"].any()


def test_statistics_use_the_full_cohort_and_include_quantiles():
    hits, traced = _mini_frames()
    featured = add_geometry_features(hits, traced)
    summary = summarize_geometry(featured)
    row = summary[(summary.cohort == "recovered_m91") & (summary.feature == "delta_z_mm")].iloc[0]
    assert row["count"] == 1
    assert row["median"] == pytest.approx(5.0)
    assert row["iqr"] == pytest.approx(0.0)
    assert row["p90"] == pytest.approx(5.0)
    assert row["p95"] == pytest.approx(5.0)


def test_per_bin_rank_is_summarized_only_when_rank_is_measurable():
    hits, traced = _mini_frames()
    featured = add_geometry_features(hits, traced)
    summary = summarize_geometry(featured)
    rank_row = summary[(summary.cohort == "missed_rank_11_20") & (summary.feature == "per_bin_rank")].iloc[0]
    assert rank_row["count"] == 1
    assert rank_row["median"] == pytest.approx(14.0)
    above20 = summary[(summary.cohort == "missed_rank_above_20") & (summary.feature == "per_bin_rank")].iloc[0]
    assert above20["count"] == 0
    assert pd.isna(above20["median"])


def test_deterministic_prefix_sampling_is_stable_and_validated():
    values = np.arange(20)
    np.testing.assert_array_equal(deterministic_prefix_sample(values, 5), np.arange(5))
    np.testing.assert_array_equal(deterministic_prefix_sample(values, 50), values)
    with pytest.raises(ValueError):
        deterministic_prefix_sample(values, -1)

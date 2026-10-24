import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from src.geometry_graph import build_candidate_graph
from src.m9_4_rank_audit import (
    build_query_stages, rank_merged_candidates, stable_query_top_k,
    stage_counts, trace_true_edges,
)
from scripts.run_m9_4_rank_audit import development_event_ids


def test_stable_extended_query_resolves_distance_ties_by_target_id():
    target_xyz = np.array([[2., 0., 1.], [2., 0., -1.], [2., 0., 1.]])
    target_indices = np.array([0, 1, 2])
    target_ids = np.array([50, 10, 20])
    tree = cKDTree(target_xyz)
    local, distances = stable_query_top_k(tree, target_indices, target_ids, target_xyz,
                                           np.array([[1., 0., 0.]]), 3)
    assert list(target_ids[target_indices[local[0]]]) == [10, 20, 50]
    assert distances[0, 0] == pytest.approx(1.41421356237)


def test_extended_query_records_per_bin_ranks_and_radial_filter_after_query():
    hits = pd.DataFrame({"hit_id": [1, 2, 3], "x": [1., 20., 190.], "y": [0., 0., 0.], "z": [0., 0., 0.]})
    stage = build_query_stages(hits)
    pair1 = stage[(stage.source_hit_id == 1) & (stage.target_hit_id == 2)].iloc[0]
    pair2 = stage[(stage.source_hit_id == 1) & (stage.target_hit_id == 3)].iloc[0]
    assert pair1.per_bin_rank == 1
    assert pair1.radial_step_pass
    assert pair2.per_bin_rank == 1
    assert not pair2.radial_step_pass
    assert pd.isna(pair2.merged_rank)


def test_final_merged_selection_keeps_only_ten_and_orders_by_distance_then_id():
    candidates = pd.DataFrame({"source_hit_id": [1]*12, "target_hit_id": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 13],
                               "distance": [float(i) for i in range(12)], "per_bin_rank": [1]*12,
                               "delta_r": [1.]*12, "radial_step_pass": [True]*12})
    ranked = rank_merged_candidates(candidates)
    assert ranked.merged_rank.max() == 12
    assert ranked.selected_final10.sum() == 10
    assert set(ranked.loc[ranked.selected_final10, "target_hit_id"]) == set(range(3, 13))


def test_trace_reports_rank_11_20_radial_and_final_cap_outcomes():
    hits = pd.DataFrame({"hit_id": [1, 2], "x": [1., 20.], "y": [0., 0.], "z": [0., 0.]})
    true_edges = pd.DataFrame({"source_hit_id": [1], "target_hit_id": [2]})
    stages = pd.DataFrame({"source_hit_id": [1], "target_hit_id": [2], "source_bucket": [0], "target_bucket": [1],
                           "per_bin_rank": [11], "distance": [19.], "delta_r": [19.], "radial_step_pass": [True],
                           "merged_rank": [1], "selected_final10": [True]})
    traced = trace_true_edges(hits, true_edges, stages, pd.DataFrame(columns=["source_hit_id", "target_hit_id"]))
    counts = stage_counts(traced)
    assert traced.iloc[0].rank_11_20
    assert traced.iloc[0].rank_11_20_radial_pass
    assert traced.iloc[0].survives_final10
    assert counts["baseline_missed_rank_11_20_final10_survivors"] == 1
    assert counts["baseline_miss_would_survive_extended_final10"] == 1


def test_trace_accounts_for_unsearched_and_radial_filter_failure_separately():
    hits = pd.DataFrame({"hit_id": [1, 2, 3], "x": [1., 10., 190.], "y": [0., 0., 0.], "z": [0., 0., 0.]})
    true_edges = pd.DataFrame({"source_hit_id": [1, 1], "target_hit_id": [2, 3]})
    stages = pd.DataFrame({"source_hit_id": [1], "target_hit_id": [3], "source_bucket": [0], "target_bucket": [11],
                           "per_bin_rank": [1], "distance": [199.], "delta_r": [199.], "radial_step_pass": [False],
                           "merged_rank": [np.nan], "selected_final10": [False]})
    traced = trace_true_edges(hits, true_edges, stages, pd.DataFrame(columns=["source_hit_id", "target_hit_id"]))
    assert traced.loc[0, "per_bin_rank_category"] == "not_searched"
    assert traced.loc[1, "per_bin_rank_category"] == "rank_1_10"
    assert not traced.loc[1, "radial_step_pass"]
    assert traced.loc[1, "baseline_miss_stage"] == "radial_filter_failed"


def test_reconstructed_baseline_behavior_matches_m91_on_small_geometry():
    hits = pd.DataFrame({"hit_id": [9, 3, 8, 4], "x": [1., 20., 22., 24.], "y": [0.]*4, "z": [0., 1., -1., 2.]})
    m91 = build_candidate_graph(hits)
    audit = build_query_stages(hits)
    reconstructed = audit.loc[audit.radial_step_pass & audit.selected_final10,
                              ["source_hit_id", "target_hit_id", "merged_rank"]].rename(columns={"merged_rank": "candidate_rank"})
    actual = m91[["source_hit_id", "target_hit_id", "candidate_rank"]]
    pd.testing.assert_frame_equal(actual.sort_values(["source_hit_id", "candidate_rank"]).reset_index(drop=True),
                                  reconstructed.sort_values(["source_hit_id", "candidate_rank"]).reset_index(drop=True),
                                  check_dtype=False)


def test_development_manifest_excludes_validation_and_final_test():
    ids = development_event_ids("experiments/results/event_split_manifest.csv")
    assert ids == [f"event{n:09d}" for n in range(1000, 1060)]
    assert all(int(event.removeprefix("event")) < 1060 for event in ids)


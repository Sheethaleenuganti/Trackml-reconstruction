import numpy as np
import pandas as pd
import pytest

from src.m9_2_diagnostics import diagnose_event, true_consecutive_edges
from scripts.run_m9_2_diagnostics import development_event_ids


def event_fixture():
    phi_s, phi_t = np.pi - 0.01, -np.pi + 0.01
    hits = pd.DataFrame({
        "hit_id": [10, 20, 30, 40],
        "x": [17*np.cos(phi_s), 36*np.cos(phi_t), 12., 25.],
        "y": [17*np.sin(phi_s), 36*np.sin(phi_t), 0., 0.],
        "z": [1., 4., 10., 10.],
    })
    truth = pd.DataFrame({"hit_id": [10, 20, 30, 40], "particle_id": [100, 100, 0, 200]})
    return hits, truth


def empty_candidates():
    return pd.DataFrame(columns=["source_hit_id", "target_hit_id", "candidate_rank"])


def test_delta_r_calculation():
    hits, truth = event_fixture()
    row = diagnose_event(hits, truth, empty_candidates()).iloc[0]
    assert row.delta_r == pytest.approx(19.)


def test_wrapped_delta_phi_calculation():
    hits, truth = event_fixture()
    row = diagnose_event(hits, truth, empty_candidates()).iloc[0]
    assert row.abs_delta_phi == pytest.approx(.02)


def test_delta_z_calculation():
    hits, truth = event_fixture()
    row = diagnose_event(hits, truth, empty_candidates()).iloc[0]
    assert row.abs_delta_z == pytest.approx(3.)


def test_xyz_distance_calculation():
    hits, truth = event_fixture()
    row = diagnose_event(hits, truth, empty_candidates()).iloc[0]
    p1 = hits.loc[hits.hit_id == 10, ["x", "y", "z"]].to_numpy()[0]
    p2 = hits.loc[hits.hit_id == 20, ["x", "y", "z"]].to_numpy()[0]
    assert row.distance_xyz == pytest.approx(np.linalg.norm(p2-p1))


def test_truth_edges_order_outward_skip_noise_and_equal_radius():
    hits, _ = event_fixture()
    hits = pd.DataFrame({"hit_id": [1, 2, 3, 4], "x": [20., 10., -10., 5.], "y": [0., 0., 0., 0.], "z": [0.]*4})
    truth = pd.DataFrame({"hit_id": [1, 2, 3, 4], "particle_id": [7, 7, 7, 0]})
    edges = true_consecutive_edges(hits, truth)
    assert list(zip(edges.source_hit_id, edges.target_hit_id)) == [(3, 1)]


def test_recovered_and_missed_classification():
    hits, truth = event_fixture()
    candidate = pd.DataFrame({"source_hit_id": [10], "target_hit_id": [20], "candidate_rank": [1]})
    result = diagnose_event(hits, truth, candidate)
    assert result.iloc[0].recovered
    assert result.iloc[0].primary_cause == "recovered"


def test_bucket_search_classification():
    hits, truth = event_fixture()
    result = diagnose_event(hits, truth, empty_candidates())
    assert result.iloc[0].within_bucket_search
    assert result.iloc[0].bucket_difference == 2


def test_candidate_rank_and_top_k_classification():
    hits, truth = event_fixture()
    result = diagnose_event(hits, truth, empty_candidates())
    assert result.iloc[0].rank_category == ">10"
    assert result.iloc[0].primary_cause == "top_k_truncation"
    candidate = pd.DataFrame({"source_hit_id": [10], "target_hit_id": [20], "candidate_rank": [3]})
    assert diagnose_event(hits, truth, candidate).iloc[0].rank_category == "rank_3"


def test_repeated_diagnostic_is_deterministic():
    hits, truth = event_fixture()
    a = diagnose_event(hits, truth, empty_candidates())
    b = diagnose_event(hits, truth, empty_candidates())
    pd.testing.assert_frame_equal(a, b)


def test_only_expected_development_ids_are_selected():
    ids = development_event_ids("experiments/results/event_split_manifest.csv")
    assert ids == [f"event{n:09d}" for n in range(1000, 1060)]
    assert all(int(e.removeprefix("event")) < 1060 for e in ids)

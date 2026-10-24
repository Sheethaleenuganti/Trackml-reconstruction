import numpy as np
import pandas as pd

from src.geometry_graph import build_candidate_graph
import src.m9_3_per_bin_query_ablation as m93


def query_fixture(n_targets=21):
    target_r = np.linspace(20.0, 35.0, n_targets)
    return pd.DataFrame({
        "hit_id": [999] + list(range(100, 100+n_targets)),
        "x": np.r_[1.0, target_r], "y": np.zeros(n_targets+1),
        "z": np.zeros(n_targets+1),
        # Deliberately present labels must not affect candidate construction.
        "particle_id": [123] * (n_targets+1),
    })


def test_per_target_bin_query_cap_is_twenty(monkeypatch):
    calls = []
    real_tree = m93.cKDTree

    class RecordingTree:
        def __init__(self, points):
            self.tree = real_tree(points)

        def query(self, points, k):
            calls.append(k)
            return self.tree.query(points, k=k)

    monkeypatch.setattr(m93, "cKDTree", RecordingTree)
    m93.build_candidate_graph_m9_3(query_fixture())
    assert 20 in calls


def test_final_source_cap_remains_ten_and_is_never_exceeded():
    graph = m93.build_candidate_graph_m9_3(query_fixture(35))
    assert graph.groupby("source_hit_id").size().max() <= 10
    assert m93.FINAL_CANDIDATE_CAP == 10


def test_deterministic_ordering_uses_target_hit_id_for_equal_distance():
    hits = pd.DataFrame({"hit_id": [1, 9, 3], "x": [1., 20., 20.], "y": [0., 0., 0.], "z": [0., 1., -1.]})
    graph = m93.build_candidate_graph_m9_3(hits)
    outgoing = graph.loc[graph.source_hit_id == 1].sort_values("candidate_rank")
    assert list(outgoing.target_hit_id) == [3, 9]
    assert list(outgoing.candidate_rank) == [1, 2]
    repeated = m93.build_candidate_graph_m9_3(hits)
    pd.testing.assert_frame_equal(graph, repeated)


def test_same_radial_bucket_is_excluded():
    hits = pd.DataFrame({"hit_id": [1, 2], "x": [2., 10.], "y": [0., 0.], "z": [0., 0.]})
    assert m93.build_candidate_graph_m9_3(hits).empty


def test_edges_point_from_smaller_to_larger_radius():
    hits = pd.DataFrame({"hit_id": [4, 1], "x": [30., 1.], "y": [0., 0.], "z": [0., 0.]})
    graph = m93.build_candidate_graph_m9_3(hits)
    radii = hits.set_index("hit_id").eval("x**2 + y**2")
    assert len(graph) == 1
    assert radii[graph.iloc[0].source_hit_id] < radii[graph.iloc[0].target_hit_id]


def test_m91_behavior_matches_when_each_bin_has_at_most_ten_targets():
    hits = pd.DataFrame({"hit_id": [8, 3, 2, 5], "x": [1., 20., 22., 24.], "y": [0., 0., 0., 0.], "z": [0., 1., -1., 2.]})
    baseline = build_candidate_graph(hits)
    variant = m93.build_candidate_graph_m9_3(hits)
    pd.testing.assert_frame_equal(baseline, variant)


def test_candidate_builder_ignores_truth_columns():
    hits = query_fixture(8)
    with_labels = m93.build_candidate_graph_m9_3(hits)
    without_labels = m93.build_candidate_graph_m9_3(hits.drop(columns="particle_id"))
    pd.testing.assert_frame_equal(with_labels, without_labels)

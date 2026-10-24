import numpy as np
import pandas as pd
import pytest

from src.geometry_graph import EDGE_COLUMNS, build_candidate_graph


def sample_hits():
    return pd.DataFrame({"hit_id": [40, 10, 30, 20], "x": [3., 1., 0., 2.],
                         "y": [0., 0., 4., 0.], "z": [0., 0., 1., 0.]})


def test_deterministic_and_hit_ids_preserved():
    hits = sample_hits()
    a = build_candidate_graph(hits, max_radial_step_mm=10, max_candidates_per_hit=2)
    b = build_candidate_graph(hits, max_radial_step_mm=10, max_candidates_per_hit=2)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.source_hit_id).issubset(set(hits.hit_id))
    assert set(a.target_hit_id).issubset(set(hits.hit_id))


def test_no_self_or_duplicate_edges_and_cap_respected():
    graph = build_candidate_graph(sample_hits(), max_radial_step_mm=10, max_candidates_per_hit=1)
    assert not (graph.source_hit_id == graph.target_hit_id).any()
    assert not graph.duplicated(["source_hit_id", "target_hit_id"]).any()
    assert graph.groupby("source_hit_id").size().max() <= 1


def test_direction_is_increasing_radius():
    hits = sample_hits().set_index("hit_id")
    graph = build_candidate_graph(sample_hits(), max_radial_step_mm=10, max_candidates_per_hit=3)
    for edge in graph.itertuples(index=False):
        s, t = hits.loc[edge.source_hit_id], hits.loc[edge.target_hit_id]
        assert np.hypot(t.x, t.y) > np.hypot(s.x, s.y)
        assert edge.delta_r > 0


def test_geometry_calculations_and_wrapped_phi():
    phi_source, phi_target = np.pi - .01, -np.pi + .01
    hits = pd.DataFrame({"hit_id": [1, 2], "x": [10*np.cos(phi_source), 20*np.cos(phi_target)],
                         "y": [10*np.sin(phi_source), 20*np.sin(phi_target)], "z": [1., 4.]})
    edge = build_candidate_graph(hits, max_radial_step_mm=20, max_candidates_per_hit=1).iloc[0]
    assert edge.source_hit_id == 1 and edge.target_hit_id == 2
    assert edge.delta_r == pytest.approx(10.)
    assert edge.delta_phi == pytest.approx(.02)
    assert edge.delta_z == pytest.approx(3.)
    assert edge.distance == pytest.approx(np.linalg.norm(hits.loc[0:1, ["x", "y", "z"]].iloc[1].to_numpy(dtype=float)-hits.loc[0:1, ["x", "y", "z"]].iloc[0].to_numpy(dtype=float)))


def test_truth_not_required_or_emitted():
    hits = sample_hits().assign(particle_id=[100, 100, 200, 200])
    graph = build_candidate_graph(hits, max_radial_step_mm=10)
    assert list(graph.columns) == list(EDGE_COLUMNS)
    assert "particle_id" not in graph.columns


@pytest.mark.parametrize("hits", [pd.DataFrame(columns=["hit_id", "x", "y", "z"]), sample_hits().iloc[:1]])
def test_empty_and_single_hit_events(hits):
    graph = build_candidate_graph(hits)
    assert graph.empty
    assert list(graph.columns) == list(EDGE_COLUMNS)


def test_candidate_mapping_is_stable_under_input_row_permutation():
    hits = sample_hits()
    a = build_candidate_graph(hits, max_radial_step_mm=10, max_candidates_per_hit=2)
    b = build_candidate_graph(hits.sample(frac=1, random_state=13), max_radial_step_mm=10, max_candidates_per_hit=2)
    pd.testing.assert_frame_equal(a, b)


def test_invalid_geometry_and_parameters_raise():
    with pytest.raises(ValueError, match="finite and positive"):
        build_candidate_graph(sample_hits(), max_radial_step_mm=0)
    with pytest.raises(ValueError, match="positive integer"):
        build_candidate_graph(sample_hits(), max_candidates_per_hit=0)
    with pytest.raises(ValueError, match="finite"):
        build_candidate_graph(sample_hits().assign(x=np.nan))

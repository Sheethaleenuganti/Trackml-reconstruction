"""Focused tests for the frozen M5 multi-event runner wiring."""
from types import SimpleNamespace

import pandas as pd

from scripts import run_m8_development_baseline as runner


def test_run_event_uses_frozen_m5_path_and_loads_truth_after_clustering(monkeypatch):
    hits = pd.DataFrame({"hit_id": [10, 11]})
    truth = pd.DataFrame({"hit_id": [10, 11]})
    calls = []

    def fake_load_event(data_dir, event_id, parts):
        calls.append(("load", event_id, parts))
        return {parts[0]: hits if parts == ("hits",) else truth}

    def fake_transform(frame):
        calls.append(("transform", frame))
        return "transformed-hits"

    def fake_extract(frame, representation):
        calls.append(("features", frame, representation))
        return "xyz-features"

    def fake_cluster(features, eps, min_samples):
        calls.append(("cluster", features, eps, min_samples))
        return {"labels": [0, -1], "runtime_seconds": 1.25}

    def fake_evaluate(truth_frame, hit_ids, labels):
        calls.append(("evaluate", truth_frame, list(hit_ids), labels))
        return {
            "official_trackml_score": 0.125,
            "n_predicted_clusters": 1,
            "n_noise_hits": 1,
            "mean_purity": 0.9,
            "mean_efficiency": 0.4,
        }

    monkeypatch.setattr(runner, "load_event", fake_load_event)
    monkeypatch.setattr(runner, "transform_hits", fake_transform)
    monkeypatch.setattr(runner, "extract_features", fake_extract)
    monkeypatch.setattr(runner, "cluster_tracks", fake_cluster)
    monkeypatch.setattr(runner, "summarize_evaluation", fake_evaluate)

    result = runner.run_event("unused-data-dir", "event000001000")

    assert calls[0] == ("load", 1000, ("hits",))
    assert calls[1][0] == "transform"
    assert calls[2] == ("features", "transformed-hits", "xyz")
    assert calls[3] == ("cluster", "xyz-features", 0.01, 3)
    assert calls[4] == ("load", 1000, ("truth",))
    assert calls[5][0] == "evaluate"
    assert result == {
        "event_id": "event000001000",
        "official_score": 0.125,
        "n_predicted_clusters": 1,
        "n_noise_hits": 1,
        "mean_purity": 0.9,
        "mean_efficiency": 0.4,
        "runtime_seconds": 1.25,
    }

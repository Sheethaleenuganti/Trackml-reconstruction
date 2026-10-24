"""Validation of the saved M8.2 event-1000 provenance artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m8_2_event1000_provenance import (
    FEATURES_PATH,
    LABELS_PATH,
    MAPPING_HASH_PATH,
    PROVENANCE_PATH,
    SUBMISSION_PATH,
    canonical_mapping_sha256,
    canonicalize_mapping,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cluster_label_artifact_has_one_row_per_unique_input_hit():
    labels = pd.read_csv(LABELS_PATH)
    assert list(labels.columns) == ["hit_id", "cluster_label"]
    assert len(labels) == 120_939
    assert labels["hit_id"].nunique() == len(labels)
    assert (labels["cluster_label"] == -1).sum() == 61_996


def test_submission_has_one_row_per_hit_and_agrees_with_labels():
    labels = pd.read_csv(LABELS_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)
    assert list(submission.columns) == ["hit_id", "track_id"]
    assert len(submission) == 120_939
    assert submission["hit_id"].nunique() == len(submission)
    assert labels["hit_id"].equals(submission["hit_id"])
    assert (submission["track_id"] >= 0).all()


def test_canonical_mapping_is_deterministic_under_cluster_id_permutation():
    hit_ids = np.array([91, 12, 44, 7, 63])
    labels_a = np.array([8, -1, 8, 3, 3])
    labels_b = np.array([40, -1, 40, 11, 11])
    assert canonicalize_mapping(hit_ids, labels_a).equals(
        canonicalize_mapping(hit_ids, labels_b)
    )
    assert canonical_mapping_sha256(hit_ids, labels_a) == canonical_mapping_sha256(
        hit_ids, labels_b
    )


def test_provenance_has_required_fields_and_numeric_score():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["event_id"] == "event000001000"
    assert provenance["configuration"]["dbscan"] == {
        "eps": 0.01,
        "metric": "euclidean",
        "min_samples": 3,
    }
    assert provenance["features"]["names"] == ["x2", "y2", "z2"]
    assert provenance["features"]["shape"] == [120_939, 3]
    assert isinstance(provenance["official_score"]["value"], (int, float))
    assert provenance["environment"]["python_version"]
    for key in (
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "scikit_learn_version",
        "trackml",
        "platform",
        "repository_commit",
    ):
        assert key in provenance["environment"]


def test_saved_artifact_hashes_reproduce():
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    labels = pd.read_csv(LABELS_PATH)
    features = np.load(FEATURES_PATH, allow_pickle=False)
    expected = provenance["fingerprints"]
    actual_mapping_hash = canonical_mapping_sha256(
        labels["hit_id"].to_numpy(), labels["cluster_label"].to_numpy()
    )
    assert actual_mapping_hash == expected["canonical_cluster_mapping_sha256"]
    assert MAPPING_HASH_PATH.read_text(encoding="ascii").strip() == actual_mapping_hash
    assert _sha256(LABELS_PATH) == expected["cluster_labels_file_sha256"]
    assert _sha256(SUBMISSION_PATH) == expected["submission_file_sha256"]
    assert _sha256(FEATURES_PATH) == expected["feature_file_sha256"]
    assert hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest() == expected[
        "feature_matrix_sha256"
    ]

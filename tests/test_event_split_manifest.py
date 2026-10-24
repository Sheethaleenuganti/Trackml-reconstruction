"""Focused tests for the fixed M7.2 whole-event split manifest."""
import pytest

from src.event_split_manifest import (
    expected_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def test_expected_manifest_has_exact_contiguous_splits():
    rows = expected_manifest()

    assert len(rows) == 100
    assert rows[0] == {"event_id": "event000001000", "split": "DEVELOPMENT"}
    assert rows[59] == {"event_id": "event000001059", "split": "DEVELOPMENT"}
    assert rows[60] == {"event_id": "event000001060", "split": "VALIDATION"}
    assert rows[79] == {"event_id": "event000001079", "split": "VALIDATION"}
    assert rows[80] == {"event_id": "event000001080", "split": "FINAL_TEST"}
    assert rows[-1] == {"event_id": "event000001099", "split": "FINAL_TEST"}
    assert validate_manifest(rows) == {
        "DEVELOPMENT": 60,
        "VALIDATION": 20,
        "FINAL_TEST": 20,
    }


def test_manifest_round_trip(tmp_path):
    path = write_manifest(tmp_path / "event_split_manifest.csv")

    assert read_manifest(path) == expected_manifest()


def test_validation_rejects_duplicate_event_ids():
    rows = expected_manifest()
    rows[-1] = dict(rows[0])

    with pytest.raises(ValueError, match="duplicate event IDs"):
        validate_manifest(rows)


def test_validation_rejects_wrong_event_set():
    rows = expected_manifest()
    rows[-1] = {"event_id": "event000001100", "split": "FINAL_TEST"}

    with pytest.raises(ValueError, match="missing expected event IDs.*unexpected event IDs"):
        validate_manifest(rows)


def test_validation_rejects_split_membership_change():
    rows = expected_manifest()
    rows[0] = {"event_id": "event000001000", "split": "VALIDATION"}

    with pytest.raises(ValueError, match="event000001000 belongs in DEVELOPMENT"):
        validate_manifest(rows)

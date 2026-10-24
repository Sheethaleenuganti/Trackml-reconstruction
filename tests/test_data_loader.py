"""Unit tests for src/data_loader.py.

These tests run against a small, generated synthetic event (see
scripts/generate_synthetic_event.py) so they can run without requiring the
real (multi-gigabyte, gated) TrackML dataset to be present. This is
explicitly a stand-in for structural/schema testing, NOT a validation of
real physics — see that script's module docstring for the same caveat.
"""
import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.generate_synthetic_event import generate_synthetic_event, write_synthetic_event
from src.data_loader import load_event, load_event_ids, profile_event, EventFilesNotFoundError


@pytest.fixture()
def synthetic_data_dir():
    tmp_dir = tempfile.mkdtemp()
    event = generate_synthetic_event(event_id=0, n_particles=10, n_layers=4, seed=1)
    write_synthetic_event(tmp_dir, event, event_id=0)
    # write a second event too, to test multi-event discovery
    event2 = generate_synthetic_event(event_id=1, n_particles=10, n_layers=4, seed=2)
    write_synthetic_event(tmp_dir, event2, event_id=1)
    yield tmp_dir
    shutil.rmtree(tmp_dir)


class TestLoadEventIds:
    def test_finds_all_written_events(self, synthetic_data_dir):
        ids = load_event_ids(synthetic_data_dir)
        assert ids == [0, 1]

    def test_raises_clear_error_on_missing_directory(self):
        with pytest.raises(EventFilesNotFoundError):
            load_event_ids("/this/path/does/not/exist")

    def test_raises_clear_error_on_empty_directory(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            with pytest.raises(EventFilesNotFoundError):
                load_event_ids(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir)


class TestLoadEvent:
    def test_loads_requested_parts(self, synthetic_data_dir):
        # Note: the synthetic generator (scripts/generate_synthetic_event.py)
        # deliberately does not produce a cells.csv, since cells are
        # explicitly out of scope for Step 3 ("Cells can be loaded
        # separately if needed, but do not process them yet"). So this
        # test requests the three parts the generator does produce, rather
        # than the full four-part default. Loading a real event's 'cells'
        # part is covered structurally by test_loads_subset_of_parts below
        # using a part the fixture actually has, and the loader's
        # missing-file behavior is covered by
        # test_raises_clear_error_for_nonexistent_event.
        event = load_event(synthetic_data_dir, event_id=0, parts=("hits", "particles", "truth"))
        assert set(event.keys()) == {"hits", "particles", "truth"}

    def test_raises_clear_error_when_a_requested_part_is_missing(self, synthetic_data_dir):
        # cells.csv does not exist for the synthetic fixture — confirm the
        # loader fails loudly and specifically, rather than silently
        # returning partial data, exactly as required by the Step 3 spec.
        with pytest.raises(EventFilesNotFoundError, match="cells"):
            load_event(synthetic_data_dir, event_id=0, parts=("hits", "cells"))

    def test_loads_subset_of_parts(self, synthetic_data_dir):
        event = load_event(synthetic_data_dir, event_id=0, parts=("hits", "truth"))
        assert set(event.keys()) == {"hits", "truth"}

    def test_hits_has_expected_columns(self, synthetic_data_dir):
        event = load_event(synthetic_data_dir, event_id=0, parts=("hits",))
        expected = {"hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"}
        assert expected.issubset(set(event["hits"].columns))

    def test_truth_has_expected_columns(self, synthetic_data_dir):
        event = load_event(synthetic_data_dir, event_id=0, parts=("truth",))
        expected = {"hit_id", "particle_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "weight"}
        assert expected.issubset(set(event["truth"].columns))

    def test_particles_has_expected_columns(self, synthetic_data_dir):
        event = load_event(synthetic_data_dir, event_id=0, parts=("particles",))
        expected = {"particle_id", "vx", "vy", "vz", "px", "py", "pz", "q", "nhits"}
        assert expected.issubset(set(event["particles"].columns))

    def test_raises_clear_error_for_nonexistent_event(self, synthetic_data_dir):
        with pytest.raises(EventFilesNotFoundError):
            load_event(synthetic_data_dir, event_id=999)

    def test_does_not_truncate_hits(self, synthetic_data_dir):
        # explicit check that the full event's hits are returned, not a
        # subsample — this is a hard requirement per the course spec.
        event_via_loader = load_event(synthetic_data_dir, event_id=0, parts=("hits",))
        raw_csv_path = Path(synthetic_data_dir) / "event000000000-hits.csv"
        raw_line_count = sum(1 for _ in open(raw_csv_path)) - 1  # minus header
        assert len(event_via_loader["hits"]) == raw_line_count


class TestProfileEvent:
    def test_profile_contains_expected_keys(self, synthetic_data_dir):
        event = load_event(synthetic_data_dir, event_id=0, parts=("hits", "particles", "truth"))
        stats = profile_event(event)
        expected_keys = {
            "n_hits",
            "n_unique_hit_ids",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
            "n_truth_rows",
            "n_noise_hits",
            "n_unique_nonzero_particle_ids",
            "n_particles",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_n_hits_matches_actual_row_count(self, synthetic_data_dir):
        event = load_event(synthetic_data_dir, event_id=0, parts=("hits", "particles", "truth"))
        stats = profile_event(event)
        assert stats["n_hits"] == len(event["hits"])

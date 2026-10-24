"""Unit tests for src/truth_checks.py."""
import shutil
import tempfile

import pytest

from scripts.generate_synthetic_event import generate_synthetic_event, write_synthetic_event
from src.data_loader import load_event
from src.truth_checks import (
    check_hit_id_uniqueness,
    check_every_hit_has_truth,
    check_noise_hits,
    hits_per_particle_histogram,
    compare_particles_nhits_with_truth,
    run_all_checks,
    assert_core_invariants,
)


@pytest.fixture()
def synthetic_event():
    tmp_dir = tempfile.mkdtemp()
    event = generate_synthetic_event(event_id=0, n_particles=15, n_layers=5, seed=7)
    write_synthetic_event(tmp_dir, event, event_id=0)
    loaded = load_event(tmp_dir, event_id=0, parts=("hits", "particles", "truth"))
    shutil.rmtree(tmp_dir)
    return loaded


class TestCheckHitIdUniqueness:
    def test_synthetic_event_has_unique_ids(self, synthetic_event):
        result = check_hit_id_uniqueness(synthetic_event["hits"], synthetic_event["truth"])
        assert result["hits_hit_id_unique"] is True
        assert result["truth_hit_id_unique"] is True

    def test_detects_duplicate_hit_ids(self, synthetic_event):
        hits = synthetic_event["hits"].copy()
        # Deliberately introduce a duplicate to confirm the check catches it
        hits = hits._append(hits.iloc[0]) if hasattr(hits, "_append") else hits
        import pandas as pd
        hits_with_dupe = pd.concat([hits, hits.iloc[[0]]], ignore_index=True)
        result = check_hit_id_uniqueness(hits_with_dupe, synthetic_event["truth"])
        assert result["hits_hit_id_unique"] is False


class TestCheckEveryHitHasTruth:
    def test_synthetic_event_is_consistent(self, synthetic_event):
        result = check_every_hit_has_truth(synthetic_event["hits"], synthetic_event["truth"])
        assert result["len_hits_equals_len_truth"] is True
        assert result["n_hit_ids_missing_from_truth"] == 0
        assert result["n_truth_hit_ids_missing_from_hits"] == 0


class TestCheckNoiseHits:
    def test_returns_sensible_counts(self, synthetic_event):
        result = check_noise_hits(synthetic_event["truth"])
        assert result["n_noise_hits_particle_id_zero"] >= 0
        assert 0.0 <= result["noise_fraction"] <= 1.0


class TestHitsPerParticleHistogram:
    def test_zero_hit_bucket_is_structurally_always_zero(self, synthetic_event):
        # documented in the source: truth only has rows for hits that
        # exist, so a particle_id appearing in truth necessarily has >=1
        # hit there by construction.
        result = hits_per_particle_histogram(synthetic_event["truth"])
        assert result["particles_with_0_hits"] == 0

    def test_bucket_counts_sum_correctly(self, synthetic_event):
        result = hits_per_particle_histogram(synthetic_event["truth"])
        summed = (
            result["particles_with_1_hit"]
            + result["particles_with_2_hits"]
            + result["particles_with_3_hits"]
            + result["particles_with_4_or_more_hits"]
        )
        assert summed == result["total_nonzero_particles_observed_in_truth"]


class TestCompareParticlesNhitsWithTruth:
    def test_synthetic_generator_produces_consistent_nhits(self, synthetic_event):
        # This is a meaningful test: it checks that our OWN synthetic
        # generator correctly fills in particles.nhits to match what's
        # actually in truth.csv — if this fails, the synthetic generator
        # itself has a bug, which would be important to know before
        # trusting any test built on top of it.
        result = compare_particles_nhits_with_truth(
            synthetic_event["particles"], synthetic_event["truth"]
        )
        assert result["n_mismatching"] == 0, (
            f"Synthetic generator produced inconsistent nhits: "
            f"{result['mismatch_sample']}"
        )

    def test_handles_duplicate_particle_id_rows_correctly(self):
        # REGRESSION TEST for a real bug found against real data (event
        # 1000, Kaggle train_sample): particle_id is NOT guaranteed unique
        # in particles.csv. A naive merge (pre-fix) compared each
        # individual row's nhits against the FULL truth-derived count,
        # producing false "mismatches" for every duplicated particle_id.
        # This test hand-constructs that exact scenario: one particle_id
        # appearing as 2 separate particles.csv rows (nhits=3 and nhits=2),
        # with 5 total matching truth rows (3+2), which is the correct,
        # well-defined comparison.
        import pandas as pd

        particles = pd.DataFrame({
            "particle_id": [777, 777],  # same ID, two distinct rows
            "vx": [0.0, 5.0],
            "vy": [0.0, 5.0],
            "vz": [0.0, 5.0],
            "px": [1.0, 2.0],
            "py": [1.0, 2.0],
            "pz": [1.0, 2.0],
            "q": [1, -1],
            "nhits": [3, 2],
        })
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4, 5],
            "particle_id": [777, 777, 777, 777, 777],
            "tx": [0.0] * 5, "ty": [0.0] * 5, "tz": [0.0] * 5,
            "tpx": [0.0] * 5, "tpy": [0.0] * 5, "tpz": [0.0] * 5,
            "weight": [0.2] * 5,
        })

        result = compare_particles_nhits_with_truth(particles, truth)
        # summed nhits (3+2=5) should match truth-derived count (5)
        assert result["n_mismatching"] == 0, (
            f"Duplicate particle_id handling regressed: {result['mismatch_sample']}"
        )
        assert result["n_particles_checked"] == 1  # one unique particle_id


class TestRunAllChecks:
    def test_returns_all_expected_sections(self, synthetic_event):
        report = run_all_checks(synthetic_event)
        expected_sections = {
            "hit_id_uniqueness",
            "every_hit_has_truth",
            "noise_hits",
            "hits_per_particle_histogram",
            "particles_nhits_vs_truth",
        }
        assert set(report.keys()) == expected_sections


class TestAssertCoreInvariants:
    def test_passes_on_valid_synthetic_event(self, synthetic_event):
        # should not raise
        assert_core_invariants(synthetic_event)

    def test_raises_on_broken_hit_truth_correspondence(self, synthetic_event):
        broken = dict(synthetic_event)
        broken["truth"] = synthetic_event["truth"].iloc[:-1]  # drop one row
        with pytest.raises(AssertionError):
            assert_core_invariants(broken)

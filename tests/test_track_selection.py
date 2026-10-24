"""Tests for src/track_selection.py — the M3 fix for the real-data finding
that particle_id does not uniquely identify a physical particle in
particles.csv.

Covers the four cases explicitly required for closing M3:
  1. unique particle_id -> accepted
  2. duplicated particle_id -> rejected
  3. unique particle_id with nhits below threshold -> rejected
  4. unique particle_id with sufficient nhits -> accepted

Uses small, hand-built DataFrames (not the full real event) so each case
is exact and independent of real-data specifics — but the module itself
has also been run against real event 1000 and manually verified (847
unambiguous IDs, 751 passing ORDER_MIN_HITS>=4), documented in
docs/PROJECT_CONTENT_MASTER.md.
"""
import pandas as pd
import pytest

from src.track_selection import (
    unambiguous_particle_ids,
    select_valid_tracks,
    get_track_hits,
    track_summary,
)


def _make_particles(rows):
    """Helper: build a minimal particles.csv-shaped DataFrame from a list
    of dicts, filling in any columns not specified with sensible defaults.
    """
    defaults = {
        "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "px": 1.0, "py": 0.0, "pz": 0.0,
        "q": 1, "nhits": 0,
    }
    full_rows = []
    for r in rows:
        row = dict(defaults)
        row.update(r)
        full_rows.append(row)
    return pd.DataFrame(full_rows)


def _make_hits_and_truth(particle_id, n_hits, radii):
    """Helper: build hits/truth for a single particle_id at increasing
    transverse radius (a physically sensible synthetic trajectory), so
    get_track_hits/track_summary have something real to compute over.
    """
    hit_ids = list(range(1, n_hits + 1))
    hits = pd.DataFrame({
        "hit_id": hit_ids,
        "x": radii,  # place along x-axis for simplicity: r = x, phi = 0
        "y": [0.0] * n_hits,
        "z": [float(i * 10) for i in range(n_hits)],
        "volume_id": [1] * n_hits,
        "layer_id": [1] * n_hits,
        "module_id": [1] * n_hits,
    })
    truth = pd.DataFrame({
        "hit_id": hit_ids,
        "particle_id": [particle_id] * n_hits,
        "tx": radii, "ty": [0.0] * n_hits, "tz": [float(i * 10) for i in range(n_hits)],
        "tpx": [1.0] * n_hits, "tpy": [0.0] * n_hits, "tpz": [0.0] * n_hits,
        "weight": [1.0 / n_hits] * n_hits,
    })
    return hits, truth


class TestUnambiguousParticleIds:
    def test_case_1_unique_particle_id_is_accepted(self):
        """Case 1: a particle_id with exactly one row IS returned as
        unambiguous."""
        particles = _make_particles([
            {"particle_id": 100, "nhits": 5},
            {"particle_id": 200, "nhits": 8},
        ])
        result = set(unambiguous_particle_ids(particles))
        assert 100 in result
        assert 200 in result

    def test_case_2_duplicated_particle_id_is_rejected(self):
        """Case 2: a particle_id with multiple rows (the real M3 finding
        — e.g. different vertices or different momentum at the same
        vertex) is EXCLUDED from the unambiguous set."""
        particles = _make_particles([
            {"particle_id": 300, "vx": 0.0, "px": 1.0, "q": 1, "nhits": 5},
            {"particle_id": 300, "vx": 5.0, "px": 2.0, "q": -1, "nhits": 3},  # same ID, different particle
            {"particle_id": 400, "nhits": 6},  # unique, unaffected
        ])
        result = set(unambiguous_particle_ids(particles))
        assert 300 not in result, "Duplicated particle_id must be rejected"
        assert 400 in result, "Unrelated unique particle_id must still be accepted"


class TestSelectValidTracks:
    def test_case_3_unique_id_below_hit_threshold_is_rejected(self):
        """Case 3: a UNIQUE particle_id with nhits below min_hits is
        rejected by select_valid_tracks, even though it's unambiguous."""
        particles = _make_particles([
            {"particle_id": 500, "nhits": 2},  # unique but too short
        ])
        result = select_valid_tracks(particles, min_hits=4)
        assert 500 not in set(result["particle_id"]), (
            "A unique particle_id with insufficient nhits must still be excluded"
        )

    def test_case_4_unique_id_with_sufficient_hits_is_accepted(self):
        """Case 4: a UNIQUE particle_id with nhits >= min_hits IS accepted."""
        particles = _make_particles([
            {"particle_id": 600, "nhits": 4},  # exactly at threshold
            {"particle_id": 700, "nhits": 10},  # well above threshold
        ])
        result = select_valid_tracks(particles, min_hits=4)
        assert 600 in set(result["particle_id"])
        assert 700 in set(result["particle_id"])

    def test_combines_both_filters_correctly(self):
        """Integration of cases 1-4 together: duplicated AND/OR
        short-hit-count particles are excluded; only unique-AND-sufficient
        particles survive."""
        particles = _make_particles([
            {"particle_id": 1, "nhits": 10},                     # keep: unique, enough hits
            {"particle_id": 2, "nhits": 1},                      # drop: unique, too few hits
            {"particle_id": 3, "vx": 0.0, "nhits": 10},           # drop: duplicated...
            {"particle_id": 3, "vx": 9.0, "nhits": 10},           # ...(different vertex)
            {"particle_id": 4, "nhits": 4},                      # keep: unique, at threshold
        ])
        result = set(select_valid_tracks(particles, min_hits=4)["particle_id"])
        assert result == {1, 4}


class TestGetTrackHits:
    def test_sorts_by_transverse_radius_not_z(self):
        # Regression test for the M2/M3 ordering bug: hits must be sorted
        # by transverse_radius (physically correct — a real trajectory's
        # radius increases monotonically outward), NOT by z.
        particle_id = 42
        # Deliberately construct hits where z-order and radius-order DIFFER,
        # so a z-based sort would produce a different (wrong) result.
        hits, truth = _make_hits_and_truth(particle_id, n_hits=4, radii=[30, 10, 20, 40])
        # z increases as [0, 10, 20, 30] (hit_id order), but radius is
        # [30, 10, 20, 40] in that same hit_id order -- these disagree.
        result = get_track_hits(hits, truth, particle_id)
        assert list(result["transverse_radius"]) == sorted([30, 10, 20, 40])

    def test_only_returns_hits_for_the_requested_particle(self):
        hits_a, truth_a = _make_hits_and_truth(1, n_hits=3, radii=[10, 20, 30])
        hits_b, truth_b = _make_hits_and_truth(2, n_hits=2, radii=[15, 25])
        hits_b["hit_id"] = [4, 5]  # avoid hit_id collisions with particle 1
        truth_b["hit_id"] = [4, 5]

        all_hits = pd.concat([hits_a, hits_b], ignore_index=True)
        all_truth = pd.concat([truth_a, truth_b], ignore_index=True)

        result = get_track_hits(all_hits, all_truth, particle_id=1)
        assert len(result) == 3
        assert set(result["hit_id"]) == {1, 2, 3}


class TestTrackSummary:
    def test_raises_on_ambiguous_particle_id(self):
        # This is the core M3 safety guarantee: track_summary must hard-fail
        # (not silently average/combine) when given an ambiguous particle_id.
        particles = _make_particles([
            {"particle_id": 999, "vx": 0.0, "px": 1.0, "nhits": 5},
            {"particle_id": 999, "vx": 9.0, "px": 3.0, "nhits": 5},
        ])
        hits, truth = _make_hits_and_truth(999, n_hits=5, radii=[10, 20, 30, 40, 50])

        with pytest.raises(ValueError, match="expected exactly 1"):
            track_summary(particles, hits, truth, particle_id=999)

    def test_returns_correct_summary_for_valid_particle(self):
        particles = _make_particles([
            {"particle_id": 111, "px": 3.0, "py": 4.0, "q": -1, "nhits": 3},
        ])
        hits, truth = _make_hits_and_truth(111, n_hits=3, radii=[10, 20, 30])

        summary = track_summary(particles, hits, truth, particle_id=111)
        assert summary["particle_id"] == 111
        assert summary["n_hits"] == 3
        assert summary["pt"] == pytest.approx(5.0)  # sqrt(3^2+4^2)
        assert summary["charge"] == -1
        assert summary["r_min"] == pytest.approx(10.0)
        assert summary["r_max"] == pytest.approx(30.0)

    def test_phi_spread_flags_contamination(self):
        # This directly tests the diagnostic signal that first exposed the
        # M3 contamination in the real notebook: a genuinely single
        # particle should have a NARROW phi_spread, since it's confined to
        # one direction from the collision point.
        particles = _make_particles([
            {"particle_id": 222, "px": 1.0, "py": 0.0, "nhits": 3},
        ])
        hits, truth = _make_hits_and_truth(222, n_hits=3, radii=[10, 20, 30])
        summary = track_summary(particles, hits, truth, particle_id=222)
        # All hits placed along the x-axis (y=0) in the helper -> phi=0
        # for every hit -> phi_spread should be ~0, not close to 2*pi.
        assert summary["phi_spread"] == pytest.approx(0.0, abs=1e-6)

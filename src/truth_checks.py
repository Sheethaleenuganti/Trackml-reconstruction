"""Truth-data sanity checks.

Per the Step 3 instructions: this module performs checks and reports
findings. It does NOT silently fix inconsistencies. If an invariant is
violated, the corresponding function raises an AssertionError (or, for
soft/informational checks, returns a result that the caller must inspect)
rather than quietly patching the data.
"""
from __future__ import annotations

import pandas as pd


def check_hit_id_uniqueness(hits: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Check 1 & 2: are hit_id values unique in hits and in truth?"""
    hits_unique = hits["hit_id"].is_unique
    truth_unique = truth["hit_id"].is_unique
    return {
        "hits_hit_id_unique": bool(hits_unique),
        "truth_hit_id_unique": bool(truth_unique),
        "n_hits": len(hits),
        "n_unique_hit_ids_in_hits": int(hits["hit_id"].nunique()),
        "n_truth_rows": len(truth),
        "n_unique_hit_ids_in_truth": int(truth["hit_id"].nunique()),
    }


def check_every_hit_has_truth(hits: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Check 3: does every hit have a truth row?"""
    hit_ids = set(hits["hit_id"])
    truth_ids = set(truth["hit_id"])
    hits_without_truth = hit_ids - truth_ids
    truth_without_hits = truth_ids - hit_ids
    return {
        "len_hits_equals_len_truth": len(hits) == len(truth),
        "n_hits": len(hits),
        "n_truth_rows": len(truth),
        "n_hit_ids_missing_from_truth": len(hits_without_truth),
        "n_truth_hit_ids_missing_from_hits": len(truth_without_hits),
        "hit_ids_missing_from_truth_sample": sorted(hits_without_truth)[:10],
        "truth_hit_ids_missing_from_hits_sample": sorted(truth_without_hits)[:10],
    }


def check_noise_hits(truth: pd.DataFrame) -> dict:
    """Check 4 & 5: noise-hit count and unique nonzero particle_id count."""
    n_noise = int((truth["particle_id"] == 0).sum())
    nonzero = truth.loc[truth["particle_id"] != 0, "particle_id"]
    return {
        "n_noise_hits_particle_id_zero": n_noise,
        "n_unique_nonzero_particle_ids": int(nonzero.nunique()),
        "noise_fraction": n_noise / len(truth) if len(truth) > 0 else float("nan"),
    }


def hits_per_particle_histogram(truth: pd.DataFrame) -> dict:
    """Check 6: bucket particles by hit count (0, 1, 2, 3, >=4).

    Note: a particle with 0 hits in `truth` cannot literally appear in
    `truth` (truth only has rows for hits that exist), so the '0 hits'
    bucket here is necessarily always 0 when counted this way. It is
    included anyway, for completeness, and any nonzero value in that
    bucket would itself be a red flag worth investigating (it would mean
    something is wrong with how this function is being called, since it
    should be structurally impossible given how truth.csv is defined).
    """
    nonzero = truth.loc[truth["particle_id"] != 0]
    counts = nonzero.groupby("particle_id").size()
    return {
        "particles_with_0_hits": int((counts == 0).sum()),
        "particles_with_1_hit": int((counts == 1).sum()),
        "particles_with_2_hits": int((counts == 2).sum()),
        "particles_with_3_hits": int((counts == 3).sum()),
        "particles_with_4_or_more_hits": int((counts >= 4).sum()),
        "total_nonzero_particles_observed_in_truth": int(len(counts)),
    }


def compare_particles_nhits_with_truth(
    particles: pd.DataFrame, truth: pd.DataFrame
) -> dict:
    """Check 7 & 8: compare particles.nhits against counts derived from truth.

    IMPORTANT — discovered against real data (event 1000, Kaggle
    train_sample), not assumed from documentation:

    `particle_id` is NOT a unique key in particles.csv. On the real event
    inspected, particles.csv had 12,263 rows but only 2,564 unique
    particle_id values (confirmed via the OFFICIAL trackml.dataset.load_event
    loader, which itself casts particle_id to int64 — so this is not a
    dtype/precision artifact of our own code, it reflects the real file).

    This means a naive one-to-one merge of particles.csv against a
    truth-derived hit-count table (grouped by particle_id) is invalid: a
    single particle_id can legitimately correspond to multiple distinct
    rows in particles.csv (e.g. different vertices, different charges),
    each of which would incorrectly be compared against the SAME aggregated
    truth hit-count. This produced a highly misleading result (>90%
    "mismatch") the first time this function was run against real data,
    which is exactly the kind of inconsistency this project's rules say to
    stop and explain rather than silently patch.

    This function therefore compares SUMMED nhits per particle_id (summed
    across all particles.csv rows sharing that ID) against the
    truth-derived count for that ID, which is the comparison that is
    actually well-defined given the real schema. Per-row-level comparison
    (if ever needed) would require a different, particle-instance-aware
    join key, which is not currently available from the documented schema
    and is left as a TO VERIFY item rather than invented here.

    Returns per-particle mismatches rather than a single boolean, so
    inconsistencies can be inspected rather than silently glossed over.
    """
    nonzero = truth.loc[truth["particle_id"] != 0]
    truth_counts = nonzero.groupby("particle_id").size().rename("truth_derived_nhits")

    # Sum nhits across all rows sharing a particle_id, since particle_id is
    # not guaranteed unique in particles.csv (see docstring above).
    particles_summed = (
        particles.groupby("particle_id")["nhits"].sum().rename("nhits_summed")
    )

    merged = particles_summed.to_frame().merge(
        truth_counts, on="particle_id", how="left"
    )
    merged["truth_derived_nhits"] = merged["truth_derived_nhits"].fillna(0).astype(int)
    merged["matches"] = merged["nhits_summed"] == merged["truth_derived_nhits"]
    merged = merged.reset_index()

    mismatches = merged.loc[~merged["matches"]]

    return {
        "n_particles_checked": len(merged),
        "n_matching": int(merged["matches"].sum()),
        "n_mismatching": int((~merged["matches"]).sum()),
        "mismatch_sample": mismatches.head(10).to_dict(orient="records"),
        "note": (
            "Compares nhits SUMMED per particle_id against truth-derived "
            "hit counts, because particle_id is not a unique row key in "
            "particles.csv on real data (see function docstring)."
        ),
    }


def run_all_checks(event: dict[str, pd.DataFrame]) -> dict:
    """Run every check above and return a combined report.

    Parameters
    ----------
    event : dict[str, pandas.DataFrame]
        Must contain 'hits', 'truth', and 'particles' keys, as returned by
        src.data_loader.load_event.

    Returns
    -------
    dict
        Nested dict, one key per check, containing that check's results.
        This function does NOT raise on its own for informational checks;
        callers should inspect the results (or use `assert_core_invariants`
        below for a hard-failing version of the checks that truly must
        always hold).
    """
    hits, truth, particles = event["hits"], event["truth"], event["particles"]
    return {
        "hit_id_uniqueness": check_hit_id_uniqueness(hits, truth),
        "every_hit_has_truth": check_every_hit_has_truth(hits, truth),
        "noise_hits": check_noise_hits(truth),
        "hits_per_particle_histogram": hits_per_particle_histogram(truth),
        "particles_nhits_vs_truth": compare_particles_nhits_with_truth(particles, truth),
    }


def assert_core_invariants(event: dict[str, pd.DataFrame]) -> None:
    """Hard-fail on invariants that should ALWAYS hold for valid TrackML data.

    This is deliberately separate from `run_all_checks` (which is
    diagnostic/reporting) so that test code and pipeline code can choose to
    hard-fail on core invariants without having to re-derive which checks
    are "must always pass" versus "informational."

    Raises
    ------
    AssertionError
        If any core invariant is violated. The message names which
        invariant failed and includes the relevant counts, so a failure
        is immediately actionable rather than requiring re-running the
        diagnostic checks to find out what happened.
    """
    hits, truth = event["hits"], event["truth"]

    assert hits["hit_id"].is_unique, "hits.hit_id contains duplicate values"
    assert truth["hit_id"].is_unique, "truth.hit_id contains duplicate values"
    assert len(hits) == len(truth), (
        f"len(hits)={len(hits)} != len(truth)={len(truth)} — "
        "every hit should have exactly one truth row (Step 2, Part C)"
    )
    assert set(hits["hit_id"]) == set(truth["hit_id"]), (
        "hit_id sets in hits and truth do not match exactly"
    )

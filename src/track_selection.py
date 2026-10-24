"""Truth-track selection for diagnostic/validation visualization.

============================================================================
WHY THIS MODULE EXISTS — real-data finding (M3)
============================================================================
On first contact with real TrackML data (event 1000, Kaggle
train_sample), we found that `particle_id` is NOT a unique identifier for
a physical particle in `particles.csv`:

  - 12,263 rows in particles.csv, but only 2,564 unique particle_id values.
  - Of those 2,564 IDs, 1,082 (42%) have rows spanning MULTIPLE distinct
    production vertices (vx, vy, vz) — these are unambiguously different
    physical particles that happen to share one raw ID.
  - A further 635 IDs have multiple rows at the exact SAME vertex, with
    different momentum and often different charge (581/635 show mixed
    charge, suggestive of pair-production-like processes, though this
    specific physical mechanism is NOT confirmed by any primary source we
    have — it is an observation, not a documented fact).
  - The official trackml.utils.decode_particle_id() does not distinguish
    these same-vertex duplicate rows either (identical vertex_id,
    primary_id, generation, secondary_id, process fields).
  - Neither particles.csv nor truth.csv contains any row-level instance
    identifier that could re-associate a specific hit to a specific one of
    several particles.csv rows sharing a particle_id.
  - Strong corroborating evidence for "each row = one real particle": the
    RAW ROW COUNT (12,263) matches the literature figure of "O(10^4)
    particles per event" (Step 2, Part J) far better than the deduplicated
    ID count (2,564, off by nearly an order of magnitude).

CONCLUSION (confident, evidence-based — not "mechanism not yet
established", since the evidence here is decisive): particle_id alone is
an insufficient key for identifying a unique physical particle in this
dataset. particles.csv ROWS, not particle_id VALUES, are the correct unit
for "one physical particle" in this dataset. However, truth.csv only
records particle_id (no row-level pointer), so hits CANNOT be reliably
re-associated to one specific particles.csv row when that ID is
ambiguous, using currently-documented fields alone.

============================================================================
THE FIX APPLIED HERE
============================================================================
Rather than attempting to solve instance-disambiguation (a genuinely open
sub-problem, out of scope for a diagnostic visualization module), this
module selects trajectories ONLY from particle_ids that are UNAMBIGUOUS
(exactly one row in particles.csv). On event 1000 this is 847 of 2,564
unique IDs (751 after the official ORDER_MIN_HITS>=4 filter) — a large,
valid, unbiased pool for visualization purposes. Their hit-count
distribution (mean 10.4 hits) matches the literature's ~12-hit average,
unlike the contaminated multi-vertex groups (which showed 90-186 "hits").

This is a DIAGNOSTIC/VALIDATION-ONLY module. It uses truth information
freely (particle_id, vertex, momentum) because its entire purpose is
human visual inspection of known-truth tracks — never reconstruction
input. See the module-level warning in every function below.

IMPORTANT SCOPE NOTE: this restricts which tracks we VISUALIZE. It is not
a claim that the underlying particle-instance ambiguity is solved for the
dataset as a whole — ambiguous particle_ids still exist in the data and
must be handled correctly (or explicitly excluded, with that exclusion
documented) wherever else particle_id is used, e.g. in the eventual
TrackML scoring implementation.
============================================================================
"""
from __future__ import annotations

import pandas as pd


def unambiguous_particle_ids(particles: pd.DataFrame) -> pd.Series:
    """Return particle_id values that correspond to exactly one row in
    particles.csv (i.e., not affected by the M3 duplicate-ID finding).

    DIAGNOSTIC USE ONLY. This function reads particles.csv, which is
    truth-adjacent information; it must never be used to filter or select
    inputs to a clustering/reconstruction algorithm — only to choose which
    truth tracks to VISUALIZE for human inspection.

    Parameters
    ----------
    particles : pandas.DataFrame
        As returned by src.data_loader.load_event(..., parts=("particles",)).

    Returns
    -------
    pandas.Series
        The subset of particle_id values with exactly one row in
        `particles`.
    """
    row_counts = particles.groupby("particle_id").size()
    return row_counts[row_counts == 1].index.to_series(name="particle_id").reset_index(drop=True)


def select_valid_tracks(
    particles: pd.DataFrame,
    min_hits: int = 4,
) -> pd.DataFrame:
    """Select particles.csv rows suitable for truth-track visualization.

    DIAGNOSTIC USE ONLY (see module docstring).

    Applies two filters:
      1. Unambiguous particle_id (see `unambiguous_particle_ids`) — so a
         selected particle_id corresponds to exactly one real particle,
         not several sharing an ID.
      2. nhits >= min_hits (default 4, matching the official TrackML
         scoring library's ORDER_MIN_HITS threshold, verified against
         trackml/weights.py in Step 3) — so degenerately short tracks
         are excluded, matching how the real scoring metric treats them.

    Parameters
    ----------
    particles : pandas.DataFrame
        As returned by the loader.
    min_hits : int, optional
        Minimum nhits to include. Default 4 (official ORDER_MIN_HITS).

    Returns
    -------
    pandas.DataFrame
        Filtered rows of `particles`, safe to sample particle_id from for
        visualization — each selected particle_id is guaranteed to
        correspond to exactly one row.
    """
    valid_ids = set(unambiguous_particle_ids(particles))
    mask = particles["particle_id"].isin(valid_ids) & (particles["nhits"] >= min_hits)
    return particles.loc[mask].reset_index(drop=True)


def get_track_hits(
    hits: pd.DataFrame,
    truth: pd.DataFrame,
    particle_id: int,
) -> pd.DataFrame:
    """Return all hits belonging to one (unambiguous) particle_id, with
    physics-derived columns attached, sorted by transverse radius.

    DIAGNOSTIC USE ONLY (see module docstring). Callers should ensure
    `particle_id` came from `select_valid_tracks` / `unambiguous_particle_ids`
    — this function does not itself verify unambiguity, since it may
    reasonably be called on `particle_id=0` (noise) for diagnostic
    purposes too.

    Ordering rationale (M3 finding): hits are sorted by transverse radius
    (r = sqrt(x^2+y^2)), NOT by z. A real particle flies OUTWARD from the
    collision point, so r increases monotonically along a genuine
    trajectory; z does not in general (a particle's z position can move in
    complex, non-monotonic ways depending on its angle). Sorting by z was
    an earlier bug (Step 3/M2) that produced visually chaotic,
    non-physical "trajectories".

    Parameters
    ----------
    hits : pandas.DataFrame
        As returned by the loader.
    truth : pandas.DataFrame
        As returned by the loader.
    particle_id : int
        Which particle's hits to retrieve.

    Returns
    -------
    pandas.DataFrame
        Hits for this particle_id, with 'transverse_radius' and 'phi'
        columns added, sorted by transverse_radius ascending.
    """
    from src.physics import add_transverse_radius_and_phi

    hits_with_truth = hits.merge(
        truth[["hit_id", "particle_id"]], on="hit_id", how="left"
    )
    track = hits_with_truth.loc[hits_with_truth["particle_id"] == particle_id]
    track = add_transverse_radius_and_phi(track)
    return track.sort_values("transverse_radius").reset_index(drop=True)


def track_summary(
    particles: pd.DataFrame,
    hits: pd.DataFrame,
    truth: pd.DataFrame,
    particle_id: int,
) -> dict:
    """Compute a short diagnostic summary for one track, for reporting.

    DIAGNOSTIC USE ONLY (see module docstring).

    Raises
    ------
    ValueError
        If `particle_id` does not correspond to exactly one row in
        `particles` — this is a deliberate hard-fail (not a silent
        best-effort), matching the project's rule of stopping and
        explaining inconsistencies rather than quietly working around
        them. Callers should select particle_id via `select_valid_tracks`
        or `unambiguous_particle_ids` first.

    Returns
    -------
    dict with keys: particle_id, n_hits, pt, charge, r_min, r_max, z_min,
    z_max, phi_spread. `phi_spread` is included specifically because it is
    what exposed the M3 contamination issue (a real single particle should
    have a narrow phi spread, not close to the full 2*pi range).
    """
    from src.physics import transverse_momentum

    particle_row = particles.loc[particles["particle_id"] == particle_id]

    if len(particle_row) != 1:
        raise ValueError(
            f"track_summary called on particle_id={particle_id}, which has "
            f"{len(particle_row)} rows in particles.csv (expected exactly 1 "
            "— use select_valid_tracks() or unambiguous_particle_ids() to "
            "choose a valid particle_id first). See module docstring for "
            "the M3 finding this guards against."
        )

    track = get_track_hits(hits, truth, particle_id)

    pt = float(transverse_momentum(
        particle_row["px"].values, particle_row["py"].values
    )[0])
    charge = int(particle_row["q"].values[0])

    return {
        "particle_id": int(particle_id),
        "n_hits": int(len(track)),
        "pt": pt,
        "charge": charge,
        "r_min": float(track["transverse_radius"].min()) if len(track) else None,
        "r_max": float(track["transverse_radius"].max()) if len(track) else None,
        "z_min": float(track["z"].min()) if len(track) else None,
        "z_max": float(track["z"].max()) if len(track) else None,
        "phi_spread": (
            float(track["phi"].max() - track["phi"].min()) if len(track) else None
        ),
    }

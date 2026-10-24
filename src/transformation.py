"""Coordinate transformation for TrackML hit clustering.

============================================================================
WHY THIS TRANSFORMATION — physics justification
============================================================================
Raw detector hits are recorded in Cartesian (x, y, z), in millimeters, with
z along the beam axis (Step 2, Part E). A charged particle moving through
the detector's magnetic field follows a HELICAL trajectory: it curves in
the transverse (x-y) plane while advancing along z. This is the direct
physical consequence of the Lorentz force on a moving charge in a magnetic
field aligned with z (Step 2, Part D).

This is a genuine obstacle for density-based clustering directly in (x,y,z):
points belonging to the SAME particle are often not close together in raw
Cartesian space (a helix can be quite spread out), while points from
DIFFERENT particles emitted in similar initial directions can start out
very close together near the collision point. DBSCAN, which groups points
by local density/proximity, does not work well on this raw representation.

============================================================================
THE TRANSFORMATION USED — official TrackML starting kit
============================================================================
This project implements the FIRST of two candidate transforms identified
in Step 2 (Part E) and re-verified directly against the primary source
during M4 (Amrouche et al. 2019, "The Tracking Machine Learning challenge:
Accuracy phase", arXiv:1904.06778, Section 2.7 "Starting kit"):

    r1 = sqrt(x^2 + y^2 + z^2)      (full 3D distance from the origin)
    x2 = x / r1
    y2 = y / r1
    r2 = sqrt(x^2 + y^2)             (transverse radius)
    z2 = z / r2

This is NOT an invented transformation — it is the exact starting-kit
formula from the official challenge documentation, confirmed to achieve
a TrackML score of approximately 0.21 in that source, and confirmed by a
second primary source (Amrouche et al. 2020, "Machine Learning scientific
competitions and datasets", arXiv:2012.08520) to have been the dominant
approach among competitors for roughly the first 30 days of the
competition, with scores clustering around 0.20-0.25.

We selected this over the second Step 2 candidate (the Hough/polar
r=2*r0*cos(phi-theta) parametrization) because it maps directly onto
"transform coordinates -> feed to DBSCAN", matching this project's
Objectives #3/#4 exactly (Content Master, Section E), whereas the Hough
candidate is a different algorithmic family (a voting scheme over track
parameters, not a direct clustering-space transform).

Physical intuition for WHY this transform helps: dividing (x, y) by the
full 3D radius r1 projects each hit toward a direction-like quantity
(similar in spirit to projecting onto a unit sphere), and scaling z by the
TRANSVERSE radius r2 (not r1) normalizes the longitudinal spread relative
to how far out, transversely, the hit already is. The intended effect is
that hits belonging to one helical trajectory become numerically closer
together in (x2, y2, z2) space than in raw (x, y, z), making
proximity-based clustering more effective.

============================================================================
WHAT IS PRESERVED, WHAT MAY BE DISTORTED
============================================================================
Preserved: hit_id (carried through unchanged, so results can always be
re-joined to the original hits/truth tables). Raw x, y, z are also kept
alongside the transformed columns — this module never overwrites or drops
the original coordinates.

Distorted/lost: the transform is undefined (division by zero) exactly at
the origin (r1=0) and along the beam axis where the transverse radius is
zero (r2=0, hits with x=y=0). Real detector hits are not expected at
r1=0 (that is the collision point itself, not a sensor), but r2=0 is a
theoretical edge case for a hit lying exactly on the z-axis, which is
physically implausible for a cylindrical detector but is handled
explicitly (not silently) below rather than assumed impossible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def transform_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Apply the official TrackML starting-kit coordinate transformation.

    Implements exactly:
        r1 = sqrt(x^2 + y^2 + z^2)
        x2 = x / r1
        y2 = y / r1
        r2 = sqrt(x^2 + y^2)
        z2 = z / r2

    as given in Amrouche et al. 2019 (arXiv:1904.06778), Section 2.7,
    re-verified directly against that primary source during M4 (see
    module docstring for full citation and physics justification).

    Parameters
    ----------
    hits : pandas.DataFrame
        Must contain 'hit_id', 'x', 'y', 'z' columns (as returned by
        src.data_loader.load_event). NOT modified in place.

    Returns
    -------
    pandas.DataFrame
        A NEW DataFrame containing:
          - 'hit_id' (preserved, unchanged, for later re-joining)
          - 'x', 'y', 'z' (original raw coordinates, preserved unchanged)
          - 'r1' (full 3D radius from origin)
          - 'r2' (transverse radius)
          - 'x2', 'y2', 'z2' (the transformed coordinates)
        Row order and row count are identical to the input — this function
        never drops or reorders hits.

    Notes on invalid/edge-case values (handled explicitly, not silently)
    -----------------------------------------------------------------
    - If any hit has r1 == 0 (x=y=z=0, i.e. exactly at the origin), x2 and
      y2 are undefined (0/0). This is set to NaN rather than raising or
      silently producing inf/nan from raw numpy division, so it is
      visible and countable downstream.
    - If any hit has r2 == 0 (x=y=0, i.e. exactly on the beam axis), z2
      is undefined (division by zero). Also set to NaN explicitly.
    - Both cases are counted and reported by `transform_hits` via the
      companion function `transform_diagnostics` below — callers should
      check these before assuming a clean transform.
    """
    required_cols = {"hit_id", "x", "y", "z"}
    missing = required_cols - set(hits.columns)
    if missing:
        raise ValueError(
            f"transform_hits requires columns {required_cols}, missing: {missing}"
        )

    x = hits["x"].to_numpy(dtype=np.float64)
    y = hits["y"].to_numpy(dtype=np.float64)
    z = hits["z"].to_numpy(dtype=np.float64)

    r1 = np.sqrt(x**2 + y**2 + z**2)
    r2 = np.sqrt(x**2 + y**2)

    # Explicit handling of division-by-zero edge cases (see docstring):
    # produce NaN rather than inf/warning-suppressed garbage, so these
    # cases are visible and countable rather than silently propagating.
    with np.errstate(divide="ignore", invalid="ignore"):
        x2 = np.where(r1 != 0, x / r1, np.nan)
        y2 = np.where(r1 != 0, y / r1, np.nan)
        z2 = np.where(r2 != 0, z / r2, np.nan)

    result = pd.DataFrame({
        "hit_id": hits["hit_id"].to_numpy(),
        "x": x,
        "y": y,
        "z": z,
        "r1": r1,
        "r2": r2,
        "x2": x2,
        "y2": y2,
        "z2": z2,
    })
    return result


def transform_diagnostics(transformed: pd.DataFrame) -> dict:
    """Compute summary diagnostics for a transformed-hits DataFrame.

    Intended to be checked immediately after `transform_hits` on any new
    dataset, real or synthetic, per this project's rule of validating
    before trusting (never assume the transform "just worked").

    Parameters
    ----------
    transformed : pandas.DataFrame
        As returned by `transform_hits`.

    Returns
    -------
    dict
        Counts of hits, NaN values per transformed column, and min/max
        ranges for each transformed column (ignoring NaN).
    """
    stats: dict = {"n_hits": len(transformed)}
    for col in ["r1", "r2", "x2", "y2", "z2"]:
        values = transformed[col]
        stats[f"{col}_n_nan"] = int(values.isna().sum())
        stats[f"{col}_n_inf"] = int(np.isinf(values.to_numpy(dtype=np.float64)).sum())
        finite = values[np.isfinite(values.to_numpy(dtype=np.float64))]
        stats[f"{col}_min"] = float(finite.min()) if len(finite) else None
        stats[f"{col}_max"] = float(finite.max()) if len(finite) else None
    return stats

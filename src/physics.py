"""Simple, verified derived physics quantities for the TrackML project.

This module intentionally implements ONLY the small set of quantities that
are:
  (a) not raw columns in the dataset (see docs/DATA_ACCESS.md / Step 2), and
  (b) needed at this stage of the project (Step 3: exploration/profiling).

It does NOT implement the TrackML coordinate transformation for clustering
(that is a later step) and does NOT implement the scoring metric (also a
later step). Keeping this module small and simple makes it easy to unit
test and easy to trust.

Conventions match the official `trackml.utils` module (verified against
github.com/LAL/trackml-library/blob/master/trackml/utils.py during Step 3),
with one deliberate naming difference explained below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def transverse_radius(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Transverse (cylindrical) radial distance from the beam axis.

        r = sqrt(x^2 + y^2)

    Physical meaning
    ----------------
    This is the distance from a hit to the beam axis (the z-axis), measured
    in the x-y ("transverse") plane. Because the magnetic field runs along
    z, a charged particle bends *within* this transverse plane, so this
    quantity — not the full 3D distance from the origin — is the natural
    radial coordinate for describing how a track curves.

    Note on naming versus the official library
    ----------------------------------------------
    The official `trackml.utils.add_position_quantities` calls this
    quantity `rho`, and reserves the name `r` for the full 3D distance
    `sqrt(x^2 + y^2 + z^2)`. Step 2 of this project used `r` informally
    for the transverse radius, following the convention used in the
    TrackML challenge paper's own prose description of the coordinate
    system. To avoid ambiguity going forward, THIS module uses the
    explicit name `transverse_radius` rather than a bare `r`, and callers
    should never assume `r` alone is unambiguous without checking which
    convention is in play. This discrepancy was discovered by directly
    reading the official library's source code in Step 3 and is recorded
    here rather than silently resolved.
    """
    return np.hypot(x, y)


def azimuthal_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Azimuthal angle phi in the transverse (x-y) plane.

        phi = atan2(y, x)

    Physical meaning
    ----------------
    This describes the *direction*, around the beam axis, in which a hit
    lies — phi = 0 points along +x, and phi sweeps through (-pi, pi].
    Two hits from the same particle, viewed early versus late in the
    particle's flight, will have similar (though not necessarily
    identical, due to bending) phi values if the particle has not curved
    too far. This matches `trackml.utils.add_position_quantities`, which
    computes `phi` identically via `np.arctan2(y, x)`.
    """
    return np.arctan2(y, x)


def transverse_momentum(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Transverse momentum pT of a particle.

        pT = sqrt(px^2 + py^2)

    Physical meaning
    ----------------
    This is the component of a particle's momentum lying in the transverse
    (x-y) plane — the plane in which the magnetic field causes bending.
    pT directly sets the radius of curvature of the helical trajectory:
    high-pT particles travel in nearly straight lines through the detector,
    while low-pT particles curl tightly. This is why pT is the natural
    variable for an efficiency-versus-momentum breakdown (Step 2, Part G):
    it is precisely the quantity that determines how hard a given track is
    to reconstruct.

    Per Step 2's Part G, this project computes pT at the *particle* level
    from `particles.px, particles.py` (one pT value per particle), not at
    the per-hit level from `truth.tpx, truth.tpy`. This matches
    `trackml.utils.add_momentum_quantities`, which computes
    `pt = np.hypot(px, py)` from whichever px/py columns are passed to it.
    """
    return np.hypot(px, py)


def add_transverse_radius_and_phi(hits: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `hits` with `transverse_radius` and `phi` columns added.

    Does not modify the input DataFrame in place, matching the Step 1
    architecture principle of never modifying raw data
    (see docs/DATA_ACCESS.md and the loader module's docstring).

    Parameters
    ----------
    hits : pandas.DataFrame
        Must contain 'x' and 'y' columns (as loaded from an event's
        hits.csv via the official trackml.dataset.load_event).

    Returns
    -------
    pandas.DataFrame
        A new DataFrame, a copy of `hits`, with two additional columns:
        'transverse_radius' and 'phi'.
    """
    result = hits.copy()
    result["transverse_radius"] = transverse_radius(result["x"].values, result["y"].values)
    result["phi"] = azimuthal_angle(result["x"].values, result["y"].values)
    return result


def add_particle_pt(particles: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `particles` with a 'pt' column added.

    Parameters
    ----------
    particles : pandas.DataFrame
        Must contain 'px' and 'py' columns (as loaded from an event's
        particles.csv via the official trackml.dataset.load_event).

    Returns
    -------
    pandas.DataFrame
        A new DataFrame, a copy of `particles`, with an additional 'pt'
        column.
    """
    result = particles.copy()
    result["pt"] = transverse_momentum(result["px"].values, result["py"].values)
    return result

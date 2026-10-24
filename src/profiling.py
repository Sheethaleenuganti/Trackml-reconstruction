"""Profiling multiple complete events.

Produces the summary table required by Step 3, using only the loader and
physics modules already validated by this repository's tests — no
duplicated CSV-parsing logic.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data_loader import load_event


def profile_events(data_dir: str, event_ids: Iterable[int]) -> pd.DataFrame:
    """Profile several complete events and return a summary table.

    For each event, computes:
      - number of hits
      - number of non-noise hits (particle_id != 0 in truth)
      - number of particles
      - number of reconstructable particles (nhits >= 1, per particles.csv)
      - average hits per reconstructable particle
      - median hits per reconstructable particle
      - maximum hits per particle
      - noise fraction (n_noise_hits / n_hits)

    Detector-layer occupancy (hits per (volume_id, layer_id) pair) is
    returned separately by `layer_occupancy`, since it does not collapse
    to a single row per event.

    Parameters
    ----------
    data_dir : str
        Directory containing the event CSV files.
    event_ids : iterable of int
        Which event IDs to profile.

    Returns
    -------
    pandas.DataFrame
        One row per event, indexed by event_id, with the columns listed
        above.
    """
    rows = []
    for event_id in event_ids:
        event = load_event(data_dir, event_id, parts=("hits", "particles", "truth"))
        hits = event["hits"]
        particles = event["particles"]
        truth = event["truth"]

        n_hits = len(hits)
        n_noise_hits = int((truth["particle_id"] == 0).sum())
        n_non_noise_hits = n_hits - n_noise_hits
        n_particles = len(particles)

        reconstructable = particles.loc[particles["nhits"] >= 1]
        n_reconstructable = len(reconstructable)
        if n_reconstructable > 0:
            avg_hits = float(reconstructable["nhits"].mean())
            median_hits = float(reconstructable["nhits"].median())
            max_hits = int(reconstructable["nhits"].max())
        else:
            avg_hits = float("nan")
            median_hits = float("nan")
            max_hits = 0

        noise_fraction = n_noise_hits / n_hits if n_hits > 0 else float("nan")

        rows.append(
            {
                "event_id": event_id,
                "n_hits": n_hits,
                "n_non_noise_hits": n_non_noise_hits,
                "n_particles": n_particles,
                "n_reconstructable_particles": n_reconstructable,
                "avg_hits_per_reconstructable_particle": avg_hits,
                "median_hits_per_reconstructable_particle": median_hits,
                "max_hits_per_particle": max_hits,
                "noise_fraction": noise_fraction,
            }
        )

    return pd.DataFrame(rows).set_index("event_id")


def layer_occupancy(data_dir: str, event_id: int) -> pd.DataFrame:
    """Return hit counts per (volume_id, layer_id) pair for one event.

    Parameters
    ----------
    data_dir : str
        Directory containing the event CSV files.
    event_id : int
        Which event to profile.

    Returns
    -------
    pandas.DataFrame
        Columns: volume_id, layer_id, n_hits. One row per observed
        (volume_id, layer_id) combination, sorted by volume_id then
        layer_id.
    """
    event = load_event(data_dir, event_id, parts=("hits",))
    hits = event["hits"]
    counts = (
        hits.groupby(["volume_id", "layer_id"])
        .size()
        .reset_index(name="n_hits")
        .sort_values(["volume_id", "layer_id"])
        .reset_index(drop=True)
    )
    return counts

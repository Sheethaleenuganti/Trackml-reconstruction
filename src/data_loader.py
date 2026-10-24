"""TrackML data loading utilities.

This module deliberately wraps the OFFICIAL trackml-library
(github.com/LAL/trackml-library) rather than re-implementing CSV parsing.
See docs/DATA_ACCESS.md for install instructions and for an important
warning about a same-named-but-unrelated package on PyPI.

Design principles (Step 1 architecture):
  - never modify raw data,
  - preserve original column names/dtypes exactly as the official library
    provides them,
  - fail loudly and clearly when files are missing, rather than silently
    returning partial or empty data.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from trackml.dataset import load_event as _official_load_event
except ImportError as exc:  # pragma: no cover - exercised only when misconfigured
    raise ImportError(
        "Could not import the official trackml library. This project "
        "requires installing it from GitHub, not from PyPI (a same-named "
        "but unrelated package exists on PyPI). Run:\n\n"
        "    pip install \"git+https://github.com/LAL/trackml-library.git\"\n\n"
        "See docs/DATA_ACCESS.md for details."
    ) from exc


class EventFilesNotFoundError(FileNotFoundError):
    """Raised when an event's expected CSV files cannot be located."""


def load_event_ids(data_dir: str) -> list[int]:
    """Return the sorted list of event IDs found in `data_dir`.

    An event is considered present if at least a `*-hits.csv` file exists
    for it, following the `eventNNNNNNNNN-hits.csv` naming convention
    documented by the official library and verified in Step 2.

    Parameters
    ----------
    data_dir : str
        Directory expected to contain files like
        `event000000000-hits.csv`, `event000000000-truth.csv`, etc.

    Returns
    -------
    list[int]
        Sorted event IDs (as integers, e.g. 0, 1, 2, ...).

    Raises
    ------
    EventFilesNotFoundError
        If `data_dir` does not exist, or if no matching hits files are
        found inside it.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise EventFilesNotFoundError(
            f"Data directory does not exist: {data_dir!s}. "
            "See docs/DATA_ACCESS.md for how to obtain and place the "
            "TrackML dataset."
        )

    pattern = str(data_path / "event*-hits.csv*")
    hit_files = sorted(glob.glob(pattern))
    if not hit_files:
        raise EventFilesNotFoundError(
            f"No files matching 'event*-hits.csv' found in {data_dir!s}. "
            "See docs/DATA_ACCESS.md for how to obtain and place the "
            "TrackML dataset, or use "
            "scripts/generate_synthetic_event.py for a clearly-labeled "
            "synthetic fixture to exercise this code path."
        )

    event_ids = []
    for f in hit_files:
        name = os.path.basename(f)
        # Expected shape: event000000000-hits.csv or event000000000-hits.csv.gz
        digits = name.split("-")[0].replace("event", "")
        event_ids.append(int(digits))
    return sorted(set(event_ids))


def load_event(
    data_dir: str,
    event_id: int,
    parts: Iterable[str] = ("hits", "cells", "particles", "truth"),
) -> dict[str, pd.DataFrame]:
    """Load a single, COMPLETE event's data.

    This function loads all requested parts of one event and returns them
    as a dict keyed by part name, which is easier to work with than the
    official library's positional-tuple return convention, while still
    delegating all actual CSV parsing to the official library.

    Parameters
    ----------
    data_dir : str
        Directory containing the event's CSV files.
    event_id : int
        The numeric event identifier (e.g. 0 for event000000000).
    parts : iterable of str, optional
        Which file types to load. Any subset of
        {'hits', 'cells', 'particles', 'truth'}. Defaults to all four.
        For test-dataset-style loading (no ground truth available), pass
        parts=('hits', 'cells').

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys are the requested part names; values are the corresponding
        DataFrames with original column names and dtypes preserved exactly
        as returned by the official library.

    Raises
    ------
    EventFilesNotFoundError
        If any requested part's file cannot be found for this event.

    Notes
    -----
    This function does NOT truncate or subsample the event's hits — it
    loads the complete event, as required for Step 3. Event-level (not
    within-event) subsampling is a separate, later concern (see Step 1
    architecture, module `data/subsampler.py`), which does not yet exist
    in this repository.
    """
    parts = list(parts)
    prefix = str(Path(data_dir) / f"event{event_id:09d}")

    # Check each requested file exists before calling into the official
    # loader, so that we can raise a clear, specific error rather than
    # whatever internal exception the official library happens to raise.
    missing = []
    for part in parts:
        expr = f"{prefix}-{part}.csv*"
        if not glob.glob(expr):
            missing.append(f"{prefix}-{part}.csv")
    if missing:
        raise EventFilesNotFoundError(
            f"Missing expected file(s) for event {event_id}: {missing}. "
            "See docs/DATA_ACCESS.md."
        )

    loaded = _official_load_event(prefix, parts=parts)
    return dict(zip(parts, loaded))


def profile_event(event: dict[str, pd.DataFrame]) -> dict:
    """Compute basic, verified descriptive statistics for one loaded event.

    This performs NO fabrication and NO assumption beyond what is directly
    computable from the DataFrames themselves — every value here is a
    direct count, min, or max over real data columns.

    Parameters
    ----------
    event : dict[str, pandas.DataFrame]
        As returned by `load_event`, must contain at least 'hits' and
        'truth' to compute the full set of statistics; 'particles' is
        used if present.

    Returns
    -------
    dict
        Flat dictionary of profiling statistics. See keys inline below.
    """
    stats: dict = {}

    if "hits" in event:
        hits = event["hits"]
        stats["n_hits"] = len(hits)
        stats["n_unique_hit_ids"] = hits["hit_id"].nunique()
        stats["x_min"] = float(hits["x"].min())
        stats["x_max"] = float(hits["x"].max())
        stats["y_min"] = float(hits["y"].min())
        stats["y_max"] = float(hits["y"].max())
        stats["z_min"] = float(hits["z"].min())
        stats["z_max"] = float(hits["z"].max())
        stats["volume_id_min"] = int(hits["volume_id"].min())
        stats["volume_id_max"] = int(hits["volume_id"].max())
        stats["layer_id_min"] = int(hits["layer_id"].min())
        stats["layer_id_max"] = int(hits["layer_id"].max())
        stats["module_id_min"] = int(hits["module_id"].min())
        stats["module_id_max"] = int(hits["module_id"].max())

    if "truth" in event:
        truth = event["truth"]
        stats["n_truth_rows"] = len(truth)
        stats["n_unique_truth_hit_ids"] = truth["hit_id"].nunique()
        stats["n_noise_hits"] = int((truth["particle_id"] == 0).sum())
        nonzero_ids = truth.loc[truth["particle_id"] != 0, "particle_id"]
        stats["n_unique_nonzero_particle_ids"] = nonzero_ids.nunique()

    if "particles" in event:
        particles = event["particles"]
        stats["n_particles"] = len(particles)
        if "nhits" in particles.columns:
            stats["n_reconstructable_particles_nhits_ge_1"] = int(
                (particles["nhits"] >= 1).sum()
            )

    return stats

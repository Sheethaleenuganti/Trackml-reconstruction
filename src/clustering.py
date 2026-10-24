"""DBSCAN-based track candidate generation.

============================================================================
SCOPE AND SAFETY BOUNDARY (read before using)
============================================================================
This module implements the M5 baseline: DBSCAN clustering on TRANSFORMED
detector-hit coordinates ONLY. It must never receive particle_id, truth
labels, truth weights, or any ground-truth-derived quantity as a feature.

Truth is used ELSEWHERE (src/evaluation.py) to score the clusters this
module produces, strictly AFTER clustering — never as an input here.

This module reuses src.transformation.transform_hits (M4) rather than
re-implementing the coordinate transform, per the project's own rule of
not duplicating an existing verified implementation.
============================================================================
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def extract_features(
    transformed_hits: pd.DataFrame,
    representation: str = "xy",
) -> np.ndarray:
    """Extract a clustering feature matrix from transformed hits.

    Parameters
    ----------
    transformed_hits : pandas.DataFrame
        As returned by src.transformation.transform_hits — must contain
        'x2', 'y2', and (if representation requires it) 'z2'.
    representation : str
        One of:
          - 'xy'  : (x2, y2) only — 2D
          - 'xyz' : (x2, y2, z2) — 3D, no scaling
        Scaling variants are handled by the caller (see
        `extract_features_scaled` below) rather than baked in here, so
        that "scaled" is always an explicit, visible experimental choice
        (per M5 rules: never silently standardize and call it an
        improvement).

    Returns
    -------
    numpy.ndarray, shape (n_hits, n_features)
        ONLY the requested transformed coordinate columns — no particle_id,
        no truth, no raw x/y/z.
    """
    if representation == "xy":
        return transformed_hits[["x2", "y2"]].to_numpy(dtype=np.float64)
    elif representation == "xyz":
        return transformed_hits[["x2", "y2", "z2"]].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown representation '{representation}'. Expected 'xy' or 'xyz'."
        )


def extract_features_scaled(
    transformed_hits: pd.DataFrame,
    representation: str = "xyz",
) -> tuple[np.ndarray, dict]:
    """Extract a STANDARDIZED (zero mean, unit variance per column) feature
    matrix, as an explicit, separately-labeled experimental variant.

    This exists specifically to test whether x2/y2's numerical scale
    ([-1,1], since they are direction cosines — see M4 real-data
    validation, docs/m4_real_data_validation.txt) differs enough from
    z2's scale (observed [-42.5, 47.9] on real event 1000) that raw
    Euclidean DBSCAN distance would let z2 dominate. Standardization is
    NOT applied by default anywhere else in this module — it must be
    explicitly requested, and every experiment using it is labeled
    "scaled" throughout results/tests/docs.

    Returns
    -------
    (numpy.ndarray, dict)
        The scaled feature matrix, and a dict of the per-column mean/std
        used (for transparency/reproducibility — no scaler object is
        hidden inside a black box).
    """
    raw = extract_features(transformed_hits, representation=representation)
    means = raw.mean(axis=0)
    stds = raw.std(axis=0)
    stds_safe = np.where(stds == 0, 1.0, stds)  # avoid div-by-zero on a degenerate column
    scaled = (raw - means) / stds_safe
    info = {
        "representation": representation,
        "means": means.tolist(),
        "stds": stds.tolist(),
    }
    return scaled, info


def cluster_tracks(
    X: np.ndarray,
    eps: float,
    min_samples: int,
) -> dict:
    """Run DBSCAN on a permitted feature matrix and return labels + timing.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_hits, n_features)
        MUST contain only detector-derived features (transformed
        coordinates, optionally scaled) — see module docstring. This
        function does not and cannot verify that truth was not smuggled
        in; that responsibility belongs to the caller (this project's
        notebooks/scripts only ever call this with output from
        `extract_features`/`extract_features_scaled`).
    eps : float
        DBSCAN eps parameter (in the units of X — i.e. transformed-space
        units, not millimeters, once X has been produced by
        extract_features).
    min_samples : int
        DBSCAN min_samples parameter.

    Returns
    -------
    dict with keys:
        'labels': numpy.ndarray of cluster labels (-1 = DBSCAN noise),
        'n_clusters': number of non-noise clusters found,
        'n_noise': number of hits labeled as noise,
        'runtime_seconds': wall-clock time for the DBSCAN fit call,
        'eps': eps used, 'min_samples': min_samples used.
    """
    t0 = time.perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(X)
    runtime = time.perf_counter() - t0

    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1})
    n_noise = int((labels == -1).sum())

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_hits": len(labels),
        "runtime_seconds": runtime,
        "eps": eps,
        "min_samples": min_samples,
    }


def knn_distance_diagnostic(X: np.ndarray, k: int) -> np.ndarray:
    """Compute the k-th nearest-neighbor distance for every point in X.

    This is the standard DBSCAN parameter-selection diagnostic (sorted
    k-NN distance plot, sometimes called a "k-distance graph" or
    "elbow plot"): the elbow/knee of the SORTED k-th-NN-distance curve is
    a commonly used heuristic starting point for choosing eps, with k
    conventionally set equal to the intended min_samples. This is a
    purely geometric diagnostic over the feature matrix X — it uses NO
    truth/particle_id information, consistent with M5's rules.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_hits, n_features)
        Permitted feature matrix (see `cluster_tracks` docstring).
    k : int
        Which neighbor rank to use (conventionally == min_samples).

    Returns
    -------
    numpy.ndarray, shape (n_hits,)
        The distance from each point to its k-th nearest neighbor
        (excluding itself), SORTED ascending (as is conventional for
        the diagnostic plot).
    """
    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 because a point is its own 0th neighbor
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    kth_distances = distances[:, k]  # column k (0-indexed) = k-th neighbor, excluding self
    return np.sort(kth_distances)

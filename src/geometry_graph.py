"""Deterministic sparse geometry-only candidate graph for TrackML hits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EDGE_COLUMNS = ("source_hit_id", "target_hit_id", "delta_r", "delta_phi", "delta_z", "distance", "candidate_rank")
REQUIRED_HIT_COLUMNS = ("hit_id", "x", "y", "z")


def build_candidate_graph(hits: pd.DataFrame, *, max_radial_step_mm: float = 180.0, max_candidates_per_hit: int = 10) -> pd.DataFrame:
    """Connect each hit to up to k nearest hits within a positive radial step.

    Only hit_id,x,y,z are read. The 180 mm default is a geometry-only
    exploratory window: event 1000's adjacent barrel-layer median-radius
    gaps span about 40–160 mm, so this leaves a 20 mm margin. The top-10
    cap is an explicitly declared sparse candidate budget, not truth-tuned.
    """
    missing = sorted(set(REQUIRED_HIT_COLUMNS) - set(hits.columns))
    if missing:
        raise ValueError(f"hits is missing required geometry columns: {missing}")
    if not np.isfinite(max_radial_step_mm) or max_radial_step_mm <= 0:
        raise ValueError("max_radial_step_mm must be finite and positive")
    if isinstance(max_candidates_per_hit, bool) or int(max_candidates_per_hit) != max_candidates_per_hit or max_candidates_per_hit < 1:
        raise ValueError("max_candidates_per_hit must be a positive integer")

    geom = hits.loc[:, list(REQUIRED_HIT_COLUMNS)].copy()
    if geom.hit_id.isna().any() or geom.hit_id.duplicated().any():
        raise ValueError("hit_id values must be present and unique")
    xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("hit coordinates must be finite")
    empty = pd.DataFrame(columns=EDGE_COLUMNS)
    if len(geom) < 2:
        return empty

    ids = geom.hit_id.to_numpy()
    x, y, z = xyz.T
    radius = np.hypot(x, y)
    phi = np.arctan2(y, x)
    width = float(max_radial_step_mm)
    # Search buckets are at most 20 mm wide and partition the configured
    # radial window into at least ten outward bins. Excluding the same-bin
    # bucket prevents same-layer/self neighbors from consuming top-k slots;
    # sensor-to-sensor edges below this resolution remain a limitation.
    bin_width = min(20.0, width / 10.0)
    bins = np.floor(radius / bin_width).astype(np.int64)
    n_outward_bins = int(np.ceil(width / bin_width))
    cap = int(max_candidates_per_hit)
    idx_by_bin = {}
    tree_by_bin = {}
    for bin_id in np.unique(bins):
        indices = np.flatnonzero(bins == bin_id)
        indices = indices[np.argsort(ids[indices], kind="stable")]
        idx_by_bin[int(bin_id)] = indices
        tree_by_bin[int(bin_id)] = cKDTree(xyz[indices])

    edge_source_parts = []
    edge_target_parts = []
    edge_distance_parts = []
    for source_bin in sorted(idx_by_bin):
        sources = idx_by_bin[source_bin]
        source_parts, target_parts, distance_parts = [], [], []
        for target_bin in range(source_bin + 1, source_bin + n_outward_bins + 1):
            targets = idx_by_bin.get(target_bin)
            tree = tree_by_bin.get(target_bin)
            if targets is None or tree is None:
                continue
            k = min(cap, len(targets))
            distances, local_targets = tree.query(xyz[sources], k=k)
            if k == 1:
                distances = np.asarray(distances)[:, None]
                local_targets = np.asarray(local_targets)[:, None]
            source_rows = np.repeat(np.arange(len(sources), dtype=np.int64), k)
            local_targets = np.asarray(local_targets).reshape(-1)
            flat_distances = np.asarray(distances).reshape(-1)
            valid = np.isfinite(flat_distances) & (local_targets < len(targets))
            source_rows = source_rows[valid]
            target_rows = targets[local_targets[valid].astype(np.int64)]
            flat_distances = flat_distances[valid]
            delta_r = radius[target_rows] - radius[sources[source_rows]]
            valid_radial = (delta_r > 0) & (delta_r <= width)
            if valid_radial.any():
                source_parts.append(sources[source_rows[valid_radial]])
                target_parts.append(target_rows[valid_radial])
                distance_parts.append(flat_distances[valid_radial])

        if not source_parts:
            continue
        source_indices = np.concatenate(source_parts)
        target_indices = np.concatenate(target_parts)
        distances = np.concatenate(distance_parts)
        # Target radial bins are disjoint. Resolve rank ties with target hit ID.
        order = np.lexsort((ids[target_indices], distances, ids[source_indices]))
        source_indices, target_indices, distances = (
            source_indices[order], target_indices[order], distances[order]
        )
        starts = np.r_[0, np.flatnonzero(source_indices[1:] != source_indices[:-1]) + 1]
        group_lengths = np.diff(np.r_[starts, len(source_indices)])
        ranks = np.arange(len(source_indices)) - np.repeat(starts, group_lengths)
        keep = ranks < cap
        edge_source_parts.append(source_indices[keep])
        edge_target_parts.append(target_indices[keep])
        edge_distance_parts.append(distances[keep])

    if not edge_source_parts:
        return empty
    source_indices = np.concatenate(edge_source_parts)
    target_indices = np.concatenate(edge_target_parts)
    distances = np.concatenate(edge_distance_parts)
    # Source indices are ordered by radial bin, not hit ID; restore stable output order.
    order = np.lexsort((ids[target_indices], distances, ids[source_indices]))
    source_indices, target_indices, distances = (
        source_indices[order], target_indices[order], distances[order]
    )
    rank_starts = np.r_[0, np.flatnonzero(source_indices[1:] != source_indices[:-1]) + 1]
    ranks = np.arange(len(source_indices)) - np.repeat(
        rank_starts, np.diff(np.r_[rank_starts, len(source_indices)])
    ) + 1
    raw_dphi = phi[target_indices] - phi[source_indices]
    graph = pd.DataFrame(
        {
            "source_hit_id": ids[source_indices],
            "target_hit_id": ids[target_indices],
            "delta_r": radius[target_indices] - radius[source_indices],
            "delta_phi": np.arctan2(np.sin(raw_dphi), np.cos(raw_dphi)),
            "delta_z": z[target_indices] - z[source_indices],
            "distance": distances,
            "candidate_rank": ranks,
        },
        columns=EDGE_COLUMNS,
    )
    if graph.empty:
        return graph
    return graph.sort_values(["source_hit_id", "candidate_rank"], kind="stable").reset_index(drop=True)

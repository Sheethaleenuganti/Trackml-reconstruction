"""M9.3 geometry-only candidate graph with a wider per-bin query cap.

This is a separate variant. ``src.geometry_graph`` and the M9.1 behavior are
left unchanged. Candidate construction reads only hit_id and x/y/z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.geometry_graph import EDGE_COLUMNS, REQUIRED_HIT_COLUMNS

MAX_RADIAL_STEP_MM = 180.0
RADIAL_BUCKET_WIDTH_MM = 18.0
OUTWARD_BUCKETS = 10
PER_BIN_QUERY_CAP = 20
FINAL_CANDIDATE_CAP = 10


def build_candidate_graph_m9_3(hits: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic M9.3 candidates; only the per-bin query cap differs.

    The unchanged M9.1 final cap, radial constraints, search buckets, XYZ
    metric, outward direction, and final distance/target-ID ordering are kept.
    """
    missing = sorted(set(REQUIRED_HIT_COLUMNS) - set(hits.columns))
    if missing:
        raise ValueError(f"hits is missing required geometry columns: {missing}")
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
    bins = np.floor(radius / RADIAL_BUCKET_WIDTH_MM).astype(np.int64)
    idx_by_bin: dict[int, np.ndarray] = {}
    tree_by_bin: dict[int, cKDTree] = {}
    for bin_id in np.unique(bins):
        indices = np.flatnonzero(bins == bin_id)
        indices = indices[np.argsort(ids[indices], kind="stable")]
        idx_by_bin[int(bin_id)] = indices
        tree_by_bin[int(bin_id)] = cKDTree(xyz[indices])

    src_parts, dst_parts, dist_parts = [], [], []
    for source_bin in sorted(idx_by_bin):
        sources = idx_by_bin[source_bin]
        source_parts, target_parts, distance_parts = [], [], []
        for target_bin in range(source_bin + 1, source_bin + OUTWARD_BUCKETS + 1):
            targets = idx_by_bin.get(target_bin)
            tree = tree_by_bin.get(target_bin)
            if targets is None or tree is None:
                continue
            k = min(PER_BIN_QUERY_CAP, len(targets))
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
            valid_radial = (delta_r > 0) & (delta_r <= MAX_RADIAL_STEP_MM)
            if valid_radial.any():
                source_parts.append(sources[source_rows[valid_radial]])
                target_parts.append(target_rows[valid_radial])
                distance_parts.append(flat_distances[valid_radial])

        if not source_parts:
            continue
        source_indices = np.concatenate(source_parts)
        target_indices = np.concatenate(target_parts)
        distances = np.concatenate(distance_parts)
        order = np.lexsort((ids[target_indices], distances, ids[source_indices]))
        source_indices, target_indices, distances = (
            source_indices[order], target_indices[order], distances[order]
        )
        starts = np.r_[0, np.flatnonzero(source_indices[1:] != source_indices[:-1]) + 1]
        group_lengths = np.diff(np.r_[starts, len(source_indices)])
        ranks = np.arange(len(source_indices)) - np.repeat(starts, group_lengths)
        keep = ranks < FINAL_CANDIDATE_CAP
        src_parts.append(source_indices[keep])
        dst_parts.append(target_indices[keep])
        dist_parts.append(distances[keep])

    if not src_parts:
        return empty
    source_indices = np.concatenate(src_parts)
    target_indices = np.concatenate(dst_parts)
    distances = np.concatenate(dist_parts)
    order = np.lexsort((ids[target_indices], distances, ids[source_indices]))
    source_indices, target_indices, distances = (
        source_indices[order], target_indices[order], distances[order]
    )
    starts = np.r_[0, np.flatnonzero(source_indices[1:] != source_indices[:-1]) + 1]
    ranks = np.arange(len(source_indices)) - np.repeat(
        starts, np.diff(np.r_[starts, len(source_indices)])
    ) + 1
    raw_dphi = phi[target_indices] - phi[source_indices]
    graph = pd.DataFrame({
        "source_hit_id": ids[source_indices], "target_hit_id": ids[target_indices],
        "delta_r": radius[target_indices] - radius[source_indices],
        "delta_phi": np.arctan2(np.sin(raw_dphi), np.cos(raw_dphi)),
        "delta_z": z[target_indices] - z[source_indices], "distance": distances,
        "candidate_rank": ranks,
    }, columns=EDGE_COLUMNS)
    return graph.sort_values(["source_hit_id", "candidate_rank"], kind="stable").reset_index(drop=True)


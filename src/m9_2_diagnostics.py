"""Post-hoc diagnostics for missed M9.1 true consecutive edges.

Candidate graphs are constructed by the unchanged M9.1 geometry-only builder.
This module receives an already-built graph before it receives truth.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MAX_RADIAL_STEP_MM = 180.0
BUCKET_WIDTH_MM = 18.0
OUTWARD_BUCKETS = 10
PER_BIN_K = 10
FINAL_K = 10
METRICS = ("delta_r", "abs_delta_phi", "abs_delta_z", "distance_xyz")


def true_consecutive_edges(hits: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Match M9.1's post-hoc truth definition; never used to build candidates."""
    geom = hits[["hit_id", "x", "y"]].copy()
    geom["r"] = np.hypot(geom.x.to_numpy(float), geom.y.to_numpy(float))
    joined = truth[["hit_id", "particle_id"]].merge(
        geom[["hit_id", "r"]], on="hit_id", how="inner", validate="one_to_one"
    )
    joined = joined.loc[joined.particle_id != 0].sort_values(
        ["particle_id", "r", "hit_id"], kind="stable"
    )
    groups = joined.groupby("particle_id", sort=False)
    joined["target_hit_id"] = groups.hit_id.shift(-1)
    joined["target_r"] = groups.r.shift(-1)
    result = joined.loc[
        joined.target_hit_id.notna() & (joined.target_r > joined.r),
        ["particle_id", "hit_id", "target_hit_id"],
    ].rename(columns={"hit_id": "source_hit_id"})
    result["target_hit_id"] = result.target_hit_id.astype(hits.hit_id.dtype)
    return result.reset_index(drop=True)


def _wrapped_phi(phi_source: np.ndarray, phi_target: np.ndarray) -> np.ndarray:
    difference = phi_target - phi_source
    return np.arctan2(np.sin(difference), np.cos(difference))


def query_neighborhood_presence(
    hits: pd.DataFrame, true_edges: pd.DataFrame, *, chunk_size: int = 8192
) -> np.ndarray:
    """Whether each eligible true target is in its exact M9.1 target-bin kNN.

    Only post-hoc eligible missed pairs should be passed. Trees contain hit
    geometry only, targets are ordered by hit_id as in M9.1, and query k is 10.
    """
    if true_edges.empty:
        return np.zeros(0, dtype=bool)
    geom = hits[["hit_id", "x", "y", "z"]].copy()
    ids = geom.hit_id.to_numpy()
    xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float64)
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    bins = np.floor(radius / BUCKET_WIDTH_MM).astype(np.int64)
    id_to_idx = {hit_id: i for i, hit_id in enumerate(ids)}
    source_idx = true_edges.source_hit_id.map(id_to_idx).to_numpy(dtype=np.int64)
    target_idx = true_edges.target_hit_id.map(id_to_idx).to_numpy(dtype=np.int64)
    target_bins = bins[target_idx]
    present = np.zeros(len(true_edges), dtype=bool)

    for bucket in np.unique(target_bins):
        rows = np.flatnonzero(target_bins == bucket)
        target_indices = np.flatnonzero(bins == bucket)
        target_indices = target_indices[np.argsort(ids[target_indices], kind="stable")]
        k = min(PER_BIN_K, len(target_indices))
        if not k:
            continue
        tree = cKDTree(xyz[target_indices])
        local_target = {int(index): pos for pos, index in enumerate(target_indices)}
        for start in range(0, len(rows), chunk_size):
            batch_rows = rows[start : start + chunk_size]
            _, local_neighbors = tree.query(xyz[source_idx[batch_rows]], k=k)
            local_neighbors = np.asarray(local_neighbors)
            if k == 1:
                local_neighbors = local_neighbors[:, None]
            desired = np.fromiter(
                (local_target[int(index)] for index in target_idx[batch_rows]),
                dtype=np.int64,
                count=len(batch_rows),
            )
            present[batch_rows] = np.any(local_neighbors == desired[:, None], axis=1)
    return present


def diagnose_event(
    hits: pd.DataFrame, truth: pd.DataFrame, candidate_edges: pd.DataFrame
) -> pd.DataFrame:
    """Return post-hoc true-edge rows with geometry, rank and miss causes."""
    edges = true_consecutive_edges(hits, truth)
    if edges.empty:
        return edges.assign(
            delta_r=pd.Series(dtype=float), abs_delta_phi=pd.Series(dtype=float),
            abs_delta_z=pd.Series(dtype=float), distance_xyz=pd.Series(dtype=float),
            recovered=pd.Series(dtype=bool), candidate_rank=pd.Series(dtype=float),
            source_bucket=pd.Series(dtype=int), target_bucket=pd.Series(dtype=int),
            bucket_difference=pd.Series(dtype=int), within_bucket_search=pd.Series(dtype=bool),
            within_radial_step=pd.Series(dtype=bool), in_queried_neighborhood=pd.Series(dtype=bool),
            rank_category=pd.Series(dtype=str), primary_cause=pd.Series(dtype=str),
        )

    geom = hits[["hit_id", "x", "y", "z"]].copy()
    ids = geom.hit_id.to_numpy()
    xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float64)
    index = {hit_id: i for i, hit_id in enumerate(ids)}
    source = edges.source_hit_id.map(index).to_numpy(dtype=np.int64)
    target = edges.target_hit_id.map(index).to_numpy(dtype=np.int64)
    radii = np.hypot(xyz[:, 0], xyz[:, 1])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    dphi = _wrapped_phi(phi[source], phi[target])
    edges["delta_r"] = radii[target] - radii[source]
    edges["abs_delta_phi"] = np.abs(dphi)
    edges["abs_delta_z"] = np.abs(xyz[target, 2] - xyz[source, 2])
    edges["distance_xyz"] = np.linalg.norm(xyz[target] - xyz[source], axis=1)
    edges["source_bucket"] = np.floor(radii[source] / BUCKET_WIDTH_MM).astype(np.int64)
    edges["target_bucket"] = np.floor(radii[target] / BUCKET_WIDTH_MM).astype(np.int64)
    edges["bucket_difference"] = edges.target_bucket - edges.source_bucket
    edges["within_bucket_search"] = edges.bucket_difference.between(1, OUTWARD_BUCKETS)
    edges["within_radial_step"] = edges.delta_r.between(0, MAX_RADIAL_STEP_MM, inclusive="right")

    candidate_view = candidate_edges[["source_hit_id", "target_hit_id", "candidate_rank"]].drop_duplicates(
        ["source_hit_id", "target_hit_id"]
    )
    edges = edges.merge(candidate_view, on=["source_hit_id", "target_hit_id"], how="left", validate="one_to_one")
    edges["recovered"] = edges.candidate_rank.notna()
    eligible = edges.within_bucket_search & edges.within_radial_step
    needs_query = eligible & ~edges.recovered
    edges["in_queried_neighborhood"] = False
    if needs_query.any():
        subset = edges.loc[needs_query, ["source_hit_id", "target_hit_id"]]
        edges.loc[needs_query, "in_queried_neighborhood"] = query_neighborhood_presence(hits, subset)

    edges["rank_category"] = "outside_search_region"
    recovered = edges.recovered
    edges.loc[recovered, "rank_category"] = "rank_" + edges.loc[recovered, "candidate_rank"].astype(int).astype(str)
    topk = needs_query & edges.in_queried_neighborhood
    edges.loc[topk, "rank_category"] = ">10"
    geom_miss = needs_query & ~edges.in_queried_neighborhood
    edges.loc[geom_miss, "rank_category"] = "not_present_in_queried_neighborhood"

    edges["radial_step_exclusion"] = ~edges.within_radial_step
    edges["bucket_search_exclusion"] = ~edges.within_bucket_search
    edges["top_k_truncation"] = topk
    edges["geometric_neighborhood_miss"] = geom_miss
    edges["primary_cause"] = "recovered"
    missed = ~edges.recovered
    edges.loc[missed & edges.radial_step_exclusion, "primary_cause"] = "radial_step_exclusion"
    edges.loc[missed & ~edges.radial_step_exclusion & edges.bucket_search_exclusion, "primary_cause"] = "bucket_search_exclusion"
    edges.loc[missed & topk, "primary_cause"] = "top_k_truncation"
    edges.loc[missed & geom_miss, "primary_cause"] = "geometric_neighborhood_miss"
    edges.loc[missed & (edges.primary_cause == "recovered"), "primary_cause"] = "other_unclassified"
    return edges


def distribution_stats(values: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "maximum": None}
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)), "maximum": float(np.max(values)),
    }


def summarize_event(diagnostic: pd.DataFrame, event_id: str) -> dict:
    total = len(diagnostic)
    recovered = int(diagnostic.recovered.sum()) if total else 0
    missed = total - recovered
    denom = missed or 1
    return {
        "event_id": event_id, "true_edge_count": total, "recovered_edge_count": recovered,
        "missed_edge_count": missed, "recall": recovered / total if total else 0.0,
        "missed_delta_r_gt_180_count": int((~diagnostic.recovered & (diagnostic.delta_r > MAX_RADIAL_STEP_MM)).sum()),
        "missed_delta_r_gt_180_fraction": int((~diagnostic.recovered & (diagnostic.delta_r > MAX_RADIAL_STEP_MM)).sum()) / denom,
        "missed_top_k_truncation_count": int((~diagnostic.recovered & diagnostic.top_k_truncation).sum()),
        "missed_top_k_truncation_fraction": int((~diagnostic.recovered & diagnostic.top_k_truncation).sum()) / denom,
        "missed_outside_bucket_search_count": int((~diagnostic.recovered & diagnostic.bucket_search_exclusion).sum()),
        "missed_outside_bucket_search_fraction": int((~diagnostic.recovered & diagnostic.bucket_search_exclusion).sum()) / denom,
        "missed_geometric_neighborhood_count": int((~diagnostic.recovered & diagnostic.geometric_neighborhood_miss).sum()),
        "missed_other_count": int((~diagnostic.recovered & (diagnostic.primary_cause == "other_unclassified")).sum()),
    }


def mechanism_counts(frame: pd.DataFrame) -> dict:
    missed = frame.loc[~frame.recovered]
    causes = Counter(missed.primary_cause)
    return {
        "missed_edge_count": int(len(missed)),
        "overlapping": {
            "radial_step_exclusion": int(missed.radial_step_exclusion.sum()),
            "bucket_search_exclusion": int(missed.bucket_search_exclusion.sum()),
            "top_k_truncation": int(missed.top_k_truncation.sum()),
            "geometric_neighborhood_miss": int(missed.geometric_neighborhood_miss.sum()),
            "other_unclassified": int((missed.primary_cause == "other_unclassified").sum()),
        },
        "primary": dict(causes),
    }


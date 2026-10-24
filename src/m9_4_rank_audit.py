"""Geometry-only M9.4 query-stage audit and post-hoc true-edge tracing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STAGE_COLUMNS = (
    "source_hit_id", "target_hit_id", "source_bucket", "target_bucket",
    "per_bin_rank", "distance", "delta_r", "radial_step_pass",
    "merged_rank", "selected_final10",
)


def stable_query_top_k(
    tree: cKDTree,
    target_indices: np.ndarray,
    target_ids: np.ndarray,
    target_xyz: np.ndarray,
    query_xyz: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cKDTree top-k, sorting returned ties by target ID.

    Target-bin rows are pre-sorted by hit ID and cKDTree is queried with one
    worker. cKDTree determines membership at a tied k-th-distance cutoff;
    returned equal-distance candidates are then ordered by target ID. This
    matches the frozen query mechanism while making recorded ranks repeatable
    under the recorded SciPy version.
    """
    if len(target_indices) == 0:
        return np.empty((len(query_xyz), 0), dtype=np.int64), np.empty((len(query_xyz), 0), dtype=float)
    k = min(int(k), len(target_indices))
    distances, locals_ = tree.query(query_xyz, k=k, workers=1)
    distances = np.asarray(distances, dtype=float)
    locals_ = np.asarray(locals_, dtype=np.int64)
    if k == 1:
        distances = distances[:, None]
        locals_ = locals_[:, None]
    order = np.lexsort((target_ids[target_indices[locals_]], distances), axis=1)
    out_indices = np.take_along_axis(locals_, order, axis=1)
    out_distances = np.take_along_axis(distances, order, axis=1)
    return out_indices, out_distances


def rank_merged_candidates(candidates: pd.DataFrame, *, final_cap: int = 10) -> pd.DataFrame:
    """Rank radial-filter-passing query candidates across bins per source."""
    if candidates.empty:
        result = candidates.copy()
        result["merged_rank"] = pd.Series(dtype="int64")
        result["selected_final10"] = pd.Series(dtype=bool)
        return result
    result = candidates.sort_values(
        ["source_hit_id", "distance", "target_hit_id"], kind="stable"
    ).copy()
    result["merged_rank"] = result.groupby("source_hit_id", sort=False).cumcount() + 1
    result["selected_final10"] = result.merged_rank <= final_cap
    return result


def build_query_stages(
    hits: pd.DataFrame,
    *,
    query_cap: int = 20,
    max_radial_step_mm: float = 180.0,
    bucket_width_mm: float = 18.0,
    outward_buckets: int = 10,
    final_cap: int = 10,
) -> pd.DataFrame:
    """Build full geometry-only query20 stage records, then merged ranks.

    The returned table includes query results even when they fail the radial
    filter. No truth-like columns are read. Events must be processed one at a
    time because this contains up to query_cap candidates per searched bin.
    """
    required = {"hit_id", "x", "y", "z"}
    missing = sorted(required - set(hits.columns))
    if missing:
        raise ValueError(f"hits is missing required geometry columns: {missing}")
    geom = hits.loc[:, ["hit_id", "x", "y", "z"]].copy()
    if geom.hit_id.isna().any() or geom.hit_id.duplicated().any():
        raise ValueError("hit_id values must be present and unique")
    xyz = geom[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("hit coordinates must be finite")
    ids = geom.hit_id.to_numpy()
    radii = np.hypot(xyz[:, 0], xyz[:, 1])
    bins = np.floor(radii / bucket_width_mm).astype(np.int64)
    bins_by_id: dict[int, np.ndarray] = {}
    trees: dict[int, cKDTree] = {}
    for bucket in np.unique(bins):
        indices = np.flatnonzero(bins == bucket)
        indices = indices[np.argsort(ids[indices], kind="stable")]
        bins_by_id[int(bucket)] = indices
        trees[int(bucket)] = cKDTree(xyz[indices])

    records = []
    for source_bucket in sorted(bins_by_id):
        sources = bins_by_id[source_bucket]
        source_records = []
        for target_bucket in range(source_bucket + 1, source_bucket + outward_buckets + 1):
            target_indices = bins_by_id.get(target_bucket)
            tree = trees.get(target_bucket)
            if target_indices is None or tree is None:
                continue
            selected_local, selected_distance = stable_query_top_k(
                tree, target_indices, ids, xyz, xyz[sources], min(query_cap, len(target_indices))
            )
            k = selected_local.shape[1]
            if not k:
                continue
            source_rows = np.repeat(np.arange(len(sources), dtype=np.int64), k)
            target_rows = target_indices[selected_local.reshape(-1)]
            source_idx = sources[source_rows]
            distance = selected_distance.reshape(-1)
            delta_r = radii[target_rows] - radii[source_idx]
            passes = (delta_r > 0) & (delta_r <= max_radial_step_mm)
            source_records.append(pd.DataFrame({
                "source_hit_id": ids[source_idx], "target_hit_id": ids[target_rows],
                "source_bucket": np.full(len(source_idx), source_bucket, dtype=np.int32),
                "target_bucket": np.full(len(source_idx), target_bucket, dtype=np.int32),
                "per_bin_rank": np.tile(np.arange(1, k + 1, dtype=np.uint8), len(sources)),
                "distance": distance.astype(np.float64), "delta_r": delta_r,
                "radial_step_pass": passes,
            }))
        if source_records:
            # Final ranking is independent for each source hit, so finish it
            # within this source bucket. This avoids a second event-sized
            # candidate table and a large many-column pandas merge.
            source_frame = pd.concat(source_records, ignore_index=True)
            passing = np.flatnonzero(source_frame.radial_step_pass.to_numpy())
            merged_rank = np.full(len(source_frame), np.nan, dtype=np.float32)
            selected_final10 = np.zeros(len(source_frame), dtype=bool)
            if len(passing):
                source_values = source_frame.source_hit_id.to_numpy()
                target_values = source_frame.target_hit_id.to_numpy()
                distances = source_frame.distance.to_numpy()
                order = np.lexsort((target_values[passing], distances[passing], source_values[passing]))
                ranked = passing[order]
                ranked_sources = source_values[ranked]
                starts = np.r_[0, np.flatnonzero(ranked_sources[1:] != ranked_sources[:-1]) + 1]
                lengths = np.diff(np.r_[starts, len(ranked)])
                ranks = np.arange(len(ranked), dtype=np.int32) - np.repeat(starts, lengths) + 1
                merged_rank[ranked] = ranks
                selected_final10[ranked] = ranks <= final_cap
            source_frame["merged_rank"] = merged_rank
            source_frame["selected_final10"] = selected_final10
            records.append(source_frame)

    if not records:
        return pd.DataFrame(columns=STAGE_COLUMNS)
    return pd.concat(records, ignore_index=True).reindex(columns=STAGE_COLUMNS)


def trace_true_edges(
    hits: pd.DataFrame,
    true_edges: pd.DataFrame,
    query_stages: pd.DataFrame,
    baseline_edges: pd.DataFrame,
    *,
    bucket_width_mm: float = 18.0,
    outward_buckets: int = 10,
    max_radial_step_mm: float = 180.0,
) -> pd.DataFrame:
    """Join truth after all geometry stages are fixed and classify stage flow."""
    result = true_edges[["source_hit_id", "target_hit_id"]].copy()
    lookup = hits.set_index("hit_id")[["x", "y"]]
    src = lookup.loc[result.source_hit_id.to_numpy()]
    dst = lookup.loc[result.target_hit_id.to_numpy()]
    source_r = np.hypot(src.x.to_numpy(float), src.y.to_numpy(float))
    target_r = np.hypot(dst.x.to_numpy(float), dst.y.to_numpy(float))
    result["delta_r"] = target_r - source_r
    result["source_bucket"] = np.floor(source_r / bucket_width_mm).astype(np.int64)
    result["target_bucket"] = np.floor(target_r / bucket_width_mm).astype(np.int64)
    result["bucket_difference"] = result.target_bucket - result.source_bucket
    result["target_bin_searched"] = result.bucket_difference.between(1, outward_buckets)
    result["radial_step_pass"] = result.delta_r.between(0, max_radial_step_mm, inclusive="right")

    query_view = query_stages[[
        "source_hit_id", "target_hit_id", "per_bin_rank", "merged_rank", "selected_final10"
    ]]
    result = result.merge(query_view, on=["source_hit_id", "target_hit_id"], how="left", validate="one_to_one")
    result["query20_present"] = result.per_bin_rank.notna()
    result["per_bin_rank_category"] = "not_searched"
    searched = result.target_bin_searched
    present = searched & result.query20_present
    result.loc[present & (result.per_bin_rank <= 10), "per_bin_rank_category"] = "rank_1_10"
    result.loc[present & result.per_bin_rank.between(11, 20), "per_bin_rank_category"] = "rank_11_20"
    result.loc[searched & ~result.query20_present, "per_bin_rank_category"] = ">20"
    result["per_bin_top10_present"] = searched & result.per_bin_rank.between(1, 10)
    result["rank_11_20"] = present & result.per_bin_rank.between(11, 20)
    result["rank_11_20_radial_pass"] = result.rank_11_20 & result.radial_step_pass
    result["in_merged_pool"] = result.query20_present & result.radial_step_pass
    result["survives_final10"] = result.in_merged_pool & result.selected_final10.fillna(False)

    baseline_view = baseline_edges[["source_hit_id", "target_hit_id"]].drop_duplicates()
    baseline_view["baseline_recovered"] = True
    result = result.merge(baseline_view, on=["source_hit_id", "target_hit_id"], how="left", validate="one_to_one")
    result["baseline_recovered"] = result.baseline_recovered.fillna(False).astype(bool)
    result["baseline_missed"] = ~result.baseline_recovered

    # Mutually exclusive explanation for baseline misses. The independent
    # stage flags above remain available for overlapping analyses.
    result["baseline_miss_stage"] = "not_missed_baseline"
    result.loc[result.baseline_missed, "baseline_miss_stage"] = "would_survive_extended_final10"
    result.loc[result.baseline_missed & ~result.target_bin_searched, "baseline_miss_stage"] = "target_bin_not_searched"
    result.loc[result.baseline_missed & result.target_bin_searched & ~result.query20_present, "baseline_miss_stage"] = "rank_above_20"
    result.loc[result.baseline_missed & result.query20_present & ~result.radial_step_pass, "baseline_miss_stage"] = "radial_filter_failed"
    result.loc[result.baseline_missed & result.in_merged_pool & ~result.survives_final10, "baseline_miss_stage"] = "final_top10_rejected"
    return result


def stage_counts(traced: pd.DataFrame) -> dict[str, int]:
    return {
        "true_edges_available": int(len(traced)),
        "baseline_recovered": int(traced.baseline_recovered.sum()),
        "baseline_missed": int(traced.baseline_missed.sum()),
        "target_bin_searched": int(traced.target_bin_searched.sum()),
        "target_bin_not_searched": int((~traced.target_bin_searched).sum()),
        "per_bin_rank_1_10": int((traced.per_bin_rank_category == "rank_1_10").sum()),
        "per_bin_rank_11_20": int((traced.per_bin_rank_category == "rank_11_20").sum()),
        "per_bin_rank_above_20": int((traced.per_bin_rank_category == ">20").sum()),
        "rank_11_20_radial_pass": int(traced.rank_11_20_radial_pass.sum()),
        "rank_11_20_radial_fail": int((traced.rank_11_20 & ~traced.radial_step_pass).sum()),
        "query20_candidates_radial_pass": int(traced.in_merged_pool.sum()),
        "query20_candidates_final10": int(traced.survives_final10.sum()),
        "query20_candidates_rejected_final10": int((traced.in_merged_pool & ~traced.survives_final10).sum()),
        "baseline_miss_target_bin_not_searched": int((traced.baseline_miss_stage == "target_bin_not_searched").sum()),
        "baseline_miss_rank_above_20": int((traced.baseline_miss_stage == "rank_above_20").sum()),
        "baseline_miss_radial_filter_failed": int((traced.baseline_miss_stage == "radial_filter_failed").sum()),
        "baseline_miss_final_top10_rejected": int((traced.baseline_miss_stage == "final_top10_rejected").sum()),
        "baseline_miss_would_survive_extended_final10": int((traced.baseline_miss_stage == "would_survive_extended_final10").sum()),
        "baseline_missed_rank_11_20": int((traced.baseline_missed & traced.rank_11_20).sum()),
        "baseline_missed_rank_11_20_radial_pass": int((traced.baseline_missed & traced.rank_11_20_radial_pass).sum()),
        "baseline_missed_rank_11_20_final10_survivors": int((traced.baseline_missed & traced.rank_11_20 & traced.survives_final10).sum()),
    }


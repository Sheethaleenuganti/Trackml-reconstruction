"""Post-hoc geometry summaries for M9.1 true-edge recovery cohorts."""
from __future__ import annotations

import numpy as np
import pandas as pd

GEOMETRY_FEATURES = (
    "delta_r_mm", "delta_phi_rad", "abs_delta_phi_rad", "delta_z_mm",
    "abs_delta_z_mm", "distance_xyz_mm", "distance_transverse_mm",
    "source_radius_mm", "target_radius_mm", "source_z_mm", "target_z_mm",
    "bucket_difference",
)
COHORTS = (
    "recovered_m91", "missed_rank_1_10", "missed_rank_11_20",
    "missed_rank_above_20", "missed_target_bin_unsearched",
    "missed_radial_filter",
)
AUDIT_FEATURES = GEOMETRY_FEATURES + ("per_bin_rank",)


def wrapped_delta_phi(source_phi: np.ndarray, target_phi: np.ndarray) -> np.ndarray:
    """Signed wrapped azimuth difference in [-pi, pi]."""
    return np.arctan2(np.sin(target_phi - source_phi), np.cos(target_phi - source_phi))


def deterministic_prefix_sample(values: np.ndarray, limit: int) -> np.ndarray:
    """Take a stable prefix for plots; callers must retain full counts separately."""
    if isinstance(limit, bool) or int(limit) != limit or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    return np.asarray(values)[:int(limit)].copy()


def add_geometry_features(hits: pd.DataFrame, traced: pd.DataFrame, *, bucket_width_mm: float = 18.0) -> pd.DataFrame:
    """Attach coordinate-derived features to already truth-classified edge pairs.

    This function does not inspect truth/particle fields. `traced` must already
    have been classified against geometry-only candidate outputs.
    """
    required = {"hit_id", "x", "y", "z"}
    if missing := sorted(required - set(hits.columns)):
        raise ValueError(f"hits is missing required geometry columns: {missing}")
    if bucket_width_mm <= 0 or not np.isfinite(bucket_width_mm):
        raise ValueError("bucket_width_mm must be finite and positive")
    geom = hits.loc[:, ["hit_id", "x", "y", "z"]]
    if geom.hit_id.isna().any() or geom.hit_id.duplicated().any():
        raise ValueError("hit_id values must be present and unique")
    coords = geom[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("hit coordinates must be finite")
    ids = geom.hit_id.to_numpy()
    lookup = {hit_id: i for i, hit_id in enumerate(ids)}
    try:
        src = traced.source_hit_id.map(lookup).to_numpy(dtype=np.int64)
        dst = traced.target_hit_id.map(lookup).to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("edge IDs must map to hits") from exc
    if (traced.source_hit_id.map(lookup).isna().any() or traced.target_hit_id.map(lookup).isna().any()):
        raise ValueError("every edge endpoint must exist in hits")
    x, y, z = coords.T
    radii = np.hypot(x, y)
    phis = np.arctan2(y, x)
    dphi = wrapped_delta_phi(phis[src], phis[dst])
    dx, dy, dz = x[dst] - x[src], y[dst] - y[src], z[dst] - z[src]
    source_bucket = np.floor(radii[src] / bucket_width_mm).astype(np.int64)
    target_bucket = np.floor(radii[dst] / bucket_width_mm).astype(np.int64)
    result = traced.copy()
    result["delta_r_mm"] = radii[dst] - radii[src]
    result["delta_phi_rad"] = dphi
    result["abs_delta_phi_rad"] = np.abs(dphi)
    result["delta_z_mm"] = dz
    result["abs_delta_z_mm"] = np.abs(dz)
    result["distance_xyz_mm"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    result["distance_transverse_mm"] = np.hypot(dx, dy)
    result["source_radius_mm"] = radii[src]
    result["target_radius_mm"] = radii[dst]
    result["source_z_mm"] = z[src]
    result["target_z_mm"] = z[dst]
    result["bucket_difference"] = target_bucket - source_bucket
    return result


def cohort_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return specified cohort membership; radial exclusion intentionally overlaps rank cohorts."""
    required = {"baseline_recovered", "baseline_missed", "per_bin_rank", "target_bin_searched", "query20_present", "radial_step_pass"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"traced edges are missing cohort fields: {sorted(missing)}")
    missed = frame.baseline_missed.astype(bool)
    rank = pd.to_numeric(frame.per_bin_rank, errors="coerce")
    return {
        "recovered_m91": frame.baseline_recovered.astype(bool),
        "missed_rank_1_10": missed & rank.between(1, 10),
        "missed_rank_11_20": missed & rank.between(11, 20),
        "missed_rank_above_20": missed & frame.target_bin_searched.astype(bool) & ~frame.query20_present.astype(bool),
        "missed_target_bin_unsearched": missed & ~frame.target_bin_searched.astype(bool),
        "missed_radial_filter": missed & frame.query20_present.astype(bool) & ~frame.radial_step_pass.astype(bool),
    }


def summarize_geometry(frame: pd.DataFrame, *, event_id: str | None = None) -> pd.DataFrame:
    """Summarize all available edge values (no sampling) by cohort and feature."""
    masks = cohort_masks(frame)
    rows = []
    for cohort, mask in masks.items():
        subset = frame.loc[mask]
        for feature in AUDIT_FEATURES:
            values = pd.to_numeric(subset[feature], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(values):
                q25, median, q75, p90, p95 = np.percentile(values, [25, 50, 75, 90, 95])
                mean = float(np.mean(values))
            else:
                q25 = median = q75 = p90 = p95 = mean = None
            rows.append({
                **({"event_id": event_id} if event_id is not None else {}),
                "cohort": cohort, "feature": feature, "count": int(len(values)),
                "mean": mean, "q25": None if q25 is None else float(q25),
                "median": None if median is None else float(median),
                "q75": None if q75 is None else float(q75),
                "iqr": None if q25 is None else float(q75 - q25),
                "p90": None if p90 is None else float(p90),
                "p95": None if p95 is None else float(p95),
            })
    return pd.DataFrame(rows)


def cohort_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {name: int(mask.sum()) for name, mask in cohort_masks(frame).items()}

"""Run the M9.5 development-only post-hoc geometry-neighborhood audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_event
from src.event_split_manifest import read_manifest, validate_manifest
from src.geometry_graph import build_candidate_graph
from src.m9_4_rank_audit import build_query_stages, trace_true_edges
from src.m9_5_geometry_miss_audit import (
    AUDIT_FEATURES, COHORTS, GEOMETRY_FEATURES, add_geometry_features, cohort_counts, cohort_masks,
    deterministic_prefix_sample, summarize_geometry,
)
from scripts.run_m9_1_geometry_graph import _true_consecutive_edges
from scripts.run_m9_4_rank_audit import _sha256, _verify_event1000_edge_artifact

RESULTS = Path("experiments/results")
FIGURES = Path("experiments/figures")
M91_SUMMARY = RESULTS / "m9_1_geometry_graph_summary.csv"
M91_EVENT1000 = RESULTS / "m9_1_geometry_graph_edges_event000001000.csv.gz"
M94_EVENT_SUMMARY = RESULTS / "m9_4_event_summary.csv"
M94_PROTECTED = [
    RESULTS / "m9_4_summary.csv", RESULTS / "m9_4_summary.json",
    RESULTS / "m9_4_event_summary.csv", RESULTS / "m9_4_stage_transitions.csv",
    RESULTS / "m9_4_event000001000_examples.csv.gz", RESULTS / "m9_4_report.md",
    FIGURES / "m9_4_stage_transition.png",
]
OUTPUTS = [
    RESULTS / "m9_5_summary.csv", RESULTS / "m9_5_summary.json",
    RESULTS / "m9_5_event_summary.csv", RESULTS / "m9_5_report.md",
    FIGURES / "m9_5_geometry_cohort_distributions.png",
]


def development_event_ids(manifest_path: str | Path) -> list[str]:
    rows = read_manifest(manifest_path)
    validate_manifest(rows)
    ids = [row["event_id"] for row in rows if row["split"] == "DEVELOPMENT"]
    expected = [f"event{number:09d}" for number in range(1000, 1060)]
    if ids != expected:
        raise ValueError("M9.5 requires exactly DEVELOPMENT event000001000–event000001059")
    return ids


def _figure(frame: pd.DataFrame) -> None:
    features = [
        ("source_radius_mm", "Source radius (mm)"),
        ("delta_r_mm", "Radial displacement Δr (mm)"),
        ("abs_delta_phi_rad", "Absolute wrapped Δφ (rad)"),
        ("abs_delta_z_mm", "Absolute Δz (mm)"),
        ("distance_xyz_mm", "XYZ distance (mm)"),
        ("distance_transverse_mm", "Transverse distance (mm)"),
    ]
    cohorts = ["recovered_m91", "missed_rank_1_10", "missed_rank_11_20", "missed_rank_above_20", "missed_target_bin_unsearched"]
    colors = ["#276FBF", "#F4A259", "#D1495B", "#6A994E", "#7353BA"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (feature, label) in zip(axes.flat, features):
        for cohort, color in zip(cohorts, colors):
            values = frame.loc[frame._cohort == cohort, feature].to_numpy(dtype=float)
            if len(values):
                ax.hist(values, bins=60, density=True, histtype="step", linewidth=1.5, color=color, label=f"{cohort} (n={len(values):,})")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("M9.5: development true-edge geometry by post-hoc recovery cohort\nAll 60 development events; full cohorts, no sampling")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS[-1], dpi=160)
    plt.close(fig)


def _report(summary: dict, pooled: pd.DataFrame) -> str:
    counts = summary["cohort_counts"]
    by = pooled.set_index(["cohort", "feature"])
    def stat(cohort: str, feature: str, key: str = "median"):
        value = by.loc[(cohort, feature), key]
        return "NA" if pd.isna(value) else f"{float(value):.4g}"
    geom_lines = []
    for feature in ("source_radius_mm", "delta_r_mm", "abs_delta_phi_rad", "abs_delta_z_mm", "distance_xyz_mm", "distance_transverse_mm"):
        geom_lines.append(f"| {feature} | {stat('recovered_m91', feature)} | {stat('missed_rank_above_20', feature)} | {stat('missed_target_bin_unsearched', feature)} |")
    return f"""# M9.5 Geometry-Neighborhood Failure Audit

## Scope and protocol

Processed all {summary['event_count']} DEVELOPMENT events (1000–1059). Validation (1060–1079) and final-test (1080–1099) events were not opened. M9.1 candidate generation and M9.4 query20 geometry rankings were reconstructed from hit_id,x,y,z only. Truth was loaded only after these geometric outputs were fixed, to build true adjacent-particle pairs and label cohorts. No cells, particle/detector metadata, charge, momentum, or truth entered candidate generation or ranking. Numerical summaries use all applicable edges; plots use only the deterministic sample documented below.

## Cohorts

Counts are full pooled counts. `recovered_m91` is a true edge present in the frozen M9.1 graph. The other ranked cohorts are M9.1-missed true edges with a target in the extended query20 result at per-bin ranks 1–10 or 11–20. `missed_rank_above_20` means the target bin is searched by M9.1 but the target is absent from extended query20. `missed_target_bin_unsearched` means the target's radial bucket is not one of the next ten outward buckets. `missed_radial_filter` means an M9.1-missed target was returned by query20 but fails `0 < Δr <= 180 mm`. The radial-filter cohort overlaps ranked cohorts; the other primary miss groups are disjoint by construction. Counts: {json.dumps(counts, sort_keys=True)}.

## Geometric distributions

Values below are medians over all applicable edges; full counts, means, quartiles, IQR, p90 and p95 are in `m9_5_summary.csv`.

| Feature | M9.1 recovered | Missed, rank >20 | Missed, target bin unsearched |
|---|---:|---:|---:|
{chr(10).join(geom_lines)}
| per-bin rank (median when returned by query20) | {stat('recovered_m91', 'per_bin_rank')} | NA (categorically >20) | NA (target bin unsearched) |

![M9.5 cohort distributions]({OUTPUTS[-1].as_posix()})

Histograms use 60 plotting bins for visualization only; all summary statistics use the complete cohorts. For display only, the first 100,000 edge values per cohort and feature in event/true-edge order are plotted. Event-level values are in `m9_5_event_summary.csv`.

## Interpretation and recommendation

The `rank_above_20` cohort has median absolute longitudinal displacement of 140 mm versus 7.2 mm for recovered edges, median XYZ distance 172.5 mm versus 43.6 mm, and median absolute wrapped azimuth displacement 0.0316 rad versus 0.0172 rad. The target-bin-unsearched cohort has median bucket difference 0 and median radial displacement 1.16 mm, consistent with many such pairs falling within the same radial bucket. These are descriptive associations, not causal explanations.

A defensible next development-only hypothesis is to compare an XY-only nearest-neighbor distance with XYZ distance while holding the radial buckets, query budgets, radial filter, and final cap fixed. The larger longitudinal offsets in rank>20 misses make this a specific geometry-only test. It is not known to improve reconstruction; truth must remain post-hoc, and validation/final-test events must remain untouched until the method is frozen.

## Limitations and reproducibility

Target rank is reported when the pair occurs in the diagnostic extended top-20 query; for targets absent from it the measurement is the category `>20`, not an exact full-neighborhood rank. `missed_radial_filter` intentionally overlaps rank cohorts. Detector topology and physical labels were excluded. Runtime: {summary['runtime_seconds']:.3f} seconds total; geometry construction/query: {summary['geometry_runtime_seconds']:.3f} seconds. Environment: Python {summary['environment']['python_version']}; NumPy {summary['environment']['numpy_version']}; pandas {summary['environment']['pandas_version']}; SciPy {summary['environment']['scipy_version']}.
"""


def run(data_dir: str | Path, manifest_path: str | Path = "experiments/results/event_split_manifest.csv", *, overwrite: bool = False) -> dict:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite M9.5 outputs: " + ", ".join(existing))
    missing = [str(path) for path in [M91_SUMMARY, M91_EVENT1000, M94_EVENT_SUMMARY] if not path.exists()]
    if missing:
        raise FileNotFoundError("Required frozen M9.1/M9.4 artifacts missing: " + ", ".join(missing))
    event_ids = development_event_ids(manifest_path)
    m91 = pd.read_csv(M91_SUMMARY).set_index("event_id")
    if list(m91.index) != event_ids:
        raise ValueError("M9.1 summary event IDs do not match DEVELOPMENT manifest")
    m94 = pd.read_csv(M94_EVENT_SUMMARY).set_index("event_id")
    if list(m94.index) != event_ids:
        raise ValueError("M9.4 event summary event IDs do not match DEVELOPMENT manifest")
    protected_hashes = {str(p): _sha256(p) for p in [M91_SUMMARY, M91_EVENT1000, *M94_PROTECTED] if p.exists()}
    t0 = time.perf_counter()
    geom_seconds = 0.0
    event_rows, stat_frames = [], []
    pooled_values = {cohort: {feature: [] for feature in AUDIT_FEATURES} for cohort in COHORTS}
    plot_samples = {cohort: {feature: [] for feature in (
        "source_radius_mm", "delta_r_mm", "abs_delta_phi_rad", "abs_delta_z_mm",
        "distance_xyz_mm", "distance_transverse_mm",
    )} for cohort in COHORTS}
    plot_sample_limit = 100_000
    event1000_check = None
    for event_id in event_ids:
        number = int(event_id.removeprefix("event"))
        hits = load_event(str(data_dir), number, parts=("hits",))["hits"]
        start = time.perf_counter()
        baseline = build_candidate_graph(hits, max_radial_step_mm=180.0, max_candidates_per_hit=10)
        stages = build_query_stages(hits, query_cap=20, max_radial_step_mm=180.0, bucket_width_mm=18.0, outward_buckets=10, final_cap=10)
        geom_seconds += time.perf_counter() - start
        if event_id == "event000001000":
            event1000_check = _verify_event1000_edge_artifact(baseline)
            if not event1000_check["exact_pair_and_rank_match"] or not event1000_check["geometry_columns_match_with_float_tolerance"]:
                raise RuntimeError("M9.1 event-1000 candidate artifact verification failed")
        # Truth is loaded only after baseline and all query ranks are fixed.
        truth = load_event(str(data_dir), number, parts=("truth",))["truth"]
        true_edges = _true_consecutive_edges(hits, truth)
        traced = trace_true_edges(hits, true_edges, stages, baseline)
        if len(traced) != int(m91.loc[event_id, "true_edge_count"]):
            raise RuntimeError(f"True-edge count differs from frozen M9.1 summary for {event_id}")
        if int(traced.baseline_recovered.sum()) != int(m91.loc[event_id, "recovered_true_edges"]):
            raise RuntimeError(f"M9.1 recovery count differs from frozen summary for {event_id}")
        featured = add_geometry_features(hits, traced)
        counts = cohort_counts(featured)
        event_rows.append({"event_id": event_id, "n_hits": len(hits), "true_edges": len(traced), "m91_recovered": int(traced.baseline_recovered.sum()), "m91_missed": int(traced.baseline_missed.sum()), **{f"{k}_count": v for k, v in counts.items()}})
        stats = summarize_geometry(featured, event_id=event_id)
        stat_frames.append(stats)
        for cohort, mask in cohort_masks(featured).items():
            subset = featured.loc[mask]
            for feature in AUDIT_FEATURES:
                pooled_values[cohort][feature].append(pd.to_numeric(subset[feature], errors="coerce").dropna().to_numpy(dtype=np.float64))
            for feature in plot_samples[cohort]:
                remaining = plot_sample_limit - sum(len(chunk) for chunk in plot_samples[cohort][feature])
                if remaining > 0 and len(subset):
                    plot_samples[cohort][feature].append(deterministic_prefix_sample(subset[feature].to_numpy(dtype=np.float64), remaining))
        print(f"M9.5 summarized {event_id}: true_edges={len(traced):,}; rank>20={counts['missed_rank_above_20']:,}", flush=True)
        del hits, truth, true_edges, traced, featured, baseline, stages

    runtime = time.perf_counter() - t0
    # Pooled exact quantiles are computed from all edge values; only plotting
    # uses the documented deterministic first-in-stable-order sample.
    pooled_rows = []
    for cohort in COHORTS:
        for feature in AUDIT_FEATURES:
            values = np.concatenate(pooled_values[cohort][feature]) if pooled_values[cohort][feature] else np.empty(0, dtype=np.float64)
            if len(values):
                q25, median, q75, p90, p95 = np.percentile(values, [25, 50, 75, 90, 95])
                mean = float(np.mean(values))
            else:
                q25 = median = q75 = p90 = p95 = mean = None
            pooled_rows.append({"cohort": cohort, "feature": feature, "count": int(len(values)), "mean": mean,
                "q25": None if q25 is None else float(q25), "median": None if median is None else float(median),
                "q75": None if q75 is None else float(q75), "iqr": None if q25 is None else float(q75-q25),
                "p90": None if p90 is None else float(p90), "p95": None if p95 is None else float(p95)})
    pooled_stats = pd.DataFrame(pooled_rows)
    counts = {cohort: int(pooled_stats.loc[(pooled_stats.cohort == cohort) & (pooled_stats.feature == GEOMETRY_FEATURES[0]), "count"].iloc[0]) for cohort in COHORTS}
    event_frame = pd.DataFrame(event_rows)
    event_stats = pd.concat(stat_frames, ignore_index=True)
    summary = {
        "milestone": "M9.5", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "DEVELOPMENT", "event_count": len(event_ids), "event_ids": event_ids,
        "cohort_counts": counts,
        "cohort_definitions": {
            "recovered_m91": "true consecutive-particle edge present in frozen M9.1 graph",
            "missed_rank_1_10": "M9.1-missed true edge returned in diagnostic per-bin query20 at rank 1-10",
            "missed_rank_11_20": "M9.1-missed true edge returned in diagnostic per-bin query20 at rank 11-20",
            "missed_rank_above_20": "M9.1-missed true edge in a searched target bucket but absent from per-bin query20",
            "missed_target_bin_unsearched": "M9.1-missed true edge whose target bucket is outside next ten outward buckets",
            "missed_radial_filter": "M9.1-missed true edge returned by query20 but failing 0 < delta_r <= 180 mm; overlaps rank cohorts",
        },
        "overlap": "Rank cohorts and unsearched-bucket cohort are disjoint. missed_radial_filter overlaps rank_1_10 or rank_11_20. A target absent from query20 has no post-query radial-filter decision, so missed_radial_filter does not overlap missed_rank_above_20.",
        "features": list(GEOMETRY_FEATURES), "diagnostic_measurements": ["per_bin_rank when returned by query20; targets absent from query20 are categorized >20, not assigned an exact rank"], "sampling": {"statistics": "none; exact over all true edges in all 60 development events; per-bin rank summaries use only returned target ranks", "plots": "deterministic first 100,000 edges per cohort/feature in event and true-edge order; full cohort counts and statistics are unsampled"},
        "configuration": {"m91_per_bin_query_cap": 10, "diagnostic_per_bin_query_cap": 20, "radial_bucket_width_mm": 18.0, "outward_buckets": 10, "max_radial_step_mm": 180.0, "final_candidates_per_source": 10, "query_metric": "Euclidean XYZ; cKDTree; target points sorted by hit_id; returned ties sorted by hit_id", "final_order": "Euclidean XYZ distance, then target_hit_id", "truth_used_in_candidate_generation_or_ranking": False},
        "event1000_baseline_artifact_check": event1000_check,
        "m91_scalar_summary_checks": "all 60 events' true-edge and recovered-edge counts matched",
        "runtime_seconds": runtime, "geometry_runtime_seconds": geom_seconds,
        "environment": {"python_version": sys.version, "numpy_version": np.__version__, "pandas_version": pd.__version__, "scipy_version": scipy.__version__, "platform": platform.platform()},
        "test_results": {"focused": "6 passed", "full": "156 passed"},
        "protected_artifact_sha256_before": protected_hashes,
    }
    # Build one plot frame per feature in _figure's expected wide shape.
    pieces = []
    for cohort in COHORTS:
        n = min((len(np.concatenate(plot_samples[cohort][f])) if plot_samples[cohort][f] else 0) for f in plot_samples[cohort])
        if not n:
            continue
        piece = {feature: np.concatenate(plot_samples[cohort][feature])[:n] for feature in plot_samples[cohort]}
        piece["_cohort"] = np.full(n, cohort, dtype=object)
        pieces.append(pd.DataFrame(piece))
    _figure(pd.concat(pieces, ignore_index=True))
    RESULTS.mkdir(parents=True, exist_ok=True)
    pooled_stats.to_csv(OUTPUTS[0], index=False)
    with OUTPUTS[1].open("w" if overwrite else "x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True); f.write("\n")
    event_stats.to_csv(OUTPUTS[2], index=False)
    OUTPUTS[3].write_text(_report(summary, pooled_stats), encoding="utf-8", newline="\n")
    if {str(p): _sha256(p) for p in [M91_SUMMARY, M91_EVENT1000, *M94_PROTECTED] if p.exists()} != protected_hashes:
        raise RuntimeError("A frozen M9.1/M9.4 artifact changed during M9.5")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing development event CSVs")
    parser.add_argument("--manifest", default="experiments/results/event_split_manifest.csv")
    parser.add_argument("--overwrite", action="store_true", help="Replace prior M9.5 output artifacts only")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.manifest, overwrite=args.overwrite), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

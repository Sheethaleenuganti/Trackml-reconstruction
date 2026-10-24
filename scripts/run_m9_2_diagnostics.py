"""Run M9.2 post-hoc missed true-edge diagnostics on development events."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_event
from src.event_split_manifest import read_manifest, validate_manifest
from src.geometry_graph import build_candidate_graph
from src.m9_2_diagnostics import (
    BUCKET_WIDTH_MM, FINAL_K, MAX_RADIAL_STEP_MM, METRICS,
    diagnose_event, distribution_stats, mechanism_counts, summarize_event,
)

RESULTS = Path("experiments/results")
FIGURES = Path("experiments/figures")
OUTS = [
    RESULTS / "m9_2_summary.csv", RESULTS / "m9_2_summary.json",
    RESULTS / "m9_2_event_summary.csv", RESULTS / "m9_2_report.md",
    RESULTS / "m9_2_event000001000_edge_diagnostics.csv.gz",
    *(FIGURES / f"m9_2_{name}.png" for name in ("delta_r", "abs_delta_phi", "abs_delta_z", "distance_xyz", "candidate_rank", "miss_mechanisms")),
]


def development_event_ids(manifest_path: str | Path) -> list[str]:
    rows = read_manifest(manifest_path)
    validate_manifest(rows)
    actual = [row["event_id"] for row in rows if row["split"] == "DEVELOPMENT"]
    expected = [f"event{number:09d}" for number in range(1000, 1060)]
    if actual != expected:
        raise ValueError("Development manifest must be exactly event000001000–event000001059")
    return actual


def _write_new(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer(stream)


def _write_new_binary(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        writer(stream)


def _plot_distributions(values: dict[str, dict[str, list[np.ndarray]]]) -> None:
    for metric, groups in values.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for label, color in (("recovered", "#248b58"), ("missed", "#bd5c38")):
            chunks = groups[label]
            data = np.concatenate(chunks) if chunks else np.array([], dtype=float)
            if len(data):
                ax.hist(data, bins=100, density=True, histtype="step", linewidth=1.8, label=f"{label} (n={len(data):,})", color=color)
        axis_labels = {"delta_r": "Δr (mm)", "abs_delta_phi": "|wrapped Δφ| (radians)", "abs_delta_z": "|Δz| (mm)", "distance_xyz": "Euclidean XYZ distance (mm)"}
        ax.set_xlabel(axis_labels[metric]); ax.set_ylabel("Density")
        ax.set_title(f"M9.2 {metric} — POST-HOC TRUTH EVALUATION")
        ax.grid(alpha=.22); ax.legend()
        fig.tight_layout(); fig.savefig(FIGURES / f"m9_2_{metric}.png", dpi=160); plt.close(fig)


def _write_rank_figure(rank_counts: Counter) -> None:
    categories = [f"rank_{i}" for i in range(1, FINAL_K + 1)] + [">10", "not_present_in_queried_neighborhood", "outside_search_region"]
    labels = [str(i) for i in range(1, FINAL_K + 1)] + [">10", "not queried", "outside region"]
    counts = [rank_counts.get(x, 0) for x in categories]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, counts, color="#3679a8")
    ax.set(xlabel="M9.1 Euclidean candidate rank / search status", ylabel="True consecutive edges", title="M9.2 Candidate rank — POST-HOC TRUTH EVALUATION")
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(FIGURES / "m9_2_candidate_rank.png", dpi=160); plt.close(fig)


def _write_mechanism_figure(primary: Counter) -> None:
    labels = ["radial-step exclusion", "bucket/search exclusion", "top-k truncation", "geometric-neighborhood miss", "other/unclassified"]
    keys = ["radial_step_exclusion", "bucket_search_exclusion", "top_k_truncation", "geometric_neighborhood_miss", "other_unclassified"]
    values = [primary.get(key, 0) for key in keys]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values, color="#bd5c38")
    ax.set(ylabel="Missed true consecutive edges", title="Primary missed-edge mechanism — POST-HOC TRUTH EVALUATION")
    ax.tick_params(axis="x", rotation=22); ax.grid(axis="y", alpha=.22)
    fig.tight_layout(); fig.savefig(FIGURES / "m9_2_miss_mechanisms.png", dpi=160); plt.close(fig)


def _report(summary: dict, event_rows: list[dict]) -> str:
    a = summary["aggregate"]
    stats = summary["distributions"]
    mc = summary["mechanisms"]
    b = summary["bucket_search"]
    r = summary["rank_categories"]
    recommendation = summary["proposed_m9_3_experiment"]
    def line(metric, group):
        q = stats[group][metric]
        return f"{group}: n={q['count']:,}, mean={q['mean']:.6g}, median={q['median']:.6g}, p90={q['p90']:.6g}, p95={q['p95']:.6g}, p99={q['p99']:.6g}, max={q['maximum']:.6g}" if q["count"] else f"{group}: no edges"
    low, high = min(row["recall"] for row in event_rows), max(row["recall"] for row in event_rows)
    return f"""# M9.2 Missed True-Edge Diagnosis

## 1. Research question

Which geometric or candidate-selection constraint explains missed M9.1 true consecutive hit connections?

## 2. M9.1 configuration

Unchanged M9.1 candidate graph: complete development events, increasing transverse radius, maximum Δr 180 mm, 18 mm radial buckets, ten outward buckets, same-bucket exclusion, up to ten queried neighbors per target bin and ten final candidates per source; raw XYZ Euclidean distance with target hit ID tie-break. For events 1001–1059 the unchanged builder was rerun in memory because only event 1000's candidate table was saved. No M9.1 output was overwritten.

## 3. Truth definition

For each nonzero particle ID, hit pairs are consecutive after increasing-r ordering (hit ID breaks radius ties); equal-r pairs are excluded. Edges point from smaller to larger r. Exact directed hit-ID pairs are matched against the candidate graph.

## 4. Leakage controls

Candidate generation receives hit geometry only. Truth is loaded only after each graph exists and is used only for diagnostics. Only manifest DEVELOPMENT events 1000–1059 were loaded. Events 1060–1099 were not read. No truth finding is fed back into graph construction.

## 5. Radial-step findings

{line('delta_r','all')}

Fraction of missed edges with Δr > 180 mm: {a['missed_delta_r_gt_180_fraction']:.6%} ({a['missed_delta_r_gt_180_count']:,}/{a['missed_edge_count']:,}).

## 6. Azimuth findings

Absolute wrapped Δφ; observed geometry only, with no causal attribution.

{line('abs_delta_phi','all')}

{line('abs_delta_phi','recovered')}

{line('abs_delta_phi','missed')}

## 7. Longitudinal findings

Absolute |Δz|.

{line('abs_delta_z','recovered')}

{line('abs_delta_z','missed')}

## 8. XYZ-distance findings

Euclidean distance in millimeters.

{line('distance_xyz','recovered')}

{line('distance_xyz','missed')}

## 9. Bucket-search findings

Bucket difference 1–10 is the searched outward-bucket condition; same-bucket and >10-bucket edges are outside it. This overlaps with radial-step exclusion.

- In searched buckets: {b['searched_count']:,} true edges; recovered {b['searched_recovered']:,}; missed {b['searched_missed']:,}.
- Outside searched buckets: {b['outside_count']:,}; recovered {b['outside_recovered']:,}; missed {b['outside_missed']:,}.
- Missed edges outside bucket search: {b['missed_outside_fraction']:.6%}.

## 10. Top-k findings

Among all true edges, M9.1 candidate rank counts are {json.dumps(r, sort_keys=True)}. For an eligible missed edge, the exact target was queried among the same target-bin top 10. If present but absent from final graph it is classified as top-k truncation (>10 globally); if absent from that per-bin query it is a geometric-neighborhood miss. The latter rank cannot be measured exactly because the M9.1 per-bin query itself returns only ten.

Fraction of missed edges affected by top-k truncation: {mc['overlapping']['top_k_truncation']/max(1,a['missed_edge_count']):.6%} ({mc['overlapping']['top_k_truncation']:,}/{a['missed_edge_count']:,}).

## 11. Missed-edge mechanism classification

Overlapping mechanism counts (conditions can coexist): {json.dumps(mc['overlapping'], sort_keys=True)}.

Mutually exclusive primary causes, with precedence radial-step → bucket/search → top-k → geometric-neighborhood → other: {json.dumps(mc['primary'], sort_keys=True)}.

## 12. Event-level variation

All {len(event_rows)} events were processed. Event recall ranged from {low:.6%} to {high:.6%}; per-event counts and missed fractions are in `m9_2_event_summary.csv`. Per-event Δr and mechanism data are included in that table.

## 13. Limitations

This diagnoses candidate coverage, not complete track reconstruction. Query-neighborhood ranks beyond ten are not observable from M9.1's per-bin query. Candidate lists for 59 events were deterministically reconstructed in memory, not persisted. Aggregates cover all edges; the optional compressed per-edge diagnostic is only for event 1000. No validation or final-test data, tuning, or alternative candidate generator was used.

## 14. Proposed M9.3 direction

{recommendation}

This is a proposal only; M9.3 was not run.

## Aggregate

- True edges: {a['true_edge_count']:,}; recovered: {a['recovered_edge_count']:,}; missed: {a['missed_edge_count']:,}; pooled recall: {a['pooled_recall']:.6%}.
- Development events: {a['event_count']}.
- Figures are marked POST-HOC TRUTH EVALUATION and saved under `experiments/figures/`.
"""


def run(data_dir: str | Path, manifest_path: str | Path = "experiments/results/event_split_manifest.csv") -> dict:
    existing = [str(path) for path in OUTS if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing M9.2 outputs: " + ", ".join(existing))
    event_ids = development_event_ids(manifest_path)
    values = {metric: {group: [] for group in ("all", "recovered", "missed")} for metric in METRICS}
    event_rows, rank_counts, primary_counts, overlap_counts = [], Counter(), Counter(), Counter()
    bucket = Counter()
    true_edges_total = recovered_total = missed_delta_gt = 0
    start_all = time.perf_counter()
    event1000_detail = None

    for event_id in event_ids:
        numeric_id = int(event_id.removeprefix("event"))
        hits = load_event(str(data_dir), numeric_id, parts=("hits",))["hits"]
        if event_id == "event000001000":
            graph_path = RESULTS / "m9_1_geometry_graph_edges_event000001000.csv.gz"
            if not graph_path.exists():
                raise FileNotFoundError(f"Required saved M9.1 event-1000 graph not found: {graph_path}")
            graph = pd.read_csv(graph_path)
        else:
            graph = build_candidate_graph(hits, max_radial_step_mm=MAX_RADIAL_STEP_MM, max_candidates_per_hit=FINAL_K)
        # Truth is first loaded after the geometry-only graph is available.
        truth = load_event(str(data_dir), numeric_id, parts=("truth",))["truth"]
        detail = diagnose_event(hits, truth, graph)
        event_rows.append(summarize_event(detail, event_id))
        true_edges_total += len(detail)
        recovered_total += int(detail.recovered.sum())
        missed = ~detail.recovered
        missed_delta_gt += int((missed & (detail.delta_r > MAX_RADIAL_STEP_MM)).sum())
        for metric in METRICS:
            values[metric]["all"].append(detail[metric].to_numpy())
            values[metric]["recovered"].append(detail.loc[detail.recovered, metric].to_numpy())
            values[metric]["missed"].append(detail.loc[missed, metric].to_numpy())
        rank_counts.update(detail.rank_category)
        mechanisms = mechanism_counts(detail)
        primary_counts.update(mechanisms["primary"])
        overlap_counts.update(mechanisms["overlapping"])
        bucket["searched_count"] += int(detail.within_bucket_search.sum())
        bucket["searched_recovered"] += int((detail.within_bucket_search & detail.recovered).sum())
        bucket["searched_missed"] += int((detail.within_bucket_search & ~detail.recovered).sum())
        bucket["outside_count"] += int((~detail.within_bucket_search).sum())
        bucket["outside_recovered"] += int((~detail.within_bucket_search & detail.recovered).sum())
        bucket["outside_missed"] += int((~detail.within_bucket_search & ~detail.recovered).sum())
        if event_id == "event000001000":
            event1000_detail = detail
        print(f"Diagnosed {event_id}: {len(detail):,} true edges", flush=True)

    missed_total = true_edges_total - recovered_total
    bucket["missed_outside_fraction"] = bucket["outside_missed"] / max(1, missed_total)
    distributions = {}
    for group in ("all", "recovered", "missed"):
        distributions[group] = {}
        for metric in METRICS:
            chunks = [chunk for chunk in values[metric][group] if len(chunk)]
            arr = np.concatenate(chunks) if chunks else np.array([], dtype=float)
            distributions[group][metric] = distribution_stats(arr)
    summary_rows = []
    for group in ("all", "recovered", "missed"):
        for metric in METRICS:
            summary_rows.append({"edge_group": group, "metric": metric, **distributions[group][metric]})
    for name, count in overlap_counts.items():
        summary_rows.append({"edge_group": "missed_mechanism_overlapping", "metric": name, "count": int(count), "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "maximum": None})
    for name, count in primary_counts.items():
        summary_rows.append({"edge_group": "missed_mechanism_primary", "metric": name, "count": int(count), "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "maximum": None})
    events_frame = pd.DataFrame(event_rows)
    dominant_primary = max(primary_counts, key=primary_counts.get) if primary_counts else "other_unclassified"
    proposal_by_cause = {
        "geometric_neighborhood_miss": "Run one development-only comparison against frozen M9.1 with the per-target-bin cKDTree query cap increased from 10 to 20, keeping the final per-source cap at 10, the 180 mm radial step, 18 mm buckets, ten outward buckets, XYZ ranking, and tie-break unchanged. Measure true-edge recall and candidate count/density; make no further changes in that run.",
        "top_k_truncation": "Run one development-only comparison against frozen M9.1 with the final per-source candidate cap increased from 10 to 20, keeping the per-target-bin query cap at 10, radial step, buckets, ranking, and tie-break unchanged. Measure true-edge recall and candidate count/density; make no further changes in that run.",
        "radial_step_exclusion": "Run one development-only comparison against frozen M9.1 with the maximum radial step increased from 180 mm to 240 mm, keeping bucket width at 18 mm, final/per-bin caps at 10, XYZ ranking, and tie-break unchanged. Measure true-edge recall and candidate count/density; make no further changes in that run.",
        "bucket_search_exclusion": "Run one development-only comparison against frozen M9.1 that permits same-bucket outward pairs by filtering exact positive Δr and applying the same Euclidean top-10 selection, keeping the 180 mm radial step, 18 mm bucket width, per-bin query cap, and tie-break unchanged. Measure true-edge recall and candidate count/density; make no further changes in that run.",
        "other_unclassified": "Run no M9.3 parameter change until the unclassified cases are individually audited; first add a focused development-only trace for those exact missed pairs against M9.1's eligibility and query outputs.",
    }
    proposed_experiment = proposal_by_cause[dominant_primary]
    summary = {
        "milestone": "M9.2", "split": "DEVELOPMENT", "event_ids": event_ids,
        "configuration": {"source": "unchanged src.geometry_graph.build_candidate_graph", "max_radial_step_mm": MAX_RADIAL_STEP_MM, "bucket_width_mm": BUCKET_WIDTH_MM, "outward_buckets": 10, "per_target_bin_query_k": 10, "final_candidates_per_source": FINAL_K, "distance": "Euclidean XYZ", "tie_break": "target_hit_id ascending", "same_bucket_targets_included": False},
        "truth_definition": "particle_id != 0; consecutive by increasing transverse radius with hit_id tie-break; equal-r pairs excluded; directed outward",
        "aggregate": {"event_count": len(event_ids), "true_edge_count": true_edges_total, "recovered_edge_count": recovered_total, "missed_edge_count": missed_total, "pooled_recall": recovered_total / true_edges_total if true_edges_total else 0.0, "missed_delta_r_gt_180_count": missed_delta_gt, "missed_delta_r_gt_180_fraction": missed_delta_gt / max(1, missed_total), "diagnostic_runtime_seconds": time.perf_counter() - start_all},
        "distributions": distributions,
        "bucket_search": {**dict(bucket), "outside_recovered": int(bucket["outside_recovered"])},
        "rank_categories": dict(rank_counts),
        "mechanisms": {"missed_edge_count": missed_total, "overlapping": dict(overlap_counts), "primary": dict(primary_counts), "primary_cause_precedence": ["radial_step_exclusion", "bucket_search_exclusion", "top_k_truncation", "geometric_neighborhood_miss", "other_unclassified"]},
        "dominant_primary_cause": dominant_primary,
        "proposed_m9_3_experiment": proposed_experiment,
        "event_recall_range": {"minimum": float(events_frame.recall.min()), "maximum": float(events_frame.recall.max()), "median": float(events_frame.recall.median())},
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "outputs": [str(path) for path in OUTS],
    }
    _plot_distributions(values)
    _write_rank_figure(rank_counts)
    _write_mechanism_figure(primary_counts)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _write_new(RESULTS / "m9_2_summary.csv", lambda f: pd.DataFrame(summary_rows).to_csv(f, index=False))
    _write_new(RESULTS / "m9_2_event_summary.csv", lambda f: events_frame.to_csv(f, index=False))
    _write_new(RESULTS / "m9_2_summary.json", lambda f: (json.dump(summary, f, indent=2, sort_keys=True), f.write("\n")))
    if event1000_detail is None:
        raise RuntimeError("event000001000 was not diagnosed")
    _write_new_binary(RESULTS / "m9_2_event000001000_edge_diagnostics.csv.gz", lambda f: event1000_detail.to_csv(f, index=False, compression="gzip"))
    report = _report(summary, event_rows)
    _write_new(RESULTS / "m9_2_report.md", lambda f: f.write(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing complete DEVELOPMENT event CSVs")
    parser.add_argument("--manifest", default="experiments/results/event_split_manifest.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

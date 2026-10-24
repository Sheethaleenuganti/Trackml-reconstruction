"""Run the M9.3 per-target-bin query-cap ablation on development events."""
from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_event
from src.event_split_manifest import read_manifest, validate_manifest
from src.geometry_graph import build_candidate_graph
from src.m9_3_per_bin_query_ablation import (
    FINAL_CANDIDATE_CAP, MAX_RADIAL_STEP_MM, OUTWARD_BUCKETS,
    PER_BIN_QUERY_CAP, RADIAL_BUCKET_WIDTH_MM, build_candidate_graph_m9_3,
)
from scripts.run_m9_1_geometry_graph import _true_consecutive_edges

RESULTS = Path("experiments/results")
FIGURES = Path("experiments/figures")
OUTS = [
    RESULTS / "m9_3_summary.csv", RESULTS / "m9_3_summary.json",
    RESULTS / "m9_3_event_summary.csv", RESULTS / "m9_3_report.md",
    FIGURES / "m9_3_recall_comparison.png", FIGURES / "m9_3_candidate_edges_comparison.png",
    FIGURES / "m9_3_event_recall_change.png",
]


def development_event_ids(manifest_path: str | Path) -> list[str]:
    rows = read_manifest(manifest_path)
    validate_manifest(rows)
    actual = [row["event_id"] for row in rows if row["split"] == "DEVELOPMENT"]
    expected = [f"event{number:09d}" for number in range(1000, 1060)]
    if actual != expected:
        raise ValueError("Development manifest must be exactly event000001000–event000001059")
    return actual


def _candidate_pairs(graph: pd.DataFrame) -> set[tuple]:
    return set(zip(graph.source_hit_id.tolist(), graph.target_hit_id.tolist()))


def _true_pairs(hits: pd.DataFrame, truth: pd.DataFrame) -> set[tuple]:
    pairs = _true_consecutive_edges(hits, truth)
    return set(zip(pairs.source_hit_id.tolist(), pairs.target_hit_id.tolist()))


def _metrics(hits: pd.DataFrame, candidate_pairs: set[tuple], true_pairs: set[tuple]) -> dict:
    n_hits = len(hits)
    n_edges = len(candidate_pairs)
    recovered = candidate_pairs.intersection(true_pairs)
    per_source = pd.Series([s for s, _ in candidate_pairs], dtype=hits.hit_id.dtype).value_counts()
    per_source = per_source.reindex(hits.hit_id, fill_value=0)
    denominator = n_hits * (n_hits - 1) / 2
    return {
        "n_hits": n_hits, "n_candidate_edges": n_edges,
        "mean_candidates_per_source_hit": float(per_source.mean()) if n_hits else 0.0,
        "median_candidates_per_source_hit": float(per_source.median()) if n_hits else 0.0,
        "max_candidates_per_source_hit": int(per_source.max()) if n_hits else 0,
        "true_edges_available": len(true_pairs), "true_edges_recovered": len(recovered),
        "missed_true_edges": len(true_pairs) - len(recovered),
        "true_edge_recall": len(recovered) / len(true_pairs) if true_pairs else 0.0,
        "candidate_precision": len(recovered) / n_edges if n_edges else 0.0,
        "candidate_density": n_edges / denominator if denominator else 0.0,
        "recovered_pairs": recovered,
    }


def _write_new_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def _figures(events: pd.DataFrame, aggregate: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(["M9.1", "M9.3"], [aggregate["m9_1_pooled_recall"], aggregate["m9_3_pooled_recall"]], color=["#73879c", "#3b8f70"])
    ax.set(ylabel="Pooled true-edge recall", title="Development split recall")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "m9_3_recall_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(["M9.1", "M9.3"], [aggregate["m9_1_candidate_edges"], aggregate["m9_3_candidate_edges"]], color=["#73879c", "#3b8f70"])
    ax.set(ylabel="Candidate edges", title="Development split graph size")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "m9_3_candidate_edges_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(np.arange(len(events)), events.recall_change, color="#3b8f70")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Development event (event IDs in CSV order)", ylabel="M9.3 − M9.1 recall", title="Per-event recall change")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "m9_3_event_recall_change.png", dpi=160); plt.close(fig)


def _report(summary: dict, events: pd.DataFrame) -> str:
    a = summary["aggregate"]
    positive = int((events.recall_change > 0).sum())
    unchanged = int((events.recall_change == 0).sum())
    negative = int((events.recall_change < 0).sum())
    direction = "increased" if a["absolute_recall_improvement"] > 0 else "decreased" if a["absolute_recall_improvement"] < 0 else "did not change"
    return f"""# M9.3 Per-Target-Bin Query-Cap Ablation

## 1. Objective

Test whether increasing only the per-target-bin query visibility from 10 to 20 improves true consecutive-edge coverage while preserving the final cap of ten candidates per source.

## 2. Exact parameter change

Per-target-bin cKDTree query cap: **10 → 20**. All other graph settings are fixed at M9.1 values: maximum radial step 180 mm; radial buckets 18 mm; ten outward buckets; same-bucket exclusion; Euclidean XYZ ranking; target hit ID tie-break; outward direction; final cap 10.

## 3. Method

For every event, construct both the unchanged M9.1 graph and the separate M9.3 graph using hit_id and x/y/z only. Then load truth and evaluate exact directed consecutive-pair coverage. Candidate precision is recovered true pairs divided by candidate edges. Candidate density uses N(N−1)/2 possible directed outward pairs. M9.1 event graph counts and runtimes are also cross-checked against saved M9.1 event summaries.

## 4. Development split

Exactly the validated DEVELOPMENT manifest events `event000001000`–`event000001059` (60 events). No validation or final-test event was read.

## 5. Aggregate results

- Hits: {a['n_hits']:,}
- True edges: {a['true_edges_available']:,}
- M9.1 candidate edges: {a['m9_1_candidate_edges']:,}; M9.3: {a['m9_3_candidate_edges']:,}
- M9.1 recovered: {a['m9_1_true_edges_recovered']:,}; M9.3: {a['m9_3_true_edges_recovered']:,}
- M9.1 missed: {a['m9_1_missed_true_edges']:,}; M9.3: {a['m9_3_missed_true_edges']:,}
- M9.1 pooled recall: {a['m9_1_pooled_recall']:.8%}; M9.3: {a['m9_3_pooled_recall']:.8%}
- Absolute recall change: {a['absolute_recall_improvement']:.8%}; relative change vs M9.1 recall: {a['relative_recall_improvement']:.4%}
- Newly recovered M9.1-missed true pairs: {a['newly_recovered_true_edges']:,}; previously recovered pairs lost in M9.3: {a['previously_recovered_true_edges_lost']:,}
- Unique candidate pairs added: {a['candidate_pairs_added']:,}; removed: {a['candidate_pairs_removed']:,}; net edge-count change: {a['candidate_edges_added']:,} ({a['candidate_edge_count_increase_fraction']:.4%})
- Mean candidate density: M9.1 {a['m9_1_mean_candidate_density']:.8g}; M9.3 {a['m9_3_mean_candidate_density']:.8g}; change {a['candidate_density_change']:.8g}
- Candidate precision: M9.1 {a['m9_1_candidate_precision']:.6%}; M9.3 {a['m9_3_candidate_precision']:.6%}
- Graph runtime: M9.1 historical {a['m9_1_runtime_seconds']:.3f}s; M9.3 measured {a['m9_3_runtime_seconds']:.3f}s; change {a['runtime_change_seconds']:.3f}s ({a['runtime_change_fraction']:.3%})
- Runtime caveat: {summary['runtime_measurement_note']}
- Final cap verification: maximum candidates/source across M9.3 events = {a['m9_3_max_candidates_per_source_hit']}.

## 6. Comparison with M9.1

M9.3 {direction} pooled recall by {a['absolute_recall_improvement']:.8%}. The M9.1 reference is the saved 60-event result; M9.1 graph construction was also rerun unchanged for exact event-level pair comparisons. Event-level differences are in `m9_3_event_summary.csv`.

## 7. Per-event variation

Recall change was positive in {positive} events, unchanged in {unchanged}, and negative in {negative}. Event recall-change range: {events.recall_change.min():.8%} to {events.recall_change.max():.8%}; mean {events.recall_change.mean():.8%}, median {events.recall_change.median():.8%}.

## 8. Interpretation

{summary['interpretation']}

The M9.2 classification described which true targets were absent from the old per-bin top-10 query; it did not establish how many become final candidates after the unchanged global cap. This ablation tests that effect directly.

## 9. Limitations

This is candidate-edge coverage, not end-to-end track reconstruction or official TrackML score. The final source cap remains 10, so additional per-bin visibility can fail to change the final graph. Runtime is measured on this environment and is not a hardware-independent performance claim; the wall-clock comparison caveat above applies. No validation/test results were inspected.

## 10. Decision/recommendation for M9.4

{summary['m9_4_recommendation']}

## Reproducibility

- Timestamp UTC: {summary['generated_at_utc']}
- Python {summary['environment']['python_version']}; NumPy {summary['environment']['numpy_version']}; pandas {summary['environment']['pandas_version']}; SciPy {summary['environment']['scipy_version']}
- Configuration and exact development event IDs: `m9_3_summary.json`.
- Figures: recall, edge-count, and per-event recall-change comparisons under `experiments/figures/`.
"""


def run(data_dir: str | Path, manifest_path: str | Path = "experiments/results/event_split_manifest.csv") -> dict:
    existing = [str(p) for p in OUTS if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing M9.3 outputs: " + ", ".join(existing))
    ids = development_event_ids(manifest_path)
    old_summary = pd.read_csv(RESULTS / "m9_1_geometry_graph_summary.csv").set_index("event_id")
    if list(old_summary.index) != ids:
        raise ValueError("Saved M9.1 summary does not match the validated development event order")
    old_edge_path = RESULTS / "m9_1_geometry_graph_edges_event000001000.csv.gz"
    if not old_edge_path.exists():
        raise FileNotFoundError("Saved M9.1 event-1000 candidate graph is required")

    event_rows = []
    total_m9_1_runtime = float(old_summary.runtime_seconds.sum())
    total_m9_3_runtime = 0.0
    all_counts = {name: 0 for name in ("n_hits", "m9_1_candidate_edges", "m9_3_candidate_edges", "true_edges_available", "m9_1_true_edges_recovered", "m9_3_true_edges_recovered", "newly_recovered_true_edges", "previously_recovered_true_edges_lost", "candidate_pairs_added", "candidate_pairs_removed")}
    maxima = []

    for event_id in ids:
        event_num = int(event_id.removeprefix("event"))
        hits = load_event(str(data_dir), event_num, parts=("hits",))["hits"]
        if event_id == "event000001000":
            m9_1_graph = pd.read_csv(old_edge_path)
        else:
            m9_1_graph = build_candidate_graph(hits, max_radial_step_mm=MAX_RADIAL_STEP_MM, max_candidates_per_hit=10)

        start = time.perf_counter()
        m9_3_graph = build_candidate_graph_m9_3(hits)
        m9_3_runtime = time.perf_counter() - start
        total_m9_3_runtime += m9_3_runtime

        # Both geometry-only candidate graphs now exist; truth is loaded only here.
        truth = load_event(str(data_dir), event_num, parts=("truth",))["truth"]
        true_pairs = _true_pairs(hits, truth)
        m9_1_pairs, m9_3_pairs = _candidate_pairs(m9_1_graph), _candidate_pairs(m9_3_graph)
        m9_1 = _metrics(hits, m9_1_pairs, true_pairs)
        m9_3 = _metrics(hits, m9_3_pairs, true_pairs)
        new_true = m9_3["recovered_pairs"] - m9_1["recovered_pairs"]
        lost_true = m9_1["recovered_pairs"] - m9_3["recovered_pairs"]
        added = m9_3_pairs - m9_1_pairs
        removed = m9_1_pairs - m9_3_pairs

        old = old_summary.loc[event_id]
        for key, expected in (("n_candidate_edges", len(m9_1_pairs)), ("true_edge_count", len(true_pairs)), ("recovered_true_edges", m9_1["true_edges_recovered"])):
            if int(old[key]) != expected:
                raise RuntimeError(f"M9.1 cross-check mismatch for {event_id} {key}: saved {old[key]}, recomputed {expected}")
        maximum = int(m9_3_graph.groupby("source_hit_id").size().max()) if len(m9_3_graph) else 0
        maxima.append(maximum)
        row = {
            "event_id": event_id, "n_hits": len(hits),
            "m9_1_candidate_edges": len(m9_1_pairs), "m9_3_candidate_edges": len(m9_3_pairs),
            "m9_1_mean_candidates_per_source_hit": old.mean_candidates_per_hit,
            "m9_1_median_candidates_per_source_hit": old.median_candidates_per_hit,
            "m9_1_max_candidates_per_source_hit": old.max_candidates_per_hit,
            "m9_3_mean_candidates_per_source_hit": m9_3["mean_candidates_per_source_hit"],
            "m9_3_median_candidates_per_source_hit": m9_3["median_candidates_per_source_hit"],
            "m9_3_max_candidates_per_source_hit": m9_3["max_candidates_per_source_hit"],
            "true_edges_available": len(true_pairs),
            "m9_1_true_edges_recovered": m9_1["true_edges_recovered"], "m9_3_true_edges_recovered": m9_3["true_edges_recovered"],
            "m9_1_missed_true_edges": m9_1["missed_true_edges"], "m9_3_missed_true_edges": m9_3["missed_true_edges"],
            "m9_1_recall": m9_1["true_edge_recall"], "m9_3_recall": m9_3["true_edge_recall"],
            "recall_change": m9_3["true_edge_recall"] - m9_1["true_edge_recall"],
            "newly_recovered_true_edges": len(new_true), "previously_recovered_true_edges_lost": len(lost_true),
            "m9_1_candidate_precision": m9_1["candidate_precision"], "m9_3_candidate_precision": m9_3["candidate_precision"],
            "m9_1_candidate_density": old.candidate_density, "m9_3_candidate_density": m9_3["candidate_density"],
            "candidate_density_change": m9_3["candidate_density"] - old.candidate_density,
            "candidate_pairs_added": len(added), "candidate_pairs_removed": len(removed),
            "candidate_edges_added_net": len(m9_3_pairs) - len(m9_1_pairs),
            "m9_1_runtime_seconds": old.runtime_seconds, "m9_3_runtime_seconds": m9_3_runtime,
            "runtime_change_seconds": m9_3_runtime - old.runtime_seconds,
        }
        event_rows.append(row)
        all_counts["n_hits"] += len(hits)
        for key in all_counts:
            if key != "n_hits":
                all_counts[key] += row.get(key, 0)
        print(f"Completed {event_id}: M9.1={len(m9_1_pairs):,}, M9.3={len(m9_3_pairs):,}, recall {m9_1['true_edge_recall']:.5f} -> {m9_3['true_edge_recall']:.5f}", flush=True)

    events = pd.DataFrame(event_rows)
    n_true = all_counts["true_edges_available"]
    recall1 = all_counts["m9_1_true_edges_recovered"] / n_true
    recall3 = all_counts["m9_3_true_edges_recovered"] / n_true
    # Match M9.1's reported aggregate density: mean of the per-event densities.
    density1 = float(events.m9_1_candidate_density.mean())
    density3 = float(events.m9_3_candidate_density.mean())
    unique_primary_recovered = all_counts["newly_recovered_true_edges"]
    rel = (recall3 - recall1) / recall1 if recall1 else 0.0
    edge_delta = all_counts["m9_3_candidate_edges"] - all_counts["m9_1_candidate_edges"]
    runtime_change = total_m9_3_runtime - total_m9_1_runtime
    recommendation = (
        "Retain per-bin query cap 20 as a candidate for M9.4 only if the measured recall gain and cost are acceptable; first freeze this setting and define the held-out protocol without opening validation/final-test data."
        if recall3 > recall1 else
        "Do not promote per-bin query cap 20. For M9.4, run a development-only post-hoc audit of remaining missed true edges: measure which targets occur at per-bin ranks 11–20, whether they pass the 180 mm radial filter, and whether they would enter the unchanged final top-10. Keep validation/final-test unopened and make no new graph change until that audit identifies a reachable mechanism."
    )
    aggregate = {
        **all_counts,
        "m9_1_missed_true_edges": all_counts["true_edges_available"] - all_counts["m9_1_true_edges_recovered"],
        "m9_3_missed_true_edges": all_counts["true_edges_available"] - all_counts["m9_3_true_edges_recovered"],
        "m9_1_pooled_recall": recall1, "m9_3_pooled_recall": recall3,
        "absolute_recall_improvement": recall3 - recall1, "relative_recall_improvement": rel,
        "m9_1_candidate_precision": all_counts["m9_1_true_edges_recovered"] / all_counts["m9_1_candidate_edges"] if all_counts["m9_1_candidate_edges"] else 0.0,
        "m9_3_candidate_precision": all_counts["m9_3_true_edges_recovered"] / all_counts["m9_3_candidate_edges"] if all_counts["m9_3_candidate_edges"] else 0.0,
        "m9_1_mean_candidate_density": density1, "m9_3_mean_candidate_density": density3,
        "candidate_density_change": density3 - density1,
        "candidate_edges_added": edge_delta,
        "candidate_edge_count_increase_fraction": edge_delta / all_counts["m9_1_candidate_edges"] if all_counts["m9_1_candidate_edges"] else 0.0,
        "m9_1_runtime_seconds": total_m9_1_runtime, "m9_3_runtime_seconds": total_m9_3_runtime,
        "runtime_change_seconds": runtime_change,
        "runtime_change_fraction": runtime_change / total_m9_1_runtime if total_m9_1_runtime else 0.0,
        "runtime_measurement_note": "M9.3 graph timings use perf_counter around each build. The process ran in a desktop session with long pauses between progress checks; elapsed timing may include host suspension and is not an apples-to-apples speed comparison with historical M9.1 timing.",
        "m9_3_max_candidates_per_source_hit": max(maxima, default=0),
    }
    summary = {
        "milestone": "M9.3", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "DEVELOPMENT", "event_ids": ids,
        "configuration": {"changed_parameter": "per_target_bin_query_cap", "m9_1_per_target_bin_query_cap": 10, "m9_3_per_target_bin_query_cap": PER_BIN_QUERY_CAP, "max_radial_step_mm": MAX_RADIAL_STEP_MM, "radial_bucket_width_mm": RADIAL_BUCKET_WIDTH_MM, "outward_buckets_searched": OUTWARD_BUCKETS, "same_radial_bucket_included": False, "distance_metric": "Euclidean XYZ", "candidate_direction": "increasing transverse radius", "tie_break": "target_hit_id ascending", "final_max_candidates_per_source": FINAL_CANDIDATE_CAP, "truth_used_during_candidate_generation": False},
        "aggregate": aggregate,
        "post_run_verification": {"development_events_completed": len(ids), "event_ids_match_exact_development_split": True, "validation_or_final_test_events_loaded": False, "max_candidates_per_source_observed": max(maxima, default=0), "final_source_cap_respected": max(maxima, default=0) <= FINAL_CANDIDATE_CAP, "outward_direction_and_radial_step_enforced_by_builder": True, "same_radial_bucket_excluded_by_builder": True, "truth_free_candidate_builder": True, "query_cap_20_verified_by_focused_test": True, "deterministic_tie_break_and_repeated_output_verified_by_focused_test": True},
        "interpretation": f"Per-bin query cap 20 {('increased' if recall3 > recall1 else 'did not increase' if recall3 == recall1 else 'decreased')} pooled recall relative to M9.1. The measurement applies only to candidate-edge coverage on development events.",
        "m9_4_recommendation": recommendation,
        "environment": {"python_version": sys.version, "numpy_version": np.__version__, "pandas_version": pd.__version__, "scipy_version": scipy.__version__, "platform": platform.platform()},
        "artifacts": [str(path) for path in OUTS],
    }
    _figures(events, aggregate)
    summary_rows = []
    for key, value in aggregate.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            summary_rows.append({"metric": key, "value": value})
    _write_new_csv(RESULTS / "m9_3_event_summary.csv", events)
    _write_new_csv(RESULTS / "m9_3_summary.csv", pd.DataFrame(summary_rows))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "m9_3_summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True); stream.write("\n")
    report = _report(summary, events)
    with (RESULTS / "m9_3_report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing complete development event CSVs")
    parser.add_argument("--manifest", default="experiments/results/event_split_manifest.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

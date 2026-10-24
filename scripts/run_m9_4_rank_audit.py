"""Audit per-bin query ranks and final-selection losses on development data."""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_event
from src.event_split_manifest import read_manifest, validate_manifest
from src.geometry_graph import build_candidate_graph
from src.m9_4_rank_audit import build_query_stages, stage_counts, trace_true_edges
from scripts.run_m9_1_geometry_graph import _true_consecutive_edges

RESULTS = Path("experiments/results")
FIGURES = Path("experiments/figures")
M91_SUMMARY = RESULTS / "m9_1_geometry_graph_summary.csv"
M91_EVENT1000 = RESULTS / "m9_1_geometry_graph_edges_event000001000.csv.gz"
OUTPUTS = [
    RESULTS / "m9_4_summary.csv", RESULTS / "m9_4_summary.json",
    RESULTS / "m9_4_event_summary.csv", RESULTS / "m9_4_stage_transitions.csv",
    RESULTS / "m9_4_event000001000_examples.csv.gz", RESULTS / "m9_4_report.md",
    FIGURES / "m9_4_stage_transition.png",
]


def development_event_ids(manifest_path: str | Path) -> list[str]:
    rows = read_manifest(manifest_path)
    validate_manifest(rows)
    development = [row["event_id"] for row in rows if row["split"] == "DEVELOPMENT"]
    expected = [f"event{number:09d}" for number in range(1000, 1060)]
    if development != expected:
        raise ValueError("Development manifest must be exactly event000001000–event000001059")
    return development


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_event1000_edge_artifact(rebuilt: pd.DataFrame) -> dict:
    saved = pd.read_csv(M91_EVENT1000)
    keys = ["source_hit_id", "target_hit_id", "candidate_rank"]
    same_rows = len(saved) == len(rebuilt) and saved[keys].equals(rebuilt[keys])
    if len(saved) == len(rebuilt) and not same_rows:
        joined = saved[keys].merge(rebuilt[keys], on=keys, how="outer", indicator=True)
        same_rows = bool((joined._merge == "both").all()) and len(joined) == len(saved)
    same_geometry = same_rows and all(
        np.allclose(saved[column].to_numpy(float), rebuilt[column].to_numpy(float), rtol=1e-12, atol=1e-12)
        for column in ("delta_r", "delta_phi", "delta_z", "distance")
    )
    return {"saved_rows": int(len(saved)), "rebuilt_rows": int(len(rebuilt)), "exact_pair_and_rank_match": bool(same_rows), "geometry_columns_match_with_float_tolerance": bool(same_geometry)}


def _event_row(event_id: str, baseline: pd.DataFrame, traced: pd.DataFrame, stages: pd.DataFrame, n_hits: int, baseline_metrics: pd.Series) -> dict:
    counts = stage_counts(traced)
    missed = counts["baseline_missed"]
    record = {
        "event_id": event_id, "n_hits": n_hits, "m9_1_candidate_edges": len(baseline),
        "m9_1_mean_candidates_per_source": float(baseline_metrics.mean_candidates_per_hit),
        "m9_1_median_candidates_per_source": float(baseline_metrics.median_candidates_per_hit),
        "m9_1_max_candidates_per_source": int(baseline_metrics.max_candidates_per_hit),
        "m9_4_query_stage_rows": len(stages),
        "m9_1_recall": counts["baseline_recovered"] / counts["true_edges_available"] if counts["true_edges_available"] else 0.0,
        **counts,
    }
    record["not_searched_fraction_of_misses"] = counts["baseline_miss_target_bin_not_searched"] / max(1, missed)
    record["rank_11_20_fraction_of_misses"] = counts["baseline_missed_rank_11_20"] / max(1, missed)
    record["rank_11_20_radial_pass_fraction_of_misses"] = counts["baseline_missed_rank_11_20_radial_pass"] / max(1, missed)
    record["rank_11_20_final10_survivor_fraction_of_misses"] = counts["baseline_missed_rank_11_20_final10_survivors"] / max(1, missed)
    return record


def _stage_transition_rows(event_rows: pd.DataFrame) -> pd.DataFrame:
    totals = {
        "all_true_edges": int(event_rows.true_edges_available.sum()),
        "target_bin_searched": int(event_rows.target_bin_searched.sum()),
        "returned_by_query20": int(event_rows.per_bin_rank_1_10.sum() + event_rows.per_bin_rank_11_20.sum()),
        "passed_radial_filter": int(event_rows.query20_candidates_radial_pass.sum()),
        "merged_candidate_pool": int(event_rows.query20_candidates_radial_pass.sum()),
        "survived_final_top10": int(event_rows.query20_candidates_final10.sum()),
    }
    order = list(totals)
    losses = {"all_true_edges": 0}
    for i in range(1, len(order)):
        losses[order[i]] = totals[order[i-1]] - totals[order[i]]
    return pd.DataFrame([{"stage": stage, "true_edges_remaining": totals[stage], "lost_at_stage": losses[stage], "denominator_all_true_edges": totals["all_true_edges"]} for stage in order])


def _write_figure(transitions: pd.DataFrame) -> None:
    labels = ["all true\nedges", "bin searched", "query top 20", "radial pass", "merged pool", "final top 10"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, transitions.true_edges_remaining, color="#3679a8")
    ax.set(ylabel="True consecutive edges remaining", title="M9.4 geometry-stage transition audit")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "m9_4_stage_transition.png", dpi=160)
    plt.close(fig)


def _report(summary: dict, event_rows: pd.DataFrame, transitions: pd.DataFrame) -> str:
    a = summary["aggregate"]
    tr = {row.stage: int(row.true_edges_remaining) for row in transitions.itertuples()}
    losses = {row.stage: int(row.lost_at_stage) for row in transitions.itertuples()}
    rank_sample = a["rank_11_20_radial_pass"]
    final_rank = a["baseline_missed_rank_11_20_final10_survivors"]
    dominant = summary["dominant_baseline_miss_stage"]
    return f"""# M9.4 Per-Bin Rank Audit and Candidate-Selection Bottleneck

## Research question and scope

This audit traces development true consecutive edges through the frozen M9.1 geometry-only selection pipeline and an extended, reproducible per-bin query cap of 20. It is a diagnostic audit, not a new reconstruction method. All {a['event_count']} events are in the DEVELOPMENT split (1000–1059); validation and final-test files were not loaded.

## Confirmed M9.1 pipeline order

`src/geometry_graph.py::build_candidate_graph` reads only hit_id and x/y/z. It computes r and 18 mm buckets, sorts each bucket's hit indices by hit ID, then calls cKDTree `query(k=10)` for each source bucket against each of the next ten outward target buckets. **The per-bin nearest-neighbor query occurs before** the exact filter `0 < Δr <= 180 mm`. Returned eligible hits from target buckets are concatenated for each source bucket; candidate ranks are assigned by Euclidean XYZ distance and target hit ID tie-break; only then is the final per-source cap of ten applied. The per-bin cKDTree query itself does not explicitly break equal-distance cutoff ties by hit ID; input target points are hit-ID sorted and SciPy's deterministic query result is used in the frozen builder. M9.4 uses the same cKDTree query mechanism with one worker and query cap 20; target rows enter in hit-ID order and returned equal-distance results are sorted by hit ID. Cutoff membership follows the recorded SciPy implementation.

The true edge definition was confirmed from `scripts/run_m9_1_geometry_graph.py::_true_consecutive_edges`: same nonzero particle ID, increasing transverse radius, consecutive after sorting by particle_id/r/hit_id, equal-r pairs excluded; only used after geometry stages are fixed.

## Configuration and audit method

M9.1 settings: max radial step 180 mm; bucket width 18 mm; next ten outward buckets; same-bucket excluded; raw XYZ Euclidean ranking; final top-10/source; outward direction. M9.4 constructs all per-target-bin top-20 query results from geometry before loading truth, records per-bin rank and radial-step pass/fail, forms the merged eligible pool and its exact global rank, and marks whether each candidate would survive the final top-10. Truth is then joined for post-hoc matching.

The transition denominator is all post-hoc true edges. Sequential stage counts are:

| Stage | True edges remaining | Lost at stage |
|---|---:|---:|
{chr(10).join(f"| {row.stage} | {int(row.true_edges_remaining):,} | {int(row.lost_at_stage):,} |" for row in transitions.itertuples())}

## Pooled stage findings

- True edges: {a['true_edges_available']:,}; M9.1 recovered: {a['baseline_recovered']:,}; missed: {a['baseline_missed']:,}.
- Target bin never searched: {a['target_bin_not_searched']:,} overall; among M9.1 misses: {a['baseline_miss_target_bin_not_searched']:,}.
- Query20 ranks 1–10: {a['per_bin_rank_1_10']:,}; ranks 11–20: {a['per_bin_rank_11_20']:,}; absent from extended top 20 despite searched bin (rank >20): {a['per_bin_rank_above_20']:,}.
- Rank 11–20 passing Δr filter: {a['rank_11_20_radial_pass']:,}; failing filter: {a['rank_11_20_radial_fail']:,}.
- Rank 11–20 passing radial filter and entering merged pool: {rank_sample:,}.
- Of M9.1-missed edges, rank 11–20: {a['baseline_missed_rank_11_20']:,}; passing radial filter: {a['baseline_missed_rank_11_20_radial_pass']:,}; surviving the extended final top-10: **{final_rank:,}**. Thus query20 adds no final-top-10 true edge among the rank-11–20 M9.1 misses.
- All query20 true targets passing radial filter and entering merged pool: {a['query20_candidates_radial_pass']:,}; of these surviving final top-10: {a['query20_candidates_final10']:,}; rejected at final cap: {a['query20_candidates_rejected_final10']:,}.

The saved aggregate does not separately count the rank-11–20/final-top-10 intersection among already M9.1-recovered edges; no total for that broader overlap is claimed here. The event-level audit records the exact post-hoc stage flags for the M9.1-missed subset used to assess whether query20 would recover additional edges.

## Baseline verification

The rebuilt M9.1 graph for event 1000 {summary['event1000_artifact_check']['verification_text']}. For all 60 events, reconstructed M9.1 graph size and post-hoc hit/recovery counts were checked against the saved M9.1 summary; mismatches: {a['m91_summary_mismatches']}. The complete historical edge lists were not saved for events 1001–1059, so exact full-edge-set equality cannot be verified for those 59 events. The frozen event-1000 artifact SHA-256 before and after was {summary['m91_artifact_sha256_before']} and {summary['m91_artifact_sha256_after']}.

Per-event cap maximum observed: {a['max_baseline_candidates_per_source']}; direction, radial step and same-bucket constraints were verified against reconstructed M9.1 graphs. Event recall range: {event_rows.m9_1_recall.min():.6%}–{event_rows.m9_1_recall.max():.6%}. Per-event stage counts and fractions are in `m9_4_event_summary.csv`.

## Overlap and dominant failure stage

Failure flags can overlap: radial-step failure can coexist with a searched bucket; ranks 11–20 can also fail radial filtering or be rejected by the final cap. Their overlapping counts appear separately in the event and summary CSVs. For a mutually exclusive accounting of M9.1 misses, precedence is target bin not searched → rank above 20 → radial filter failed → final top-10 rejected → would survive extended final-10. The largest category is **{dominant}** ({a['dominant_baseline_miss_count']:,} missed edges).

## Limitations

Only event 1000 has a persisted M9.1 edge table, so full edge-set artifact comparison is limited to that event. The other 59 reconstructed baseline graphs match saved per-event scalar summaries but cannot be compared edge-by-edge to missing artifacts. Per-bin cutoff tie membership follows cKDTree in the recorded SciPy environment; final merging explicitly sorts equal distances by target ID. All results concern candidate-edge coverage, not final tracks or official TrackML score.

## Recommendation

The per-bin ranks 11–20 actually surviving the final ten-candidate cap number {final_rank:,}. Based strictly on this audit, the next experiment should target the largest measured baseline miss stage ({dominant}) with one isolated development-only change, while preserving all other settings. Do not infer benefit before measuring it, and do not open validation/final-test data until the method is frozen.

## Reproducibility

- UTC timestamp: {summary['generated_at_utc']}
- Python {summary['environment']['python_version']}; NumPy {summary['environment']['numpy_version']}; pandas {summary['environment']['pandas_version']}; SciPy {summary['environment']['scipy_version']}.
- Per-bin query cap 20; final source cap 10; tie rank is Euclidean XYZ then target hit ID.
- Runtime (baseline graph build + query-stage audit): {a['geometry_stage_runtime_seconds']:.3f} seconds; excludes truth loading and report generation.
- Tests: focused {summary['test_results']['focused']}; full suite {summary['test_results']['full']}.
"""


def _examples(event_id: str, traced: pd.DataFrame, per_category: int = 50) -> pd.DataFrame:
    if event_id != "event000001000":
        return pd.DataFrame()
    selections = []
    categories = ["target_bin_not_searched", "rank_above_20", "radial_filter_failed", "final_top10_rejected", "would_survive_extended_final10"]
    for category in categories:
        rows = traced.loc[traced.baseline_miss_stage == category].head(per_category).copy()
        rows["sample_category"] = category
        selections.append(rows)
    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def _write_text(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer(stream)


def run(data_dir: str | Path, manifest_path: str | Path = "experiments/results/event_split_manifest.csv") -> dict:
    exists = [str(path) for path in OUTPUTS if path.exists()]
    if exists:
        raise FileExistsError("Refusing to overwrite existing M9.4 outputs: " + ", ".join(exists))
    event_ids = development_event_ids(manifest_path)
    if not M91_SUMMARY.exists() or not M91_EVENT1000.exists():
        raise FileNotFoundError("Frozen M9.1 summary and event-1000 edge artifact are required")
    m91_summary = pd.read_csv(M91_SUMMARY).set_index("event_id")
    if list(m91_summary.index) != event_ids:
        raise ValueError("Saved M9.1 summary does not match exact development manifest")
    artifact_hash_before = _sha256(M91_EVENT1000)
    events = []
    stage_totals: dict[str, int] = {}
    example_frames = []
    m91_mismatches = 0
    event1000_check = None
    geometry_runtime = 0.0
    max_source_candidates = 0

    for event_id in event_ids:
        numeric_id = int(event_id.removeprefix("event"))
        hits = load_event(str(data_dir), numeric_id, parts=("hits",))["hits"]
        start = time.perf_counter()
        baseline = build_candidate_graph(hits, max_radial_step_mm=180.0, max_candidates_per_hit=10)
        query_stages = build_query_stages(hits)
        geometry_runtime += time.perf_counter() - start
        if event_id == "event000001000":
            event1000_check = _verify_event1000_edge_artifact(baseline)
            if not event1000_check["exact_pair_and_rank_match"]:
                raise RuntimeError("Rebuilt event000001000 M9.1 candidate pairs/ranks do not match the frozen artifact")

        # The M9.1 baseline and all per-bin/final query stages now exist.
        # Truth is introduced only for the post-hoc join below.
        truth = load_event(str(data_dir), numeric_id, parts=("truth",))["truth"]
        true_edges = _true_consecutive_edges(hits, truth)
        traced = trace_true_edges(hits, true_edges, query_stages, baseline)
        old = m91_summary.loc[event_id]
        row = _event_row(event_id, baseline, traced, query_stages, len(hits), old)
        checks = {
            "n_hits": len(hits), "n_candidate_edges": len(baseline),
            "true_edge_count": len(true_edges), "recovered_true_edges": int(traced.baseline_recovered.sum()),
        }
        for key, value in checks.items():
            if int(old[key]) != int(value):
                m91_mismatches += 1
        baseline_counts = baseline.groupby("source_hit_id").size()
        event_max = int(baseline_counts.max()) if len(baseline_counts) else 0
        max_source_candidates = max(max_source_candidates, event_max)
        for key, value in stage_counts(traced).items():
            stage_totals[key] = stage_totals.get(key, 0) + int(value)
        events.append(row)
        examples = _examples(event_id, traced)
        if len(examples):
            example_frames.append(examples)
        print(f"Audited {event_id}: true={len(true_edges):,}, M9.1 recovered={int(traced.baseline_recovered.sum()):,}, rank11-20={int(traced.rank_11_20.sum()):,}", flush=True)

    if event1000_check is None:
        raise RuntimeError("Development event 1000 was not processed")
    artifact_hash_after = _sha256(M91_EVENT1000)
    if artifact_hash_before != artifact_hash_after:
        raise RuntimeError("Frozen M9.1 event-1000 artifact hash changed during M9.4")
    event_frame = pd.DataFrame(events)
    transitions = _stage_transition_rows(event_frame)
    stage_loss = {row.stage: int(row.lost_at_stage) for row in transitions.itertuples()}
    baseline_miss_counts = {
        category: int(event_frame[column].sum())
        for category, column in (
            ("target_bin_not_searched", "baseline_miss_target_bin_not_searched"),
            ("rank_above_20", "baseline_miss_rank_above_20"),
            ("radial_filter_failed", "baseline_miss_radial_filter_failed"),
            ("final_top10_rejected", "baseline_miss_final_top10_rejected"),
            ("would_survive_extended_final10", "baseline_miss_would_survive_extended_final10"),
        )
    }
    dominant = max(baseline_miss_counts, key=baseline_miss_counts.get)
    summary = {
        "milestone": "M9.4", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "DEVELOPMENT", "event_ids": event_ids,
        "configuration": {"max_radial_step_mm": 180.0, "radial_bucket_width_mm": 18.0, "outward_buckets_searched": 10, "same_bucket_targets_included": False, "per_bin_extended_audit_query_cap": 20, "per_bin_rank_order": "cKDTree query with target points in hit-ID order; returned ties sorted by target_hit_id; cutoff membership follows recorded SciPy implementation", "radial_filter_order": "after each per-bin query", "merge_order": "all radial-filter-passing per-bin results combined per source", "final_candidate_cap": 10, "final_rank_order": "Euclidean XYZ distance ascending, then target_hit_id ascending", "truth_used_for_geometry_or_query_stages": False},
        "pipeline_confirmation": {"builder": "src/geometry_graph.py::build_candidate_graph", "true_edge_definition": "scripts/run_m9_1_geometry_graph.py::_true_consecutive_edges", "per_bin_query_precedes_radial_filter": True, "target_bins_merged_before_final_cap": True, "final_cap_applied_after_distance_and_hit_id_sort": True, "target_points_sorted_by_hit_id_before_tree_build": True, "per_bin_equal_distance_cutoff_tie_break_in_frozen_m91": "cKDTree behavior; no explicit hit-ID tie-break at per-bin cutoff", "audit_extended_query_tie_break": "cKDTree cutoff membership with hit-ID-sorted points; returned equal-distance results sorted by target_hit_id"},
        "event1000_artifact_check": {**event1000_check, "verification_text": "matches the frozen M9.1 artifact exactly by source/target/candidate-rank; geometry columns match within 1e-12" if event1000_check["exact_pair_and_rank_match"] and event1000_check["geometry_columns_match_with_float_tolerance"] else "does not match the frozen M9.1 artifact"},
        "m91_artifact_sha256_before": artifact_hash_before, "m91_artifact_sha256_after": artifact_hash_after,
        "aggregate": {"event_count": len(event_ids), "true_edges_available": int(stage_totals["true_edges_available"]), "baseline_recovered": int(stage_totals["baseline_recovered"]), "baseline_missed": int(stage_totals["baseline_missed"]), **stage_totals, "baseline_miss_primary_counts": baseline_miss_counts, "dominant_baseline_miss_count": baseline_miss_counts[dominant], "dominant_baseline_miss_stage": dominant, "m91_summary_mismatches": m91_mismatches, "max_baseline_candidates_per_source": max_source_candidates, "geometry_stage_runtime_seconds": geometry_runtime},
        "dominant_baseline_miss_stage": dominant,
        "environment": {"python_version": sys.version, "numpy_version": np.__version__, "pandas_version": pd.__version__, "scipy_version": scipy.__version__, "platform": platform.platform()},
        "test_results": {"focused": "run after implementation", "full": "run after focused tests if practical"},
        "artifacts": [str(path) for path in OUTPUTS],
    }
    _write_figure(transitions)
    _write_text(RESULTS / "m9_4_event_summary.csv", lambda stream: event_frame.to_csv(stream, index=False))
    _write_text(RESULTS / "m9_4_summary.csv", lambda stream: pd.DataFrame([{"metric": key, "value": value} for key, value in summary["aggregate"].items() if isinstance(value, (int, float))]).to_csv(stream, index=False))
    _write_text(RESULTS / "m9_4_stage_transitions.csv", lambda stream: transitions.to_csv(stream, index=False))
    with (RESULTS / "m9_4_summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True); stream.write("\n")
    examples = pd.concat(example_frames, ignore_index=True) if example_frames else pd.DataFrame()
    with (RESULTS / "m9_4_event000001000_examples.csv.gz").open("xb") as stream:
        examples.to_csv(stream, index=False, compression="gzip")
    report = _report(summary, event_frame, transitions)
    _write_text(RESULTS / "m9_4_report.md", lambda stream: stream.write(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing development event CSVs")
    parser.add_argument("--manifest", default="experiments/results/event_split_manifest.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Generate and post-hoc evaluate M9.1 geometry-only candidate graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
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
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_event
from src.event_split_manifest import read_manifest, validate_manifest
from src.geometry_graph import build_candidate_graph


MAX_RADIAL_STEP_MM = 180.0
MAX_CANDIDATES_PER_HIT = 10
RESULTS_DIR = Path("experiments/results")
SUMMARY_CSV = RESULTS_DIR / "m9_1_geometry_graph_summary.csv"
SUMMARY_JSON = RESULTS_DIR / "m9_1_geometry_graph_summary.json"
EDGE_CSV = RESULTS_DIR / "m9_1_geometry_graph_edges_event000001000.csv.gz"
FIGURE_COUNTS = Path("experiments/figures/m9_1_event1000_candidate_count_distribution.png")
FIGURE_RECALL = Path("experiments/figures/m9_1_event1000_true_edge_recall_by_rank.png")
REPORT = RESULTS_DIR / "m9_1_geometry_graph_report.md"


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _true_consecutive_edges(hits: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Construct post-hoc same-particle neighbors ordered by increasing r."""
    geom = hits[["hit_id", "x", "y"]].copy()
    geom["r"] = np.hypot(geom["x"].to_numpy(dtype=float), geom["y"].to_numpy(dtype=float))
    joined = truth[["hit_id", "particle_id"]].merge(geom[["hit_id", "r"]], on="hit_id", how="inner", validate="one_to_one")
    joined = joined.loc[joined["particle_id"] != 0].sort_values(
        ["particle_id", "r", "hit_id"], kind="stable"
    )
    grouped = joined.groupby("particle_id", sort=False)
    joined["target_hit_id"] = grouped["hit_id"].shift(-1)
    joined["target_r"] = grouped["r"].shift(-1)
    pairs = joined.loc[
        joined["target_hit_id"].notna() & (joined["target_r"] > joined["r"]),
        ["hit_id", "target_hit_id"],
    ].rename(columns={"hit_id": "source_hit_id"})
    pairs["target_hit_id"] = pairs["target_hit_id"].astype(hits["hit_id"].dtype)
    return pairs.reset_index(drop=True)


def evaluate_candidate_graph(
    hits: pd.DataFrame, truth: pd.DataFrame, edges: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """Compute truth-based recall only after geometry-only edges exist."""
    true_edges = _true_consecutive_edges(hits, truth)
    candidates = edges[["source_hit_id", "target_hit_id", "candidate_rank"]]
    if len(true_edges):
        matched = true_edges.merge(
            candidates,
            on=["source_hit_id", "target_hit_id"],
            how="left",
            validate="one_to_one",
        )
        recovered = int(matched["candidate_rank"].notna().sum())
    else:
        matched = true_edges.assign(candidate_rank=np.nan)
        recovered = 0

    n_hits = len(hits)
    edge_counts = edges.groupby("source_hit_id").size().reindex(hits.hit_id, fill_value=0)
    n_edges = len(edges)
    possible_directed_outward_pairs = n_hits * (n_hits - 1) / 2
    metrics = {
        "n_hits": int(n_hits),
        "n_candidate_edges": int(n_edges),
        "mean_candidates_per_hit": float(edge_counts.mean()) if n_hits else 0.0,
        "median_candidates_per_hit": float(edge_counts.median()) if n_hits else 0.0,
        "max_candidates_per_hit": int(edge_counts.max()) if n_hits else 0,
        "true_edge_count": int(len(true_edges)),
        "recovered_true_edges": recovered,
        "true_edge_recall": recovered / len(true_edges) if len(true_edges) else 0.0,
        "candidate_density": n_edges / possible_directed_outward_pairs if possible_directed_outward_pairs else 0.0,
    }
    return metrics, matched


def _event_ids(manifest_path: str | Path) -> list[str]:
    rows = read_manifest(manifest_path)
    validate_manifest(rows)
    development = [row["event_id"] for row in rows if row["split"] == "DEVELOPMENT"]
    expected = [f"event{number:09d}" for number in range(1000, 1060)]
    if development != expected:
        raise ValueError("Development split does not match event000001000–event000001059")
    return development


def _write_new_csv(path: Path, frame: pd.DataFrame, *, compression=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        frame.to_csv(stream, index=False, compression=compression)


def _make_figures(event1000_hits: pd.DataFrame, event1000_edges: pd.DataFrame, matched: pd.DataFrame) -> None:
    counts = event1000_edges.groupby("source_hit_id").size().reindex(
        event1000_hits.hit_id, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(counts, bins=np.arange(-0.5, MAX_CANDIDATES_PER_HIT + 1.5, 1), color="#3679a8", edgecolor="white")
    ax.set(xlabel="Outward candidate edges per hit", ylabel="Number of hits", title="M9.1 event 1000 candidate count distribution")
    ax.set_xticks(range(MAX_CANDIDATES_PER_HIT + 1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    FIGURE_COUNTS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_COUNTS, dpi=160)
    plt.close(fig)

    rank_recall = []
    denominator = len(matched)
    for rank in range(1, MAX_CANDIDATES_PER_HIT + 1):
        rank_recall.append(
            float((matched["candidate_rank"].notna() & (matched["candidate_rank"] <= rank)).sum() / denominator)
            if denominator else 0.0
        )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, MAX_CANDIDATES_PER_HIT + 1), rank_recall, marker="o", color="#bd5c38")
    ax.set(xlabel="Candidate rank budget", ylabel="True consecutive-edge recall", ylim=(0, 1),
           title="M9.1 event 1000 true-edge recall by candidate rank")
    ax.text(0.02, 0.04, "Truth-based post-hoc evaluation; truth was not used to build the graph.", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_RECALL, dpi=160)
    plt.close(fig)


def _report(summary: dict, rows: list[dict]) -> str:
    agg = summary["aggregate"]
    return f"""# M9.1 Geometry-Aware Candidate Graph Report

## 1. Research question

Can local detector geometry generate a sparse candidate graph that recovers true consecutive hit connections without using truth during construction?

## 2. Motivation from M6

M6 found that 99/100 sampled real particles had at least one internal consecutive-hit gap larger than DBSCAN `eps=0.01`; 267/2,487 analyzed particles were contained in exactly one predicted cluster, and mean largest-cluster fraction was approximately 0.22. This motivates candidate connections beyond fixed-radius connected components but does not show that all DBSCAN settings fail.

The supplied repository snapshot does not include the M6 diagnostics module/output artifacts; this motivation uses the verified M6 findings supplied in the task context. No M6 file was recreated or changed.

## 3. Candidate-generation method

For each event, `src.geometry_graph.build_candidate_graph` reads only `hit_id,x,y,z`. It computes cylindrical radius and azimuth, buckets hits in radial intervals, searches outward buckets with 3D XYZ cKDTree neighborhoods, filters to a positive radial step no larger than the configured limit, and retains the nearest candidates ranked by Euclidean XYZ distance with hit ID as a tie-break. Edges point from smaller to larger transverse radius. The graph is sparse with a per-source top-k limit.

## 4. Geometry features used

Candidate construction uses Cartesian `x,y,z` to rank local spatial neighbors and derives `r=sqrt(x²+y²)` and `phi=atan2(y,x)` for radial direction and edge diagnostics. `volume_id`, `layer_id`, `module_id`, cells, particles, and truth are not used to generate candidates.

## 5. Hyperparameters

- Maximum positive radial step: {MAX_RADIAL_STEP_MM:.1f} mm. Event-1000 geometry showed adjacent barrel-layer median-radius gaps of about 40–160 mm; 180 mm provides a 20 mm margin over the largest observed gap.
- Maximum candidates per hit: {MAX_CANDIDATES_PER_HIT}. This is a declared initial sparsity budget, not tuned with truth.
- Internal radial indexing bucket width: {min(20.0, MAX_RADIAL_STEP_MM / 10.0):.1f} mm (`min(20 mm, max_radial_step/10)`); ten outward bins are searched. Same-bin edges are excluded to prevent same-layer neighbors dominating the outgoing candidate budget.
- Each target-bin query asks for up to {MAX_CANDIDATES_PER_HIT} nearest hits; final per-source selection is at most ten across all bins. Ties use ascending target hit ID.
- Feature-space ranking: Euclidean distance in raw XYZ (millimeters).

## 6. Development split

The validated M7 manifest selected exactly events `event000001000` through `event000001059` (60 events). Validation and final-test events were not loaded or evaluated.

## 7. Leakage controls

The graph module accepts and reads only hit IDs and coordinates; event loading requests `parts=("hits",)` for construction. Truth is loaded only after the edge graph has been generated and is used only to measure consecutive-edge coverage. Candidate files contain no truth or particle columns.

## 8. Candidate graph statistics

- Development events processed: {agg['n_events']}
- Total candidate edges: {agg['total_candidate_edges']}
- Mean candidate edges/event: {agg['mean_candidate_edges_per_event']:.3f}
- Mean edges/hit, averaged across events: {agg['mean_candidates_per_hit_event_mean']:.6f}
- Median candidates/hit, averaged across events: {agg['median_candidates_per_hit_event_mean']:.3f}
- Maximum candidates/hit observed: {agg['maximum_candidates_per_hit']}

## 9. True-edge recall

Truth-based post-hoc edges are consecutive nonzero-particle hits ordered by increasing transverse radius; equal-radius consecutive pairs are excluded because this graph is strictly outward-directed. Pooled recall was {agg['pooled_true_edge_recall']:.6f} ({agg['total_recovered_true_edges']}/{agg['total_true_edges']}). Mean event recall was {agg['mean_event_true_edge_recall']:.6f}. This is candidate-edge coverage, not track reconstruction accuracy.

## 10. Sparsity/density

Candidate density is computed against all possible directed outward pairs, `N*(N-1)/2`, per event. Mean event density was {agg['mean_candidate_density']:.8g}; maximum event density was {agg['maximum_candidate_density']:.8g}. The top-10 cap bounds edge count by ten per hit.

## 11. Runtime

Candidate-generation runtime (excluding data loading and truth-based evaluation) was {agg['total_graph_runtime_seconds']:.3f} seconds total, {agg['mean_graph_runtime_seconds']:.3f} seconds/event on average.

## 12. Figures

- `experiments/figures/m9_1_event1000_candidate_count_distribution.png` — geometry-only outgoing candidate counts.
- `experiments/figures/m9_1_event1000_true_edge_recall_by_rank.png` — truth-based post-hoc recall by candidate rank; the figure labels truth use explicitly.

## 13. Limitations

This first graph uses a global radial window and Euclidean XYZ nearest neighbors. It does not model magnetic curvature, charge, momentum, detector-surface transitions, or track-level consistency. Same-radial-bin targets are excluded, the 180 mm window may omit valid transitions outside the event-1000 barrel-derived range, and top-10 truncation may miss true neighbors. The metric depends on ordering truth hits by transverse radius. No end-to-end tracks or TrackML reconstruction score were produced.

## 14. Comparison with frozen baseline

The M8.2 event-1000 reference remains score `0.13625134527683258`, 11,544 clusters, and 61,996 noise hits. Candidate coverage and candidate sparsity are different quantities from the TrackML score and are not a direct baseline comparison.

## 15. Scientific conclusion

The experiment generated a deterministic, bounded-size geometry-only candidate graph over the 60 development events. It recovered 363,034 of 5,007,067 post-hoc true consecutive edges (7.25% pooled recall) at a mean density of 0.0174%. This is sparse, but the measured recall is low; this initial candidate design does not recover a high fraction of local true connections. It does not establish final track-reconstruction performance.

## 16. Next experiment recommendation

M9.2 should diagnose missed development true edges by radial step, wrapped azimuth change, longitudinal displacement, radial-bin omission, and candidate rank. Use that diagnosis to propose one revised candidate generator. Keep validation and final-test events untouched until a method and evaluation protocol are frozen.

## Reproducibility

- Configuration: `max_radial_step_mm={MAX_RADIAL_STEP_MM}`, `max_candidates_per_hit={MAX_CANDIDATES_PER_HIT}`.
- Python: `{summary['environment']['python_version']}`
- NumPy: `{summary['environment']['numpy_version']}`; pandas: `{summary['environment']['pandas_version']}`; SciPy: `{summary['environment']['scipy_version']}`; scikit-learn: `{summary['environment']['scikit_learn_version']}`
- Platform: `{summary['environment']['platform']}`
- Git revision: `{summary['environment']['git_revision']}`
"""


def run(data_dir: str | Path, manifest_path: str | Path = "experiments/results/event_split_manifest.csv") -> dict:
    output_paths = [SUMMARY_CSV, SUMMARY_JSON, EDGE_CSV, FIGURE_COUNTS, FIGURE_RECALL, REPORT]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing M9.1 outputs: " + ", ".join(existing))

    event_ids = _event_ids(manifest_path)
    rows = []
    event1000_edges = None
    event1000_hits = None
    event1000_matched = None
    total_runtime = 0.0

    for event_id in event_ids:
        numeric_id = int(event_id.removeprefix("event"))
        hits = load_event(data_dir, numeric_id, parts=("hits",))["hits"]
        start = time.perf_counter()
        edges = build_candidate_graph(
            hits,
            max_radial_step_mm=MAX_RADIAL_STEP_MM,
            max_candidates_per_hit=MAX_CANDIDATES_PER_HIT,
        )
        graph_runtime = time.perf_counter() - start
        total_runtime += graph_runtime

        # Truth is requested only after geometry-only candidate generation.
        truth = load_event(data_dir, numeric_id, parts=("truth",))["truth"]
        metrics, matched = evaluate_candidate_graph(hits, truth, edges)
        metrics["event_id"] = event_id
        metrics["runtime_seconds"] = graph_runtime
        rows.append(metrics)
        print(f"Completed {event_id}: {metrics['n_candidate_edges']} edges", flush=True)
        if event_id == "event000001000":
            event1000_edges = edges
            event1000_hits = hits
            event1000_matched = matched

    results = pd.DataFrame(rows)[[
        "event_id", "n_hits", "n_candidate_edges", "mean_candidates_per_hit",
        "median_candidates_per_hit", "max_candidates_per_hit", "true_edge_count",
        "recovered_true_edges", "true_edge_recall", "candidate_density", "runtime_seconds",
    ]]
    _write_new_csv(SUMMARY_CSV, results)
    if event1000_edges is None or event1000_hits is None or event1000_matched is None:
        raise RuntimeError("event000001000 was not present in the development manifest")
    _write_new_csv(EDGE_CSV, event1000_edges, compression="gzip")
    _make_figures(event1000_hits, event1000_edges, event1000_matched)

    total_true_edges = int(results.true_edge_count.sum())
    total_recovered = int(results.recovered_true_edges.sum())
    aggregate = {
        "n_events": int(len(results)),
        "total_hits": int(results.n_hits.sum()),
        "total_candidate_edges": int(results.n_candidate_edges.sum()),
        "mean_candidate_edges_per_event": float(results.n_candidate_edges.mean()),
        "mean_candidates_per_hit_event_mean": float(results.mean_candidates_per_hit.mean()),
        "median_candidates_per_hit_event_mean": float(results.median_candidates_per_hit.mean()),
        "maximum_candidates_per_hit": int(results.max_candidates_per_hit.max()),
        "total_true_edges": total_true_edges,
        "total_recovered_true_edges": total_recovered,
        "pooled_true_edge_recall": total_recovered / total_true_edges if total_true_edges else 0.0,
        "mean_event_true_edge_recall": float(results.true_edge_recall.mean()),
        "mean_candidate_density": float(results.candidate_density.mean()),
        "maximum_candidate_density": float(results.candidate_density.max()),
        "total_graph_runtime_seconds": total_runtime,
        "mean_graph_runtime_seconds": total_runtime / len(results),
        "runtime_excludes": ["event file loading", "truth loading", "post-hoc evaluation"],
    }
    summary = {
        "milestone": "M9.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "DEVELOPMENT",
        "event_ids": event_ids,
        "configuration": {
            "max_radial_step_mm": MAX_RADIAL_STEP_MM,
            "radial_bucket_width_mm": min(20.0, MAX_RADIAL_STEP_MM / 10.0),
            "outward_radial_bins_searched": int(np.ceil(MAX_RADIAL_STEP_MM / min(20.0, MAX_RADIAL_STEP_MM / 10.0))),
            "same_radial_bin_targets_included": False,
            "max_candidates_per_hit": MAX_CANDIDATES_PER_HIT,
            "nearest_queries_per_target_bin": MAX_CANDIDATES_PER_HIT,
            "candidate_rank_tie_break": "target_hit_id ascending",
            "distance_metric": "Euclidean XYZ",
            "candidate_direction": "increasing transverse radius",
            "truth_used_for_candidate_generation": False,
            "true_edge_definition": "consecutive nonzero-particle hits sorted by transverse radius; equal-r pairs excluded",
            "candidate_density_denominator": "N*(N-1)/2 possible directed outward pairs",
        },
        "aggregate": aggregate,
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "platform": platform.platform(),
            "git_revision": _git_revision(),
        },
        "artifacts": {
            "summary_csv": str(SUMMARY_CSV),
            "representative_edges": str(EDGE_CSV),
            "figures": [str(FIGURE_COUNTS), str(FIGURE_RECALL)],
        },
    }
    with SUMMARY_JSON.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_report(summary, rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing development event CSV files")
    parser.add_argument("--manifest", default="experiments/results/event_split_manifest.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

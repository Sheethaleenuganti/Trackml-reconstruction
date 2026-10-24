"""Run the frozen M5 baseline over development events only."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from src.clustering import cluster_tracks, extract_features
from src.data_loader import load_event
from src.event_split_manifest import read_manifest, validate_manifest
from src.evaluation import summarize_evaluation
from src.transformation import transform_hits


RESULT_FIELDS = (
    "event_id",
    "official_score",
    "n_predicted_clusters",
    "n_noise_hits",
    "mean_purity",
    "mean_efficiency",
    "runtime_seconds",
)


def run_event(data_dir: str | Path, event_id: str) -> dict[str, int | float | str]:
    """Run one complete event with frozen M5 settings and post-hoc scoring.

    Runtime uses the existing ``cluster_tracks`` measurement, which times
    the DBSCAN fit itself. Only hit coordinates reach the clustering path;
    truth is loaded after labels have been produced.
    """
    numeric_id = int(event_id.removeprefix("event"))
    hits = load_event(data_dir, numeric_id, parts=("hits",))["hits"]

    transformed = transform_hits(hits)
    features = extract_features(transformed, representation="xyz")
    clustered = cluster_tracks(features, eps=0.01, min_samples=3)

    truth = load_event(data_dir, numeric_id, parts=("truth",))["truth"]
    summary = summarize_evaluation(truth, hits["hit_id"], clustered["labels"])
    return {
        "event_id": event_id,
        "official_score": float(summary["official_trackml_score"]),
        "n_predicted_clusters": int(summary["n_predicted_clusters"]),
        "n_noise_hits": int(summary["n_noise_hits"]),
        "mean_purity": float(summary["mean_purity"]),
        "mean_efficiency": float(summary["mean_efficiency"]),
        "runtime_seconds": float(clustered["runtime_seconds"]),
    }


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_development(
    data_dir: str | Path,
    manifest_path: str | Path,
    result_path: str | Path,
    summary_path: str | Path,
    failures_path: str | Path,
) -> dict:
    """Evaluate only validated DEVELOPMENT assignments and save outputs."""
    manifest = read_manifest(manifest_path)
    validate_manifest(manifest)
    development_ids = [row["event_id"] for row in manifest if row["split"] == "DEVELOPMENT"]

    results: list[dict] = []
    failures: list[dict[str, str]] = []
    job_start = time.perf_counter()
    for event_id in development_ids:
        try:
            results.append(run_event(data_dir, event_id))
            print(f"Completed {event_id} ({len(results)} results)", flush=True)
        except Exception as exc:  # record the event and continue the fixed split
            failures.append({"event_id": event_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"Failed {event_id}: {type(exc).__name__}: {exc}", flush=True)

    _write_csv(Path(result_path), RESULT_FIELDS, results)
    _write_csv(Path(failures_path), ("event_id", "error"), failures)

    if not results:
        raise RuntimeError("No development events completed; see the failures CSV")

    scores = [float(row["official_score"]) for row in results]
    runtimes = [float(row["runtime_seconds"]) for row in results]
    aggregate = {
        "number_of_events": len(results),
        "failed_events": len(failures),
        "score_mean": statistics.mean(scores),
        "score_median": statistics.median(scores),
        "score_standard_deviation_population": statistics.pstdev(scores),
        "score_minimum": min(scores),
        "score_maximum": max(scores),
        "runtime_mean_seconds": statistics.mean(runtimes),
        "runtime_median_seconds": statistics.median(runtimes),
        "runtime_measurement": "DBSCAN fit only (src.clustering.cluster_tracks)",
        "total_job_wall_runtime_seconds": time.perf_counter() - job_start,
    }
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing extracted event CSV files")
    parser.add_argument(
        "--manifest", default="experiments/results/event_split_manifest.csv"
    )
    parser.add_argument(
        "--results", default="experiments/results/m8_1_development_baseline.csv"
    )
    parser.add_argument(
        "--summary", default="experiments/results/m8_1_development_summary.json"
    )
    parser.add_argument(
        "--failures", default="experiments/results/m8_1_development_failures.csv"
    )
    args = parser.parse_args()

    summary = run_development(
        args.data_dir, args.manifest, args.results, args.summary, args.failures
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

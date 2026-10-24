"""Run the frozen M5 configuration twice on event 1000 and save provenance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import trackml
from trackml.score import score_event

# Direct ``python scripts/<runner>.py`` execution puts scripts/ first on
# sys.path; add the repository root so the established src package resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import cluster_tracks, extract_features
from src.data_loader import load_event
from src.evaluation import labels_to_submission
from src.transformation import transform_hits


EVENT_ID = "event000001000"
NUMERIC_EVENT_ID = 1000
EPS = 0.01
MIN_SAMPLES = 3
FEATURE_NAMES = ["x2", "y2", "z2"]
RESULTS_DIR = Path("experiments/results")
LABELS_PATH = RESULTS_DIR / "m8_2_event1000_cluster_labels.csv"
SUBMISSION_PATH = RESULTS_DIR / "m8_2_event1000_submission.csv"
FEATURES_PATH = RESULTS_DIR / "m8_2_event1000_features.npy"
MAPPING_HASH_PATH = RESULTS_DIR / "m8_2_event1000_cluster_mapping_sha256.txt"
PROVENANCE_PATH = RESULTS_DIR / "m8_2_event1000_provenance.json"
DETERMINISM_PATH = RESULTS_DIR / "m8_2_event1000_determinism.md"
REPORT_PATH = RESULTS_DIR / "m8_2_event1000_provenance_report.md"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(contiguous.tobytes(order="C"))


def canonicalize_mapping(hit_ids, labels) -> pd.DataFrame:
    """Return hit-ordered labels with non-noise IDs keyed by minimum hit ID."""
    hits = np.asarray(hit_ids)
    raw_labels = np.asarray(labels)
    if hits.ndim != 1 or raw_labels.ndim != 1 or len(hits) != len(raw_labels):
        raise ValueError("hit_ids and labels must be one-dimensional and equally sized")
    if len(np.unique(hits)) != len(hits):
        raise ValueError("hit_ids must be unique")

    frame = pd.DataFrame(
        {"hit_id": hits.astype(np.int64), "cluster_label": raw_labels.astype(np.int64)}
    )
    cluster_minima = (
        frame.loc[frame["cluster_label"] != -1]
        .groupby("cluster_label", sort=False)["hit_id"]
        .min()
        .sort_values(kind="stable")
    )
    canonical_ids = {label: index for index, label in enumerate(cluster_minima.index)}
    frame["cluster_label"] = frame["cluster_label"].map(
        lambda label: -1 if label == -1 else canonical_ids[label]
    )
    return frame.sort_values("hit_id", kind="stable").reset_index(drop=True)


def canonical_mapping_sha256(hit_ids, labels) -> str:
    mapping = canonicalize_mapping(hit_ids, labels)
    canonical_array = mapping[["hit_id", "cluster_label"]].to_numpy(dtype=np.int64)
    return _array_sha256(canonical_array)


def _feature_hash(features: np.ndarray) -> str:
    return _array_sha256(features)


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _trackml_provenance() -> dict:
    module_path = Path(trackml.__file__).resolve()
    source_revision = _git_revision(module_path.parent)
    if source_revision is None:
        source_revision = _git_revision(module_path.parent.parent)
    distributions = importlib.metadata.packages_distributions().get("trackml", [])
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "module_path": str(module_path),
        "distribution_versions": versions,
        "source_revision": source_revision,
    }


def _load_features(data_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    # Ground truth and particles are deliberately not requested on this path.
    hits = load_event(data_dir, NUMERIC_EVENT_ID, parts=("hits",))["hits"]
    transformed = transform_hits(hits)
    features = extract_features(transformed, representation="xyz")
    return hits, features


def _cluster(features: np.ndarray) -> np.ndarray:
    result = cluster_tracks(features, eps=EPS, min_samples=MIN_SAMPLES)
    return np.asarray(result["labels"])


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        frame.to_csv(stream, index=False)


def _mapping_equal(left_ids, left_labels, right_ids, right_labels) -> bool:
    left = canonicalize_mapping(left_ids, left_labels)
    right = canonicalize_mapping(right_ids, right_labels)
    return left.equals(right)


def _determinism_report(
    *,
    raw_labels_equal: bool,
    canonical_equal: bool,
    submissions_equal: bool,
    scores_equal: bool,
    input_hashes_equal: bool,
    first_label_hash: str,
    second_label_hash: str,
    first_mapping_hash: str,
    second_mapping_hash: str,
    first_submission_mapping_hash: str,
    second_submission_mapping_hash: str,
    first_score: float | None,
    second_score: float | None,
) -> str:
    return "\n".join(
        [
            "# M8.2 Event-1000 Determinism Check",
            "",
            "Two complete event-1000 loads, feature constructions, and frozen DBSCAN runs were compared.",
            "",
            f"- Raw DBSCAN labels exactly equal: **{raw_labels_equal}**",
            f"- Canonical hit-to-cluster mappings equal: **{canonical_equal}**",
            f"- Canonical hit-to-submission-track mappings equal: **{submissions_equal}**",
            f"- Scores equal: **{scores_equal}**",
            f"- Raw hit-ID and feature input hashes equal: **{input_hashes_equal}**",
            f"- First raw label SHA-256: `{first_label_hash}`",
            f"- Second raw label SHA-256: `{second_label_hash}`",
            f"- First canonical mapping SHA-256: `{first_mapping_hash}`",
            f"- Second canonical mapping SHA-256: `{second_mapping_hash}`",
            f"- First saved-submission mapping SHA-256: `{first_submission_mapping_hash}`",
            f"- Second submission mapping SHA-256: `{second_submission_mapping_hash}`",
            f"- First score: `{first_score}`",
            f"- Second score: `{second_score}`",
            "",
        ]
    )


def _provenance_report(provenance: dict) -> str:
    counts = provenance["clustering_output"]
    score = provenance["official_score"]
    env = provenance["environment"]
    fp = provenance["fingerprints"]
    return f"""# M8.2 Event-1000 Provenance Report

## 1. Objective

Create a reproducible provenance record for the current frozen M5 implementation on event 1000. The historical M5 score discrepancy is not being repaired or retuned.

## 2. Configuration

- Event: `{provenance['event_id']}`
- Complete event, no hit subsampling
- Representation: unscaled XYZ
- DBSCAN: `eps={EPS}`, `min_samples={MIN_SAMPLES}`, Euclidean metric
- Truth and particle data are excluded from feature construction and clustering.

## 3. Input event

- Hit rows: {provenance['input']['hit_rows']}
- Unique hit IDs: {provenance['input']['unique_hit_ids']}
- Input hit-ID SHA-256: `{fp['raw_input_hit_id_sha256']}`

## 4. Feature construction

The existing M4 `transform_hits` and M5 `extract_features(..., representation="xyz")` functions were reused. Feature order: `{', '.join(provenance['features']['names'])}`. Dtype: `{provenance['features']['dtype']}`. Shape: `{provenance['features']['shape']}`. Feature matrix SHA-256: `{fp['feature_matrix_sha256']}`. The exact matrix is saved as `m8_2_event1000_features.npy`.

## 5. DBSCAN configuration

The existing `src.clustering.cluster_tracks` was called with `eps=0.01` and `min_samples=3`. Only the feature matrix entered clustering. Truth was loaded after both clustering runs.

## 6. Clustering output

- Non-noise clusters: {counts['n_clusters']}
- Noise hits (raw DBSCAN label `-1`): {counts['n_noise_hits']}
- Raw label rows: {counts['n_labels']}
- Cluster-size distribution (non-noise): min {counts['cluster_size_min']}, median {counts['cluster_size_median']}, mean {counts['cluster_size_mean']:.6f}, max {counts['cluster_size_max']}
- Cluster-label artifact: `m8_2_event1000_cluster_labels.csv`

## 7. Submission construction

The existing `labels_to_submission` function generated `hit_id,track_id`. The CSV was saved, read back from disk, and that read-back dataframe was passed to the official scorer. Its row count is {score['submission_rows']} with {score['submission_unique_hit_ids']} unique hit IDs.

## 8. Official score

- Official TrackML score: **{score['value']:.17g}**
- Scorer: `trackml.score.score_event`
- Truth rows: {score['truth_rows']}
- Truth unique hit IDs: {score['truth_unique_hit_ids']}
- Scorer truth-input SHA-256: `{fp['scorer_truth_input_sha256']}`
- Saved submission file SHA-256: `{fp['submission_file_sha256']}`

## 9. Artifact inventory

- `m8_2_event1000_cluster_labels.csv`
- `m8_2_event1000_submission.csv`
- `m8_2_event1000_features.npy`
- `m8_2_event1000_cluster_mapping_sha256.txt`
- `m8_2_event1000_provenance.json`
- `m8_2_event1000_determinism.md`
- `m8_2_event1000_provenance_report.md`

## 10. SHA-256 fingerprints

- Raw label array: `{fp['raw_dbscan_labels_sha256']}`
- Canonical hit-to-cluster mapping: `{fp['canonical_cluster_mapping_sha256']}`
- Feature matrix: `{fp['feature_matrix_sha256']}`
- Input hit IDs: `{fp['raw_input_hit_id_sha256']}`
- Saved submission file: `{fp['submission_file_sha256']}`
- Scorer truth input: `{fp['scorer_truth_input_sha256']}`

## 11. Environment

- Python: `{env['python_version']}`
- NumPy: `{env['numpy_version']}`
- pandas: `{env['pandas_version']}`
- SciPy: `{env['scipy_version']}`
- scikit-learn: `{env['scikit_learn_version']}`
- TrackML package/source: `{json.dumps(env['trackml'], sort_keys=True)}`
- Platform: `{env['platform']}`
- Repository commit: `{env['repository_commit']}`
- Runner timestamp (UTC): `{provenance['runner_timestamp_utc']}`

## 12. Determinism check

See `m8_2_event1000_determinism.md`. Two complete runs had identical canonical cluster mappings, canonical submission mappings, scores, and input hashes. Raw DBSCAN labels were {'identical' if provenance['determinism']['raw_labels_equal'] else 'not identical'}. The first-run label, submission, and feature artifacts were preserved. Their original timestamp was not captured because the first invocation stopped at a dtype-sensitive comparison; `first_run_timestamp_utc` is therefore null. The subsequent type-normalized comparison verified the saved mappings.

## 13. Relationship to historical M5

- Historical M5 recorded score = `0.026887569576501846`.
- Current reproducible event-1000 score = `{score['value']:.17g}`.
- Historical per-hit mapping is unavailable, so direct mapping-level comparison is impossible.
- The discrepancy remains unresolved.
- M8.2 does NOT invalidate M5.
- M8.2 does NOT establish that M8.1 or M5 is wrong.

## 14. Scientific conclusion

This run establishes a new reproducible provenance point for the current implementation only. It does not reproduce or explain the historical M5 score.
"""


def run(data_dir: str | Path, *, verify_existing_first_run: bool = False) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    first_artifacts = (LABELS_PATH, SUBMISSION_PATH, FEATURES_PATH)
    final_artifacts = (MAPPING_HASH_PATH, PROVENANCE_PATH, DETERMINISM_PATH, REPORT_PATH)
    present_first = [path.exists() for path in first_artifacts]
    present_final = [path.exists() for path in final_artifacts]

    if verify_existing_first_run:
        if not all(present_first) or PROVENANCE_PATH.exists() or REPORT_PATH.exists():
            raise FileExistsError(
                "--verify-existing-first-run requires first-run labels, submission, "
                "and features, with provenance/master report absent. An existing "
                "mapping fingerprint will be checked and preserved."
            )
        saved_labels = pd.read_csv(LABELS_PATH)
        saved_submission = pd.read_csv(SUBMISSION_PATH)
        first_labels = saved_labels["cluster_label"].to_numpy(dtype=np.int64)
        first_features = np.load(FEATURES_PATH, allow_pickle=False)
        hit_ids = saved_labels["hit_id"].to_numpy(dtype=np.int64)
        first_timestamp = None
        input_hit_id_dtype = None
    else:
        if any(present_first) or any(present_final):
            raise FileExistsError(
                "Refusing to overwrite existing M8.2 artifacts. Use "
                "--verify-existing-first-run only to preserve and verify existing "
                "first-run artifacts."
            )
        hits, first_features = _load_features(data_dir)
        if first_features.shape != (len(hits), 3):
            raise ValueError(f"Unexpected feature matrix shape: {first_features.shape}")
        first_labels = _cluster(first_features)
        if len(first_labels) != len(hits):
            raise ValueError("DBSCAN label count does not match the input hit count")
        hit_ids = hits["hit_id"].to_numpy()
        input_hit_id_dtype = str(hits["hit_id"].dtype)
        first_submission = labels_to_submission(hits["hit_id"], first_labels)

        # Persist first-run outputs before starting the independent second run.
        labels_frame = pd.DataFrame({"hit_id": hit_ids, "cluster_label": first_labels})
        _write_csv_exclusive(LABELS_PATH, labels_frame)
        _write_csv_exclusive(SUBMISSION_PATH, first_submission[["hit_id", "track_id"]])
        with FEATURES_PATH.open("xb") as stream:
            np.save(stream, first_features, allow_pickle=False)
        saved_labels = pd.read_csv(LABELS_PATH)
        saved_submission = pd.read_csv(SUBMISSION_PATH)
        first_timestamp = timestamp

    if len(saved_labels) != len(hit_ids) or saved_labels["hit_id"].duplicated().any():
        raise ValueError("Saved cluster-label artifact failed row/identity validation")
    if len(saved_submission) != len(hit_ids) or saved_submission["hit_id"].duplicated().any():
        raise ValueError("Saved submission failed row/identity validation")
    if not np.array_equal(saved_labels["hit_id"].to_numpy(dtype=np.int64), hit_ids):
        raise ValueError("Saved label hit IDs are not in original input order")
    if not np.array_equal(saved_submission["hit_id"].to_numpy(dtype=np.int64), hit_ids):
        raise ValueError("Saved submission hit IDs are not in original input order")

    second_hits, second_features = _load_features(data_dir)
    second_labels = _cluster(second_features)
    second_submission = labels_to_submission(second_hits["hit_id"], second_labels)
    if input_hit_id_dtype is None:
        input_hit_id_dtype = str(second_hits["hit_id"].dtype)

    input_hashes_equal = (
        _array_sha256(hit_ids.astype(np.int64, copy=False))
        == _array_sha256(second_hits["hit_id"].to_numpy(dtype=np.int64))
        and _feature_hash(first_features) == _feature_hash(second_features)
    )
    first_label_hash = _array_sha256(np.asarray(first_labels, dtype=np.int64))
    second_label_hash = _array_sha256(np.asarray(second_labels, dtype=np.int64))
    first_mapping_hash = canonical_mapping_sha256(hit_ids, first_labels)
    second_mapping_hash = canonical_mapping_sha256(
        second_hits["hit_id"].to_numpy(), second_labels
    )
    first_submission_canonical = canonicalize_mapping(
        saved_submission["hit_id"], saved_submission["track_id"]
    )
    second_submission_canonical = canonicalize_mapping(
        second_submission["hit_id"], second_submission["track_id"]
    )
    submissions_equal = first_submission_canonical.equals(second_submission_canonical)
    raw_labels_equal = np.array_equal(first_labels, second_labels)
    canonical_equal = first_mapping_hash == second_mapping_hash

    # Ground truth is loaded only after both unsupervised clustering runs.
    truth = load_event(data_dir, NUMERIC_EVENT_ID, parts=("truth",))["truth"]
    scorer_truth = truth[["hit_id", "particle_id", "weight"]]
    score_value = float(score_event(scorer_truth, saved_submission))
    second_score_value = float(score_event(scorer_truth, second_submission))
    scores_equal = score_value == second_score_value

    first_submission_mapping_hash = canonical_mapping_sha256(
        saved_submission["hit_id"].to_numpy(), saved_submission["track_id"].to_numpy()
    )
    second_submission_mapping_hash = canonical_mapping_sha256(
        second_submission["hit_id"].to_numpy(), second_submission["track_id"].to_numpy()
    )
    determinism_text = _determinism_report(
        raw_labels_equal=raw_labels_equal,
        canonical_equal=canonical_equal,
        submissions_equal=submissions_equal,
        scores_equal=scores_equal,
        input_hashes_equal=input_hashes_equal,
        first_label_hash=first_label_hash,
        second_label_hash=second_label_hash,
        first_mapping_hash=first_mapping_hash,
        second_mapping_hash=second_mapping_hash,
        first_submission_mapping_hash=first_submission_mapping_hash,
        second_submission_mapping_hash=second_submission_mapping_hash,
        first_score=score_value,
        second_score=second_score_value,
    )
    DETERMINISM_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if verify_existing_first_run and DETERMINISM_PATH.exists() else "x"
    with DETERMINISM_PATH.open(mode, encoding="utf-8", newline="\n") as stream:
        if mode == "a":
            stream.write("\n---\n\n## Corrected dtype-normalized comparison\n\n")
            stream.write(
                "The initial comparison above treated TrackML input `hit_id` values "
                "(int32) and CSV-round-tripped values (int64) as unequal because "
                "pandas `DataFrame.equals` is dtype-sensitive. The verification below "
                "normalizes IDs and compares mapping identity. First-run artifacts "
                "were preserved unchanged.\n\n"
            )
        stream.write(determinism_text)

    if not (canonical_equal and submissions_equal and scores_equal and input_hashes_equal):
        raise RuntimeError(
            "M8.2 determinism check failed; first-run artifacts were preserved. "
            f"See {DETERMINISM_PATH}."
        )

    non_noise_sizes = pd.Series(first_labels)[pd.Series(first_labels) != -1].value_counts()
    canonical_text = f"{first_mapping_hash}\n"
    if MAPPING_HASH_PATH.exists():
        if MAPPING_HASH_PATH.read_text(encoding="ascii") != canonical_text:
            raise RuntimeError(
                "Existing first-run mapping fingerprint disagrees with the preserved labels."
            )
    else:
        with MAPPING_HASH_PATH.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(canonical_text)

    feature_bytes_hash = _feature_hash(first_features)
    truth_hash_frame = scorer_truth.sort_values("hit_id", kind="stable").reset_index(drop=True)
    truth_hash = _sha256_bytes(
        "\n".join(
            f"{int(row.hit_id)}\t{int(row.particle_id)}\t{float(row.weight).hex()}"
            for row in truth_hash_frame.itertuples(index=False)
        ).encode("utf-8")
    )
    label_artifact_hash = _sha256_bytes(LABELS_PATH.read_bytes())
    feature_file_hash = _sha256_bytes(FEATURES_PATH.read_bytes())
    provenance = {
        "event_id": EVENT_ID,
        "runner_timestamp_utc": timestamp,
        "first_run_timestamp_utc": first_timestamp,
        "configuration": {
            "event_id": EVENT_ID,
            "complete_event": True,
            "hit_subsampling": False,
            "representation": "unscaled XYZ",
            "feature_names": FEATURE_NAMES,
            "feature_order": FEATURE_NAMES,
            "dbscan": {"eps": EPS, "min_samples": MIN_SAMPLES, "metric": "euclidean"},
            "truth_used_during_clustering": False,
        },
        "input": {
            "hit_rows": int(len(saved_labels)),
            "unique_hit_ids": int(saved_labels["hit_id"].nunique()),
            "raw_input_hit_id_dtype": input_hit_id_dtype,
        },
        "features": {
            "names": FEATURE_NAMES,
            "dtype": str(first_features.dtype),
            "shape": list(first_features.shape),
            "order": "x2, y2, z2; input hit row order preserved",
        },
        "clustering_output": {
            "n_labels": int(len(first_labels)),
            "n_clusters": int(len(non_noise_sizes)),
            "n_noise_hits": int((np.asarray(first_labels) == -1).sum()),
            "cluster_size_min": int(non_noise_sizes.min()),
            "cluster_size_median": float(non_noise_sizes.median()),
            "cluster_size_mean": float(non_noise_sizes.mean()),
            "cluster_size_max": int(non_noise_sizes.max()),
            "cluster_label_dtype": str(np.asarray(first_labels).dtype),
        },
        "submission": {
            "rows": int(len(saved_submission)),
            "unique_hit_ids": int(saved_submission["hit_id"].nunique()),
            "columns": ["hit_id", "track_id"],
            "artifact_scored": str(SUBMISSION_PATH),
        },
        "official_score": {
            "value": score_value,
            "scorer": "trackml.score.score_event",
            "submission_rows": int(len(saved_submission)),
            "submission_unique_hit_ids": int(saved_submission["hit_id"].nunique()),
            "truth_rows": int(len(truth)),
            "truth_unique_hit_ids": int(truth["hit_id"].nunique()),
        },
        "fingerprints": {
            "raw_input_hit_id_sha256": _array_sha256(hit_ids.astype(np.int64, copy=False)),
            "feature_matrix_sha256": feature_bytes_hash,
            "feature_file_sha256": feature_file_hash,
            "raw_dbscan_labels_sha256": first_label_hash,
            "cluster_labels_file_sha256": label_artifact_hash,
            "canonical_cluster_mapping_sha256": first_mapping_hash,
            "canonical_mapping_sha256_file_sha256": _sha256_bytes(canonical_text.encode("ascii")),
            "submission_file_sha256": _sha256_bytes(SUBMISSION_PATH.read_bytes()),
            "canonical_submission_mapping_sha256": first_submission_mapping_hash,
            "scorer_truth_input_sha256": truth_hash,
        },
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "trackml": _trackml_provenance(),
            "platform": platform.platform(),
            "repository_commit": _git_revision(Path(__file__).resolve().parent.parent),
        },
        "determinism": {
            "raw_labels_equal": raw_labels_equal,
            "canonical_mapping_equal": canonical_equal,
            "submission_mapping_equal": submissions_equal,
            "scores_equal": scores_equal,
            "input_hashes_equal": input_hashes_equal,
            "first_score": score_value,
            "second_score": second_score_value,
        },
    }
    with PROVENANCE_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with REPORT_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_provenance_report(provenance))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing event000001000 CSV files")
    parser.add_argument(
        "--verify-existing-first-run",
        action="store_true",
        help="Keep existing first-run outputs and compare a new complete run against them.",
    )
    args = parser.parse_args()
    result = run(args.data_dir, verify_existing_first_run=args.verify_existing_first_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# M8.1-R2 — Historical M5 Artifact Recovery

**Scope:** read-only recovery/search. No reconstruction was run, no M5/M6/M8.1 result was changed, and no validation or final-test event was evaluated.

## 1. Repository/Git status

The workspace is inside a Git working tree. `git status` reports the supplied project files as untracked and `master` has no commits. `git rev-list --all --count` reports zero reachable commits. Two Codex-managed refs exist, but both resolve to tree objects rather than commits; they provide no commit history or historical files. Therefore there were no relevant commits to search, and no Git-history artifact was recovered. No checkout, reset, or history mutation was performed.

## 2. Local artifact search

Searched the available project tree for likely prediction/submission filenames and binary or tabular artifact types, excluding `.git`, raw data, caches, and environments. Candidate result files were:

- `experiments/results/m5_dbscan_results.csv`
- `experiments/results/m5_best_config_cluster_table.csv`
- `experiments/results/m8_1_development_baseline.csv`
- `experiments/results/m8_1_development_summary.json`

The M5 results CSV records the historical event-1000 run (run 19): 120,939 hits, 11,544 clusters, 61,996 noise hits, score `0.026887569576501846`, and runtime `2.935142788999997` seconds. It contains aggregate run metrics, not per-hit labels. The best-configuration cluster table has 11,544 rows and columns for cluster label and truth-derived cluster summaries, but no `hit_id`; it cannot recover hit membership or a hit-to-cluster mapping. The M8.1 files are the already-existing current results, not historical predictions.

No `.npy`, `.npz`, pickle, Parquet, Feather, or joblib prediction/submission artifact was found in the available project tree.

## 3. Notebook/source search

Searched project notebooks and source for DBSCAN, score invocation, submission conversion, evaluation functions, the historical score, and event 1000. The notebook does not contain a saved historical M5 label vector, submission, or run output. The M5 log describes the historical sequence as `extract_features(..., "xyz")` → `cluster_tracks(eps=0.01, min_samples=3)` → `summarize_evaluation`; the saved evidence does not include the executable sweep/reproduction driver or its label output.

The M5 archive contains `src/clustering.py`, `src/transformation.py`, `src/evaluation.py`, and `src/track_selection.py`. SHA-256 comparisons show each of these files is byte-identical to the corresponding supplied workspace file. Thus the available M5 core implementation does not reveal a material difference from the implementation used by the current M8.1 path. This establishes implementation-file identity only; it does not prove that the historical run used the same exact in-memory hit order, environment, truth input, or execution state.

## 4. Archive/backup search

The available Desktop `TrackML` folder contained the M3, M4, M5, real-data, and step3 project archives, plus the dataset archives and presentation file; it contained no additional project directory or backup folder. Inspected project archive inventories and searched candidate artifact names. M3, M4, real-data, and step3 archives contained no prediction/submission artifact candidates. The M5 archive contains the same two aggregate M5 CSVs as the workspace and no per-hit prediction or submission file. The archived M5 result CSV and cluster table are byte-identical to the workspace copies.

Raw training/test archives were not inspected for predictions; they are dataset sources, not project-run artifacts.

## 5. Historical score search

The precise historical score is recorded in `experiments/results/m5_dbscan_results.csv` (also present in the M5 archive) and in `experiments/results/m8_1_r_event1000_score_discrepancy.md`. Rounded `0.026888` references appear in `experiments/EXPERIMENT_LOG.md` and `docs/PROJECT_CONTENT_MASTER.md` (both also present in the M5 archive). These are result/log references, not per-hit evidence.

## 6. Recovery result

**C. Historical code recovered but prediction artifact missing.** The M5 archive preserves the core implementation, aggregate results, and experiment documentation. It does not preserve the historical per-hit predicted labels, a submission dataframe, or a run driver that would establish how the saved score was produced at the per-hit level.

## 7. Mapping comparison

Not possible: no historical `hit_id → cluster_id` mapping or `hit_id → track_id` submission was recovered. Therefore permutation-invariant equivalence to M8.1, differing hit/cluster counts, and noise-assignment differences cannot be determined from the available evidence. As established in the prior M8.1-R report, the current M8.1 path's own hit alignment was checked; that does not establish historical mapping equivalence.

## 8. Scientific conclusion

Established: the archived M5 run record reports score `0.026887569576501846`, 11,544 clusters, and 61,996 noise hits. The M8.1 record reports the same hit, cluster, and noise counts but score `0.13625134527683258`. The supplied M5 core source files match the current workspace source files byte-for-byte, but the historical labels/submission and exact historical execution path are unavailable.

**No recoverable historical per-hit prediction/submission artifact was found in the available repository, Git history, archives, notebooks, or project files.** The discrepancy remains unresolved. Equal cluster/noise counts and matching source files do not prove that the hit-to-cluster assignments or scorer inputs were identical. The evidence does not establish that the environment, M8.1, or the historical result caused the discrepancy.

## 9. Recommendation

Treat the historical score as a recorded but unverified historical result. Accept that its per-hit artifact is unrecoverable from the materials searched, then establish a new documented baseline with saved per-hit labels or a saved submission and explicit provenance. Do not alter either existing score to force agreement. No controlled environment reproduction is justified by a specific implementation difference found in this recovery phase.

## Tests

No tests were run because no code changed. For context, the previously reported M8.1-R test state is 20 focused tests passed and 112 tests passed in the available full suite; the supplied snapshot lacks later M6 files/tests. These are prior results, not executions in this recovery phase.

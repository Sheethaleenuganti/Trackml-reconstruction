# M8.1-R — Event 1000 historical M5 score discrepancy

**Date:** 2026-10-08  
**Scope:** diagnosis only; event `event000001000` only. No M8.1 output,
M5/M6 implementation, parameters, or scientific conclusions were changed.

## Conclusion

The exact historical M5 hit-to-cluster assignment and scorer input cannot be
recovered from the supplied repository artifacts. The historical M5
predicted-label artifact is **not available**, so permutation-invariant
hit-to-cluster equivalence is **not determinable**. The cause of the score
discrepancy is therefore **unresolved**. Matching cluster and noise counts
alone do not establish equivalent assignments.

## Historical M5 path and retained artifacts

- `experiments/results/m5_dbscan_results.csv`, run 19, records event 1000,
  representation `xyz`, scaling `none`, `eps=0.01`, `min_samples=3`,
  120,939 hits, 11,544 clusters, 61,996 noise hits, score
  `0.026887569576501846`, and DBSCAN runtime `2.935142788999997` seconds.
- `experiments/results/m5_best_config_cluster_table.csv` has 11,544 rows
  and per-cluster summary columns (`cluster_label`, `n_hits`, majority
  particle/count, purity, efficiency, and fragmentation flag). It has no
  `hit_id` column and cannot recover any cluster's hit membership.
- The M5 section of `experiments/EXPERIMENT_LOG.md` describes the historical
  route as `extract_features(..., "xyz")` →
  `cluster_tracks(eps=0.01, min_samples=3)` → `summarize_evaluation`, and
  says a baseline rerun reproduced score `0.026888`. The supplied M5 archive
  contains the M5 modules and aggregate outputs, but no saved per-hit label
  array, submission dataframe, M5 sweep runner, or serialized scorer input.
  The dataset-exploration notebook also contains no saved M5 label output.
- Searches of the repository, the supplied M5 archive, and the local TrackML
  project files found no historical prediction/submission artifact with
  per-hit labels.

Thus the exact historical `hit_id → cluster_id` map and exact historical
submission passed to `score_event` are unavailable. The saved run row proves
the recorded numeric result and parameter/count summary, but not the scorer
input that generated it.

## M8.1 current-path trace

`scripts/run_m8_development_baseline.py::run_event` performs this sequence:

1. `load_event(..., parts=("hits",))` loads event 1000's complete hits.
2. `transform_hits(hits)` preserves row order and `hit_id`.
3. `extract_features(..., representation="xyz")` selects `x2`, `y2`, and
   `z2` in that same row order.
4. `cluster_tracks(..., eps=0.01, min_samples=3)` returns one positional
   label for each feature row.
5. Only after labels exist, `load_event(..., parts=("truth",))` loads truth
   and `summarize_evaluation` calls `official_trackml_score`.
6. `official_trackml_score` calls `labels_to_submission(hits["hit_id"],
   labels)`, then passes truth columns `hit_id`, `particle_id`, `weight` and
   submission columns `hit_id`, `track_id` to official
   `trackml.score.score_event`.

For the M8.1 event-1000 reconstruction, the current trace verified:

- hits: 120,939 rows, 120,939 unique, increasing `hit_id`s;
- transformed hit IDs preserve the loaded hit-ID order;
- feature rows and prediction labels: 120,939 each;
- 11,544 non-noise clusters and 61,996 noise hits;
- the generated submission has 120,939 rows and unique hit IDs in the same
  order as the input hits; all submitted track IDs are nonnegative;
- `labels_to_submission` maps each DBSCAN `-1` noise hit to its own
  singleton nonnegative track ID (61,996 singleton noise tracks);
- the current score is `0.13625134527683258`, matching the saved M8.1 row.

Permutation-invariant fingerprint for the current hit-to-cluster map
(cluster IDs canonicalized by each cluster's minimum hit ID; noise remains
`-1`):

```text
SHA-256 1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8
```

Hashes of the current ordered scorer inputs, using UTF-8 tab-separated rows
(`hit_id, track_id` for submission; `hit_id, particle_id, float.hex(weight)`
for truth), are:

```text
submission 106028cc703767137e18ac3c97d39a501e36a3b5dcb70279377483db622e0d48
truth      d17fea8538f5bb763a9ce987bc4ed710c9df83ac073b901730e39d1ad1f1500e
```

These are current-path fingerprints only. No historical fingerprints exist
for comparison.

## Equivalence and alignment findings

| Question | Finding |
|---|---|
| Are historical and M8.1 hit-to-cluster assignments equivalent up to cluster-ID permutation? | **Not determinable.** Historical per-hit labels are absent. |
| Is hit-ID ordering/alignment correct in M8.1? | **Yes for the current path.** Hit IDs are unique and increasing; transform, features, labels, and submission preserve positional alignment. Historical row alignment cannot be compared without its label artifact. |
| Are historical and M8.1 submission labels equivalent? | **Not determinable.** The historical submission is absent. Current conversion preserves hit order and uses singleton IDs for noise. |
| Are exact scorer inputs equivalent? | **Not determinable.** Current truth/submission inputs were fingerprinted above; historical scorer arrays/dataframes were not saved. |
| Was a sort/reindex mismatch found in M8.1? | **No.** The current path resets the hit-ID and label Series indexes in `labels_to_submission` without reordering their values. The official scorer then joins on `hit_id` with one-to-one validation and sorts by `track_id`, `particle_id`. |

The saved M5 cluster table provides limited aggregate evidence: its cluster
size multiset matches the current one, and each saved `cluster_label` has the
same `n_hits` as its current counterpart. The `majority_hits_in_cluster`
value matches for 11,439 of 11,544 labels. These summaries do not contain
hit memberships, and numerical cluster IDs cannot substitute for comparing
the per-hit map; this evidence cannot establish equivalence.

## Submission conversion, noise, and official score semantics

The current `labels_to_submission` function resets indexes on both series
but does not sort or reindex values. Real cluster labels are retained; each
`-1` noise value is replaced with a distinct ID above the maximum real label.
For the current event-1000 labels, evaluating noise as one shared track
instead of singleton tracks produced the same score
`0.13625134527683258`; this tested noise grouping does not explain the
historical/current score difference for the current label map.

The installed official `score_event` calls `_analyze_tracks`, joins truth
and submission one-to-one on `hit_id`, sorts by `track_id` and `particle_id`,
and computes each submitted track's reconstruction purity and majority
particle purity. A track contributes its normalized majority-particle
weight only when both purities are strictly greater than 0.5. The event
score is the sum of those weights. Renumbering cluster IDs alone or changing
submission row order while preserving the same `hit_id → track_id` mapping
does not change this score.

## Environment comparison and remaining possibilities

| Package/runtime | Recorded M5 environment | M8.1 diagnostic environment |
|---|---:|---:|
| Python | 3.12.3 | 3.13.5 |
| NumPy | 2.4.4 | 2.5.3 |
| pandas | 3.0.2 | 3.0.6 |
| SciPy | 1.17.1 | 1.18.1 |
| scikit-learn | 1.8.0 | 1.8.0 |
| TrackML library | official library, version recorded as unknown | official GitHub source, commit `53a165e15a2c885f54c2bef1bd1ed53db6ed9648` |

The scikit-learn versions match; Python, NumPy, pandas, and SciPy differ.
Those differences could affect borderline DBSCAN membership or data
conversion, but no historical labels are available to demonstrate that they
did. The historical TrackML library commit/version is also not recorded, so
the exact scorer implementation cannot be proven identical. The supplied
historical artifacts also contain no per-file hashes to establish bytewise
identity of the event-1000 hits and truth used in M5 versus M8.1.

**Cause:** not established. The evidence does not justify attributing the
discrepancy to cluster-ID permutation, row order, noise conversion, a
scoring-formula difference, environment, or event data.

## Next diagnostic needed

To establish the cause, recover either the historical event-1000 predicted
label vector together with its hit-ID order, or the exact submission
dataframe passed to `score_event`; also record the historical event input
hashes and official TrackML source revision if available. Compare the
historical and current canonical `hit_id → cluster` mappings, then compare
their scorer inputs. Existing M8.1 result files were not overwritten.

M8.1 does **not** need an immediate rerun based on this diagnosis: a rerun
with the same available artifacts and environment would not recover the
missing historical comparison target. Keep the M8.1 run marked unconfirmed
against the historical M5 score until the missing artifact/source details
are recovered; rerun only if that comparison identifies a concrete change
to correct.

## Tests

- Focused existing evaluation and runner tests: **20 passed**.
- Available full suite: **112 passed**. The supplied repository snapshot
  does not include the later M6 tests/files.

# M5 — DBSCAN Baseline

## Research Question

Can DBSCAN recover particle tracks from M4-transformed TrackML hit
coordinates without using truth information, and how does performance
change with clustering parameters and coordinate representation?

## Real Dataset Used

Kaggle `train_sample`, event `000001000`. **Complete event: 120,939
hits, no hit-level subsampling.**

## Input Representation

Three tested, all derived from M4's `transform_hits()` output — never raw
x,y,z alone, never particle_id/truth:
- `(x2, y2)` — 2D
- `(x2, y2, z2)` — 3D, unscaled
- `(x2, y2, z2)` — 3D, standardized (zero mean, unit variance per column)

A real k-NN diagnostic (see figure below) found z2's std (8.57) is ~22x
larger than x2/y2's std (0.38) — this measured scale mismatch is why the
scaled variant was tested, not assumed necessary.

## DBSCAN Method

`sklearn.cluster.DBSCAN`, Euclidean metric. Truth (`particle_id`,
`truth.csv`) never enters clustering — used only afterward, for scoring.

## Parameter Search

45-run grid: 3 representations x up to 5 eps values (chosen from real
k-NN distance percentiles for each representation) x 3 min_samples
(3, 4, 6). Every run recorded, including poor ones. Full table:
`experiments/results/m5_dbscan_results.csv`.

## Quantitative Results

**Honest headline: the baseline score is LOW.**

| Representation | Max TrackML score achieved |
|---|---|
| xy | 0.0160 |
| xyz (unscaled) | **0.0269** |
| xyz (scaled) | 0.0268 |

Best result (2.7%) is far below the literature's ~0.20-0.22 starting-kit
reference score for this same transform (Step 2/M4). Scaling did **not**
improve on unscaled xyz, despite the measured scale mismatch — reported
as observed, not smoothed over.

## Baseline Configuration

**Selection criterion: highest TrackML score within the tested 45-run
grid** (not claimed optimal — the grid is limited, see Limitations).

`representation=xyz (unscaled), eps=0.01, min_samples=3`

- TrackML score: 0.0269
- Predicted clusters: 11,544
- Noise fraction: 51.3%
- Mean cluster purity: 0.637
- Mean track efficiency: 0.074
- Fragmented particles: 1,644 of ~2,200 recovered

## Failure Modes

All four found as **real, non-manufactured examples within this single
event** (see Fig D):

1. **Success** — cluster 24: purity=0.94, efficiency=1.00 (15/15 true
   hits recovered).
2. **Fragmentation** (dominant failure mode) — particle
   22526000000000000: one smooth, extended real trajectory split across
   2 separate DBSCAN clusters.
3. **Merging** — cluster 1178 (49 hits, purity=0.24): many different real
   particles mixed together.
4. **Noise rejection** — particle 4504560000000000: 8 of 10 true hits
   labeled DBSCAN noise.

**Root cause of the low score, evidenced not speculated**: best-run
cluster sizes have median 4 hits (mean 5.1) — DBSCAN is finding tiny,
tight local density pockets rather than whole ~10-hit tracks. This is
directly visible in Fig A.

## Figures

All in `experiments/figures/`:
- `m5_knn_diagnostic_all_representations.png` — k-NN distance diagnostic,
  all 3 representations
- `m5_score_vs_eps.png` — TrackML score vs. eps, all 45 runs
- `m5_cluster_size_distribution.png` — predicted cluster sizes, best run
- `m5_figA_clusters_xy.png` — 15 real DBSCAN clusters in (x2,y2)
- `m5_figB_noise.png` — clustered vs. noise points, (x2,y2)
- `m5_figD_failure_modes.png` — all 4 failure modes, real examples

## What We Learned

- The full M4→M5 pipeline is implementable and runs efficiently on a
  complete real event (full sweep: 45 runs in well under a minute total).
- The official TrackML scorer can be called directly and was validated
  against our own independently hand-derived Step 2 example (exact
  match: 0.90).
- **Fragmentation, not merging or pure noise-rejection, is the dominant,
  evidenced failure mode** — DBSCAN chops coherent tracks into small
  pieces.
- Recovery rate rises with pT (54%→79%→86% across low/mid/high bins) and
  with true hit count (42%→57%→89%), both matching physical expectation
  — descriptive, single-event, not a causal claim.
- Adding z2 helped (xyz > xy); scaling z2 did not help further, a
  genuine, somewhat counter-intuitive finding.

## Limitations

- Single event — no claim generalizes without further testing.
- Eps grids were bounded by real k-NN percentiles but not exhaustive;
  `xy`'s score was still rising at the smallest eps tested.
- pT/hit-count analysis restricted to 847 unambiguous particles (M3
  constraint), not all 2,564 unique IDs on this event.
- A real bug (noise miscounted as "fragmented particle") was found and
  fixed mid-milestone — documented, not hidden (see Experiment Log).

## Next Step

**M6 proposed question**: why does DBSCAN fragment real, coherent,
extended tracks into small pieces even in the improved (xyz) transformed
space — is this a local hit-density artifact, a metric/geometry mismatch,
or simply an eps/min_samples regime not yet explored in this grid?

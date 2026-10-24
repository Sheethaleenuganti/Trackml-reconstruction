# M9.3 Per-Target-Bin Query-Cap Ablation

## 1. Objective

Test whether increasing only the per-target-bin query visibility from 10 to 20 improves true consecutive-edge coverage while preserving the final cap of ten candidates per source.

## 2. Exact parameter change

Per-target-bin cKDTree query cap: **10 → 20**. All other graph settings are fixed at M9.1 values: maximum radial step 180 mm; radial buckets 18 mm; ten outward buckets; same-bucket exclusion; Euclidean XYZ ranking; target hit ID tie-break; outward direction; final cap 10.

## 3. Method

For every event, construct both the unchanged M9.1 graph and the separate M9.3 graph using hit_id and x/y/z only. Then load truth and evaluate exact directed consecutive-pair coverage. Candidate precision is recovered true pairs divided by candidate edges. Candidate density uses N(N−1)/2 possible directed outward pairs. M9.1 event graph counts and runtimes are also cross-checked against saved M9.1 event summaries.

## 4. Development split

Exactly the validated DEVELOPMENT manifest events `event000001000`–`event000001059` (60 events). No validation or final-test event was read.

## 5. Aggregate results

- Hits: 6,632,611
- True edges: 5,007,067
- M9.1 candidate edges: 63,326,540; M9.3: 63,326,540
- M9.1 recovered: 363,034; M9.3: 363,034
- M9.1 missed: 4,644,033; M9.3: 4,644,033
- M9.1 pooled recall: 7.25043224%; M9.3: 7.25043224%
- Absolute recall change: 0.00000000%; relative change vs M9.1 recall: 0.0000%
- Newly recovered M9.1-missed true pairs: 2; previously recovered pairs lost in M9.3: 2
- Unique candidate pairs added: 444; removed: 444; net edge-count change: 0 (0.0000%)
- Mean candidate density: M9.1 0.00017423138; M9.3 0.00017423138; change 0
- Candidate precision: M9.1 0.573273%; M9.3 0.573273%
- Graph runtime: M9.1 historical 983.364s; M9.3 measured 22587.879s; change 21604.515s (2197.001%)
- Runtime caveat: M9.3 graph timings use perf_counter around each build. The process ran in a desktop session with long pauses between progress checks; elapsed timing may include host suspension and is not an apples-to-apples speed comparison with historical M9.1 timing.
- Final cap verification: maximum candidates/source across M9.3 events = 10.

## 6. Comparison with M9.1

M9.3 did not change pooled recall by 0.00000000%. The M9.1 reference is the saved 60-event result; M9.1 graph construction was also rerun unchanged for exact event-level pair comparisons. Event-level differences are in `m9_3_event_summary.csv`.

## 7. Per-event variation

Recall change was positive in 2 events, unchanged in 56, and negative in 2. Event recall-change range: -0.00133349% to 0.00115664%; mean -0.00000385%, median 0.00000000%.

## 8. Interpretation

Per-bin query cap 20 did not increase pooled recall relative to M9.1. The measurement applies only to candidate-edge coverage on development events.

The M9.2 classification described which true targets were absent from the old per-bin top-10 query; it did not establish how many become final candidates after the unchanged global cap. This ablation tests that effect directly.

## 9. Limitations

This is candidate-edge coverage, not end-to-end track reconstruction or official TrackML score. The final source cap remains 10, so additional per-bin visibility can fail to change the final graph. Runtime is measured on this environment and is not a hardware-independent performance claim; the wall-clock comparison caveat above applies. No validation/test results were inspected.

## 10. Decision/recommendation for M9.4

Do not promote per-bin query cap 20. For M9.4, run a development-only post-hoc audit of remaining missed true edges: measure which targets occur at per-bin ranks 11–20, whether they pass the 180 mm radial filter, and whether they would enter the unchanged final top-10. Keep validation/final-test unopened and make no new graph change until that audit identifies a reachable mechanism.

## Reproducibility

- Timestamp UTC: 2026-10-08T18:23:21.531901+00:00
- Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]; NumPy 2.5.3; pandas 3.0.6; SciPy 1.18.1
- Configuration and exact development event IDs: `m9_3_summary.json`.
- Figures: recall, edge-count, and per-event recall-change comparisons under `experiments/figures/`.

# M9.3 — Per-Target-Bin Query-Cap Ablation

## Objective

Test whether increasing only per-target-bin query visibility from 10 to 20 improves true consecutive-edge coverage while retaining the final ten-candidate/source cap.

## Frozen protocol

- Use exactly the validated DEVELOPMENT events `event000001000`–`event000001059`.
- Keep M9.1 unchanged. The M9.3 implementation is a separate geometry-only builder.
- Change only per-target-bin cKDTree query cap: 10 → 20.
- Keep max radial step 180 mm, radial buckets 18 mm, ten outward buckets, same-bucket exclusion, XYZ Euclidean distance, target hit ID tie-break, outward direction, and final per-source cap 10.
- Construct both M9.1 and M9.3 graphs using hit geometry only. Load truth after both graphs exist for exact event-level comparison.
- Compare exact true-pair and candidate-pair sets to report newly recovered and lost pairs. Cross-check M9.1 recomputation with the saved event summary.
- Do not read validation/final-test data or tune another setting.

## Metrics

Per event and pooled: hit count, graph edge count, mean/median/maximum candidates/source, true-edge availability/recovery/misses/recall, candidate precision, candidate density, unique candidate pairs added/removed, net graph-size change, M9.1 historical and M9.3 graph runtimes, and per-event recall change.

## Tests

Tests verify that the query cap reaches 20, the final cap stays at 10, ordering is deterministic with target-ID tie-breaks, same-bucket pairs are excluded, edge direction is outward, truth columns do not affect construction, and M9.1 behavior remains unchanged when the query cap cannot bind.

## Outputs

- `experiments/results/m9_3_summary.csv`
- `experiments/results/m9_3_summary.json`
- `experiments/results/m9_3_event_summary.csv`
- `experiments/results/m9_3_report.md`
- Recall, candidate-edge, and event-recall-change figures under `experiments/figures/`

No M9.4 experiment is run automatically.

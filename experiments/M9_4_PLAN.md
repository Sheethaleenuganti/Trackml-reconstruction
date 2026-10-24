# M9.4 — Per-Bin Rank Audit and Candidate-Selection Bottleneck

## Objective

Trace development true consecutive edges through the exact M9.1 stages and an extended, geometry-only per-bin query20 audit. Quantify which are never searched, rank 1–10, 11–20, >20, lost by radial filtering, retained in the merged pool, or removed by the final source-level top-10.

## Confirmed pipeline

The unchanged M9.1 builder sorts target-bin points by hit ID and runs cKDTree `query(k=10)` per source/target-bin pairing. It performs exact positive Δr filtering only after each bin query, combines candidates from the searched bins, sorts the pool per source by XYZ distance then target hit ID, and applies the final cap of 10. The per-bin query cutoff itself uses cKDTree tie selection without an explicit hit-ID cutoff tie-break. The extended audit uses the same cKDTree mechanism with k=20 and one worker; target points enter in hit-ID order and returned equal-distance results are ordered by hit ID. Cutoff membership is reproducible for the recorded SciPy version and hit order, while final merged ranking explicitly uses target ID.

True edges match M9.1 evaluation: same nonzero particle, sorted by particle ID, transverse radius, hit ID, consecutive pairs only, equal-r excluded, directed outward. Truth is loaded only after both the frozen baseline graph and all geometry-only audit query/merge/final ranks exist.

## Controls

- Exact validated DEVELOPMENT events 1000–1059 only.
- Unchanged M9.1 builder/config and event-1000 saved edge artifact are used for baseline comparison.
- Check the event-1000 saved M9.1 pair/rank edge set and SHA-256; for the other 59 events, compare reconstructed M9.1 graph and truth metrics to saved per-event scalar results. Their full edge lists are not archived, so edge-by-edge verification is impossible for those events.
- Process one event at a time; write event summaries and only a bounded sample of event-1000 edge traces.
- Never load validation/final-test files or use truth to select/rank candidates.

## Definitions and output denominators

Every stage count uses all true consecutive edges as denominator unless a label explicitly says `baseline_miss`. `target_bin_searched` means target bucket is among the next ten buckets. `per_bin_rank` follows the extended cKDTree query order (Euclidean XYZ distance; returned ties ordered by target hit ID, with cutoff membership selected by the recorded SciPy implementation). The radial filter is `0 < Δr <= 180 mm`. The merged pool is all returned query20 candidates passing that filter, across searched bins, ranked per source with the same final distance/ID ordering. A true edge survives final selection when its merged rank is at most ten. Stage loss rows form sequential, mutually exclusive transitions; additional flags/counts may overlap and are labeled as such.

## Outputs

- `experiments/results/m9_4_summary.csv` / `.json`
- `experiments/results/m9_4_event_summary.csv`
- `experiments/results/m9_4_stage_transitions.csv`
- `experiments/results/m9_4_event000001000_examples.csv.gz` (at most 50 examples per baseline miss stage)
- `experiments/results/m9_4_report.md`
- `experiments/figures/m9_4_stage_transition.png`

No M9.5 or alternate candidate method is run.

# M9.5 — Geometry-Neighborhood Failure Audit

## Objective

Describe how coordinate-derived geometry differs between true consecutive-particle edges recovered by frozen M9.1 and missed true edges in the geometry-neighborhood failure cohorts. This is post-hoc diagnosis only; no ranking or candidate method changes are permitted.

## Data and separation

- Use only manifest DEVELOPMENT events `event000001000`–`event000001059`.
- Build M9.1 baseline candidates and the diagnostic per-bin top-20 query from `hit_id,x,y,z` only.
- Fix candidate pairs and ranks before loading truth.
- Load truth only to form same-nonzero-particle consecutive outward edge pairs and label the diagnostic cohorts.
- Do not use cells or particle/detector metadata as features or ranking inputs.

## Frozen rules used

The M9.1 builder uses 18 mm radial buckets, searches the next ten outward buckets (excluding the same bucket), queries per-target-bin cKDTree nearest neighbors with `k=10`, applies `0 < Δr <= 180 mm` after each query, merges eligible candidates, ranks by Euclidean XYZ distance then target hit ID, and retains at most ten candidates per source. M9.5 reuses the M9.4 geometry-only query20 audit for rank cohorts; its top-20 cutoff uses cKDTree membership and returned equal-distance rows sorted by target hit ID. The final merged ordering remains XYZ distance then target ID. True edges match the existing evaluation rule: nonzero particle, sorted by particle ID/radius/hit ID, consecutive pairs only, equal-radius pairs excluded, directed outward.

## Cohorts

- `recovered_m91`: true edge present in M9.1 graph.
- `missed_rank_1_10`: M9.1-missed true edge returned in diagnostic query20 at per-bin rank 1–10.
- `missed_rank_11_20`: M9.1-missed true edge returned in diagnostic query20 at rank 11–20.
- `missed_rank_above_20`: target bucket searched, target absent from query20.
- `missed_target_bin_unsearched`: target bucket outside the next ten outward buckets.
- `missed_radial_filter`: M9.1-missed true edge returned by query20 but failing `0 < Δr <= 180 mm`; overlaps rank cohorts.

## Measurements and artifacts

Calculate Δr, signed and absolute wrapped Δφ, Δz and absolute Δz, XYZ and transverse distance, source/target radius, source/target z, and radial bucket difference. Report count, mean, median, quartiles/IQR, p90 and p95 for complete cohorts. Event summaries retain event-level counts and distribution statistics. No sampling is used for statistics. Plots use the deterministic first 100,000 edge values per cohort/feature in manifest and true-edge order; full counts and statistics remain unsampled.

Files: `src/m9_5_geometry_miss_audit.py`, `scripts/run_m9_5_geometry_miss_audit.py`, `tests/test_m9_5_geometry_miss_audit.py`, `experiments/results/m9_5_summary.csv`, `m9_5_summary.json`, `m9_5_event_summary.csv`, `m9_5_report.md`, and `experiments/figures/m9_5_geometry_cohort_distributions.png`.

## Verification

The runner must validate exact manifest membership, match M9.1 true-edge/recovery scalar counts for all 60 events, compare event 1000 against the persisted M9.1 edge artifact, and verify hashes of M9.1/M9.4 artifacts before and after. Focused and full test outcomes are recorded only after actual test runs. Validation and final-test events must remain unopened.

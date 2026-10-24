# M9.2 — Missed True-Edge Diagnosis

## Research question

Identify which fixed M9.1 geometric/search constraint accounts for missed true consecutive hit pairs, using post-hoc truth diagnostics on development events only.

## Frozen inputs and controls

- Process only events `event000001000` through `event000001059` from the validated M7 development manifest.
- Preserve the M9.1 graph configuration: positive outward direction, maximum radial step 180 mm, 18 mm radial buckets, ten outward buckets, same-bin exclusion, per-bin query k=10, final per-source k=10, Euclidean XYZ ranking, target hit ID tie-break.
- Load hits and construct/load the geometry-only graph before loading truth.
- For event 1000, use its saved M9.1 graph. For the other 59 development events, invoke the unchanged M9.1 builder in memory; do not persist or overwrite M9.1 outputs.
- Do not load validation or final-test events. Do not test alternate parameters or methods.

## Definitions

For each nonzero particle ID, sort hits by increasing transverse radius (hit ID breaks ties), form consecutive outward directed pairs, and exclude equal-radius pairs. For each pair, calculate Δr, absolute wrapped Δφ, |Δz|, and Euclidean XYZ distance. Determine exact directed hit-ID pair membership in the M9.1 graph.

Bucket search membership is bucket difference 1–10. Radial-step membership is `0 < Δr <= 180 mm`. For missed edges inside both restrictions, reproduce the M9.1 target-bin cKDTree query to determine whether the true target was among the ten queried neighbors. A target returned by that query but omitted from the final graph is classified as top-k truncated; one absent from that per-bin query is a geometric-neighborhood miss. Exact rank above 10 is not available because M9.1 limits each per-bin query to ten; report it as `>10`.

Overlapping counts report each applicable mechanism independently. Primary cause uses precedence radial-step exclusion, bucket/search exclusion, top-k truncation, geometric-neighborhood miss, then other/unclassified.

## Outputs

- `experiments/results/m9_2_summary.csv` and `.json`
- `experiments/results/m9_2_event_summary.csv`
- `experiments/results/m9_2_event000001000_edge_diagnostics.csv.gz` (representative per-edge detail only)
- `experiments/results/m9_2_report.md`
- Six figures under `experiments/figures/`, all explicitly marked POST-HOC TRUTH EVALUATION

The runner refuses to overwrite existing M9.2 outputs. No raw data is copied, moved, or committed.

## Tests

Focused tests cover the four geometric quantities, truth ordering/equal-r behavior, recovered/missed membership, bucket/search categorization, candidate rank/top-k categorization, determinism, and development-only ID selection.

## M9.3

No revised candidate generator is implemented or evaluated in M9.2. A single next experiment will be proposed from the dominant observed miss category in the final report.

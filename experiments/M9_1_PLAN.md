# M9.1 — Geometry-Aware Candidate Graph Plan

## Question

Can local hit geometry produce a sparse outward candidate graph that contains a useful fraction of same-particle consecutive hit pairs, without using truth during graph generation?

## Frozen scope

- Use only the 60 `DEVELOPMENT` events in the validated M7 manifest: events 1000–1059.
- Do not read validation events 1060–1079 or final-test events 1080–1099.
- Do not alter M5, M6, M7, M8.1, or M8.2 code/results.
- Do not build final tracks or report a TrackML score for this graph.

## Initial method

`src.geometry_graph.build_candidate_graph` receives hits and reads only `hit_id,x,y,z`. It computes transverse radius and azimuth, indexes outward radial bins, finds local neighbors with a SciPy cKDTree over XYZ, filters to strictly increasing radius and a maximum radial step, then retains at most ten nearest candidates per source hit. Candidate rank is Euclidean XYZ distance, with target hit ID as a stable tie-break. The output contains identifiers, geometry deltas, distance, and rank only.

## Configuration rationale

- `max_radial_step_mm=180`: a geometry-only survey of event 1000 found adjacent barrel-layer median-radius gaps of approximately 40–160 mm. The 180 mm window adds a 20 mm margin above the largest observed gap. It is a starting point, not a truth-tuned optimum.
- `max_candidates_per_hit=10`: an explicit initial sparsity budget. It is not selected from validation or final-test results.
- Radial indexing uses 18 mm bins for the 180 mm radial window (`min(20 mm, max_radial_step/10)`) and searches ten outward bins. Same-bin edges are excluded so same-layer proximity cannot consume the outward candidate budget.
- Euclidean distance in XYZ (millimeters) ranks candidates; wrapped azimuth is recorded for analysis but does not gate construction.

## Evaluation protocol

For each event, load hits only, construct and time the candidate graph, then load truth for post-hoc evaluation. Define true consecutive pairs as nonzero-particle hits ordered by increasing transverse radius; skip equal-radius pairs because the graph is strictly outward-directed. Calculate edge recall, graph size, per-hit candidate counts, density against `N*(N-1)/2`, and generation runtime.

Save one summary row per development event and one compressed representative edge file for event 1000. Produce one graph-size plot and one true-edge recall-by-rank plot clearly marked as truth-based post-hoc evaluation.

## Planned outputs

- `experiments/results/m9_1_geometry_graph_summary.csv`
- `experiments/results/m9_1_geometry_graph_summary.json`
- `experiments/results/m9_1_geometry_graph_edges_event000001000.csv.gz`
- `experiments/results/m9_1_geometry_graph_report.md`
- `experiments/figures/m9_1_event1000_candidate_count_distribution.png`
- `experiments/figures/m9_1_event1000_true_edge_recall_by_rank.png`

No conclusions about final track reconstruction will be drawn from candidate recall alone.

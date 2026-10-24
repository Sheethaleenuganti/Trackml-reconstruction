# M9.1 Geometry-Aware Candidate Graph Report

## 1. Research question

Can local detector geometry generate a sparse candidate graph that recovers true consecutive hit connections without using truth during construction?

## 2. Motivation from M6

M6 found that 99/100 sampled real particles had at least one internal consecutive-hit gap larger than DBSCAN `eps=0.01`; 267/2,487 analyzed particles were contained in exactly one predicted cluster, and mean largest-cluster fraction was approximately 0.22. This motivates candidate connections beyond fixed-radius connected components but does not show that all DBSCAN settings fail.

The supplied repository snapshot does not include the M6 diagnostics module/output artifacts; this motivation uses the verified M6 findings supplied in the task context. No M6 file was recreated or changed.

## 3. Candidate-generation method

For each event, `src.geometry_graph.build_candidate_graph` reads only `hit_id,x,y,z`. It computes cylindrical radius and azimuth, buckets hits in radial intervals, searches outward buckets with 3D XYZ cKDTree neighborhoods, filters to a positive radial step no larger than the configured limit, and retains the nearest candidates ranked by Euclidean XYZ distance with hit ID as a tie-break. Edges point from smaller to larger transverse radius. The graph is sparse with a per-source top-k limit.

## 4. Geometry features used

Candidate construction uses Cartesian `x,y,z` to rank local spatial neighbors and derives `r=sqrt(x²+y²)` and `phi=atan2(y,x)` for radial direction and edge diagnostics. `volume_id`, `layer_id`, `module_id`, cells, particles, and truth are not used to generate candidates.

## 5. Hyperparameters

- Maximum positive radial step: 180.0 mm. Event-1000 geometry showed adjacent barrel-layer median-radius gaps of about 40–160 mm; 180 mm provides a 20 mm margin over the largest observed gap.
- Maximum candidates per hit: 10. This is a declared initial sparsity budget, not tuned with truth.
- Internal radial indexing bucket width: 18 mm (`min(20 mm, max_radial_step/10)`); ten outward bins are searched. Same-bin edges are excluded to prevent same-layer neighbors dominating the outgoing candidate budget.
- Each target-bin query asks for up to 10 nearest hits; final per-source selection is at most 10 across all bins. Ties use ascending target hit ID.
- Feature-space ranking: Euclidean distance in raw XYZ (millimeters).

## 6. Development split

The validated M7 manifest selected exactly events `event000001000` through `event000001059` (60 events). Validation and final-test events were not loaded or evaluated.

## 7. Leakage controls

The graph module accepts and reads only hit IDs and coordinates; event loading requests `parts=("hits",)` for construction. Truth is loaded only after the edge graph has been generated and is used only to measure consecutive-edge coverage. Candidate files contain no truth or particle columns.

## 8. Candidate graph statistics

- Development events processed: 60
- Total candidate edges: 63326540
- Mean candidate edges/event: 1055442.333
- Mean edges/hit, averaged across events: 9.545580
- Median candidates/hit, averaged across events: 10.000
- Maximum candidates/hit observed: 10

## 9. True-edge recall

Truth-based post-hoc edges are consecutive nonzero-particle hits ordered by increasing transverse radius; equal-radius consecutive pairs are excluded because this graph is strictly outward-directed. Pooled recall was 0.072504 (363034/5007067). Mean event recall was 0.072812. This is candidate-edge coverage, not track reconstruction accuracy.

## 10. Sparsity/density

Candidate density is computed against all possible directed outward pairs, `N*(N-1)/2`, per event. Mean event density was 0.00017423138; maximum event density was 0.00021505471. The top-10 cap bounds edge count by ten per hit.

## 11. Runtime

Candidate-generation runtime (excluding data loading and truth-based evaluation) was 983.364 seconds total, 16.389 seconds/event on average.

## 12. Figures

- `experiments/figures/m9_1_event1000_candidate_count_distribution.png` — geometry-only outgoing candidate counts.
- `experiments/figures/m9_1_event1000_true_edge_recall_by_rank.png` — truth-based post-hoc recall by candidate rank; the figure labels truth use explicitly.

## 13. Limitations

This first graph uses a global radial window and Euclidean XYZ nearest neighbors. It does not model magnetic curvature, charge, momentum, detector-surface transitions, or track-level consistency. Same-radial-bin targets are excluded, the 180 mm window may omit valid transitions outside the event-1000 barrel-derived range, and top-10 truncation may miss true neighbors. The metric depends on ordering truth hits by transverse radius. No end-to-end tracks or TrackML reconstruction score were produced.

## 14. Comparison with frozen baseline

The M8.2 event-1000 reference remains score `0.13625134527683258`, 11,544 clusters, and 61,996 noise hits. Candidate coverage and candidate sparsity are different quantities from the TrackML score and are not a direct baseline comparison.

## 15. Scientific conclusion

The experiment generated a deterministic, bounded-size geometry-only candidate graph over the 60 development events. It recovered 363,034 of 5,007,067 post-hoc true consecutive edges (7.25% pooled recall) at a mean density of 0.0174%. This is sparse, but the measured recall is low; this initial candidate design does not recover a high fraction of local true connections. It does not establish final track-reconstruction performance.

## 16. Next experiment recommendation

M9.2 should diagnose missed development true edges by radial step, wrapped azimuth change, longitudinal displacement, radial-bin omission, and candidate rank. Use that diagnosis to propose one revised candidate generator. Keep validation and final-test events untouched until a method and evaluation protocol are frozen.

## Reproducibility

- Configuration: `max_radial_step_mm=180.0`, `max_candidates_per_hit=10`.
- Python: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- NumPy: `2.5.3`; pandas: `3.0.6`; SciPy: `1.18.1`; scikit-learn: `1.8.0`
- Platform: `Windows-11-10.0.26200-SP0`
- Git revision: `None`

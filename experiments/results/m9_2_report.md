# M9.2 Missed True-Edge Diagnosis

## 1. Research question

Which geometric or candidate-selection constraint explains missed M9.1 true consecutive hit connections?

## 2. M9.1 configuration

Unchanged M9.1 candidate graph: complete development events, increasing transverse radius, maximum Δr 180 mm, 18 mm radial buckets, ten outward buckets, same-bucket exclusion, up to ten queried neighbors per target bin and ten final candidates per source; raw XYZ Euclidean distance with target hit ID tie-break. For events 1001–1059 the unchanged builder was rerun in memory because only event 1000's candidate table was saved. No M9.1 output was overwritten.

## 3. Truth definition

For each nonzero particle ID, hit pairs are consecutive after increasing-r ordering (hit ID breaks radius ties); equal-r pairs are excluded. Edges point from smaller to larger r. Exact directed hit-ID pairs are matched against the candidate graph.

## 4. Leakage controls

Candidate generation receives hit geometry only. Truth is loaded only after each graph exists and is used only for diagnostics. Only manifest DEVELOPMENT events 1000–1059 were loaded. Events 1060–1099 were not read. No truth finding is fed back into graph construction.

## 5. Radial-step findings

all: n=5,007,067, mean=55.3134, median=40.0371, p90=144.896, p95=161.287, p99=215.065, max=934.212

Fraction of missed edges with Δr > 180 mm: 3.529109% (163,893/4,644,033).

## 6. Azimuth findings

Absolute wrapped Δφ; observed geometry only, with no causal attribution.

all: n=5,007,067, mean=0.0584225, median=0.0183176, p90=0.129877, p95=0.200781, p99=0.575397, max=3.14142

recovered: n=363,034, mean=0.0303465, median=0.0172418, p90=0.0800192, p95=0.100365, p99=0.141159, max=0.50296

missed: n=4,644,033, mean=0.0606173, median=0.0183949, p90=0.136081, p95=0.209533, p99=0.628044, max=3.14142

## 7. Longitudinal findings

Absolute |Δz|.

recovered: n=363,034, mean=13.2057, median=7.2, p90=27.6, p95=38.4, p99=136.42, max=1212.1

missed: n=4,644,033, mean=155.901, median=109.824, p90=363.463, p95=408.3, p99=1046.5, max=5104

## 8. XYZ-distance findings

Euclidean distance in millimeters.

recovered: n=363,034, mean=56.9451, median=43.5652, p90=143.679, p95=165.409, p99=179.292, max=1216.92

missed: n=4,644,033, mean=180.358, median=135.686, p90=400.642, p95=444.705, p99=1056.14, max=5157.07

## 9. Bucket-search findings

Bucket difference 1–10 is the searched outward-bucket condition; same-bucket and >10-bucket edges are outside it. This overlaps with radial-step exclusion.

- In searched buckets: 3,379,018 true edges; recovered 363,034; missed 3,015,984.
- Outside searched buckets: 1,628,049; recovered 0; missed 1,628,049.
- Missed edges outside bucket search: 35.056792%.

## 10. Top-k findings

Among all true edges, M9.1 candidate rank counts are {">10": 348229, "not_present_in_queried_neighborhood": 2657579, "outside_search_region": 1638225, "rank_1": 149631, "rank_10": 16172, "rank_2": 48110, "rank_3": 30686, "rank_4": 24734, "rank_5": 21225, "rank_6": 19776, "rank_7": 18388, "rank_8": 17725, "rank_9": 16587}. For an eligible missed edge, the exact target was queried among the same target-bin top 10. If present but absent from final graph it is classified as top-k truncation (>10 globally); if absent from that per-bin query it is a geometric-neighborhood miss. The latter rank cannot be measured exactly because the M9.1 per-bin query itself returns only ten.

Fraction of missed edges affected by top-k truncation: 7.498418% (348,229/4,644,033).

## 11. Missed-edge mechanism classification

Overlapping mechanism counts (conditions can coexist): {"bucket_search_exclusion": 1628049, "geometric_neighborhood_miss": 2657579, "other_unclassified": 0, "radial_step_exclusion": 163893, "top_k_truncation": 348229}.

Mutually exclusive primary causes, with precedence radial-step → bucket/search → top-k → geometric-neighborhood → other: {"bucket_search_exclusion": 1474332, "geometric_neighborhood_miss": 2657579, "radial_step_exclusion": 163893, "top_k_truncation": 348229}.

## 12. Event-level variation

All 60 events were processed. Event recall ranged from 6.696276% to 8.084104%; per-event counts and missed fractions are in `m9_2_event_summary.csv`. Per-event Δr and mechanism data are included in that table.

## 13. Limitations

This diagnoses candidate coverage, not complete track reconstruction. Query-neighborhood ranks beyond ten are not observable from M9.1's per-bin query. Candidate lists for 59 events were deterministically reconstructed in memory, not persisted. Aggregates cover all edges; the optional compressed per-edge diagnostic is only for event 1000. No validation or final-test data, tuning, or alternative candidate generator was used.

## 14. Proposed M9.3 direction

Run one development-only comparison against frozen M9.1 with the per-target-bin cKDTree query cap increased from 10 to 20, keeping the final per-source cap at 10, the 180 mm radial step, 18 mm buckets, ten outward buckets, XYZ ranking, and tie-break unchanged. Measure true-edge recall and candidate count/density; make no further changes in that run.

This is a proposal only; M9.3 was not run.

## Aggregate

- True edges: 5,007,067; recovered: 363,034; missed: 4,644,033; pooled recall: 7.250432%.
- Development events: 60.
- Figures are marked POST-HOC TRUTH EVALUATION and saved under `experiments/figures/`.

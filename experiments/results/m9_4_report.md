# M9.4 Per-Bin Rank Audit and Candidate-Selection Bottleneck

## Research question and scope

This audit traces development true consecutive edges through the frozen M9.1 geometry-only selection pipeline and an extended, reproducible per-bin query cap of 20. It is a diagnostic audit, not a new reconstruction method. All 60 events are in the DEVELOPMENT split (1000–1059); validation and final-test files were not loaded.

## Confirmed M9.1 pipeline order

`src/geometry_graph.py::build_candidate_graph` reads only hit_id and x/y/z. It computes r and 18 mm buckets, sorts each bucket's hit indices by hit ID, then calls cKDTree `query(k=10)` for each source bucket against each of the next ten outward target buckets. **The per-bin nearest-neighbor query occurs before** the exact filter `0 < Δr <= 180 mm`. Returned eligible hits from target buckets are concatenated for each source bucket; candidate ranks are assigned by Euclidean XYZ distance and target hit ID tie-break; only then is the final per-source cap of ten applied. The per-bin cKDTree query itself does not explicitly break equal-distance cutoff ties by hit ID; input target points are hit-ID sorted and SciPy's deterministic query result is used in the frozen builder. M9.4 uses the same cKDTree query mechanism with one worker and query cap 20; target rows enter in hit-ID order and returned equal-distance results are sorted by hit ID. Cutoff membership follows the recorded SciPy implementation.

The true edge definition was confirmed from `scripts/run_m9_1_geometry_graph.py::_true_consecutive_edges`: same nonzero particle ID, increasing transverse radius, consecutive after sorting by particle_id/r/hit_id, equal-r pairs excluded; only used after geometry stages are fixed.

## Configuration and audit method

M9.1 settings: max radial step 180 mm; bucket width 18 mm; next ten outward buckets; same-bucket excluded; raw XYZ Euclidean ranking; final top-10/source; outward direction. M9.4 constructs all per-target-bin top-20 query results from geometry before loading truth, records per-bin rank and radial-step pass/fail, forms the merged eligible pool and its exact global rank, and marks whether each candidate would survive the final top-10. Truth is then joined for post-hoc matching.

The transition denominator is all post-hoc true edges. Sequential stage counts are:

| Stage | True edges remaining | Lost at stage |
|---|---:|---:|
| all_true_edges | 5,007,067 | 0 |
| target_bin_searched | 3,379,018 | 1,628,049 |
| returned_by_query20 | 1,082,755 | 2,296,263 |
| passed_radial_filter | 1,076,708 | 6,047 |
| merged_candidate_pool | 1,076,708 | 0 |
| survived_final_top10 | 363,034 | 713,674 |

## Pooled stage findings

- True edges: 5,007,067; M9.1 recovered: 363,034; missed: 4,644,033.
- Target bin never searched: 1,628,049 overall; among M9.1 misses: 1,628,049.
- Query20 ranks 1–10: 716,817; ranks 11–20: 365,938; absent from extended top 20 despite searched bin (rank >20): 2,296,263.
- Rank 11–20 passing Δr filter: 365,446; failing filter: 492.
- Rank 11–20 passing radial filter and entering merged pool: 365,446.
- Of M9.1-missed edges, rank 11–20: 365,936; passing radial filter: 365,444; surviving the extended final top-10: **0**. Thus query20 adds no final-top-10 true edge among the rank-11–20 M9.1 misses.
- All query20 true targets passing radial filter and entering merged pool: 1,076,708; of these surviving final top-10: 363,034; rejected at final cap: 713,674.

The saved aggregate does not separately count the rank-11–20/final-top-10 intersection among already M9.1-recovered edges; no total for that broader overlap is claimed here. The event-level audit records the exact post-hoc stage flags for the M9.1-missed subset used to assess whether query20 would recover additional edges.

## Baseline verification

The rebuilt M9.1 graph for event 1000 matches the frozen M9.1 artifact exactly by source/target/candidate-rank; geometry columns match within 1e-12. For all 60 events, reconstructed M9.1 graph size and post-hoc hit/recovery counts were checked against the saved M9.1 summary; mismatches: 0. The complete historical edge lists were not saved for events 1001–1059, so exact full-edge-set equality cannot be verified for those 59 events. The frozen event-1000 artifact SHA-256 before and after was 5e9eba41f75f0d6b6334a95d76da8416bba4d3ff70e954db80032ac7bf3cb4fe and 5e9eba41f75f0d6b6334a95d76da8416bba4d3ff70e954db80032ac7bf3cb4fe.

Per-event cap maximum observed: 10; direction, radial step and same-bucket constraints were verified against reconstructed M9.1 graphs. Event recall range: 6.696276%–8.084104%. Per-event stage counts and fractions are in `m9_4_event_summary.csv`.

## Overlap and dominant failure stage

Failure flags can overlap: radial-step failure can coexist with a searched bucket; ranks 11–20 can also fail radial filtering or be rejected by the final cap. Their overlapping counts appear separately in the event and summary CSVs. For a mutually exclusive accounting of M9.1 misses, precedence is target bin not searched → rank above 20 → radial filter failed → final top-10 rejected → would survive extended final-10. The largest category is **rank_above_20** (2,296,263 missed edges).

## Limitations

Only event 1000 has a persisted M9.1 edge table, so full edge-set artifact comparison is limited to that event. The other 59 reconstructed baseline graphs match saved per-event scalar summaries but cannot be compared edge-by-edge to missing artifacts. Per-bin cutoff tie membership follows cKDTree in the recorded SciPy environment; final merging explicitly sorts equal distances by target ID. All results concern candidate-edge coverage, not final tracks or official TrackML score.

## Recommendation

Among M9.1-missed true edges in per-bin ranks 11–20, none survived the final ten-candidate cap. The broader overlap among M9.1-recovered edges was not separately aggregated. Based strictly on this audit, the next experiment should target the largest measured baseline miss stage (rank_above_20) with one isolated development-only change, while preserving all other settings. Do not infer benefit before measuring it, and do not open validation/final-test data until the method is frozen.

## Reproducibility

- UTC timestamp: 2026-10-11T14:07:42.689043+00:00
- Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]; NumPy 2.5.3; pandas 3.0.6; SciPy 1.18.1.
- Per-bin query cap 20; final source cap 10; tie rank is Euclidean XYZ then target hit ID.
- Runtime (baseline graph build + query-stage audit): 5162.671 seconds; excludes truth loading and report generation.
- Tests: focused 7 passed; full suite 150 passed.

# M9.5 Geometry-Neighborhood Failure Audit

## Scope and protocol

Processed all 60 DEVELOPMENT events (1000–1059). Validation (1060–1079) and final-test (1080–1099) events were not opened. M9.1 candidate generation and M9.4 query20 geometry rankings were reconstructed from hit_id,x,y,z only. Truth was loaded only after these geometric outputs were fixed, to build true adjacent-particle pairs and label cohorts. No cells, particle/detector metadata, charge, momentum, or truth entered candidate generation or ranking. Numerical summaries use all applicable edges; plots use only the deterministic sample documented below.

## Cohorts

Counts are full pooled counts. `recovered_m91` is a true edge present in the frozen M9.1 graph. The other ranked cohorts are M9.1-missed true edges with a target in the extended query20 result at per-bin ranks 1–10 or 11–20. `missed_rank_above_20` means the target bin is searched by M9.1 but the target is absent from extended query20. `missed_target_bin_unsearched` means the target's radial bucket is not one of the next ten outward buckets. `missed_radial_filter` means an M9.1-missed target was returned by query20 but fails `0 < Δr <= 180 mm`. The radial-filter cohort overlaps ranked cohorts; the other primary miss groups are disjoint by construction. Counts: {"missed_radial_filter": 6047, "missed_rank_11_20": 365936, "missed_rank_1_10": 353785, "missed_rank_above_20": 2296263, "missed_target_bin_unsearched": 1628049, "recovered_m91": 363034}.

## Geometric distributions

Values below are medians over all applicable edges; full counts, means, quartiles, IQR, p90 and p95 are in `m9_5_summary.csv`.

| Feature | M9.1 recovered | Missed, rank >20 | Missed, target bin unsearched |
|---|---:|---:|---:|
| source_radius_mm | 172.9 | 117.9 | 72.46 |
| delta_r_mm | 42.71 | 49.62 | 1.16 |
| abs_delta_phi_rad | 0.01724 | 0.0316 | 0.0009021 |
| abs_delta_z_mm | 7.2 | 140 | 4.5 |
| distance_xyz_mm | 43.57 | 172.5 | 4.513 |
| distance_transverse_mm | 42.9 | 52.03 | 1.344 |
| per-bin rank (median when returned by query20) | 2 | NA (categorically >20) | NA (target bin unsearched) |

![M9.5 cohort distributions](experiments/figures/m9_5_geometry_cohort_distributions.png)

Histograms use 60 plotting bins for visualization only; all summary statistics use the complete cohorts. For display only, the first 100,000 edge values per cohort and feature in event/true-edge order are plotted. Event-level values are in `m9_5_event_summary.csv`.

## Interpretation and recommendation

The `rank_above_20` cohort has median absolute longitudinal displacement of 140 mm versus 7.2 mm for recovered edges, median XYZ distance 172.5 mm versus 43.6 mm, and median absolute wrapped azimuth displacement 0.0316 rad versus 0.0172 rad. The target-bin-unsearched cohort has median bucket difference 0 and median radial displacement 1.16 mm, consistent with many such pairs falling within the same radial bucket. These are descriptive associations, not causal explanations.

A defensible next development-only hypothesis is to compare an XY-only nearest-neighbor distance with XYZ distance while holding the radial buckets, query budgets, radial filter, and final cap fixed. The larger longitudinal offsets in rank>20 misses make this a specific geometry-only test. It is not known to improve reconstruction; truth must remain post-hoc, and validation/final-test events must remain untouched until the method is frozen.

## Limitations and reproducibility

Target rank is reported when the pair occurs in the diagnostic extended top-20 query; for targets absent from it the measurement is the category `>20`, not an exact full-neighborhood rank. `missed_radial_filter` intentionally overlaps rank cohorts. Detector topology and physical labels were excluded. Runtime: 5840.810 seconds total; geometry construction/query: 4638.529 seconds. Environment: Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]; NumPy 2.5.3; pandas 3.0.6; SciPy 1.18.1.

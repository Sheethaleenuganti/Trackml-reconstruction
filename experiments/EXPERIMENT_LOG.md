# Experiment Log

Every entry here corresponds to an actually-executed piece of work, with
real output captured in `docs/` or `notebooks/`. Nothing here is projected
or planned — planned work belongs in `docs/PROJECT_CONTENT_MASTER.md`
(Sections R/T).

---

## Experiment ID: M3

**Date:** current session (real-data phase)
**Dataset:** Kaggle `train_sample`, event `000001000`
**Purpose:** Validate truth-track selection and visualization; resolve the
real-data finding that `particle_id` does not uniquely identify a
physical particle in `particles.csv`.

**Problem discovered:** Truth-colored trajectory visualization (built
against synthetic data in Step 3, first run on real data in the M2
session) produced chaotic, non-physical zigzagging plots instead of
smooth curves. Initial fix (sort by `transverse_radius` instead of `z`)
improved but did not resolve it — several selected "particles" had
90-186 hits and `phi` spanning almost the full `[-π,π]` range, physically
impossible for a single real particle.

**Diagnosis:** `particles.csv` for event 1000 has 12,263 rows but only
2,564 unique `particle_id` values. Investigated systematically:
- 1,082/2,564 IDs (42%) span multiple distinct production vertices
  (`vx,vy,vz`) — confirmed genuinely different physical particles sharing
  one ID.
- 635/2,564 IDs repeat at the identical vertex with different momentum
  and (581/635 cases) different charge — plausibly consistent with
  pair-production-like physics, but **not confirmed** by any primary
  source; stated as an observation only.
- `trackml.utils.decode_particle_id()` (official library) does not
  distinguish the same-vertex duplicates.
- Raw row count (12,263) matches literature's "O(10⁴) particles/event"
  far better than the deduplicated ID count (2,564) — decisive evidence
  that `particles.csv` rows, not `particle_id` values, are the correct
  unit of "one physical particle."
- Neither `particles.csv` nor `truth.csv` provides a row-level instance
  identifier to resolve which specific row an ambiguous hit belongs to.

**Fix:** New module `src/track_selection.py`. Visualization restricted to
`particle_id` values with exactly one row in `particles.csv` (847/2,564
unambiguous; 751 after official `ORDER_MIN_HITS>=4` filter). Hard-fails
(`ValueError`) if ever called on an ambiguous ID, rather than silently
combining data from multiple particles.

**Tests:** 10 new tests in `tests/test_track_selection.py`, covering:
unique-ID acceptance, duplicate-ID rejection, unique-but-short-ID
rejection, unique-and-sufficient-ID acceptance, correct hit isolation,
transverse-radius ordering (regression test for the M2 bug), hard-fail
on ambiguous input, correct pT/charge/radius computation, and a
`phi_spread`-based contamination-detection test. **Full suite: 50/50
passing** (was 40 before this experiment).

**Result:** Notebook re-executed end-to-end against real event 1000, 0
errors, 9 plots regenerated. All 8 sampled trajectories: 5-16 hits
(matches ~12-hit literature average), `phi_spread` 0.06-0.90 rad (vs.
near-2π before fix). Visual inspection confirms smooth, physically
sensible curved trajectories — see `notebooks/01_dataset_exploration.ipynb`,
truth-colored plot cell.

**Observations:** This is a genuine, dataset-level data-quality
characteristic, not a one-off bug in our code — worth carrying forward
into the eventual TrackML scoring implementation (M6+), since the
official scoring algorithm also keys on `particle_id` majority-matching
and this ambiguity could affect score interpretation. Flagged as a risk
to revisit at that milestone.

**Next change:** Proceed to coordinate transformation (M4). The
particle-instance-disambiguation problem itself remains open and is
explicitly out of scope for now (see `src/track_selection.py` docstring,
"Scope note").

---

## Experiment ID: M4

**Date:** current session (real-data phase)
**Dataset:** Kaggle `train_sample`, event `000001000`
**Purpose:** Implement and validate the coordinate transformation stage
(official TrackML starting-kit transform), before/after visualization,
physics sanity checks.

**Transform used:** `r1=sqrt(x²+y²+z²)`, `x2=x/r1`, `y2=y/r1`,
`r2=sqrt(x²+y²)`, `z2=z/r2` — verified directly against Amrouche et al.
2019 (arXiv:1904.06778, Section 2.7) during this milestone, not assumed
from memory of Step 2.

**Tests:** 20 new (`tests/test_transformation.py`), all hand-computed
analytical values. **Full suite: 70/70 passing** (was 50 before M4).

**Real-data validation:** 120,939 input hits → 120,939 output hits, exact
hit_id alignment, **zero NaN/inf** (no real hit hit the origin/beam-axis
edge cases in this event). x2/y2 in [-1,1]; z2 compressed to [-42.5,47.9]
vs raw z's [-2955,2955]. Full output: `docs/m4_real_data_validation.txt`.

**Physics validation:** 10 real tracks checked — r2 monotonically
non-decreasing in 10/10 (consistent with M3's radius-based ordering);
r1 monotonic in 9/10 (one plausible non-monotonic exception, not a bug);
zero cross-track hit_id collisions. Full output:
`docs/m4_physics_validation.txt`.

**Observations:** Before/after visualization on a random unambiguous-track
sample (seed 42) showed spatial compression but not dramatic visual
"straightening" — these happened to be higher-pT, already-fairly-straight
tracks. A second, deliberately low-pT sample (pT<0.3 GeV/c) showed one
track that remained visibly curved even after transformation — an honest,
undoctored finding, not smoothed over. Whether the observed compression
still meaningfully improves DBSCAN's ability to separate tracks by density
is an open empirical question, explicitly deferred to M5.

**Next change:** M5 — DBSCAN baseline implementation, using the `x2, y2,
z2` transformed representation as input (never raw x,y,z alone, and never
particle_id/truth).

---

## Experiment ID: M5

**Date:** current session (real-data phase)
**Dataset:** Kaggle `train_sample`, event `000001000` (120,939 hits, complete
event, no hit-level subsampling)
**Purpose:** First real, quantitative DBSCAN baseline. Research question:
can DBSCAN recover particle tracks from M4-transformed coordinates without
truth, and how does performance vary with representation/parameters?

**Method:** `src/clustering.py` (feature extraction, DBSCAN wrapper, k-NN
diagnostic) and `src/evaluation.py` (wraps official `trackml.score.score_event`
directly, validated against the exact Step 2 hand-worked 8-hit example —
reproduced 0.90 exactly; adds original purity/efficiency/fragmentation/
merging diagnostics).

**k-NN diagnostic (real, k=4):** revealed a genuine scale mismatch — z2
std (8.57) is ~22x larger than x2/y2 std (0.38), confirmed visually (raw
xyz k-NN max distance 6.21 vs xy's 0.019). This justified testing a
scaled xyz variant, per the task's explicit "don't assume, verify" rule.
Figure: `experiments/figures/m5_knn_diagnostic_all_representations.png`.

**Sweep:** 45 runs, 3 representations (xy / xyz / xyz_scaled) x up to 5
eps values (chosen from real k-NN percentiles) x 3 min_samples (3,4,6).
Full results: `experiments/results/m5_dbscan_results.csv`. ALL 45 runs
recorded, including poor ones (e.g. eps=0.012 on xy: 20 clusters,
score=0.0000 — total under-segmentation).

**Best configuration found within tested grid:** representation=xyz
(unscaled), eps=0.01, min_samples=3. TrackML score=0.0269 (2.7%),
n_clusters=11,544, noise=51.3%, mean_purity=0.637, mean_efficiency=0.074,
n_fragmented_particles=1,644 (of ~2,200 recovered).

**Honest headline result:** score is LOW — nowhere near the literature's
~0.20-0.22 starting-kit baseline (Step 2/M4). Root cause investigated and
confirmed via cluster-size distribution (median 4 hits/cluster) and visual
inspection (Fig A): DBSCAN at these parameters finds tiny, tight local
clusters, not extended tracks — severe FRAGMENTATION is the dominant
failure mode, not merging or pure noise-rejection (though both also occur
— see Fig D for one real example of each: success, fragmentation, merging,
noise-rejection, all four found in this single event, none manufactured).

**Representation comparison (max score per representation):** xyz=0.0269
> xyz_scaled=0.0268 > xy=0.0160. Scaling did NOT improve on unscaled xyz
in this sweep — this contradicts the naive expectation from the scale-
mismatch finding and is reported as-is, not smoothed over.

**Bug found and fixed during this milestone:** `cluster_purity_and_efficiency`
initially counted `majority_particle_id == 0` (noise, per the documented
convention) as if it were a "fragmented particle" (895 of 11,544 clusters
in the best run were noise-dominated and were being miscounted). Fixed in
`src/evaluation.py`; 2 regression tests added
(`test_noise_majority_particle_not_counted_as_fragmentation`,
`test_noise_dominated_cluster_excluded_from_recovered_particle_count`).
Sweep rerun after the fix; score/purity/efficiency numbers unaffected,
`n_fragmented_particles` and `n_truth_particles_recovered` corrected
downward slightly (e.g. 1645→1644 for the best run).

**pT / hit-count analysis (best config, unambiguous particles only, n=847):**
recovery rate rises with pT (low pT<0.3: 54.3%, mid 0.3-1.0: 78.7%, high
pT>1.0: 85.6%) and with true hit count (4-6 hits: 42.1%, 7-10: 57.3%, 11+:
88.7%) — both consistent with Step 2's physical predictions (low-pT tracks
curl tightly and are harder). Single-event sample; descriptive only, no
causal claim. Full output: `docs/m5_pt_and_hitcount_analysis.txt`.

**Tests:** 32 new (`tests/test_clustering.py`, `tests/test_evaluation.py`),
plus 2 regression tests for the fragmentation-counting bug. **Full suite:
104/104 passing** (was 70 before M5).

**Reproducibility check:** baseline rerun from documented command
(`extract_features(..., "xyz")` → `cluster_tracks(eps=0.01, min_samples=3)`
→ `summarize_evaluation`) — exact match: 11,544 clusters, 61,996 noise,
score=0.026888.

**Observations:** The transform (M4) does appear to help somewhat (xyz >
xy), but DBSCAN's fundamental behavior — finding tight local density
pockets rather than extended curved structures — is the dominant
limitation, not the coordinate representation. This suggests the M6
research direction should investigate parameter regions or preprocessing
that specifically address fragmentation (e.g., larger min_samples with
carefully tuned eps, or a fundamentally different distance metric/feature
engineering that better captures "along-track" proximity), rather than
further representation-scaling experiments.

**Next change:** M6 — investigate fragmentation directly (why do
DBSCAN clusters stay so small even when true tracks have 10+ hits nearby
in transformed space?).

---

*(Future experiments — M6 onward — will be appended below as they are
actually run.)*

## M7.1 — Dataset inventory (2026-10-06)

**Purpose:** establish the complete-event inventory before any multi-event
baseline or split design. The inventory reads the TrackML CSV records only;
it does not use particle/truth values for reconstruction or clustering.

**Input:** the uploaded `train_sample.zip`, containing events 1000–1099.
The archive was read in place and was not extracted or copied into the
repository. Reproduce from the repository root with:

```text
python -m scripts.build_dataset_inventory "<path-to-train_sample.zip>"
```

**Observed result:** 100 unique events inventoried; all 100 have hits, cells,
particles, and truth files. Aggregate data-row counts were 10,967,467 hits,
60,159,209 cells, 1,079,146 particles, and 10,967,467 truth rows. Per-event
counts are saved in `experiments/results/dataset_inventory.csv`. The event
1000 counts (120,939 hits, 664,996 cells, 12,263 particles, 120,939 truth
rows) match the existing M5 reference-event profile.

**Validation:** focused inventory tests: 2 passed. Full suite available in
the supplied M5 archive plus M7.1: 106 passed. The M6 diagnostic files/tests
were not present in the supplied M5 archive, so this run cannot substantiate
the project-context figure of 120/120 tests.

## M7.2 — Deterministic event split manifest (2026-10-06)

**Purpose:** define whole-event development, validation, and final-test
groups before multi-event reconstruction experiments. Assignment is fixed by
the requested contiguous event-ID ranges; there was no shuffle, raw-data
scan, hit-level split, or use of event contents to make the assignments.

**Manifest:** `experiments/results/event_split_manifest.csv` (`event_id`,
`split`). It contains events 1000–1059 as DEVELOPMENT (60), 1060–1079 as
VALIDATION (20), and 1080–1099 as FINAL_TEST (20). The M5 reference event
1000 remains in DEVELOPMENT.

**Reproduction and validation:** from the repository root, run
`python -m scripts.event_split_manifest create` to write the manifest and
`python -m scripts.event_split_manifest validate` to check its exact IDs,
uniqueness, split membership, and counts.

**Tests:** focused M7.2 tests: 5 passed. Available full suite: 111 passed.
This supplied project snapshot does not include the later M6 files/tests, so
the full-suite count here covers only files present in this snapshot and is
not the stated 120-test suite.

## M8.1 — Frozen M5 baseline on DEVELOPMENT (2026-10-08)

**Configuration:** transformed unscaled XYZ features; DBSCAN with the frozen
`eps=0.01`, `min_samples=3`; complete event hits with no hit-level sampling.
The run used only event IDs assigned DEVELOPMENT by the validated M7.2
manifest (1000–1059). Hits alone entered transformation/features/clustering;
truth was loaded after predicted labels were produced and used only by the
existing post-hoc evaluation functions. Runtime is the existing DBSCAN-fit
timer returned by `src.clustering.cluster_tracks`.

**Result files:** per-event metrics in
`experiments/results/m8_1_development_baseline.csv`; aggregate statistics in
`experiments/results/m8_1_development_summary.json`; failed-event records in
`experiments/results/m8_1_development_failures.csv`.

**Observed aggregate (n=60; 0 failed):** mean official score 0.1343528684,
median 0.1341028884, population standard deviation 0.0092223615, minimum
0.1140100658, maximum 0.1598230600. Mean DBSCAN runtime 0.6623 seconds and
median 0.4501 seconds. Total job wall time was 272.5885 seconds.

**Reference reproducibility issue:** event 1000 produced score 0.1362513453
with 11,544 predicted clusters and 61,996 noise hits. The cluster and noise
counts match the saved M5 reference, but its saved official score is
0.0268875696. The current official scorer was called through the existing
M5 evaluation code; no M5 implementation or configuration was changed. This
score discrepancy remains unresolved and should be investigated before the
M8.1 scores are treated as a reproduced extension of the historical M5
score. The run used Python 3.13.5 and scikit-learn 1.8.0; the recorded M5
environment used Python 3.12.3, so the execution environments were not
identical.

**Validation:** focused M8.1 runner test: 1 passed. Available full suite:
112 passed. The supplied snapshot does not include later M6 tests/files, so
this is not a run of the stated 120-test suite. No validation or final-test
events were evaluated.

## M9.1 — Geometry-aware candidate graph

**Question:** can local detector geometry produce a sparse directed candidate
graph that recovers true consecutive-hit connections without using truth during
generation?

**Method/configuration:** `src.geometry_graph.build_candidate_graph` reads only
`hit_id,x,y,z`. It uses a transverse-radius sweep, 180 mm maximum outward radial
step, outward radial indexing bins no wider than 20 mm (at least ten bins per
search window), cKDTree queries in Cartesian XYZ, and a top-10 per-source edge
budget. Candidates are ranked by Euclidean XYZ distance with target hit ID as a
tie-break. Same-bin targets are excluded. The 180 mm initial window was chosen
from an unsupervised event-1000 geometry survey: adjacent barrel-layer median
radius gaps were approximately 40–160 mm. The 10-edge cap is a declared
sparsity budget, not selected against truth or held-out data.

**Leakage boundary:** each graph was generated after loading hits only. Truth
was loaded after candidate generation and used only for the evaluation of
consecutive nonzero-particle hits ordered by increasing transverse radius.
Equal-radius pairs were excluded because edges are strictly outward-directed.
No validation or final-test event was loaded.

**Observed results (60 development events; 0 failed):** 6,632,611 hits and
63,326,540 candidate edges total; mean 1,055,442 edges/event, mean 9.5456
edges/hit (event-level means averaged), and maximum 10 edges/hit. Pooled
post-hoc true-edge recall was 0.072504 (363,034 / 5,007,067); mean event recall
was 0.072812. Mean event candidate density against `N*(N-1)/2` possible
outward pairs was 0.00017423 (0.0174%). Geometry-generation runtime was
983.364 seconds total, 16.389 seconds/event on average, excluding event/truth
loading and evaluation.

For event 1000, the graph contains 1,157,210 edges and recovers 6,580 of
92,740 evaluated true consecutive edges (recall 0.070951). The representative
edge file is compressed; event-level results are in
`experiments/results/m9_1_geometry_graph_summary.csv` and JSON aggregate
metadata is in `experiments/results/m9_1_geometry_graph_summary.json`.

**Interpretation:** the graph is sparse relative to all possible outward pairs,
but the measured 7.25% pooled recall is low. This initial design does not
recover a high fraction of true local connections and does not constitute an
end-to-end track reconstruction result. The missed-edge distribution should
be diagnosed on development data before proposing a revised candidate method.

**Artifacts:** `experiments/M9_1_PLAN.md`,
`experiments/results/m9_1_geometry_graph_report.md`, the summary CSV/JSON,
`experiments/results/m9_1_geometry_graph_edges_event000001000.csv.gz`, and two
event-1000 figures under `experiments/figures/`.

**Validation:** focused geometry graph tests: 9 passed. No full-suite run was
performed for this milestone.

## M9.2 — Missed true-edge diagnosis

Post-hoc diagnosis of true consecutive edge misses using only the 60 validated
development events. M9.1 candidate construction and parameters remain
unchanged. The event-1000 saved graph is reused; the unchanged M9.1 builder is
run in memory for the other development events because no full edge artifacts
were retained for them. Truth is loaded only after each graph exists. The
analysis reports radial step, wrapped azimuth, longitudinal displacement, XYZ
distance, bucket/search exclusions, per-bin query membership, final top-k
truncation, overlapping mechanisms, and primary causes. No validation or
final-test data or alternative graph configurations are used. Results are
saved in `experiments/results/m9_2_*`; see `experiments/M9_2_PLAN.md` and the
generated report. M9.3 is not run.

## M9.3 — Per-target-bin query-cap ablation

Controlled development-only ablation of the M9.2-proposed query visibility
change. A separate builder raises the per-target-bin query cap from 10 to 20;
the final per-source cap remains 10, and all other M9.1 geometry and ranking
settings remain fixed. Candidate graphs are generated from hit geometry only;
truth is loaded only after both M9.1 and M9.3 graphs exist. Compare exact true
pair and candidate pair sets across the 60 development events, with results
under `experiments/results/m9_3_*`. Validation and final-test events remain
untouched. See `experiments/M9_3_PLAN.md`; M9.4 is not run automatically.

## M9.4 — Per-bin rank audit

Audit the frozen M9.1 candidate-selection stages on the 60 development events.
Confirm query-before-radial-filter ordering, measure deterministic per-bin
query20 ranks and exact merged/final-top10 survival, compare the reconstructed
baseline to the saved M9.1 artifact/summary, and load truth only after all
geometry-only ranks are fixed. Validation/final-test events remain untouched.
See `experiments/M9_4_PLAN.md` and `experiments/results/m9_4_report.md`.

**Actual outcome:** 60/60 development events processed; no M9.1 per-event
summary mismatches. Rebuilt event 1000 matched the saved candidate source/target
and rank table exactly (1,157,210 rows; geometry tolerance 1e-12); the saved
artifact SHA-256 was unchanged. Across 5,007,067 true edges, M9.1 recovered
363,034. The largest mutually exclusive miss stage was per-bin rank >20
(2,296,263); 1,628,049 were in target buckets never searched; 6,047 failed
the radial filter after query20; 713,672 were rejected by the final top-10;
2 missed edges would survive the extended top-10. Among M9.1-missed edges
ranked 11–20, 365,444 passed the radial filter and 0 survived the final
top-10. The full rank-11–20/final-top-10 overlap among already recovered
edges was not separately aggregated. Geometry-stage runtime was 5,162.671 s.
Focused tests: 7 passed. Full suite: 150 passed. Validation and final-test
events were not evaluated. See the M9.4 report and machine-readable outputs
for the complete breakdown and limitations.

## M9.5 — Geometry-neighborhood failure audit

Post-hoc comparison of coordinate-derived properties for frozen M9.1 true-edge
recovery cohorts on DEVELOPMENT events 1000–1059 only. M9.1 candidates and
M9.4 per-bin query20 ranks were fixed using hit ID and XYZ geometry before
truth was loaded. No cells, detector identifiers, particle features, or labels
entered candidate generation or ranking. Complete cohort statistics were
computed without sampling; plots use the deterministic first 100,000 edge
values per cohort/feature. Exact per-bin rank statistics include only targets
returned by query20; absent targets remain categorized as rank >20 rather than
being assigned an exact rank. See `experiments/M9_5_PLAN.md` and
`experiments/results/m9_5_report.md`.

**Actual outcome:** all 60 development events processed; M9.1 true-edge and
recovery counts matched saved event summaries, and the rebuilt event-1000
candidate pairs/ranks matched the frozen artifact. Cohort counts were 363,034
recovered, 353,785 missed at query20 ranks 1–10, 365,936 missed at ranks
11–20, 2,296,263 beyond query20, 1,628,049 in unsearched target buckets, and
6,047 excluded by the radial filter (the last cohort overlaps returned-rank
cohorts). Rank >20 misses had median absolute Δz 140 mm and median XYZ
distance 172.5 mm, compared with 7.2 mm and 43.6 mm for recovered edges.
Unsearched-bucket misses had median bucket difference 0 and median Δr 1.16
mm. These are descriptive associations, not causal findings. A testable
follow-up is an XY-only versus XYZ neighborhood metric comparison with all
other graph settings fixed; no benefit is established. Total runtime was
5,840.810 s (geometry construction/query: 4,638.529 s). Focused tests: 6
passed; full suite: 156 passed. Validation/final-test data were not opened.

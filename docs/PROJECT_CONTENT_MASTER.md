# Project Content Master — Track Reconstruction (TrackML)

> **Historical planning snapshot.** This document contains milestone-era
> status statements that became stale as later experiments were completed.
> Use `README.md` for the publication summary and `experiments/EXPERIMENT_LOG.md`
> plus the linked result artifacts for current verified M1–M9.5 outcomes.
> Earlier text below is preserved as project history and must not be read as
> a current status report without checking its cited artifacts.

**Status key used throughout: COMPLETED / IN PROGRESS / PROPOSED / FUTURE WORK / TO BE COMPLETED**

This document is the single source of truth for both the mid-sem slides and
the final report. Every claim in either document must trace back to a line
here. Nothing here is invented; every "COMPLETED" item has been actually
run and its output inspected in this project's working sessions.

---

## A. Project Title

**Track Reconstruction (TrackML)** — Project 35, Pinnacle 3: Machine
Learning (Astrophysics, Particle Physics and Drug Design), Section B.
**Status: COMPLETED** (fixed by the official course specification).

## B. Background

Particle colliders like the LHC smash protons together at close to light
speed. Each collision ("event") produces thousands of new particles flying
outward through a magnetic field. A layered silicon detector records the
3D position each particle occupied as it crossed each layer — a "hit."
**Status: COMPLETED.**

## C. Problem Statement

Given only the unordered set of hits in one event (no particle labels),
group the hits into clusters, each cluster hypothesized to be the
trajectory of one particle. This is a **clustering problem, not
classification** — the number of true clusters (particles) varies event to
event and is not known in advance. **Status: COMPLETED.**

## D. Motivation

Future colliders (HL-LHC) will produce far denser events. Existing
combinatorial tracking algorithms scale worse than quadratically with hit
count. Faster, ML-based grouping is an active, real research need — not a
solved textbook problem. **Status: COMPLETED.**

## E. Objectives

1. Load and validate the TrackML dataset structure (hits, cells,
   particles, truth). **Status: IN PROGRESS** — code complete and
   unit-tested; real-data validation pending (see Section P).
2. Understand and implement a coordinate transformation that makes
   helical trajectories approximately linear. **Status: COMPLETED**
   (implementation + tests + real-data validation + before/after
   visualization — see Section J.1 below). Two candidate transforms were
   identified from the official starting kit (Step 2); the DBSCAN-direct
   candidate was implemented (not the Hough/polar candidate).
3. Cluster transformed hits with a density-based algorithm (DBSCAN).
   **Status: COMPLETED (baseline established)** — see Section J.2. First
   real quantitative result obtained; score is low (2.7%), root cause
   (fragmentation) identified with real evidence.
4. Implement and unit-test the official TrackML weighted hit-matching
   score. **Status: PROPOSED** — exact formula and a hand-worked toy
   example already derived from primary sources; not yet coded.
5. Compare against a random-assignment baseline. **Status: PROPOSED.**
6. Analyze reconstruction efficiency vs. transverse momentum (pT) and
   event occupancy. **Status: PROPOSED.**

## F. Scope

Laptop-CPU feasible; ready-made tools only (pandas, scikit-learn,
optionally HDBSCAN pending course approval); no novel neural architecture;
whole-event subsampling only (never truncate within an event). Matches the
official Project 35 specification exactly. **Status: COMPLETED**
(definition), **PROPOSED** (execution).

## G. Literature Review

Real, verified sources (7 total — within the faculty-required 5–10 range).
All retrieved and read (abstracts/relevant sections) during this project,
not assumed from memory.

1. **Amrouche, S. et al. (2019).** *The Tracking Machine Learning
   challenge: Accuracy phase.* arXiv:1904.06778. — Primary source for the
   challenge design, the official DBSCAN starting-kit transform
   (~0.20–0.22 baseline score), and the exact scoring metric definition
   this project implements.
2. **Rousseau, D. et al.** *The TrackML high-energy physics tracking
   challenge on Kaggle.* EPJ Web of Conferences (CHEP 2018). — Confirms
   dataset scale (~10⁴ particles, ~10⁵ hits/event) and the ≥4-hits
   scoring threshold.
3. **Farrell, S. et al. (2018).** *Novel deep learning methods for track
   reconstruction.* arXiv:1810.06111 (HEP.TrkX). — Foundational graph
   neural network approach to TrackML tracking; establishes the
   GNN-based alternative to our clustering-based approach, useful for
   framing the research gap.
4. **Ju, X. et al. (2021).** *Performance of a geometric deep learning
   pipeline for HL-LHC particle tracking.* Eur. Phys. J. C 81, 876
   (Exa.TrkX). — Shows GNN pipelines reach efficiency/purity comparable
   to production tracking algorithms, at higher computational cost —
   directly relevant to motivating why a lightweight DBSCAN-based
   baseline remains worth studying.
5. **DeZoort, G. et al. (2021).** *Charged particle tracking via
   edge-classifying interaction networks.* arXiv:2103.16701. — Explicitly
   documents that TrackML detector geometry (9 barrel layers + disks,
   pixel + short-strip + long-strip sub-detectors) and confirms the
   <0.5% hit-merging rate referenced in our Step 2 research.
6. **Liaopeiyuan (GitHub, TrackML-Particle-Tracking-Challenge repository).**
   *Clustering LHC detector hits by Hough projection, DBSCAN, classifier
   merging, outlier elimination and track extension.* — A public
   Kaggle-competition solution combining DBSCAN with the Hough-transform
   candidate identified in our Step 2 research; useful as an
   implementation reference (not to be copied, but informs our parameter
   choices).
7. **Tüysüz, C. et al. (2023).** *Hybrid quantum classical graph neural
   networks for particle track reconstruction.* Quantum Machine
   Intelligence. — Reiterates the HEP.TrkX preprocessing pipeline as a
   community standard reference point, useful for contextualizing our
   own preprocessing choices in the report's methodology chapter.

**Status: COMPLETED** (7 sources identified and read; 3-5 more can be
added for the final report if the faculty wants closer to the 10-paper
ceiling — **FUTURE WORK**).

## H. Existing Approaches (from literature)

Two broad families exist in the literature: (1) **classical/geometric**
— coordinate transforms + density-based clustering (DBSCAN) or Hough
transforms, achieving ~20-25% TrackML score in the original starting kit
[1,6]; (2) **graph neural network** approaches (HEP.TrkX, Exa.TrkX and
successors) [3,4,5], achieving much higher efficiency/purity but at
significantly higher computational and implementation cost, and explicitly
excluded from our course's MVP scope (novel neural architectures are
disallowed; GNN refinement is only an *optional* extension, and only a
"ready-made" published implementation would be permitted). **Status:
COMPLETED.**

## I. Research Gap

Framed honestly as a **proposed research question**, not a claimed
novelty (per course rules and Step 3's "no premature novelty claims"):

> *How strongly does the choice of coordinate representation affect
> density-based track reconstruction quality across different momentum
> and event-occupancy regimes, on a laptop-feasible subsample?*

This is not claimed as unanswered in the wider literature (GNN papers
implicitly answer versions of it), but it is unanswered **by us**, and is
answerable within this project's scope using controlled, reproducible
experiments. **Status: PROPOSED (research question), not yet
investigated.**

## J. Proposed Approach

```
TrackML hits (raw x,y,z)
      ↓
Coordinate transformation (starting-kit candidate, Step 2 Part E)
      ↓
DBSCAN clustering (permitted inputs only — no truth)
      ↓
Track candidates (hit_id → predicted track_id)
      ↓
TrackML weighted hit-matching score (vs. truth, evaluation-only)
      ↓
Efficiency analysis: vs. pT, vs. occupancy
```
**Status: PARTIALLY IMPLEMENTED** — data loading/validation (M1-M3) and
coordinate transformation (M4) are real and tested; clustering onward
remains PROPOSED.

### J.1 — M4: Coordinate Transformation (COMPLETED)

**Transform implemented** (`src/transformation.py`), exactly the official
TrackML starting-kit formula, re-verified directly against the primary
source during M4 (Amrouche et al. 2019, arXiv:1904.06778, Section 2.7):

```
r1 = sqrt(x^2 + y^2 + z^2)
x2 = x / r1
y2 = y / r1
r2 = sqrt(x^2 + y^2)
z2 = z / r2
```

**Physics justification**: charged particles follow helical trajectories
in the transverse plane due to the magnetic field (Step 2, Part D). Raw
(x,y,z) clustering is hampered because same-particle hits can be spread
out while different-particle hits near the collision point can start
close together. This transform projects (x,y) toward a direction-like
quantity via division by the full 3D radius, and normalizes z by the
*transverse* radius specifically — intended to bring same-helix hits
closer together in the transformed space than in raw Cartesian space.
Selected over the Step 2 Hough/polar candidate because it maps directly
onto this project's "transform → DBSCAN" objective, matching Amrouche et
al.'s reported ~0.21 starting-kit score for this exact transform.

**Tests**: 20 new tests (`tests/test_transformation.py`), all with
hand-computed analytical expected values (e.g. x=3,y=4,z=0 → r1=5,
x2=0.6, y2=0.8 by hand), covering known-value cases, zero/origin edge
cases (explicit NaN, never silent inf), negative coordinates, multiple
hits, hit_id preservation/ordering, no-unexpected-NaN on normal data,
invalid-input rejection, and output schema. **Full suite: 70/70 passing**
(was 50 before M4).

**Real-data validation** (event 1000, 120,939 hits): input/output hit
counts match exactly (120,939 → 120,939); hit_id sets and order both
preserved exactly. **Zero NaN and zero inf values produced** — no real
hit in this event landed at the origin or exactly on the beam axis, so
the explicit edge-case handling in the code was not exercised by this
particular event (but remains necessary and tested synthetically). x2/y2
ranges: [-1,1] as expected for direction-cosine-like quantities. z2
range: [-42.5, 47.9], much more compressed than raw z's [-2955, 2955].
Full output saved: `docs/m4_real_data_validation.txt`.

**Before/after visualization**: generated on real event 1000, two
samples. Figures saved to `experiments/figures/`:
- `m4_before_after_xy_highpt_sample.png` / `_zr_highpt_sample.png`: 6
  randomly-selected unambiguous tracks (seed 42). These happened to be
  higher-pT (straighter) tracks; the transform visibly compresses their
  spatial extent (raw x span ~973mm → x2 span ~1.07, unitless) but does
  not show dramatic qualitative "straightening", since these tracks were
  already close to straight in raw space.
- `m4_before_after_xy_lowpt_sample.png`: 6 deliberately-selected low-pT
  tracks (pT<0.3 GeV/c, expected to show visible helical curvature).
  **Honest finding, not smoothed over**: one track (pT=0.28 GeV/c) shows
  clearly visible curvature in raw x-y space, and **remains visibly
  curved** after transformation — the transform did not fully straighten
  it in this instance. The other 5 low-pT tracks became compact,
  near-point clusters in transformed space (potentially more separable
  by proximity, but via compression toward the unit circle rather than
  visible "straightening" per se).

**Physics validation** (`docs/m4_physics_validation.txt`, 10 tracks
checked): transverse radius (r2) is monotonically non-decreasing along
every checked track (10/10) — consistent with a particle flying outward,
and confirms the transform did not disrupt the physically-correct
ordering established in M3. Full 3D radius (r1) monotonicity holds for
9/10 tracks; one track has a small non-monotonic segment in r1 while
remaining r2-monotonic, which is physically plausible (a particle can
move slightly toward/away from the exact 3D origin while its transverse
radius still strictly increases) rather than a bug. Zero NaN/inf on all
checked real tracks. Zero cross-track hit_id collisions confirmed (no
accidental mixing between selected tracks' transformed data).

**Limitations discovered**:
1. The transform's qualitative "helix → line" benefit is not uniformly
   visible across all tracks in this real sample — it appears more as a
   spatial-extent compression than a guaranteed visual straightening,
   at least for the specific tracks sampled here. Whether this compression
   still meaningfully helps DBSCAN separate tracks by density is an
   empirical question for M5 (DBSCAN baseline), not yet answered.
2. No real hit in event 1000 exercised the origin/beam-axis edge cases —
   these remain validated only against synthetic hand-built cases, not
   real data. Should be re-checked if a future event does contain such
   hits.
3. `x2, y2` alone (without `z2`) discard z-direction information into a
   single normalized value combined with x,y through r1 — whether this
   loses information relevant to clustering is not yet empirically tested.

**M4 status: COMPLETED** (code + tests + real-data validation +
visualization + documentation, per the milestone's own completion
criteria).

### J.2 — M5: DBSCAN Baseline (COMPLETED)

**Research question:** Can DBSCAN recover particle tracks from
M4-transformed coordinates without truth, and how does performance vary
with clustering parameters and coordinate representation?

**Dataset:** Event 1000, complete (120,939 hits, no hit-level subsampling).

**Method:** `src/clustering.py` (feature extraction from M4's
`transform_hits` output, DBSCAN wrapper via scikit-learn, k-NN distance
diagnostic for eps selection) and `src/evaluation.py`, which calls the
**official** `trackml.score.score_event()` directly (validated: reproduced
the exact Step 2 hand-worked 8-hit toy score of 0.90) rather than
reimplementing it, plus original purity/efficiency/fragmentation/merging
diagnostics.

**Representations tested:** `(x2,y2)`, `(x2,y2,z2)` unscaled, `(x2,y2,z2)`
standardized. The k-NN diagnostic revealed z2's std (8.57) is ~22x
x2/y2's std (0.38) — a real, measured scale mismatch, confirmed visually
(unscaled xyz k-NN max distance 6.21 vs. xy's 0.019) — justifying the
scaled variant as a genuine experimental question, not an assumption.

**Parameter search:** 45-run grid (3 representations x up to 5 eps values,
chosen from real k-NN percentiles, x 3 min_samples). Full results in
`experiments/results/m5_dbscan_results.csv` — every run recorded, good
and bad.

**RESULT (honest, not smoothed over): the baseline score is LOW.** Best
configuration in the tested grid: `xyz` (unscaled), eps=0.01,
min_samples=3 → **TrackML score = 0.0269 (2.7%)**, far below the
literature's ~0.20-0.22 starting-kit reference (Step 2/M4) for this same
transform. This is reported as the genuine result, not adjusted or
cherry-picked.

**Representation comparison:** max score achieved — xyz=0.0269 >
xyz_scaled=0.0268 > xy=0.0160. Scaling did **not** improve on unscaled
xyz, despite the real scale-mismatch finding — a genuine, somewhat
counter-intuitive result, reported as observed.

**Root-cause diagnosis (with real evidence, not speculation):** the best
run's predicted clusters have a median size of 4 hits (mean 5.1, max 49)
— far too small to represent whole ~10-hit tracks. Visual inspection
(`experiments/figures/m5_figA_clusters_xy.png`) confirms DBSCAN is
finding tiny, tight local density pockets, not extended curved
structures. **Fragmentation is the dominant failure mode**: 1,644 of
~2,200 real particles recovered were split across multiple predicted
clusters. Concrete example found and visualized
(`experiments/figures/m5_figD_failure_modes.png`): a single real particle
with a clean, smooth, extended trajectory got split into 2 separate
DBSCAN clusters at these parameters.

**Failure modes — all four found as real, non-manufactured examples in
this one event** (Fig D): (1) SUCCESS — cluster 24, purity=0.94,
efficiency=1.00 (15/15 true hits recovered); (2) FRAGMENTATION — particle
22526000000000000 split across 2 clusters; (3) MERGING — cluster 1178, 49
hits, purity=0.24 (many different true particles mixed); (4) NOISE
REJECTION — particle 4504560000000000, 8/10 true hits became DBSCAN
noise.

**Bug found and fixed during M5:** the purity/efficiency table initially
mis-flagged noise-dominated clusters (`majority_particle_id == 0`) as
"fragmented particles" (895 of 11,544 clusters in the best run). Fixed
with 2 new regression tests. This is recorded honestly as part of the
scientific process, not hidden.

**pT / hit-count dependence (descriptive, single-event, n=847 unambiguous
particles):** recovery rate (recovered as *some* cluster's majority at
all) rises with pT: low (pT<0.3 GeV/c) 54.3%, mid (0.3-1.0) 78.7%, high
(>1.0) 85.6%. Also rises with true hit count: 4-6 hits 42.1%, 7-10 hits
57.3%, 11+ hits 88.7%. Both trends match Step 2's physical prediction that
low-pT (tightly curving) and short tracks are harder to reconstruct.
**No causal claim** — single event, descriptive only.

**Tests:** 32 new + 2 regression tests (bug fix). **Full suite: 104/104
passing** (was 70 pre-M5).

**Figures generated (all real, saved to `experiments/figures/`):**
`m5_knn_diagnostic_all_representations.png`, `m5_score_vs_eps.png`,
`m5_cluster_size_distribution.png`, `m5_figA_clusters_xy.png`,
`m5_figB_noise.png`, `m5_figD_failure_modes.png` (combines the
truth-comparison figure C's purpose into the 4-panel failure-mode figure,
since each panel already overlays truth hits against predicted clusters
— a separate standalone Fig C was judged redundant given this overlap,
noted here rather than silently omitted).

**Reproducibility:** baseline rerun from the documented command
(`extract_features(transformed, "xyz")` →
`cluster_tracks(X, eps=0.01, min_samples=3)` → `summarize_evaluation`)
reproduces the exact reported numbers (11,544 clusters, 61,996 noise,
score=0.026888).

**Limitations:**
1. Single real event — no claim generalizes beyond event 1000 without
   further testing.
2. The eps grids were bounded by the real k-NN percentiles but were not
   exhaustive; the `xy` representation's score was still rising at the
   smallest eps tested (0.001), suggesting its true optimum may lie at
   even smaller eps, not yet located.
3. pT/hit-count analysis restricted to unambiguous particles (M3
   constraint) — 847 of 2,564 unique particle_ids on this event; results
   may not generalize to the ambiguous-ID particles excluded from this
   analysis.

**Interpretation — what M5 demonstrates and does not demonstrate:**
DEMONSTRATES: the M4 transform + DBSCAN pipeline is implementable, runs
efficiently on a full real event, and its output can be rigorously scored
against real truth via the official metric; a real, reproducible baseline
now exists; fragmentation (not merging or pure noise) is the dominant,
evidenced failure mode. DOES NOT DEMONSTRATE: that this approach is
competitive with the literature baseline (it is currently ~8x lower);
that scaling is never useful (only that it wasn't useful in this specific
grid); that any single eps/min_samples choice is optimal (only
best-within-tested-grid).

**Next research question (M6):** Why does DBSCAN fragment real,
coherent, extended tracks into small pieces even when transformed
representation should bring same-track hits closer together — is this a
local-density artifact (real tracks have locally sparse hit spacing that
falls below min_samples density in places), a metric/geometry mismatch,
or an eps/min_samples regime not yet explored in this grid?

## K. Dataset

TrackML Particle Tracking Challenge, Kaggle, `train_sample.zip`. Confirmed
via direct visual inspection of the extracted folder
(`train_sample/train_100_events/`), showing all four file types
(`-hits`, `-cells`, `-particles`, `-truth`) present per event, matching the
schema verified against the official `trackml-library` documentation.
**Status: COMPLETED** (dataset selection and download), **IN PROGRESS**
(loading/validation — see Section P).

## L. Data Structure

| File | Role | Verified against |
|---|---|---|
| `hits.csv` | hit_id, x, y, z, volume_id, layer_id, module_id | Official `trackml-library` README (Step 2) |
| `cells.csv` | hit_id, ch0, ch1, value (charge deposition) | same — out of MVP scope |
| `particles.csv` | particle_id, px/py/pz, vx/vy/vz, q, nhits | same |
| `truth.csv` | hit_id → particle_id mapping + per-hit weight | same |

**Status: COMPLETED** (documentation-level verification); **TO BE
COMPLETED** (verification against an actually-opened real file — see
Section P).

## M. Methodology

Coordinate transform → DBSCAN → TrackML score → efficiency breakdown, with
strict train/evaluation separation and a hard rule that the clustering
step never receives `particle_id` or truth weights. **Status: PROPOSED.**

## N. Technologies

Python 3.12, pandas, numpy, scipy, scikit-learn, matplotlib, the official
`trackml` library (installed from `github.com/LAL/trackml-library` — **not**
the unrelated same-named PyPI package, a real trap discovered and
documented in this project). **Status: COMPLETED** (environment set up and
verified — real `check_environment.py` output on file).

## O. Experimental Setup

Config-driven (`configs/data_exploration.yaml`), seeded, reproducible.
Repository structure follows the Step 1 architecture (see
`docs/design_decisions.md`... — **TO BE COMPLETED**, this specific log file
does not yet exist and should be created before final report submission).
**Status: PARTIALLY COMPLETE.**

## P. Work Completed (real, verified)

- Repository architecture designed and built.
- Official `trackml` library correctly installed (from source, not the
  wrong PyPI package) and verified working.
- `src/data_loader.py`, `src/physics.py`, `src/truth_checks.py`,
  `src/profiling.py` — all implemented.
- **39 unit tests, all passing** (real `pytest` run, output captured in
  `docs/test_run_output.txt`).
- Exploration notebook (`notebooks/01_dataset_exploration.ipynb`) — 17
  code cells, actually executed end-to-end, 0 errors, 9 real plots
  produced.
- **All of the above tested exclusively against `[SYNTHETIC DATA]`** — a
  small, clearly-labeled synthetic event generator built specifically
  because real data was not yet available inside the development
  sandbox.
- Real dataset (Kaggle `train_sample.zip`) downloaded by the student and
  confirmed, via screenshot, to contain the correct four-file-per-event
  structure with truth included.

**Status: reflects reality as of this document's writing.**

### P.1 — Real Data Incident Log (event 1000, first real load)

This subsection exists because the project's own rules require stopping
and explaining inconsistencies rather than silently patching them. Three
things were found, in this order, on first contact with real data:

**Finding 1 — Real data matches documentation well overall.** Event 1000:
120,939 hits, 12,263 particle rows, 120,939 truth rows. `len(hits) ==
len(truth)` holds exactly. Every hit has a truth row and vice versa.
Noise fraction = 14.6% (17,634/120,939), closely matching the ~15%
figure from the literature (Step 2, Part J). volume_id spans {7,8,9,12,
13,14,16,17,18} — real multi-volume detector geometry, confirming our
synthetic generator's single-volume simplification was, as flagged at the
time, a real limitation now visibly different from reality.

**Finding 2 — Our own `compare_particles_nhits_with_truth` function had a
bug**, exposed the moment it ran on real data: it merged `particles.csv`
against a truth-derived hit-count table assuming `particle_id` was a
unique row key. It is not (see Limitation 6 above). This produced a
false "92% mismatch" result. **Root cause confirmed** via the official
`trackml.dataset.load_event` loader itself (not just our code) — real
`particles.csv` genuinely has 12,263 rows but only 2,564 unique IDs.
**Fixed** by summing `nhits` per `particle_id` before comparing (see
`src/truth_checks.py` docstring for the full explanation). After the fix:
2,564/2,564 matching, 0 mismatches — full internal consistency confirmed.
A regression test (`test_handles_duplicate_particle_id_rows_correctly`)
was added so this cannot silently regress.

**Finding 3 — The exploration notebook's trajectory-ordering had a
compounding issue.** The first fix (sort hits by `z` → sort by
`transverse_radius`, since a real trajectory's radius increases
monotonically outward while `z` does not) was necessary but not
sufficient: several "particles" selected for the truth-colored
visualization turned out to have 90-186 hits each (vs. the literature's
~12-hit average) with `phi` spanning almost the entire `[-π,π]` range —
a physical impossibility for one real particle. This traces directly back
to Finding 2's root cause: naive selection/grouping by `particle_id` pulls
together hits from multiple distinct particles that happen to share an ID.

**M3 — RESOLVED.** Root-cause investigation (see `src/track_selection.py`
module docstring for full detail) established, with strong evidence:
- 1,082 of 2,564 unique `particle_id` values (42%) span multiple distinct
  production vertices — unambiguously different physical particles
  sharing one ID.
- A further 635 IDs repeat at the exact same vertex with different
  momentum/charge (581/635 show mixed charge — suggestive of
  pair-production-like processes, but this specific physical mechanism is
  **not confirmed** by any primary source available to us; stated as an
  observation, not a fact).
- The official `trackml.utils.decode_particle_id()` cannot distinguish
  these same-vertex duplicates either.
- Decisive corroborating evidence: raw row count in `particles.csv`
  (12,263) matches the literature's "O(10⁴) particles/event" figure far
  better than the deduplicated ID count (2,564) — confirming that
  `particles.csv` ROWS, not `particle_id` VALUES, are the correct unit of
  "one physical particle" in this dataset.
- Neither `particles.csv` nor `truth.csv` contains a row-level instance
  identifier to re-associate ambiguous hits to a specific row, using
  currently-documented fields alone. This is **explicitly not solved** —
  it is worked around for diagnostic purposes (see fix below).

**Fix implemented**: new module `src/track_selection.py` provides
`unambiguous_particle_ids()`, `select_valid_tracks()`, `get_track_hits()`,
and `track_summary()`. Visualization now draws only from `particle_id`
values with exactly one row in `particles.csv` (847 of 2,564 on event
1000; 751 after the official `ORDER_MIN_HITS>=4` filter) — a large, valid,
unbiased pool. `track_summary()` hard-fails (raises `ValueError`) if ever
called on an ambiguous `particle_id`, rather than silently averaging or
combining. **10 new tests added** (`tests/test_track_selection.py`),
covering the four required cases (unique-accepted, duplicate-rejected,
unique-but-short-rejected, unique-and-sufficient-accepted) plus ordering,
isolation, hard-fail, and a `phi_spread`-based contamination-detection
regression test. **Result after fix, re-executed on real event 1000**: all
8 sampled trajectories now show 5-16 hits (matching literature) and
`phi_spread` between 0.06-0.90 rad (vs. near-2π before) — visually
confirmed as smooth, physically sensible curved trajectories, not
zigzags. **Full test suite: 50/50 passing** (was 40 before this fix).

## Q. Actual Results

**None exist yet.** No coordinate transform, clustering, or scoring has
been run — on synthetic or real data. Any number that looks like a
"result" elsewhere in this project (e.g. synthetic profiling stats) is
explicitly a `[SYNTHETIC DATA]` infrastructure-testing artifact, not a
project result. **Status: NONE — do not fabricate.**

## R. Expected Results

Per the official course success criteria: "a working reconstruction
clearly above the random baseline... a modest score properly obtained is
a full-credit outcome." The starting-kit literature baseline is ~20-22%
TrackML score [1] — this is a reasonable order-of-magnitude expectation
for our own DBSCAN baseline, **stated here as an expectation grounded in
literature, not a claim of what we will achieve.** **Status: EXPECTED,
not actual.**

## S. Limitations (current, honest)

1. ~~No real event has yet been loaded and validated by our own code.~~
   **RESOLVED**: event 1000 (Kaggle `train_sample`) loaded and validated —
   see Section P.1 below for what was found.
2. Literature review currently at 7 sources; could be extended.
3. Licence/citation terms for the Kaggle dataset not yet formally
   recorded (flagged in `docs/DATA_ACCESS.md`).
4. HDBSCAN's acceptability under "ready-made tools only" not yet
   confirmed with the course instructor.
5. Whether pT should be computed from `particles.px/py` or per-hit
   `truth.tpx/tpy` remains a team decision, not yet finalized.
6. **NEW — significant, dataset-wide finding**: `particle_id` is NOT a
   unique identifier for a physical particle in the real Kaggle
   `train_sample` data. On event 1000: 12,263 rows in `particles.csv` but
   only 2,564 unique `particle_id` values — up to 16 distinct rows
   (different vertex, momentum, and charge) share the same ID. This means
   any code that groups or merges by `particle_id` alone (ours included,
   originally) can silently produce nonsensical aggregates — e.g. a
   "particle" that appears to have 90-186 hits and a `phi` spanning nearly
   the full `[-π, π]` range, when in reality it's several unrelated
   particles whose IDs happen to collide. See Section P.1 for the full
   incident writeup. **This is now suspected to be an artifact of how
   `train_sample.zip` differs from the full training archives** — i.e., it
   may be specific to the sample-extraction process rather than a property
   of the full dataset — and is flagged as **TO VERIFY** once a full
   `train_N.zip` event is available for comparison.

## T. Future Work

Coordinate transform implementation and visualization → DBSCAN baseline →
TrackML score implementation + unit test → baseline experiment on real
events → efficiency vs. pT/occupancy analysis → (optional) parameter
tuning / HDBSCAN comparison / graph refinement.

## U. References

See Section G (Literature Review) above, plus the two dataset-adjacent
primary sources already cited throughout this project's `docs/` folder.
**A formatted bibliography (course-required citation style) is
TO BE COMPLETED** — the citation style required by the report template
was not specified in the uploaded format documents and should be
confirmed with the faculty mentor.

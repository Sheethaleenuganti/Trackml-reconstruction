# TrackML Particle Track Reconstruction

An ongoing research project exploring particle-track reconstruction from the TrackML Particle Tracking Challenge. The repository contains a frozen DBSCAN reference, event-level experiment manifests, a geometry-only candidate-edge study, diagnostic reports, figures, and tests. It is a developing study; it does not claim state-of-the-art performance.

## 1. Project overview

Each detector event contains unordered hit coordinates. The project studies how to group or connect those hits into candidate particle trajectories and how to evaluate the results against TrackML truth after reconstruction. The work is organized as milestones so that later diagnostics and hypotheses do not silently change earlier baselines.

## 2. Research question and motivation

Can simple geometry-based methods produce useful particle-track candidates from dense collider events, and what limits their reconstruction quality? The current work first establishes a reproducible DBSCAN baseline, then investigates sparse outward hit-to-hit candidate graphs and diagnoses the graph's missed true edges.

## 3. Current methodology

- M5 uses transformed, unscaled XYZ features (`x2`, `y2`, `z2`) and DBSCAN.
- M9.1 builds directed candidate edges using only `hit_id,x,y,z`. It groups hits into radial buckets, searches outward buckets with local XYZ nearest-neighbor queries, filters by positive radial displacement, merges candidates, and keeps at most ten outgoing candidates per source.
- M9.2–M9.5 analyze candidate coverage and geometric properties. These are diagnostic studies, not improved reconstruction models.
- `truth` and `particles` are excluded from candidate generation and clustering. They are joined only after predictions/rankings exist for post-hoc scoring and diagnostics.

## 4. Experiment milestones

| Milestone | Verified status and result |
|---|---|
| M1–M4 | Data-loading and validation foundation, physics helpers, track selection, and coordinate transformation; see source, tests, and the historical experiment log. |
| M5 | Frozen event-1000 DBSCAN: unscaled XYZ, `eps=0.01`, `min_samples=3`. Historical score `0.0268875696` is not independently verified against a recoverable per-hit prediction artifact. |
| M6 | Diagnostic finding: 99/100 sampled tracks had an internal gap larger than DBSCAN `eps`; recorded as evidence of structural fragmentation. |
| M7 | Inventory of 100 complete events and deterministic whole-event split manifest. |
| M8.1 | Frozen M5 evaluation on 60 development events; mean recorded official score `0.1343528684`. Event-1000 score discrepancy remains unresolved. |
| M8.2 | Reproducible event-1000 run: score `0.1362513453`, 11,544 predicted clusters, 61,996 noise hits. This is not a held-out result and does not reconcile the historical M5 score. |
| M9.1–M9.2 | Geometry-only candidate graph and missed-edge diagnosis on the development split. |
| M9.3 | Per-target-bin query cap ablation from 10 to 20; no meaningful aggregate recall improvement was observed. |
| M9.4 | Rank audit found 2,296,263 true edges beyond the extended per-bin top-20 query. |
| M9.5 | Geometry-neighborhood failure audit. Rank-above-20 misses had median absolute Δz 140 mm and median XYZ distance 172.5 mm, versus 7.2 mm and 43.6 mm for recovered edges. These are descriptive differences, not causal explanations. |

M9.1–M9.5 used development events only. The M9.5 report proposes an XY-only versus XYZ neighbor-distance comparison as a testable hypothesis; it has not established that XY-only ranking improves reconstruction.

## 5. Dataset and split protocol

The data used is Kaggle's `trackml-particle-identification` competition `train_sample` subset, covering complete events 1000–1099. Each event includes hits, cells, particles, and truth. The dataset is not included in this repository. See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) for access notes and the unresolved license check.

The fixed manifest is [`experiments/results/event_split_manifest.csv`](experiments/results/event_split_manifest.csv):

- Development: events 1000–1059 (60 events)
- Validation: events 1060–1079 (20 events)
- Final test: events 1080–1099 (20 events)

The events are assigned as whole units, with no shuffle or hit-level split. Do not use validation or final-test results to tune a development method. The final-test set has not been evaluated in the recorded M8/M9 experiments.

## 6. Reproducibility and environment setup

The dependency ranges are in [`requirements.txt`](requirements.txt). The project uses the official TrackML utility library from `LAL/trackml-library`; a same-named package on PyPI is unrelated. From a Python environment with the declared dependencies installed:

```powershell
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/LAL/trackml-library.git"
```

Place extracted event CSVs in a local directory and pass that directory to a runner. Do not place raw events in Git. The latest saved M9.5 run records Python 3.13.5, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, and Windows 11. This is a recorded run environment, not a claim that a clean install has been verified on every platform. `docs/environment_verified.txt` is an older Linux environment snapshot.

The experiment code records fixed split membership and deterministic output ordering. M9.5 checks its rebuilt event-1000 candidate artifact and per-event scalar counts against saved M9.1 results and hashes protected M9.1/M9.4 artifacts before and after the run.

## 7. Tests

Tests use pytest and are located in `tests/`:

```powershell
python -m pytest -q
```

The current M9.5 result artifact records 6 focused tests and 156 tests in the full suite as passing. Those saved outcomes were produced in the recorded experiment environment. The current publication-preparation environment did not have SciPy installed, so those tests were not rerun as part of this documentation update.

## 8. Available experiment commands

Run commands from the repository root, after installing dependencies and making the local dataset available. Long M9 runners process all 60 development events and can take a substantial time.

```powershell
# Validate the existing split manifest
python -m scripts.event_split_manifest validate

# Reproduce M8.1 development baseline (writes its default result files)
python -m scripts.run_m8_development_baseline "<path-to-train_100_events>"

# Reproduce M8.2 event-1000 provenance point
python scripts/run_m8_2_event1000_provenance.py "<path-to-train_100_events>"

# M9 development-only geometry diagnostics; the manifest defaults to the saved manifest
python scripts/run_m9_1_geometry_graph.py "<path-to-train_100_events>"
python scripts/run_m9_2_diagnostics.py "<path-to-train_100_events>"
python scripts/run_m9_3_per_bin_query_ablation.py "<path-to-train_100_events>"
python scripts/run_m9_4_rank_audit.py "<path-to-train_100_events>"
python scripts/run_m9_5_geometry_miss_audit.py "<path-to-train_100_events>"
```

The M9.5 runner refuses to overwrite existing outputs unless passed `--overwrite`; use that option only when intentionally regenerating M9.5 artifacts. These commands do not evaluate validation or final-test events.

## 9. Evaluation methodology

M8 reports the official TrackML event score via the official scoring library, using truth only after clustering. M9 reports candidate-edge coverage: a true edge is a consecutive pair of hits with the same nonzero particle ID, ordered by transverse radius and directed outward. M9's pooled recall is recovered true edges divided by all true consecutive edges across the 60 development events. This candidate-edge recall is not a full track-reconstruction score.

## 10. Results and limitations

- M8.1's recorded 60-event mean official score is `0.1343528684`.
- M8.2 event-1000's reproducible official score is `0.1362513453`; its cluster and noise counts match the historical M5 record, but its score differs.
- M9.1 recovered 363,034 of 5,007,067 true consecutive edges: pooled candidate-edge recall `7.250432%` (about 7.25%). This is not a TrackML official score.
- M9.3's larger per-bin query did not meaningfully improve aggregate recall.
- M9.4/M9.5 classified 2,296,263 edges as beyond the diagnostic per-bin top-20 query. M9.5's medians show larger longitudinal offsets and XYZ distances in that cohort than among recovered edges.

The historical M5 event-1000 score (`0.0268875696`) and M8.2 event-1000 score (`0.1362513453`) remain unreconciled because the historical per-hit prediction artifact is unavailable. Candidate-edge recall is low and measures a different stage from complete track scoring. The geometry graph does not use detector topology or learned ranking. Query cutoff behavior depends on the recorded SciPy implementation for tied neighbor distances.

## 11. Repository structure

```text
configs/                 Dataset exploration configuration
docs/                    Data access, validation, and project documentation
experiments/              Milestone plans, logs, result tables, and figures
notebooks/                Dataset exploration notebook
scripts/                  Reproducible experiment runners and utilities
src/                      Data, physics, clustering, evaluation, and diagnostics
tests/                    Unit and focused milestone tests
```

Raw data and the local official-library checkout are excluded. Large generated per-hit files are also locally retained but ignored by Git; small result tables, manifests, reports, and figures remain eligible for review.

## 12. Data access and licensing notes

Obtain `train_sample.zip` through the Kaggle TrackML competition after reviewing the current competition rules. The exact reuse, redistribution, and publication terms remain **TO VERIFY** in this project. Do not upload the raw archive or extracted event files to this repository. Do not treat the previously mentioned Zenodo DOI or CC-BY-4.0 claim as valid; the project has recorded that DOI as incorrect. See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md).

## 13. Future work

The next proposed controlled experiment is an XY-only versus XYZ neighborhood metric comparison on the development split, holding the other graph settings fixed. It must be evaluated before any claim of improvement. Later work should freeze a method and protocol before accessing validation data, and reserve final-test events for a single held-out evaluation. No M9.6 result exists yet.

## 14. Citation and contact

Follow the citation and data-use requirements stated by the Kaggle competition and cite the TrackML challenge methodology using the verified bibliographic details in the project report. Those exact requirements are still being checked; no dataset license or permission to redistribute is asserted here. Add a maintainer contact or formal citation record before publication if required.

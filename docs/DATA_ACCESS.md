# Data Access — TrackML Particle Tracking Challenge

## Dataset used

- **Competition:** Kaggle TrackML Particle Identification, Accuracy phase.
- **Archive:** `train_sample.zip`.
- **Project subset:** 100 complete events, `event000001000` through `event000001099`.
- **Event parts:** hits, cells, particles, and truth.
- **Access:** obtain the archive through the Kaggle competition after reviewing its current rules.
- **License/reuse terms:** **TO VERIFY**. This repository does not claim a license or redistribution right.

The raw archive and extracted event CSV files are not included in this repository. Keep them in a local data directory and pass that path to scripts that require data. Do not commit the raw archive or event files.

The M7.1 inventory recorded 100/100 events complete. Aggregate table row counts were 10,967,467 hits, 60,159,209 cells, 1,079,146 particles, and 10,967,467 truth rows. The machine-readable per-event inventory is `experiments/results/dataset_inventory.csv`.

## Data provenance correction

An earlier project draft gave Zenodo DOI `10.5281/zenodo.4730167` and claimed CC-BY-4.0. Those claims were not correct for this dataset and must not be cited. The project uses Kaggle's TrackML `train_sample.zip`; its current competition terms still need direct verification before public data redistribution or publication claims are made.

## Event split

The whole-event split is fixed in `experiments/results/event_split_manifest.csv`:

- DEVELOPMENT: events 1000–1059 (60)
- VALIDATION: events 1060–1079 (20)
- FINAL_TEST: events 1080–1099 (20)

M8 and M9 experiments recorded to date used the development events. Truth and particle labels are reserved for post-hoc evaluation and diagnostics; they are not inputs to candidate generation or clustering.

## Loading and setup

The code uses the official `LAL/trackml-library` utility. Do not install the unrelated same-named PyPI package. From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/LAL/trackml-library.git"
```

Example data directory layout (local only):

```text
<data-dir>/
├── event000001000-hits.csv
├── event000001000-cells.csv
├── event000001000-particles.csv
├── event000001000-truth.csv
└── ...
```

Pass `<data-dir>` as the positional argument to the selected experiment runner. `src/data_loader.py` requests complete events and fails if requested parts are missing.

## Citation and license status

Check the current Kaggle competition rules for required attribution, reuse, and redistribution terms before sharing derived or raw data. Exact rules have not yet been recorded here. The previous draft listed the TrackML challenge papers by Amrouche et al. and Rousseau et al.; verify their full bibliographic details against primary sources before using them in a final paper. No unverified DOI or license statement should be copied into publication materials.

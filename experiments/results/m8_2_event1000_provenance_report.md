# M8.2 Event-1000 Provenance Report

## 1. Objective

Create a reproducible provenance record for the current frozen M5 implementation on event 1000. The historical M5 score discrepancy is not being repaired or retuned.

## 2. Configuration

- Event: `event000001000`
- Complete event, no hit subsampling
- Representation: unscaled XYZ
- DBSCAN: `eps=0.01`, `min_samples=3`, Euclidean metric
- Truth and particle data are excluded from feature construction and clustering.

## 3. Input event

- Hit rows: 120939
- Unique hit IDs: 120939
- Input hit-ID SHA-256: `97aa8ebb7a873339f7fac9a2bb71e6f1c741410ca50f1e3e7ea24f0bd00439e1`

## 4. Feature construction

The existing M4 `transform_hits` and M5 `extract_features(..., representation="xyz")` functions were reused. Feature order: `x2, y2, z2`. Dtype: `float64`. Shape: `[120939, 3]`. Feature matrix SHA-256: `3796a60b262fd7207475ad9e97e23d195d1e04a344fd3b893220753046e1ce49`. The exact matrix is saved as `m8_2_event1000_features.npy`.

## 5. DBSCAN configuration

The existing `src.clustering.cluster_tracks` was called with `eps=0.01` and `min_samples=3`. Only the feature matrix entered clustering. Truth was loaded after both clustering runs.

## 6. Clustering output

- Non-noise clusters: 11544
- Noise hits (raw DBSCAN label `-1`): 61996
- Raw label rows: 120939
- Cluster-size distribution (non-noise): min 3, median 4.0, mean 5.105942, max 49
- Cluster-label artifact: `m8_2_event1000_cluster_labels.csv`

## 7. Submission construction

The existing `labels_to_submission` function generated `hit_id,track_id`. The CSV was saved, read back from disk, and that read-back dataframe was passed to the official scorer. Its row count is 120939 with 120939 unique hit IDs.

## 8. Official score

- Official TrackML score: **0.13625134527683258**
- Scorer: `trackml.score.score_event`
- Truth rows: 120939
- Truth unique hit IDs: 120939
- Scorer truth-input SHA-256: `13f077030ddaef0522c874e8ee5875ac2b1a75086779fd599cb09299844945e6`
- Saved submission file SHA-256: `21991d5e4085b0b2615a17f5d0462fbd7bcd4c45f9f48be5c80c58b5ff8e4283`

## 9. Artifact inventory

- `m8_2_event1000_cluster_labels.csv`
- `m8_2_event1000_submission.csv`
- `m8_2_event1000_features.npy`
- `m8_2_event1000_cluster_mapping_sha256.txt`
- `m8_2_event1000_provenance.json`
- `m8_2_event1000_determinism.md`
- `m8_2_event1000_provenance_report.md`

## 10. SHA-256 fingerprints

- Raw label array: `bcc35e90f970aabef5c3ac91acdd96c3c4cf5f0c1359cf9b87e51f3f077c1167`
- Canonical hit-to-cluster mapping: `1f226df86951923f741d9b22d21b0f66dbaa634818ab662cf7ec355776616ad8`
- Feature matrix: `3796a60b262fd7207475ad9e97e23d195d1e04a344fd3b893220753046e1ce49`
- Input hit IDs: `97aa8ebb7a873339f7fac9a2bb71e6f1c741410ca50f1e3e7ea24f0bd00439e1`
- Saved submission file: `21991d5e4085b0b2615a17f5d0462fbd7bcd4c45f9f48be5c80c58b5ff8e4283`
- Scorer truth input: `13f077030ddaef0522c874e8ee5875ac2b1a75086779fd599cb09299844945e6`

## 11. Environment

- Python: `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]`
- NumPy: `2.5.3`
- pandas: `3.0.6`
- SciPy: `1.18.1`
- scikit-learn: `1.8.0`
- TrackML package/source: `{"distribution_versions": {"trackml": "3"}, "module_path": "<local-venv>/Lib/site-packages/trackml/__init__.py", "source_revision": null}`
- Platform: `Windows-11-10.0.26200-SP0`
- Repository commit: `None`
- Runner timestamp (UTC): `2026-10-08T03:23:38.376320+00:00`

## 12. Determinism check

See `m8_2_event1000_determinism.md`. Two complete runs had identical canonical cluster mappings, canonical submission mappings, scores, and input hashes. Raw DBSCAN labels were identical. The first-run label, submission, and feature artifacts were preserved. Their original timestamp was not captured because the first invocation stopped at a dtype-sensitive comparison; `first_run_timestamp_utc` is therefore null. The subsequent type-normalized comparison verified the saved mappings.

## 13. Relationship to historical M5

- Historical M5 recorded score = `0.026887569576501846`.
- Current reproducible event-1000 score = `0.13625134527683258`.
- Historical per-hit mapping is unavailable, so direct mapping-level comparison is impossible.
- The discrepancy remains unresolved.
- M8.2 does NOT invalidate M5.
- M8.2 does NOT establish that M8.1 or M5 is wrong.

## 14. Scientific conclusion

This run establishes a new reproducible provenance point for the current implementation only. It does not reproduce or explain the historical M5 score.

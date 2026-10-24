"""TrackML-style evaluation of predicted hit-to-track clusters.

============================================================================
WHY THIS MODULE CALLS THE OFFICIAL trackml.score.score_event() DIRECTLY
============================================================================
The official trackml-library (github.com/LAL/trackml-library, installed
from source per docs/DATA_ACCESS.md) already implements the exact TrackML
weighted hit-matching score, verified against its own source code during
M2 and re-verified during M5: majority-particle matching per predicted
track, the 50%/50% dual-purity "good track" threshold, and the weighted
sum over good tracks' majority-matched hits.

Rather than re-implementing this (risking a subtly incorrect duplicate —
exactly the failure mode the course spec explicitly warns about), this
module calls `trackml.score.score_event()` directly. This was validated
during M5 against the exact hand-worked 8-hit toy example from Step 2
(Part H): the official function reproduced the expected score of 0.90
exactly (floating-point rounding aside). See tests/test_evaluation.py for
this as a formal regression test.

This module ADDS diagnostics the official library does not provide
out-of-the-box: purity/efficiency per predicted cluster, fragmentation
(one truth particle split across multiple predicted clusters) and merging
(multiple truth particles combined into one predicted cluster) detection.
These are original to this project, clearly distinguished from the
official score itself.
============================================================================
TRUTH USAGE BOUNDARY
============================================================================
Every function in this module takes truth as an EVALUATION input. None of
these functions produce cluster ASSIGNMENTS — they only ever consume
already-produced cluster labels (from src.clustering) and truth, to
measure quality after the fact. This mirrors the module-level warning in
src/clustering.py.
============================================================================
"""
from __future__ import annotations

import pandas as pd
from trackml.score import score_event


def labels_to_submission(hit_ids: pd.Series, labels) -> pd.DataFrame:
    """Convert DBSCAN output (hit_ids + integer labels) into the
    'submission'-shaped DataFrame the official score_event expects.

    DBSCAN noise (label == -1) is remapped to a distinct, very large
    track_id per noise POINT (i.e. every noise hit becomes its own
    singleton "track") rather than all sharing track_id -1 — this matches
    the semantics the official scorer expects (ASSUMPTION: 0 <= track_id
    in trackml/score.py) and avoids accidentally treating all noise hits
    as one giant "track" that could spuriously become a majority-matched
    track for whichever truth particle contributes the most noise hits.

    Parameters
    ----------
    hit_ids : pandas.Series or array-like
        hit_id values, in the same order as `labels`.
    labels : array-like of int
        DBSCAN cluster labels (-1 = noise), same length/order as hit_ids.

    Returns
    -------
    pandas.DataFrame
        Columns: hit_id, track_id. track_id >= 0 for all rows (required
        by the official scorer's documented assumption).
    """
    import numpy as np

    hit_ids = pd.Series(hit_ids).reset_index(drop=True)
    labels = pd.Series(labels).reset_index(drop=True)

    # Real DBSCAN cluster labels are 0, 1, 2, ... — shift noise points to
    # singleton track_ids starting after the highest real cluster id, so
    # every noise hit is its own one-hit "track" rather than a shared
    # bucket. This deliberately produces many singleton "tracks" for
    # noise, which the official scorer will naturally score as not-good
    # (nhits=1 fails the >=4-hit implications of the weighting scheme;
    # more directly, a 1-hit predicted track essentially never reaches
    # >50% purity against a real multi-hit particle by chance).
    max_real_label = int(labels[labels != -1].max()) if (labels != -1).any() else -1
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    noise_track_ids = np.arange(max_real_label + 1, max_real_label + 1 + n_noise)

    track_id = labels.copy()
    track_id[noise_mask] = noise_track_ids

    return pd.DataFrame({"hit_id": hit_ids, "track_id": track_id.astype(np.int64)})


def official_trackml_score(truth: pd.DataFrame, hit_ids, labels) -> float:
    """Compute the OFFICIAL TrackML weighted hit-matching score for a set
    of predicted cluster labels, by calling trackml.score.score_event
    directly (see module docstring — not reimplemented here).

    Parameters
    ----------
    truth : pandas.DataFrame
        Must have hit_id, particle_id, weight columns (as returned by
        src.data_loader.load_event(..., parts=("truth",))). This is the
        ONLY place truth enters this module's clustering-adjacent code —
        strictly for scoring, after cluster labels already exist.
    hit_ids : array-like
        hit_id values corresponding to `labels`, same order.
    labels : array-like of int
        Predicted cluster labels (e.g. from src.clustering.cluster_tracks).

    Returns
    -------
    float
        The official TrackML score for this event's predicted clustering,
        in [0, 1] (see Step 2, Part H for the metric's defined range).
    """
    submission = labels_to_submission(hit_ids, labels)
    return float(score_event(truth[["hit_id", "particle_id", "weight"]], submission))


def cluster_purity_and_efficiency(
    truth: pd.DataFrame, hit_ids, labels
) -> pd.DataFrame:
    """Compute per-predicted-cluster purity and per-truth-particle
    efficiency, plus fragmentation/merging flags.

    Definitions (original to this project, distinguished from the
    official score above):

    - PURITY of a predicted cluster: (# hits belonging to that cluster's
      majority truth particle) / (total # hits in that predicted
      cluster). How "clean" is this cluster?
    - EFFICIENCY of a truth particle (a.k.a. completeness): (# of that
      particle's hits that ended up in ITS majority-matching predicted
      cluster) / (total true # hits that particle has, per truth.csv).
      How much of the real particle did we recover into one cluster?
    - FRAGMENTATION: a truth particle whose hits are the majority
      contributor to MORE THAN ONE predicted cluster (the true track got
      split across multiple DBSCAN clusters).
    - MERGING: a predicted cluster whose hits come from many different
      truth particles with no single dominant majority (i.e. purity is
      low) — multiple true tracks collapsed into one DBSCAN cluster.

    IMPORTANT — real-data finding (M5): a predicted cluster's "majority
    particle" can be `particle_id == 0`, which per the documented
    convention (Step 2, Part C) means NOISE / not from a reconstructible
    particle — not a real physical particle at all. Such a cluster
    reflects DBSCAN grouping several noise hits together, not a
    recovered/fragmented real track. This function EXCLUDES
    `majority_particle_id == 0` from the `is_fragment_of_split_particle`
    flag and from any "particles recovered" style count derived from this
    table's `majority_particle_id` column (see
    `summarize_evaluation`, which was fixed alongside this docstring
    after this was discovered against real event 1000: 895 of 11,544
    predicted clusters in one baseline run had majority_particle_id==0,
    which had been silently inflating the naive "recovered particles"
    count before this fix). The row itself is NOT dropped from the
    table (so purity/efficiency for noise-dominated clusters remain
    inspectable), only excluded from these particular derived counts.

    DBSCAN noise (label == -1) hits are EXCLUDED from this table (they are
    not "clusters" in any meaningful sense) but are reported separately by
    the caller via labels==-1 counts — see src.clustering.cluster_tracks
    output, which already reports n_noise.

    Parameters
    ----------
    truth : pandas.DataFrame
        hit_id, particle_id, weight (weight unused here, only for the
        official score above).
    hit_ids : array-like
    labels : array-like of int

    Returns
    -------
    pandas.DataFrame
        One row per predicted (non-noise) cluster:
        cluster_label, n_hits, majority_particle_id,
        majority_particle_true_nhits, majority_hits_in_cluster, purity,
        efficiency, is_fragment_of_split_particle (always False for rows
        where majority_particle_id == 0 — see note above)
    """
    df = pd.DataFrame({"hit_id": hit_ids, "label": labels}).merge(
        truth[["hit_id", "particle_id"]], on="hit_id", how="left"
    )
    df = df.loc[df["label"] != -1]  # exclude noise from cluster-level table

    true_nhits_per_particle = truth["particle_id"].value_counts()

    rows = []
    for cluster_label, group in df.groupby("label"):
        majority_particle_id = group["particle_id"].value_counts().idxmax()
        majority_hits_in_cluster = int(
            (group["particle_id"] == majority_particle_id).sum()
        )
        n_hits = len(group)
        purity = majority_hits_in_cluster / n_hits

        majority_particle_true_nhits = int(
            true_nhits_per_particle.get(majority_particle_id, 0)
        )
        efficiency = (
            majority_hits_in_cluster / majority_particle_true_nhits
            if majority_particle_true_nhits > 0
            else float("nan")
        )

        rows.append({
            "cluster_label": cluster_label,
            "n_hits": n_hits,
            "majority_particle_id": majority_particle_id,
            "majority_particle_true_nhits": majority_particle_true_nhits,
            "majority_hits_in_cluster": majority_hits_in_cluster,
            "purity": purity,
            "efficiency": efficiency,
        })

    result = pd.DataFrame(rows)
    if len(result) == 0:
        return result

    # Fragmentation: does the same majority_particle_id appear as the
    # majority of MORE THAN ONE predicted cluster? EXCLUDES
    # majority_particle_id == 0 (noise — see docstring note above), since
    # several DBSCAN clusters legitimately dominated by noise hits is not
    # the same phenomenon as a real track being split.
    real_particle_mask = result["majority_particle_id"] != 0
    dup_counts = result.loc[real_particle_mask, "majority_particle_id"].value_counts()
    fragmented_particle_ids = set(dup_counts[dup_counts > 1].index)
    result["is_fragment_of_split_particle"] = (
        result["majority_particle_id"].isin(fragmented_particle_ids) & real_particle_mask
    )

    return result.sort_values("n_hits", ascending=False).reset_index(drop=True)


def summarize_evaluation(
    truth: pd.DataFrame,
    hit_ids,
    labels,
    cluster_table: pd.DataFrame | None = None,
) -> dict:
    """Produce a single-dict summary combining the official score with
    aggregate purity/efficiency/fragmentation/merging statistics.

    Parameters
    ----------
    truth, hit_ids, labels : as in the other functions above.
    cluster_table : pandas.DataFrame, optional
        Pre-computed output of `cluster_purity_and_efficiency`, to avoid
        recomputing it if the caller already has it.

    Returns
    -------
    dict
        official_trackml_score, n_predicted_clusters, n_noise_hits,
        mean_purity, median_purity, mean_efficiency, median_efficiency,
        n_fragmented_particles (REAL particles split across >1 cluster;
        majority_particle_id==0 [noise] excluded — see
        cluster_purity_and_efficiency docstring), n_low_purity_clusters
        (purity < 0.5 — candidate "merged" clusters, matching the official
        scorer's own 50% purity threshold for consistency),
        n_reconstructable_truth_particles_recovered (REAL truth particles,
        with >=1 hit, that appear as SOME predicted cluster's majority
        particle at all, regardless of quality; majority_particle_id==0
        excluded, since noise is not a "recovered particle" — this
        exclusion was added after being discovered as a real counting bug
        against event 1000, see experiments/EXPERIMENT_LOG.md M5 entry).
    """
    if cluster_table is None:
        cluster_table = cluster_purity_and_efficiency(truth, hit_ids, labels)

    n_noise = int((pd.Series(labels) == -1).sum())
    official_score = official_trackml_score(truth, hit_ids, labels)

    if len(cluster_table) == 0:
        return {
            "official_trackml_score": official_score,
            "n_predicted_clusters": 0,
            "n_noise_hits": n_noise,
            "mean_purity": float("nan"),
            "median_purity": float("nan"),
            "mean_efficiency": float("nan"),
            "median_efficiency": float("nan"),
            "n_fragmented_particles": 0,
            "n_low_purity_clusters": 0,
            "n_reconstructable_truth_particles_recovered": 0,
        }

    real_particle_rows = cluster_table.loc[cluster_table["majority_particle_id"] != 0]

    return {
        "official_trackml_score": official_score,
        "n_predicted_clusters": int(len(cluster_table)),
        "n_noise_hits": n_noise,
        "mean_purity": float(cluster_table["purity"].mean()),
        "median_purity": float(cluster_table["purity"].median()),
        "mean_efficiency": float(cluster_table["efficiency"].mean()),
        "median_efficiency": float(cluster_table["efficiency"].median()),
        "n_fragmented_particles": int(
            cluster_table.loc[cluster_table["is_fragment_of_split_particle"], "majority_particle_id"].nunique()
        ),
        "n_low_purity_clusters": int((cluster_table["purity"] < 0.5).sum()),
        "n_reconstructable_truth_particles_recovered": int(
            real_particle_rows["majority_particle_id"].nunique()
        ),
    }

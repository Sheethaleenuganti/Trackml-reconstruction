"""Tests for src/evaluation.py.

Includes a formal regression test that trackml.score.score_event (called
by our official_trackml_score) reproduces the exact hand-worked 8-hit
example from Step 2, Part H (expected score: 0.90) -- this is the
official evaluator validated against our own independently-derived
by-hand calculation, not just re-deriving the same code twice.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    labels_to_submission,
    official_trackml_score,
    cluster_purity_and_efficiency,
    summarize_evaluation,
)


class TestLabelsToSubmission:
    def test_noise_hits_become_singleton_tracks(self):
        hit_ids = [1, 2, 3, 4]
        labels = [0, 0, -1, -1]  # two real cluster-0 hits, two noise hits
        sub = labels_to_submission(hit_ids, labels)
        # both noise hits must get DIFFERENT track_ids (singletons), not
        # share a single "-1" or shared bucket track_id.
        noise_rows = sub[sub["hit_id"].isin([3, 4])]
        assert noise_rows["track_id"].nunique() == 2

    def test_all_track_ids_nonnegative(self):
        hit_ids = [1, 2, 3]
        labels = [0, -1, -1]
        sub = labels_to_submission(hit_ids, labels)
        assert (sub["track_id"] >= 0).all()

    def test_real_clusters_keep_their_label_as_track_id(self):
        hit_ids = [1, 2, 3, 4]
        labels = [0, 0, 1, 1]
        sub = labels_to_submission(hit_ids, labels)
        assert set(sub.loc[sub["hit_id"].isin([1, 2]), "track_id"]) == {0}
        assert set(sub.loc[sub["hit_id"].isin([3, 4]), "track_id"]) == {1}

    def test_hit_id_order_preserved(self):
        hit_ids = [5, 6, 7]
        labels = [0, 0, 0]
        sub = labels_to_submission(hit_ids, labels)
        assert list(sub["hit_id"]) == [5, 6, 7]


class TestOfficialTrackmlScore:
    def test_matches_step2_hand_worked_example(self):
        # Exactly the 8-hit toy example from Step 2, Part H. Expected
        # score, derived independently BY HAND in Step 2: 0.90.
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "particle_id": [101, 101, 101, 101, 202, 202, 202, 202],
            "weight": [0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.15, 0.10],
        })
        # Our submission: track 5 = {1,2,3} (all particle 101, purity=100%,
        # particle-purity=3/4=75%, both >50% -> GOOD, contributes 0.45);
        # track 9 = {4,5,6,7,8} (majority particle 202 with 4/5 hits,
        # purity=80%, particle-purity=4/4=100%, both >50% -> GOOD,
        # contributes 0.10+0.10+0.15+0.10=0.45). Total = 0.90.
        hit_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        labels = [5, 5, 5, 9, 9, 9, 9, 9]

        score = official_trackml_score(truth, hit_ids, labels)
        assert score == pytest.approx(0.90, abs=1e-9)

    def test_perfect_clustering_scores_close_to_one(self):
        # If every predicted cluster exactly matches a truth particle
        # (using all their hits with nonzero weight), score should be 1.0.
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4, 5, 6],
            "particle_id": [1, 1, 1, 2, 2, 2],
            "weight": [1 / 6] * 6,
        })
        hit_ids = [1, 2, 3, 4, 5, 6]
        labels = [0, 0, 0, 1, 1, 1]  # exact match to truth grouping
        score = official_trackml_score(truth, hit_ids, labels)
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_all_noise_scores_near_zero(self):
        # If every hit is DBSCAN noise (all singleton "tracks"), no
        # predicted track can reach the required hit-matching majority
        # against a multi-hit truth particle in any meaningful way -- for
        # particles with >1 true hit, a singleton predicted track has
        # purity_rec=1.0 (perfect within itself) but the particle
        # (majority) purity will be 1/true_nhits, likely <=50% unless the
        # truth particle itself only has 1 hit. This test uses truth
        # particles with 2 hits each, so singleton predictions cannot
        # reach the required >50% majority-particle purity.
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4],
            "particle_id": [1, 1, 2, 2],
            "weight": [0.25] * 4,
        })
        hit_ids = [1, 2, 3, 4]
        labels = [-1, -1, -1, -1]  # all noise -> all singleton tracks
        score = official_trackml_score(truth, hit_ids, labels)
        assert score == pytest.approx(0.0, abs=1e-9)


class TestClusterPurityAndEfficiency:
    def test_perfect_cluster_has_purity_and_efficiency_one(self):
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3],
            "particle_id": [1, 1, 1],
            "weight": [1 / 3] * 3,
        })
        hit_ids = [1, 2, 3]
        labels = [0, 0, 0]
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)
        assert len(table) == 1
        assert table.iloc[0]["purity"] == pytest.approx(1.0)
        assert table.iloc[0]["efficiency"] == pytest.approx(1.0)

    def test_impure_cluster_purity_below_one(self):
        # Cluster 0 contains 3 hits from particle 1 and 1 hit from particle 2
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4],
            "particle_id": [1, 1, 1, 2],
            "weight": [0.25] * 4,
        })
        hit_ids = [1, 2, 3, 4]
        labels = [0, 0, 0, 0]
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)
        assert table.iloc[0]["purity"] == pytest.approx(0.75)  # 3/4
        assert table.iloc[0]["majority_particle_id"] == 1

    def test_incomplete_recovery_efficiency_below_one(self):
        # particle 1 truly has 4 hits total, but only 2 end up in this
        # predicted cluster (efficiency = 2/4 = 0.5)
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4, 5],
            "particle_id": [1, 1, 1, 1, 2],
            "weight": [0.2] * 5,
        })
        hit_ids = [1, 2, 5]  # only 2 of particle 1's 4 hits, plus 1 unrelated
        labels = [0, 0, 0]
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)
        row = table.iloc[0]
        assert row["majority_particle_id"] == 1
        assert row["majority_particle_true_nhits"] == 4
        assert row["majority_hits_in_cluster"] == 2
        assert row["efficiency"] == pytest.approx(0.5)

    def test_noise_excluded_from_cluster_table(self):
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3],
            "particle_id": [1, 1, 2],
            "weight": [1 / 3] * 3,
        })
        hit_ids = [1, 2, 3]
        labels = [0, 0, -1]  # hit 3 is noise
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)
        assert len(table) == 1  # only cluster 0, noise excluded
        assert -1 not in table["cluster_label"].values

    def test_fragmentation_detected(self):
        # particle 1's hits are split as majority across TWO predicted
        # clusters (0 and 1) -- this should be flagged as fragmented.
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4],
            "particle_id": [1, 1, 1, 1],
            "weight": [0.25] * 4,
        })
        hit_ids = [1, 2, 3, 4]
        labels = [0, 0, 1, 1]  # particle 1 split across cluster 0 and cluster 1
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)
        assert len(table) == 2
        assert table["is_fragment_of_split_particle"].all()

    def test_no_fragmentation_when_particles_are_cleanly_separated(self):
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4],
            "particle_id": [1, 1, 2, 2],
            "weight": [0.25] * 4,
        })
        hit_ids = [1, 2, 3, 4]
        labels = [0, 0, 1, 1]
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)
        assert not table["is_fragment_of_split_particle"].any()

    def test_noise_majority_particle_not_counted_as_fragmentation(self):
        # REGRESSION TEST for a real bug found against real data (event
        # 1000, M5 DBSCAN sweep): several predicted clusters can each be
        # dominated by noise hits (particle_id == 0, per the documented
        # convention that 0 means "not from a reconstructible particle").
        # Before the fix, this caused particle_id==0 to appear as the
        # "most fragmented particle" with hundreds of clusters, which is
        # meaningless (noise is not a real particle that can be
        # "fragmented"). This test hand-constructs 3 separate noise-
        # dominated clusters sharing majority_particle_id==0, plus one
        # genuinely fragmented REAL particle (id=7, split across 2
        # clusters), and checks that only the real particle is flagged.
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "particle_id": [0, 0, 0, 0, 0, 0, 7, 7, 7, 7],
            "weight": [0.1] * 10,
        })
        hit_ids = list(range(1, 11))
        # 3 noise-dominated clusters (0,1,2), particle 7 split across clusters 3,4
        labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)

        noise_rows = table.loc[table["majority_particle_id"] == 0]
        assert not noise_rows["is_fragment_of_split_particle"].any(), (
            "Noise-dominated clusters must never be flagged as fragmentation"
        )
        real_rows = table.loc[table["majority_particle_id"] == 7]
        assert real_rows["is_fragment_of_split_particle"].all(), (
            "A genuinely fragmented REAL particle must still be flagged"
        )

    def test_empty_when_all_noise(self):
        truth = pd.DataFrame({
            "hit_id": [1, 2],
            "particle_id": [1, 1],
            "weight": [0.5, 0.5],
        })
        hit_ids = [1, 2]
        labels = [-1, -1]
        table = cluster_purity_and_efficiency(truth, hit_ids, labels)
        assert len(table) == 0


class TestSummarizeEvaluation:
    def test_returns_expected_keys(self):
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3],
            "particle_id": [1, 1, 2],
            "weight": [1 / 3] * 3,
        })
        hit_ids = [1, 2, 3]
        labels = [0, 0, -1]
        summary = summarize_evaluation(truth, hit_ids, labels)
        expected_keys = {
            "official_trackml_score", "n_predicted_clusters", "n_noise_hits",
            "mean_purity", "median_purity", "mean_efficiency", "median_efficiency",
            "n_fragmented_particles", "n_low_purity_clusters",
            "n_reconstructable_truth_particles_recovered",
        }
        assert expected_keys.issubset(summary.keys())

    def test_noise_count_correct(self):
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4],
            "particle_id": [1, 1, 2, 2],
            "weight": [0.25] * 4,
        })
        hit_ids = [1, 2, 3, 4]
        labels = [0, 0, -1, -1]
        summary = summarize_evaluation(truth, hit_ids, labels)
        assert summary["n_noise_hits"] == 2

    def test_handles_completely_empty_clustering_gracefully(self):
        # Degenerate case: every hit is noise. Must not crash, and
        # cluster-dependent stats should be NaN/0 rather than raising.
        truth = pd.DataFrame({
            "hit_id": [1, 2],
            "particle_id": [1, 1],
            "weight": [0.5, 0.5],
        })
        hit_ids = [1, 2]
        labels = [-1, -1]
        summary = summarize_evaluation(truth, hit_ids, labels)
        assert summary["n_predicted_clusters"] == 0
        assert summary["n_noise_hits"] == 2

    def test_noise_dominated_cluster_excluded_from_recovered_particle_count(self):
        # REGRESSION TEST (same real bug as
        # TestClusterPurityAndEfficiency.test_noise_majority_particle_not_counted_as_fragmentation,
        # checked here at the summarize_evaluation level): a predicted
        # cluster whose majority particle is noise (particle_id==0) must
        # not inflate n_reconstructable_truth_particles_recovered.
        truth = pd.DataFrame({
            "hit_id": [1, 2, 3, 4],
            "particle_id": [0, 0, 5, 5],
            "weight": [0.25] * 4,
        })
        hit_ids = [1, 2, 3, 4]
        labels = [0, 0, 1, 1]  # cluster 0 = noise-dominated, cluster 1 = real particle 5
        summary = summarize_evaluation(truth, hit_ids, labels)
        assert summary["n_predicted_clusters"] == 2  # both clusters still counted as clusters
        assert summary["n_reconstructable_truth_particles_recovered"] == 1  # only particle 5, not particle 0

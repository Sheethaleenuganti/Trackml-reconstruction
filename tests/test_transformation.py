"""Tests for src/transformation.py — the official TrackML starting-kit
coordinate transformation, verified against Amrouche et al. 2019
(arXiv:1904.06778, Section 2.7), re-checked during M4.

    r1 = sqrt(x^2+y^2+z^2), x2=x/r1, y2=y/r1, r2=sqrt(x^2+y^2), z2=z/r2

All expected values below are computed BY HAND (shown in comments), not
just re-derived from the same code being tested.
"""
import numpy as np
import pandas as pd
import pytest

from src.transformation import transform_hits, transform_diagnostics


def _hits_df(rows):
    """rows: list of (hit_id, x, y, z) tuples."""
    return pd.DataFrame(rows, columns=["hit_id", "x", "y", "z"])


class TestKnownCoordinateToExpectedValue:
    def test_case_3_4_0(self):
        # x=3, y=4, z=0
        # r1 = sqrt(3^2+4^2+0^2) = sqrt(25) = 5
        # x2 = 3/5 = 0.6 ; y2 = 4/5 = 0.8
        # r2 = sqrt(3^2+4^2) = 5
        # z2 = 0/5 = 0.0
        hits = _hits_df([(1, 3.0, 4.0, 0.0)])
        result = transform_hits(hits)
        row = result.iloc[0]
        assert row["r1"] == pytest.approx(5.0)
        assert row["x2"] == pytest.approx(0.6)
        assert row["y2"] == pytest.approx(0.8)
        assert row["r2"] == pytest.approx(5.0)
        assert row["z2"] == pytest.approx(0.0)

    def test_case_0_3_4(self):
        # x=0, y=3, z=4
        # r1 = sqrt(0+9+16) = sqrt(25) = 5
        # x2 = 0/5 = 0.0 ; y2 = 3/5 = 0.6
        # r2 = sqrt(0+9) = 3
        # z2 = 4/3
        hits = _hits_df([(2, 0.0, 3.0, 4.0)])
        result = transform_hits(hits)
        row = result.iloc[0]
        assert row["r1"] == pytest.approx(5.0)
        assert row["x2"] == pytest.approx(0.0)
        assert row["y2"] == pytest.approx(0.6)
        assert row["r2"] == pytest.approx(3.0)
        assert row["z2"] == pytest.approx(4.0 / 3.0)

    def test_case_6_8_10(self):
        # x=6, y=8, z=10
        # r1 = sqrt(36+64+100) = sqrt(200) = 14.142135...
        # r2 = sqrt(36+64) = sqrt(100) = 10
        # x2 = 6/14.142135... ; y2 = 8/14.142135... ; z2 = 10/10 = 1.0
        hits = _hits_df([(3, 6.0, 8.0, 10.0)])
        result = transform_hits(hits)
        row = result.iloc[0]
        r1_expected = np.sqrt(200)
        assert row["r1"] == pytest.approx(r1_expected)
        assert row["r2"] == pytest.approx(10.0)
        assert row["x2"] == pytest.approx(6.0 / r1_expected)
        assert row["y2"] == pytest.approx(8.0 / r1_expected)
        assert row["z2"] == pytest.approx(1.0)


class TestZeroAndSpecialCases:
    def test_origin_produces_nan_not_error(self):
        # x=y=z=0: r1=0, r2=0 -> x2,y2,z2 all undefined (0/0) -> NaN,
        # not an exception and not silently 0 or inf.
        hits = _hits_df([(4, 0.0, 0.0, 0.0)])
        result = transform_hits(hits)  # must not raise
        row = result.iloc[0]
        assert row["r1"] == pytest.approx(0.0)
        assert row["r2"] == pytest.approx(0.0)
        assert np.isnan(row["x2"])
        assert np.isnan(row["y2"])
        assert np.isnan(row["z2"])

    def test_on_beam_axis_z_nonzero_produces_nan_z2_only(self):
        # x=0, y=0, z=100: r2=0 (on beam axis) but r1=100 (nonzero).
        # x2, y2 are well-defined (0/100=0), but z2=100/0 is undefined.
        hits = _hits_df([(5, 0.0, 0.0, 100.0)])
        result = transform_hits(hits)
        row = result.iloc[0]
        assert row["r1"] == pytest.approx(100.0)
        assert row["r2"] == pytest.approx(0.0)
        assert row["x2"] == pytest.approx(0.0)
        assert row["y2"] == pytest.approx(0.0)
        assert np.isnan(row["z2"]), "z2 must be NaN (not inf) when r2=0"

    def test_no_inf_values_produced_for_zero_cases(self):
        # Explicit check that we get NaN, never inf, for the r=0 cases
        # (a naive x/0 in numpy would silently produce inf with a warning
        # rather than an explicit, countable NaN).
        hits = _hits_df([(6, 0.0, 0.0, 0.0), (7, 0.0, 0.0, 50.0)])
        result = transform_hits(hits)
        for col in ["x2", "y2", "z2"]:
            assert not np.isinf(result[col].to_numpy(dtype=np.float64)).any()


class TestNegativeCoordinates:
    def test_negative_x_and_y(self):
        # x=-3, y=-4, z=0 -> r1=5 (distance is always positive)
        # x2 = -3/5 = -0.6 ; y2 = -4/5 = -0.8 (sign preserved)
        hits = _hits_df([(8, -3.0, -4.0, 0.0)])
        result = transform_hits(hits)
        row = result.iloc[0]
        assert row["r1"] == pytest.approx(5.0)
        assert row["x2"] == pytest.approx(-0.6)
        assert row["y2"] == pytest.approx(-0.8)

    def test_negative_z(self):
        # x=3, y=4, z=-10 -> r1 = sqrt(9+16+100) = sqrt(125)
        # r2 = sqrt(9+16) = 5 ; z2 = -10/5 = -2.0 (sign preserved)
        hits = _hits_df([(9, 3.0, 4.0, -10.0)])
        result = transform_hits(hits)
        row = result.iloc[0]
        assert row["r1"] == pytest.approx(np.sqrt(125))
        assert row["r2"] == pytest.approx(5.0)
        assert row["z2"] == pytest.approx(-2.0)


class TestMultipleHits:
    def test_multiple_hits_each_transformed_independently(self):
        hits = _hits_df([
            (1, 3.0, 4.0, 0.0),   # r1=5, x2=0.6, y2=0.8
            (2, 0.0, 3.0, 4.0),   # r1=5, r2=3, z2=4/3
            (3, 6.0, 8.0, 10.0),  # r1=sqrt(200)
        ])
        result = transform_hits(hits)
        assert len(result) == 3
        assert result.iloc[0]["x2"] == pytest.approx(0.6)
        assert result.iloc[1]["z2"] == pytest.approx(4.0 / 3.0)
        assert result.iloc[2]["r1"] == pytest.approx(np.sqrt(200))

    def test_row_order_preserved(self):
        hits = _hits_df([(10, 1.0, 0.0, 0.0), (20, 0.0, 1.0, 0.0), (30, 0.0, 0.0, 1.0)])
        result = transform_hits(hits)
        assert list(result["hit_id"]) == [10, 20, 30]


class TestHitIdPreservation:
    def test_hit_ids_preserved_exactly(self):
        hits = _hits_df([(101, 1.0, 2.0, 3.0), (202, 4.0, 5.0, 6.0)])
        result = transform_hits(hits)
        assert list(result["hit_id"]) == [101, 202]

    def test_does_not_mutate_input(self):
        hits = _hits_df([(1, 3.0, 4.0, 0.0)])
        original = hits.copy()
        _ = transform_hits(hits)
        pd.testing.assert_frame_equal(hits, original)

    def test_raw_coordinates_preserved_alongside_transformed(self):
        # The module docstring promises raw x,y,z are kept, not overwritten.
        hits = _hits_df([(1, 3.0, 4.0, 5.0)])
        result = transform_hits(hits)
        assert result.iloc[0]["x"] == pytest.approx(3.0)
        assert result.iloc[0]["y"] == pytest.approx(4.0)
        assert result.iloc[0]["z"] == pytest.approx(5.0)


class TestNoUnexpectedNaNOrInf:
    def test_normal_hits_produce_no_nan(self):
        # Hits away from the origin/beam-axis edge cases should never
        # produce NaN.
        rng = np.random.default_rng(0)
        n = 200
        # keep away from x=y=0 and x=y=z=0 by adding an offset
        x = rng.uniform(50, 500, n)
        y = rng.uniform(50, 500, n)
        z = rng.uniform(-1000, 1000, n)
        hits = pd.DataFrame({"hit_id": np.arange(n), "x": x, "y": y, "z": z})
        result = transform_hits(hits)
        for col in ["r1", "r2", "x2", "y2", "z2"]:
            assert result[col].isna().sum() == 0
            assert not np.isinf(result[col].to_numpy(dtype=np.float64)).any()


class TestInvalidInputHandling:
    def test_raises_on_missing_required_column(self):
        hits = pd.DataFrame({"hit_id": [1], "x": [1.0], "y": [2.0]})  # missing 'z'
        with pytest.raises(ValueError, match="missing"):
            transform_hits(hits)

    def test_raises_on_completely_wrong_input(self):
        hits = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError):
            transform_hits(hits)


class TestOutputShapeAndSchema:
    def test_output_has_expected_columns(self):
        hits = _hits_df([(1, 1.0, 1.0, 1.0)])
        result = transform_hits(hits)
        expected_cols = {"hit_id", "x", "y", "z", "r1", "r2", "x2", "y2", "z2"}
        assert expected_cols.issubset(set(result.columns))

    def test_output_row_count_matches_input(self):
        hits = _hits_df([(i, float(i), float(i), float(i)) for i in range(1, 51)])
        result = transform_hits(hits)
        assert len(result) == len(hits) == 50


class TestTransformDiagnostics:
    def test_counts_nan_correctly(self):
        hits = _hits_df([
            (1, 3.0, 4.0, 0.0),   # clean
            (2, 0.0, 0.0, 0.0),   # origin -> x2,y2,z2 all NaN
            (3, 0.0, 0.0, 50.0),  # on beam axis -> z2 NaN only
        ])
        transformed = transform_hits(hits)
        diag = transform_diagnostics(transformed)
        assert diag["n_hits"] == 3
        assert diag["x2_n_nan"] == 1  # only hit 2
        assert diag["y2_n_nan"] == 1  # only hit 2
        assert diag["z2_n_nan"] == 2  # hits 2 and 3
        assert diag["x2_n_inf"] == 0
        assert diag["y2_n_inf"] == 0
        assert diag["z2_n_inf"] == 0

    def test_min_max_ignore_nan(self):
        hits = _hits_df([(1, 3.0, 4.0, 0.0), (2, 0.0, 0.0, 0.0)])
        transformed = transform_hits(hits)
        diag = transform_diagnostics(transformed)
        # x2 for hit 1 is 0.6, hit 2 is NaN -> min/max should both be 0.6
        assert diag["x2_min"] == pytest.approx(0.6)
        assert diag["x2_max"] == pytest.approx(0.6)

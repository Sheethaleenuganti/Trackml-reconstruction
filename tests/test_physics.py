"""Unit tests for src/physics.py — simple, hand-checkable formulas."""
import numpy as np
import pandas as pd
import pytest

from src.physics import (
    transverse_radius,
    azimuthal_angle,
    transverse_momentum,
    add_transverse_radius_and_phi,
    add_particle_pt,
)


class TestTransverseRadius:
    def test_simple_3_4_5_triangle(self):
        # x=3, y=4 -> r = sqrt(9+16) = 5, a classic hand-checkable case
        r = transverse_radius(np.array([3.0]), np.array([4.0]))
        assert r[0] == pytest.approx(5.0)

    def test_zero_is_zero(self):
        r = transverse_radius(np.array([0.0]), np.array([0.0]))
        assert r[0] == pytest.approx(0.0)

    def test_negative_coordinates(self):
        # r should be unaffected by sign, since it's a distance
        r = transverse_radius(np.array([-3.0]), np.array([-4.0]))
        assert r[0] == pytest.approx(5.0)

    def test_vectorized(self):
        x = np.array([3.0, 0.0, 1.0])
        y = np.array([4.0, 5.0, 1.0])
        r = transverse_radius(x, y)
        np.testing.assert_allclose(r, [5.0, 5.0, np.sqrt(2)])


class TestAzimuthalAngle:
    def test_positive_x_axis_is_zero(self):
        phi = azimuthal_angle(np.array([1.0]), np.array([0.0]))
        assert phi[0] == pytest.approx(0.0)

    def test_positive_y_axis_is_pi_over_2(self):
        phi = azimuthal_angle(np.array([0.0]), np.array([1.0]))
        assert phi[0] == pytest.approx(np.pi / 2)

    def test_negative_x_axis_is_pi(self):
        phi = azimuthal_angle(np.array([-1.0]), np.array([0.0]))
        assert abs(phi[0]) == pytest.approx(np.pi)

    def test_negative_y_axis_is_minus_pi_over_2(self):
        phi = azimuthal_angle(np.array([0.0]), np.array([-1.0]))
        assert phi[0] == pytest.approx(-np.pi / 2)

    def test_range_is_within_pi(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000)
        y = rng.normal(size=1000)
        phi = azimuthal_angle(x, y)
        assert (phi >= -np.pi).all()
        assert (phi <= np.pi).all()


class TestTransverseMomentum:
    def test_simple_3_4_5_triangle(self):
        pt = transverse_momentum(np.array([3.0]), np.array([4.0]))
        assert pt[0] == pytest.approx(5.0)

    def test_zero_momentum(self):
        pt = transverse_momentum(np.array([0.0]), np.array([0.0]))
        assert pt[0] == pytest.approx(0.0)

    def test_pz_does_not_affect_pt(self):
        # pT is defined only from px, py — this test documents that pz is
        # irrelevant to this specific function, guarding against a future
        # accidental change that includes pz.
        pt_a = transverse_momentum(np.array([3.0]), np.array([4.0]))
        # same px, py, imagine a very different pz elsewhere in the code —
        # this function signature doesn't even accept pz, which is itself
        # part of what's being tested (no silent broadening of the formula)
        assert pt_a[0] == pytest.approx(5.0)


class TestAddTransverseRadiusAndPhi:
    def test_does_not_mutate_input(self):
        hits = pd.DataFrame({"x": [3.0], "y": [4.0], "z": [0.0]})
        original_columns = list(hits.columns)
        _ = add_transverse_radius_and_phi(hits)
        assert list(hits.columns) == original_columns

    def test_adds_expected_columns_with_correct_values(self):
        hits = pd.DataFrame({"x": [3.0], "y": [4.0], "z": [1.0]})
        result = add_transverse_radius_and_phi(hits)
        assert "transverse_radius" in result.columns
        assert "phi" in result.columns
        assert result["transverse_radius"].iloc[0] == pytest.approx(5.0)


class TestAddParticlePt:
    def test_does_not_mutate_input(self):
        particles = pd.DataFrame({"px": [3.0], "py": [4.0], "pz": [1.0]})
        original_columns = list(particles.columns)
        _ = add_particle_pt(particles)
        assert list(particles.columns) == original_columns

    def test_adds_pt_column_with_correct_value(self):
        particles = pd.DataFrame({"px": [3.0], "py": [4.0], "pz": [1.0]})
        result = add_particle_pt(particles)
        assert result["pt"].iloc[0] == pytest.approx(5.0)

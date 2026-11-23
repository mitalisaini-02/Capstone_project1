"""
Unit tests for kinematic calculations, 2D vector extrapolation, and boundary crossing scenarios.
"""
import unittest
import math
import numpy as np

from src.features.kinematics import (
    euclidean_distance,
    estimate_velocity_vector,
    relative_displacement,
    relative_velocity,
    relative_speed,
    radial_closing_speed
)
from src.evaluation.baselines import KinematicExtrapolationBaseline


class TestKinematics(unittest.TestCase):

    def test_euclidean_distance(self):
        p1 = (0.0, 0.0)
        p2 = (300.0, 400.0)
        self.assertAlmostEqual(euclidean_distance(p1, p2), 500.0)

    def test_estimate_velocity_vector(self):
        p_prev = (100.0, 200.0)
        p_curr = (110.0, 200.0)
        dt = 1.0
        vx, vy = estimate_velocity_vector(p_curr, p_prev, dt)
        self.assertAlmostEqual(vx, 10.0)
        self.assertAlmostEqual(vy, 0.0)

    # -------------------------------------------------------------
    # Scenario 1: Two vehicles moving apart
    # -------------------------------------------------------------
    def test_scenario_moving_apart(self):
        pos_p = (600.0, 0.0)
        pos_c = (0.0, 0.0)
        vel_p = (10.0, 0.0)   # Moving East away from consumer
        vel_c = (-10.0, 0.0)  # Moving West away from provider

        # Distance is 600m. Rate of separation is +20 m/s. Closing rate is -20 m/s.
        closing = radial_closing_speed(pos_p, pos_c, vel_p, vel_c)
        self.assertAlmostEqual(closing, -20.0)

        # Baseline projection over 30s: dx increases from 600m to 600 + (20 * 30) = 1200m
        # Dummy sequence tensor of shape (1, 2, 7) where features 4, 5 are dx, dy
        # Step t-1: dx=580, dy=0. Step t: dx=600, dy=0 -> dvx = 20, dvy = 0
        X = np.zeros((1, 2, 7))
        X[0, 0, 4] = 580.0
        X[0, 1, 4] = 600.0
        baseline = KinematicExtrapolationBaseline(comm_range_m=1000.0, horizon_s=30.0)
        d_proj = baseline.compute_projected_distance(X)[0]
        self.assertAlmostEqual(d_proj, 1200.0)
        # Should predict link break (0) since 1200m > 1000m
        pred = baseline.predict(X)[0]
        self.assertEqual(pred, 0)

    # -------------------------------------------------------------
    # Scenario 2: Two vehicles approaching
    # -------------------------------------------------------------
    def test_scenario_approaching(self):
        pos_p = (800.0, 0.0)
        pos_c = (0.0, 0.0)
        vel_p = (-10.0, 0.0)  # Moving West towards consumer
        vel_c = (10.0, 0.0)   # Moving East towards provider

        # Distance is 800m. Closing rate is +20 m/s.
        closing = radial_closing_speed(pos_p, pos_c, vel_p, vel_c)
        self.assertAlmostEqual(closing, 20.0)

        # Baseline projection over 30s: dx decreases from 800m to 800 - 600 = 200m
        X = np.zeros((1, 2, 7))
        X[0, 0, 4] = 820.0
        X[0, 1, 4] = 800.0
        baseline = KinematicExtrapolationBaseline(comm_range_m=1000.0, horizon_s=30.0)
        d_proj = baseline.compute_projected_distance(X)[0]
        self.assertAlmostEqual(d_proj, 200.0)
        pred = baseline.predict(X)[0]
        self.assertEqual(pred, 1)  # Stays within range

    # -------------------------------------------------------------
    # Scenario 3: Two vehicles moving at constant separation (parallel)
    # -------------------------------------------------------------
    def test_scenario_constant_separation(self):
        pos_p = (400.0, 300.0)  # dist = 500m
        pos_c = (0.0, 0.0)
        vel_p = (15.0, 15.0)    # Both moving parallel with identical velocity
        vel_c = (15.0, 15.0)

        closing = radial_closing_speed(pos_p, pos_c, vel_p, vel_c)
        self.assertAlmostEqual(closing, 0.0)

        X = np.zeros((1, 2, 7))
        X[0, 0, 4] = 400.0
        X[0, 0, 5] = 300.0
        X[0, 1, 4] = 400.0
        X[0, 1, 5] = 300.0
        baseline = KinematicExtrapolationBaseline(comm_range_m=1000.0, horizon_s=30.0)
        d_proj = baseline.compute_projected_distance(X)[0]
        self.assertAlmostEqual(d_proj, 500.0)
        pred = baseline.predict(X)[0]
        self.assertEqual(pred, 1)

    # -------------------------------------------------------------
    # Scenario 4: A vehicle pair that crosses the communication boundary
    # -------------------------------------------------------------
    def test_scenario_boundary_crossing(self):
        # Starts at 950m (inside range <= 1000m)
        # Moving apart at 5 m/s. After 30s: 950 + 150 = 1100m (crosses boundary out)
        X = np.zeros((1, 2, 7))
        X[0, 0, 4] = 945.0
        X[0, 1, 4] = 950.0
        baseline = KinematicExtrapolationBaseline(comm_range_m=1000.0, horizon_s=30.0)
        d_proj = baseline.compute_projected_distance(X)[0]
        self.assertAlmostEqual(d_proj, 1100.0)
        pred = baseline.predict(X)[0]
        self.assertEqual(pred, 0)  # Correctly identifies boundary crossing disconnect!


if __name__ == "__main__":
    unittest.main()

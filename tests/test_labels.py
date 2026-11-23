"""
Unit tests for link survival labeling ground truth and censoring rules.
"""
import unittest
from src.features.labels import compute_link_survival_label


class TestLabels(unittest.TestCase):

    def setUp(self):
        self.p_id = 1
        self.c_id = 31
        self.comm_range = 1000.0
        self.horizon = 30.0

    def test_surviving_link(self):
        # Both vehicles active from 0 to 100s
        lifecycles = {
            self.p_id: {"start_t": 0.0, "end_t": 100.0},
            self.c_id: {"start_t": 0.0, "end_t": 100.0}
        }
        # Fixed positions within 500m across all future seconds
        traj = {}
        for t in range(10, 45):
            traj[(float(t), self.p_id)] = (500.0, 0.0)
            traj[(float(t), self.c_id)] = (0.0, 0.0)

        label = compute_link_survival_label(
            self.p_id, self.c_id, current_time_s=10.0,
            future_horizon_s=self.horizon, comm_range_m=self.comm_range,
            lifecycles=lifecycles, trajectory_lookup=traj
        )
        self.assertEqual(label, 1)

    def test_breaking_link(self):
        lifecycles = {
            self.p_id: {"start_t": 0.0, "end_t": 100.0},
            self.c_id: {"start_t": 0.0, "end_t": 100.0}
        }
        # Starts at 500m, but at t=25 moves to 1200m
        traj = {}
        for t in range(10, 45):
            traj[(float(t), self.c_id)] = (0.0, 0.0)
            if t < 25:
                traj[(float(t), self.p_id)] = (500.0, 0.0)
            else:
                traj[(float(t), self.p_id)] = (1200.0, 0.0)  # out of range!

        label = compute_link_survival_label(
            self.p_id, self.c_id, current_time_s=10.0,
            future_horizon_s=self.horizon, comm_range_m=self.comm_range,
            lifecycles=lifecycles, trajectory_lookup=traj
        )
        self.assertEqual(label, 0)

    def test_vehicle_departure_censoring(self):
        # Provider departs at t=35s (only 25s after current_time=10s, less than 30s horizon)
        lifecycles = {
            self.p_id: {"start_t": 0.0, "end_t": 35.0},  # departs early
            self.c_id: {"start_t": 0.0, "end_t": 100.0}
        }
        traj = {}
        for t in range(10, 36):
            traj[(float(t), self.p_id)] = (500.0, 0.0)
            traj[(float(t), self.c_id)] = (0.0, 0.0)

        label = compute_link_survival_label(
            self.p_id, self.c_id, current_time_s=10.0,
            future_horizon_s=self.horizon, comm_range_m=self.comm_range,
            lifecycles=lifecycles, trajectory_lookup=traj
        )
        # Must be censored (None), not false positive or negative
        self.assertIsNone(label)

    def test_missing_intermediate_timestamp_censoring(self):
        lifecycles = {
            self.p_id: {"start_t": 0.0, "end_t": 100.0},
            self.c_id: {"start_t": 0.0, "end_t": 100.0}
        }
        traj = {}
        for t in range(10, 45):
            if t == 20:
                continue  # missing second 20
            traj[(float(t), self.p_id)] = (500.0, 0.0)
            traj[(float(t), self.c_id)] = (0.0, 0.0)

        label = compute_link_survival_label(
            self.p_id, self.c_id, current_time_s=10.0,
            future_horizon_s=self.horizon, comm_range_m=self.comm_range,
            lifecycles=lifecycles, trajectory_lookup=traj
        )
        self.assertIsNone(label)


if __name__ == "__main__":
    unittest.main()

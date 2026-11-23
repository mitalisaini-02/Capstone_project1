"""
Unit tests for data validator: timestamps, missing data, and vehicle lifecycles.
"""
import unittest
import pandas as pd
from src.data.validator import MobilityValidator, MobilityValidationError


class TestValidator(unittest.TestCase):

    def test_valid_schema(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0],
            "vehicle_id": [0, 0],
            "role": ["provider", "provider"],
            "x": [100.0, 110.0],
            "y": [200.0, 200.0],
            "speed_mps": [10.0, 10.0]
        })
        # Should not raise
        MobilityValidator.validate_schema(df)
        bounds = MobilityValidator.validate_coordinates(df)
        self.assertEqual(bounds["x"], (100.0, 110.0))

    def test_missing_column(self):
        df = pd.DataFrame({
            "time_s": [1.0],
            "vehicle_id": [0]
        })
        with self.assertRaises(MobilityValidationError):
            MobilityValidator.validate_schema(df)

    def test_null_values(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0],
            "vehicle_id": [0, 0],
            "role": ["provider", "provider"],
            "x": [100.0, None],
            "y": [200.0, 200.0],
            "speed_mps": [10.0, 10.0]
        })
        with self.assertRaises(MobilityValidationError):
            MobilityValidator.validate_schema(df)

    def test_duplicate_timestamps(self):
        df = pd.DataFrame({
            "time_s": [1.0, 1.0],  # duplicate
            "vehicle_id": [0, 0],
            "role": ["provider", "provider"],
            "x": [100.0, 110.0],
            "y": [200.0, 200.0],
            "speed_mps": [10.0, 10.0]
        })
        with self.assertRaises(MobilityValidationError):
            MobilityValidator.validate_timestamps_and_lifecycles(df)

    def test_non_monotonic_timestamps(self):
        df = pd.DataFrame({
            "time_s": [2.0, 1.0],  # non-monotonic
            "vehicle_id": [0, 0],
            "role": ["provider", "provider"],
            "x": [100.0, 110.0],
            "y": [200.0, 200.0],
            "speed_mps": [10.0, 10.0]
        })
        with self.assertRaises(MobilityValidationError):
            MobilityValidator.validate_timestamps_and_lifecycles(df)

    def test_lifecycle_and_gaps(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0, 5.0],  # gap between 2.0 and 5.0
            "vehicle_id": [0, 0, 0],
            "role": ["provider", "provider", "provider"],
            "x": [100.0, 110.0, 140.0],
            "y": [200.0, 200.0, 200.0],
            "speed_mps": [10.0, 10.0, 10.0]
        })
        lifecycles = MobilityValidator.validate_timestamps_and_lifecycles(df, expected_step_s=1.0)
        self.assertEqual(lifecycles[0]["start_t"], 1.0)
        self.assertEqual(lifecycles[0]["end_t"], 5.0)
        self.assertTrue(lifecycles[0]["has_gaps"])


if __name__ == "__main__":
    unittest.main()

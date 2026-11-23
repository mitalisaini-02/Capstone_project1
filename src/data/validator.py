"""
Validation module for mobility datasets, timestamps, coordinates, and vehicle lifecycles.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


class MobilityValidationError(Exception):
    """Raised when mobility data violates structural or physical constraints."""
    pass


class MobilityValidator:
    """Validates mobility data structure, coordinates, timestamps, and vehicle lifecycles."""

    REQUIRED_COLUMNS = ["time_s", "vehicle_id", "role", "x", "y", "speed_mps"]

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> None:
        """Validates that all required columns are present with non-null values."""
        missing_cols = [c for c in MobilityValidator.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise MobilityValidationError(f"Missing required columns in mobility dataframe: {missing_cols}")

        null_counts = df[MobilityValidator.REQUIRED_COLUMNS].isnull().sum()
        if null_counts.any():
            raise MobilityValidationError(f"Found null values in columns:\n{null_counts[null_counts > 0]}")

    @staticmethod
    def validate_coordinates(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Validates Cartesian coordinates and speed bounds.
        Returns coordinate bounds dict: {'x': (min, max), 'y': (min, max), 'speed': (min, max)}
        """
        if (df["speed_mps"] < 0).any():
            raise MobilityValidationError("Negative speed values detected in mobility dataframe.")

        bounds = {
            "x": (float(df["x"].min()), float(df["x"].max())),
            "y": (float(df["y"].min()), float(df["y"].max())),
            "speed": (float(df["speed_mps"].min()), float(df["speed_mps"].max()))
        }
        return bounds

    @staticmethod
    def validate_timestamps_and_lifecycles(
        df: pd.DataFrame,
        expected_step_s: float = 1.0
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyzes per-vehicle timelines, checking for duplicate timestamps,
        temporal monotonicity, and departure/arrival lifecycles.
        Returns: {vehicle_id: {'start_t': float, 'end_t': float, 'lifespan_s': float, 'steps': int, 'has_gaps': bool}}
        """
        lifecycles = {}
        for vid, group in df.groupby("vehicle_id"):
            raw_times = group["time_s"].values

            # Check duplicates
            if len(raw_times) != len(np.unique(raw_times)):
                raise MobilityValidationError(f"Duplicate timestamps detected for vehicle {vid}.")

            # Check monotonicity in input stream
            diffs = np.diff(raw_times)
            if (diffs <= 0).any():
                raise MobilityValidationError(f"Non-monotonic timestamps detected for vehicle {vid}.")

            sorted_g = group.sort_values("time_s")
            times = sorted_g["time_s"].values

            has_gaps = bool((diffs > (expected_step_s + 1e-3)).any())
            start_t = float(times[0])
            end_t = float(times[-1])

            lifecycles[int(vid)] = {
                "start_t": start_t,
                "end_t": end_t,
                "lifespan_s": end_t - start_t,
                "steps": len(times),
                "has_gaps": has_gaps
            }
        return lifecycles

    @staticmethod
    def is_future_horizon_complete(
        lifecycles: Dict[int, Dict[str, Any]],
        vehicle_id: int,
        current_time_s: float,
        horizon_s: float
    ) -> bool:
        """
        Returns True if the vehicle remains active and observed for at least current_time_s + horizon_s.
        """
        if vehicle_id not in lifecycles:
            return False
        return lifecycles[vehicle_id]["end_t"] >= (current_time_s + horizon_s - 1e-4)

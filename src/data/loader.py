"""
Data loading and indexing module for mobility traces and UE roles.
"""
import os
from typing import Dict, Tuple, Optional
import pandas as pd

from src.data.validator import MobilityValidator, MobilityValidationError


def load_ue_roles(roles_filepath: str = "ue_roles.csv") -> Dict[int, str]:
    """Loads vehicle role assignments from ue_roles.csv."""
    if not os.path.exists(roles_filepath):
        raise FileNotFoundError(f"UE roles file '{roles_filepath}' not found.")
    df = pd.read_csv(roles_filepath)
    if "ue_id" not in df.columns or "role" not in df.columns:
        raise MobilityValidationError("ue_roles.csv must contain 'ue_id' and 'role' columns.")
    return dict(zip(df["ue_id"].astype(int), df["role"].astype(str)))


def load_mobility_data(
    mobility_filepath: str = "mobility.csv",
    roles_filepath: Optional[str] = "ue_roles.csv",
    max_time_s: float = 300.0
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, any]]]:
    """
    Loads, filters (time <= max_time_s), validates, and indexes mobility traces.
    Returns: (cleaned_dataframe, vehicle_lifecycles)
    """
    if not os.path.exists(mobility_filepath):
        raise FileNotFoundError(f"Mobility file '{mobility_filepath}' not found.")

    df = pd.read_csv(mobility_filepath)

    # Validate structure before filtering
    MobilityValidator.validate_schema(df)

    # Filter to simulation window
    df = df[df["time_s"] <= max_time_s].copy()
    df["vehicle_id"] = df["vehicle_id"].astype(int)

    # Validate coordinates and lifecycle bounds
    MobilityValidator.validate_coordinates(df)
    lifecycles = MobilityValidator.validate_timestamps_and_lifecycles(df)

    # Confirm roles match ue_roles.csv if provided
    if roles_filepath and os.path.exists(roles_filepath):
        expected_roles = load_ue_roles(roles_filepath)
        for vid in df["vehicle_id"].unique():
            if vid in expected_roles:
                actual_role = df[df["vehicle_id"] == vid]["role"].iloc[0]
                if actual_role != expected_roles[vid]:
                    raise MobilityValidationError(
                        f"Role mismatch for vehicle {vid}: actual '{actual_role}' != expected '{expected_roles[vid]}'"
                    )

    # Sort deterministically
    df.sort_values(by=["time_s", "vehicle_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, lifecycles

"""
Reusable SUMO mobility-trace loading (moved from src/datagen/mobility_source.py).

Loads one SUMO run's mobility trace and exposes roles and lifetimes taken
directly from the data. Never infers roles from vehicle ID ranges.
The raw-data directory is a parameter; DEFAULT_RAW_DIR is only a default.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.validator import MobilityValidator, MobilityValidationError

DEFAULT_RAW_DIR = "data/raw/sumo"
RAW_DIR = DEFAULT_RAW_DIR   # backward-compatible alias


@dataclass
class RunMobility:
    run_id: str
    df: pd.DataFrame                                  # time_s, vehicle_id, role, x, y, speed_mps
    roles: Dict[int, str]                             # vehicle_id -> 'provider' | 'consumer'
    start_t: Dict[int, float]
    end_t: Dict[int, float]
    times: np.ndarray                                 # sorted unique timestamps
    lookup: Dict[Tuple[float, int], Tuple[float, float, float]] = field(repr=False, default_factory=dict)

    @property
    def providers(self) -> List[int]:
        return sorted(v for v, r in self.roles.items() if r == "provider")

    @property
    def consumers(self) -> List[int]:
        return sorted(v for v, r in self.roles.items() if r == "consumer")

    def state(self, t: float, vid: int) -> Tuple[float, float, float]:
        """(x, y, speed_mps) of vehicle `vid` at time `t`, exactly as in the SUMO trace."""
        return self.lookup[(t, vid)]


def load_run(run_id: str, raw_dir: str = DEFAULT_RAW_DIR) -> RunMobility:
    path = os.path.join(raw_dir, run_id, "mobility.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mobility trace not found: {path}")
    df = pd.read_csv(path)
    MobilityValidator.validate_schema(df)
    MobilityValidator.validate_coordinates(df)
    df["vehicle_id"] = df["vehicle_id"].astype(int)

    role_counts = df.groupby("vehicle_id")["role"].nunique()
    if (role_counts > 1).any():
        bad = role_counts[role_counts > 1].index.tolist()
        raise MobilityValidationError(f"{run_id}: vehicles with more than one role: {bad}")
    if not set(df["role"].unique()) <= {"provider", "consumer"}:
        raise MobilityValidationError(f"{run_id}: unexpected roles {set(df['role'].unique())}")
    if (df["time_s"] % 1 != 0).any():
        raise MobilityValidationError(f"{run_id}: non-integer timestamps (expected 1 s steps)")

    lifecycles = MobilityValidator.validate_timestamps_and_lifecycles(df, expected_step_s=1.0)
    gaps = [v for v, lc in lifecycles.items() if isinstance(v, int) and lc["has_gaps"]]
    if gaps:
        raise MobilityValidationError(f"{run_id}: vehicles with timestamp gaps: {gaps}")

    df = df[["time_s", "vehicle_id", "role", "x", "y", "speed_mps"]].sort_values(
        ["time_s", "vehicle_id"]).reset_index(drop=True)
    roles = df.groupby("vehicle_id")["role"].first().to_dict()
    g = df.groupby("vehicle_id")["time_s"]
    rm = RunMobility(run_id=run_id, df=df, roles={int(k): v for k, v in roles.items()},
                     start_t={int(k): float(v) for k, v in g.min().items()},
                     end_t={int(k): float(v) for k, v in g.max().items()},
                     times=np.sort(df["time_s"].unique()))
    rm.lookup = {(t, v): (x, y, s) for t, v, x, y, s in
                 zip(df.time_s.values, df.vehicle_id.values, df.x.values, df.y.values, df.speed_mps.values)}
    return rm


def load_road_ids(run_id: str, raw_dir: str = DEFAULT_RAW_DIR) -> Dict[Tuple[float, int], str]:
    """(time_s, vehicle_id) -> SUMO road/edge id, read from the same trace. Additive helper;
    RunMobility is unchanged. Edge ids are supporting information only (not proof of LOS)."""
    path = os.path.join(raw_dir, run_id, "mobility.csv")
    df = pd.read_csv(path, usecols=["time_s", "vehicle_id", "road_id"])
    return {(float(t), int(v)): str(e) for t, v, e in zip(df.time_s.values, df.vehicle_id.values, df.road_id.values)}

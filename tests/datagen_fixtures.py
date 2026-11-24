"""Small synthetic SUMO-like traces for datagen unit tests (written to temp dirs only)."""
import os

import pandas as pd


def write_trace(root, run_id, rows):
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows, columns=["time_s", "vehicle_id", "role", "x", "y", "speed_mps", "road_id"]).to_csv(
        os.path.join(d, "mobility.csv"), index=False)


def two_vehicle_rows(t_end=80, p_id=40, c_id=3, p_end=None, c_end=None, speed=20.0):
    """Provider (id 40) fixed at origin; consumer (id 3) drives away at `speed` m/s.
    IDs deliberately violate the old '0-29 provider' assumption."""
    rows = []
    for t in range(1, t_end + 1):
        if p_end is None or t <= p_end:
            rows.append((float(t), p_id, "provider", 0.0, 0.0, 0.0, "e"))
        if c_end is None or t <= c_end:
            rows.append((float(t), c_id, "consumer", speed * t, 0.0, speed, "e"))
    return rows

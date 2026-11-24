"""
Supply/demand accounting used for calibration and dataset statistics.

These are simple capacity-accounting measures, NOT allocation results:
no optimisation, no LSTM, no one-PUE-per-CUE constraint is applied here.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.mobility.trace import RunMobility
from src.network.coverage import nearest_site


def active_request_seconds(cue_df: pd.DataFrame, run: RunMobility) -> pd.DataFrame:
    """One row per (time, request) while the request is active and the CUE is present."""
    rows = []
    for r in cue_df.itertuples(index=False):
        t_end = min(r.request_time_s + r.service_duration_s - 1, run.end_t[r.cue_id])
        for t in np.arange(r.request_time_s, t_end + 1.0):
            rows.append((float(t), r.request_id, r.cue_id, r.required_rate_mbps))
    return pd.DataFrame(rows, columns=["time_s", "request_id", "cue_id", "rate_mbps"])


def serving_cell(run: RunMobility, active: pd.DataFrame, sites: List[Tuple[str, float, float]],
                 radius_m: float) -> pd.Series:
    """Nearest gNodeB within radius for each active row (None if uncovered).
    Thin adapter: positions come from the trace; coverage logic is in src.network.coverage."""
    if active.empty:
        return pd.Series([], dtype=object)
    xy = np.array([run.state(t, c)[:2] for t, c in zip(active.time_s.values, active.cue_id.values)])
    return pd.Series(nearest_site(xy[:, 0], xy[:, 1], sites, radius_m), index=active.index)


def gnb_demand_coverage(active: pd.DataFrame, cells: pd.Series, gnb_df: pd.DataFrame) -> Dict[str, float]:
    """Share of demanded Mbps·s that the gNodeB ALONE could carry (coverage + per-cell capacity)."""
    total = float(active.rate_mbps.sum())
    if total == 0:
        return {"demand_mbps_s": 0.0, "covered_share": float("nan"), "gnb_carry_share": float("nan")}
    a = active.assign(cell=cells.values)
    covered = a[a.cell.notna()]
    dem = covered.groupby(["time_s", "cell"]).rate_mbps.sum().rename("demand").reset_index()
    cap = gnb_df.rename(columns={"gnb_id": "cell"})[["time_s", "cell", "available_capacity_mbps"]]
    m = dem.merge(cap, on=["time_s", "cell"], how="left")
    carried = float(np.minimum(m.demand, m.available_capacity_mbps).sum())
    return {"demand_mbps_s": total, "covered_share": float(covered.rate_mbps.sum()) / total,
            "gnb_carry_share": carried / total}


def pue_supply_demand(active: pd.DataFrame, pue_df: pd.DataFrame, conn_df: pd.DataFrame) -> Dict[str, float]:
    """
    Ratio of potential PUE supply to demand. Supply at t = sum over AVAILABLE PUEs that are
    geographically in range of at least one active CUE of min(backhaul C5, aggregate access C4).
    Ignores C3, reliability, and the one-PUE-per-CUE constraint (an upper-bound style measure).
    """
    total = float(active.rate_mbps.sum())
    if total == 0:
        return {"supply_mbps_s": 0.0, "supply_demand_ratio": float("nan")}
    act_cues = active[["time_s", "cue_id"]].drop_duplicates()
    cand = conn_df.merge(act_cues, on=["time_s", "cue_id"])[["time_s", "pue_id"]].drop_duplicates()
    avail = pue_df[pue_df.available][["time_s", "pue_id", "backhaul_capacity_mbps", "access_capacity_mbps"]]
    s = cand.merge(avail, on=["time_s", "pue_id"])
    supply = float(np.minimum(s.backhaul_capacity_mbps, s.access_capacity_mbps).sum())
    return {"supply_mbps_s": supply, "supply_demand_ratio": supply / total}

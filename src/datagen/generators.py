"""
Synthetic PUE, CUE and gNodeB generators (SYNTHETIC simulation data).

All positions/speeds/timestamps/roles/lifetimes are copied from the SUMO run.
Only marketplace attributes are synthetic. Each entity draws a FIXED set of
uniforms/normals (independent of parameters) that are then transformed, so
configurations share common random numbers (see scenarios.py).
"""
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.mobility.trace import RunMobility
from src.network.coverage import load_gnb_sites as _coverage_load_gnb_sites
from src.common.units import MBIT_PER_GB  # single source of truth (re-exported for compatibility)
from src.datagen.scenarios import entity_rng

MAX_REQUESTS_PER_CUE = 64   # fixed draw budget per consumer (never reached in practice; checked)

PUE_COLUMNS = ["scenario_id", "run_id", "time_s", "pue_id", "x_m", "y_m", "speed_mps",
               "registered", "available", "backhaul_capacity_mbps", "access_capacity_mbps",
               "link_peak_mbps", "price_per_gb"]
WITHDRAWAL_COLUMNS = ["scenario_id", "pue_id", "withdraw_time_s", "reason"]
CUE_COLUMNS = ["scenario_id", "run_id", "request_id", "cue_id", "request_time_s", "x_m", "y_m",
               "qos_class", "required_rate_mbps", "min_rate_mbps", "service_duration_s",
               "data_volume_gb", "max_price_per_gb", "max_budget", "priority_weight"]
GNB_COLUMNS = ["scenario_id", "time_s", "gnb_id", "total_capacity_mbps", "background_load_mbps",
               "available_capacity_mbps", "price_per_gb"]
SITE_COLUMNS = ["gnb_id", "x_m", "y_m", "coverage_radius_m", "source"]



# --------------------------------------------------------------------------- PUE
def generate_pues(run: RunMobility, params: Dict, scenario_id: str, base_seed: int,
                  replicate: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P = params["pue"]
    gp = params["gnb_price_per_gb"]
    rows, wd, static = [], [], []
    for vid in run.providers:
        start, end = run.start_t[vid], run.end_t[vid]
        n = int(end - start) + 1
        rng = entity_rng(base_seed, run.run_id, replicate, "pue", vid)
        # ---- fixed draws (order and count never depend on parameters) ----
        u_reg, u_delay, u_bh, u_acc, u_peak, u_price = rng.random(6)
        eps_bh = rng.standard_normal(n)
        u_state = rng.random(n)
        u_wd = rng.random(n)

        lo, hi = P["registration_delay_s"]
        delay = int(lo + math.floor(u_delay * (hi - lo + 1)))
        reg_time = start + delay
        registers = (u_reg < P["p_register"]) and (reg_time <= end)

        base_bh = P["backhaul_base_mbps"][0] + u_bh * (P["backhaul_base_mbps"][1] - P["backhaul_base_mbps"][0])
        access = P["access_capacity_mbps"][0] + u_acc * (P["access_capacity_mbps"][1] - P["access_capacity_mbps"][0])
        peak = P["link_peak_mbps"][0] + u_peak * (P["link_peak_mbps"][1] - P["link_peak_mbps"][0])
        mlo, mhi = P["price_multiplier"]
        price = gp * (mlo + u_price * (mhi - mlo))

        rho, sig = P["backhaul_ar1_rho"], P["backhaul_log_sigma"]
        z = np.empty(n)
        z[0] = sig * eps_bh[0]
        for k in range(1, n):
            z[k] = rho * z[k - 1] + math.sqrt(1 - rho ** 2) * sig * eps_bh[k]
        clo, chi = P["backhaul_multiplier_clip"]
        bh = base_bh * np.clip(np.exp(z), clo, chi)

        p_off, p_on = 1.0 / P["mean_on_s"], 1.0 / P["mean_off_s"]
        withdrawn_at = None
        on = True
        for k in range(n):
            t = start + k
            registered = registers and t >= reg_time and withdrawn_at is None
            if registered and t > reg_time:
                if P["withdrawal_hazard_per_s"] > 0 and u_wd[k] < P["withdrawal_hazard_per_s"]:
                    withdrawn_at = t
                    registered = False
                    wd.append((scenario_id, vid, float(t), "voluntary_withdrawal"))
                else:
                    on = (u_state[k] >= p_off) if on else (u_state[k] < p_on)
            available = registered and on
            x, y, s = run.state(t, vid)
            rows.append((scenario_id, run.run_id, float(t), vid, x, y, s, registered, available,
                         float(bh[k]) if registered else np.nan,
                         float(access) if registered else np.nan,
                         float(peak) if registered else np.nan,
                         float(price) if registered else np.nan))
        static.append({"pue_id": vid, "registers": bool(registers), "registration_time_s": float(reg_time),
                       "base_backhaul_mbps": base_bh, "access_capacity_mbps": access,
                       "link_peak_mbps": peak, "price_per_gb": price,
                       "withdrawn_at_s": withdrawn_at})
    pue_df = pd.DataFrame(rows, columns=PUE_COLUMNS)
    wd_df = pd.DataFrame(wd, columns=WITHDRAWAL_COLUMNS)
    return pue_df, wd_df, pd.DataFrame(static)


# --------------------------------------------------------------------------- CUE
def generate_cues(run: RunMobility, params: Dict, scenario_id: str, base_seed: int,
                  replicate: int) -> pd.DataFrame:
    C = params["cue"]
    gp = params["gnb_price_per_gb"]
    classes = list(C["qos_classes"].keys())
    mix = np.array([C["class_mix"].get(c, 0.0) for c in classes], dtype=float)
    if abs(mix.sum() - 1.0) > 1e-9:
        raise ValueError(f"class_mix must sum to 1 (got {mix.sum()})")
    cum = np.cumsum(mix)
    lam = C["arrival_rate_per_s"]
    rows = []
    for vid in run.consumers:
        start, end = run.start_t[vid], run.end_t[vid]
        rng = entity_rng(base_seed, run.run_id, replicate, "cue", vid)
        u_active = rng.random()
        draws = rng.random((MAX_REQUESTS_PER_CUE, 5))   # gap, class, rate, duration, price
        if u_active >= C["p_ever_requests"]:
            continue
        t_free = start      # earliest time a new request may arrive (no overlapping requests)
        for k in range(MAX_REQUESTS_PER_CUE):
            u_gap, u_cls, u_rate, u_dur, u_price = draws[k]
            gap = math.ceil(-math.log(1.0 - u_gap) / lam)       # exponential idle time (s)
            t_req = t_free + gap
            if t_req > end:
                break
            if k == MAX_REQUESTS_PER_CUE - 1:
                raise RuntimeError("MAX_REQUESTS_PER_CUE reached; increase the draw budget")
            cls = classes[int(np.searchsorted(cum, u_cls, side="right"))]
            spec = C["qos_classes"][cls]
            rate = round(spec["rate_mbps"][0] + u_rate * (spec["rate_mbps"][1] - spec["rate_mbps"][0]), 3)
            dlo, dhi = spec["duration_s"]
            dur = int(dlo + math.floor(u_dur * (dhi - dlo + 1)))
            volume = rate * dur / MBIT_PER_GB
            plo, phi = C["max_price_multiplier"]
            max_price = round(gp * (plo + u_price * (phi - plo)), 6)
            x, y, _ = run.state(float(t_req), vid)
            rows.append((scenario_id, run.run_id, f"{run.run_id}-c{vid}-{k:03d}", vid, float(t_req), x, y,
                         cls, rate, rate, dur, volume, max_price, max_price * volume,
                         C["priority_weight"]))
            t_free = t_req + dur     # nominal end; open-loop arrivals (independent of outcomes)
    return pd.DataFrame(rows, columns=CUE_COLUMNS)


# --------------------------------------------------------------------------- gNodeB
def load_gnb_sites(xml_path: str) -> List[Tuple[str, float, float]]:
    """Backward-compatible wrapper; coverage logic lives in src.network.coverage."""
    return _coverage_load_gnb_sites(xml_path)


def generate_gnbs(run: RunMobility, params: Dict, scenario_id: str, base_seed: int, replicate: int,
                  sites: List[Tuple[str, float, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    G = params["gnb"]
    times = run.times
    n = len(times)
    rows = []
    for idx, (gid, _, _) in enumerate(sites):
        rng = entity_rng(base_seed, run.run_id, replicate, "gnb", idx)
        eps = rng.standard_normal(n)
        rho, sig = G["background_ar1_rho"], G["background_log_sigma"]
        z = np.empty(n)
        z[0] = sig * eps[0]
        for k in range(1, n):
            z[k] = rho * z[k - 1] + math.sqrt(1 - rho ** 2) * sig * eps[k]
        tot = G["total_capacity_mbps"]
        bg = np.clip(G["load_fraction"] * tot * np.exp(z), 0.0, G["background_max_fraction"] * tot)
        for k, t in enumerate(times):
            rows.append((scenario_id, float(t), gid, tot, float(bg[k]), float(tot - bg[k]),
                         params["gnb_price_per_gb"]))
    site_df = pd.DataFrame([(gid, x, y, G["coverage_radius_m"],
                             "infrastructure.add.xml (hand-placed POI, not a real deployment)")
                            for gid, x, y in sites], columns=SITE_COLUMNS)
    return pd.DataFrame(rows, columns=GNB_COLUMNS), site_df

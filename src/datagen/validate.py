"""
Validation of generated scenarios and derived connectivity. Any failed check is
reported with details; the generation script stops on failure.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from src.datagen.connectivity import FUTURE_ONLY_COLUMNS, MEC_VISIBLE_COLUMNS
from src.datagen.generators import (CUE_COLUMNS, GNB_COLUMNS, MBIT_PER_GB, PUE_COLUMNS,
                                    SITE_COLUMNS, WITHDRAWAL_COLUMNS)
from src.mobility.trace import RunMobility


class Checks:
    def __init__(self):
        self.items: List[Dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append({"check": name, "passed": bool(ok), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.items)

    def failures(self) -> List[Dict]:
        return [c for c in self.items if not c["passed"]]


def _sumo_frame(run: RunMobility) -> pd.DataFrame:
    return run.df.rename(columns={"vehicle_id": "vid", "x": "sx", "y": "sy", "speed_mps": "ss"})


def validate_connectivity(run: RunMobility, vis: pd.DataFrame, gt: pd.DataFrame,
                          comm_range_m: float, history_window_s: int) -> Checks:
    ck = Checks()
    ck.add("conn.columns", list(vis.columns) == MEC_VISIBLE_COLUMNS, str(list(vis.columns)))
    ck.add("conn.no_future_columns", not (set(vis.columns) & FUTURE_ONLY_COLUMNS),
           str(set(vis.columns) & FUTURE_ONLY_COLUMNS))
    ck.add("conn.roles", vis.pue_id.map(run.roles).eq("provider").all()
           and vis.cue_id.map(run.roles).eq("consumer").all(), "pue=provider, cue=consumer per SUMO")
    sf = _sumo_frame(run)
    m = vis.merge(sf[["time_s", "vid", "sx", "sy"]].rename(columns={"vid": "pue_id", "sx": "px", "sy": "py"}),
                  on=["time_s", "pue_id"], how="left").merge(
        sf[["time_s", "vid", "sx", "sy"]].rename(columns={"vid": "cue_id", "sx": "cx", "sy": "cy"}),
        on=["time_s", "cue_id"], how="left")
    ck.add("conn.both_present", m[["px", "cx"]].notna().all().all(), "both vehicles present in SUMO at t")
    d = np.hypot(m.px - m.cx, m.py - m.cy)
    ck.add("conn.distance_matches_sumo", np.allclose(d, m.distance_m, rtol=0, atol=1e-9), "")
    ck.add("conn.in_range", (vis.distance_m <= comm_range_m).all() and vis.geo_in_range.all(), "")
    ok_hist = (vis.pue_observed_s >= history_window_s) & (vis.cue_observed_s >= history_window_s)
    ck.add("conn.history_flag", (ok_hist == vis.history_ok).all(), "")
    ck.add("conn.gt_aligned", len(gt) == len(vis) and
           (gt[["time_s", "pue_id", "cue_id"]].values == vis[["time_s", "pue_id", "cue_id"]].values).all(), "")
    causes = {"out_of_range", "pue_departure", "cue_departure", "both_departure", "censored_trace_end"}
    ck.add("gt.cause_values", set(gt.first_break_cause.unique()) <= causes, str(gt.first_break_cause.unique()))
    # departure vs trace-end censoring are distinguished
    dep_p = gt.first_break_cause.eq("pue_departure")
    ck.add("gt.pue_departure_not_trace_end", (~gt.loc[dep_p, "pue_end_is_trace_end"]).all(), "")
    dep_c = gt.first_break_cause.eq("cue_departure")
    ck.add("gt.cue_departure_not_trace_end", (~gt.loc[dep_c, "cue_end_is_trace_end"]).all(), "")
    return ck


def validate_scenario(run: RunMobility, params: Dict, files: Dict[str, pd.DataFrame],
                      conn: pd.DataFrame) -> Checks:
    ck = Checks()
    pue, cue, gnb, sites, wd = (files["pue_resources"], files["cue_requests"], files["gnodeb_resources"],
                                files["gnodeb_sites"], files["pue_withdrawals"])
    for name, df, cols in (("pue_resources", pue, PUE_COLUMNS), ("cue_requests", cue, CUE_COLUMNS),
                           ("gnodeb_resources", gnb, GNB_COLUMNS), ("gnodeb_sites", sites, SITE_COLUMNS),
                           ("pue_withdrawals", wd, WITHDRAWAL_COLUMNS)):
        ck.add(f"{name}.columns", list(df.columns) == cols, str(list(df.columns)))
        ck.add(f"{name}.no_future_columns", not (set(df.columns) & FUTURE_ONLY_COLUMNS), "")

    sf = _sumo_frame(run)
    # ---- PUE ----
    ck.add("pue.roles_match_sumo", pue.pue_id.map(run.roles).eq("provider").all(), "")
    ck.add("pue.all_providers_present", set(pue.pue_id.unique()) == set(run.providers), "")
    m = pue.merge(sf, left_on=["time_s", "pue_id"], right_on=["time_s", "vid"], how="left")
    ck.add("pue.positions_match_sumo", m.sx.notna().all() and (m.x_m == m.sx).all()
           and (m.y_m == m.sy).all() and (m.speed_mps == m.ss).all(), "exact equality with SUMO trace")
    ck.add("pue.one_row_per_present_second", len(pue) == int(sf[sf.role == "provider"].shape[0])
           and not pue.duplicated(["time_s", "pue_id"]).any(), "")
    ck.add("pue.available_implies_registered", (~pue.available | pue.registered).all(), "")
    reg = pue[pue.registered]
    cap_cols = ["backhaul_capacity_mbps", "access_capacity_mbps", "link_peak_mbps", "price_per_gb"]
    ck.add("pue.registered_values_present", reg[cap_cols].notna().all().all(), "")
    ck.add("pue.unregistered_values_blank", pue[~pue.registered][cap_cols].isna().all().all(), "")
    P = params["pue"]
    bl, bh = P["backhaul_base_mbps"]
    clo, chi = P["backhaul_multiplier_clip"]
    ck.add("pue.backhaul_range", reg.backhaul_capacity_mbps.between(bl * clo - 1e-9, bh * chi + 1e-9).all(), "")
    ck.add("pue.access_range", reg.access_capacity_mbps.between(*P["access_capacity_mbps"]).all(), "")
    ck.add("pue.peak_range", reg.link_peak_mbps.between(*P["link_peak_mbps"]).all(), "")
    gp = params["gnb_price_per_gb"]
    plo, phi = P["price_multiplier"]
    ck.add("pue.price_range", reg.price_per_gb.between(gp * plo - 1e-12, gp * phi + 1e-12).all()
           and (reg.price_per_gb > 0).all(), "")
    ck.add("pue.static_per_vehicle", (reg.groupby("pue_id")[["access_capacity_mbps", "link_peak_mbps",
                                                          "price_per_gb"]].nunique() <= 1).all().all(), "")
    if len(wd):
        later = pue.merge(wd[["pue_id", "withdraw_time_s"]], on="pue_id")
        later = later[later.time_s >= later.withdraw_time_s]
        ck.add("pue.withdrawal_is_permanent", (~later.registered).all(), f"{len(wd)} withdrawals")
    else:
        ck.add("pue.withdrawal_is_permanent", True, "no withdrawals")
    if P["withdrawal_hazard_per_s"] == 0:
        ck.add("pue.no_withdrawals_when_disabled", len(wd) == 0, "")

    # ---- CUE ----
    if len(cue):
        ck.add("cue.roles_match_sumo", cue.cue_id.map(run.roles).eq("consumer").all(), "")
        m = cue.merge(sf, left_on=["request_time_s", "cue_id"], right_on=["time_s", "vid"], how="left")
        ck.add("cue.positions_match_sumo", m.sx.notna().all() and (m.x_m == m.sx).all()
               and (m.y_m == m.sy).all(), "")
        ck.add("cue.request_within_lifetime",
               cue.apply(lambda r: run.start_t[r.cue_id] <= r.request_time_s <= run.end_t[r.cue_id], axis=1).all(), "")
        ck.add("cue.unique_request_id", cue.request_id.is_unique, "")
        ov = False
        for _, g in cue.sort_values("request_time_s").groupby("cue_id"):
            ends = g.request_time_s.values + g.service_duration_s.values
            if (g.request_time_s.values[1:] < ends[:-1]).any():
                ov = True
        ck.add("cue.at_most_one_active_request", not ov, "")
        ck.add("cue.volume_consistent", np.allclose(cue.data_volume_gb,
               cue.required_rate_mbps * cue.service_duration_s / MBIT_PER_GB, rtol=1e-12, atol=0), "")
        ck.add("cue.budget_consistent", np.allclose(cue.max_budget, cue.max_price_per_gb * cue.data_volume_gb,
                                                    rtol=1e-12, atol=0), "max_budget = max_price_per_gb * volume")
        ck.add("cue.min_rate_equals_required", (cue.min_rate_mbps == cue.required_rate_mbps).all(), "v1: no elastic")
        ok_cls = True
        for cls, spec in params["cue"]["qos_classes"].items():
            g = cue[cue.qos_class == cls]
            ok_cls &= g.required_rate_mbps.between(spec["rate_mbps"][0], spec["rate_mbps"][1]).all()
            ok_cls &= g.service_duration_s.between(*spec["duration_s"]).all()
        ck.add("cue.class_ranges", bool(ok_cls), "")
        ck.add("cue.integer_times", (cue.request_time_s % 1 == 0).all(), "")
        mlo, mhi = params["cue"]["max_price_multiplier"]
        ck.add("cue.max_price_range", cue.max_price_per_gb.between(gp * mlo - 1e-9, gp * mhi + 1e-9).all(), "")
    else:
        ck.add("cue.non_empty", False, "no requests generated")

    # ---- gNodeB ----
    G = params["gnb"]
    ck.add("gnb.times_cover_run", set(gnb.time_s.unique()) == set(run.times.tolist()), "")
    ck.add("gnb.capacity_consistent", np.allclose(gnb.available_capacity_mbps,
           gnb.total_capacity_mbps - gnb.background_load_mbps) and (gnb.background_load_mbps >= 0).all()
           and (gnb.available_capacity_mbps > 0).all()
           and (gnb.available_capacity_mbps <= gnb.total_capacity_mbps).all(), "")
    ck.add("gnb.total_matches_params", (gnb.total_capacity_mbps == G["total_capacity_mbps"]).all(), "")
    ck.add("gnb.price_positive", (gnb.price_per_gb > 0).all(), "")
    ck.add("gnb.sites_radius", (sites.coverage_radius_m == G["coverage_radius_m"]).all()
           and (sites.coverage_radius_m > 0).all(), "")
    return ck

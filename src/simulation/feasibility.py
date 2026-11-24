"""
Per-scenario link feasibility (no pricing, no provider selection, no allocation).

For every second in which a CUE has an active request:
  * request_seconds: present PUEs, how many are in / out of geographic range, gNodeB coverage,
    serving cell and that cell's available capacity (not shared out: sharing is allocation).
  * candidate_links: every geographically in-range (PUE, requesting CUE) pair x channel scenario,
    with an independent flag per stage and the FIRST failing stage:
        provider_unavailable (not_registered | unavailable | withdrawn)
        -> insufficient_history (no LSTM window)
        -> reliability_below_threshold (p_survive < theta)
        -> wireless_outage_<cause> (REALIZED radio capacity = 0)
        -> resource_zero (min of C3, C4, C5 is not positive)
        -> feasible
    Geographic range failures are counted per request-second (out-of-range pairs are not rows).
    Provider/CUE departures end link episodes (see the per-run episode table).

Capacities (Mbps), all REALIZED unless named expected:
  realized_link_capacity_mbps = min(link_peak_i, realized_radio_capacity)   (C3)
  access_capacity_mbps (C4), backhaul_capacity_mbps (C5)
  pue_max_rate_mbps = min(C3, C4, C5): upper bound for this single link if it were the PUE's only
  link; joint sharing of C4/C5 across CUEs is decided by the (later) optimiser.
  expected_radio_capacity_mbps is carried for reporting only and is NEVER used below.
"""
import numpy as np
import pandas as pd

from src.network.coverage import nearest_site

STAGE_ORDER = ["provider_unavailable", "insufficient_history", "reliability_below_threshold",
               "wireless_outage", "resource_zero"]


def active_request_seconds(cue: pd.DataFrame, end_t) -> pd.DataFrame:
    rows = []
    for r in cue.itertuples(index=False):
        t_end = min(r.request_time_s + r.service_duration_s - 1, end_t[r.cue_id])
        for t in np.arange(r.request_time_s, t_end + 1.0):
            rows.append((float(t), r.request_id, int(r.cue_id), float(r.required_rate_mbps)))
    return pd.DataFrame(rows, columns=["time_s", "request_id", "cue_id", "required_rate_mbps"])


def provider_status(pue: pd.DataFrame, withdrawals: pd.DataFrame) -> pd.Series:
    wd = dict(zip(withdrawals.pue_id, withdrawals.withdraw_time_s)) if len(withdrawals) else {}
    w_t = pue["pue_id"].map(wd)
    withdrawn = w_t.notna() & (pue["time_s"] >= w_t)
    return pd.Series(np.select([pue.available.to_numpy(bool), withdrawn.to_numpy(bool), pue.registered.to_numpy(bool)],
                               ["available", "withdrawn", "unavailable"], default="not_registered"),
                     index=pue.index)


def wireless_cause(df: pd.DataFrame) -> np.ndarray:
    fb = df["fallback"].to_numpy(bool)
    st = df["realized_state"].to_numpy(object)
    blocked = df["building_blocked"].fillna(False).to_numpy(bool)
    geo = df["classification_source"].to_numpy(object) == "geometry"
    return np.select(
        [fb, (st == "NLOS") & blocked & geo, st == "NLOS", st == "NLOSv", st == "LOS"],
        ["wireless_outage_fallback", "wireless_outage_building_blocked", "wireless_outage_nlos_assumed",
         "wireless_outage_vehicle_blocked", "wireless_outage_los"], default="wireless_outage_unknown")


def first_failure(flags: dict, wireless_reason: np.ndarray) -> np.ndarray:
    out = np.array(["feasible"] * len(wireless_reason), dtype=object)
    for stage in reversed(STAGE_ORDER):   # earliest failing stage wins
        fail = ~flags[stage]
        out[fail] = wireless_reason[fail] if stage == "wireless_outage" else stage
    return out


def evaluate_scenario(scenario_id: str, run, files: dict, sites, channel: pd.DataFrame,
                      reliability: pd.DataFrame, conn: pd.DataFrame, thresholds: dict,
                      discovery_range_m: float):
    pue, cue, gnb, wd = files["pue_resources"], files["cue_requests"], files["gnodeb_resources"], files["pue_withdrawals"]
    radius = float(files["gnodeb_sites"]["coverage_radius_m"].iloc[0])
    act = active_request_seconds(cue, run.end_t)

    # ---- request_seconds: geographic range counts + gNodeB coverage ----
    present = pue.groupby("time_s").size().rename("n_present_pue")
    inr = conn[conn.distance_m <= discovery_range_m].groupby(["time_s", "cue_id"]).size().rename("n_in_range_pue")
    rs = act.merge(present, left_on="time_s", right_index=True, how="left")
    rs = rs.merge(inr, left_on=["time_s", "cue_id"], right_index=True, how="left")
    rs[["n_present_pue", "n_in_range_pue"]] = rs[["n_present_pue", "n_in_range_pue"]].fillna(0).astype(int)
    rs["n_out_of_range_pue"] = rs["n_present_pue"] - rs["n_in_range_pue"]
    xy = np.array([run.state(t, c)[:2] for t, c in zip(rs.time_s, rs.cue_id)]) if len(rs) else np.empty((0, 2))
    cells = nearest_site(xy[:, 0], xy[:, 1], sites, radius) if len(rs) else np.array([], dtype=object)
    rs["gnb_covered"] = pd.Series(cells).notna().to_numpy()
    rs["gnb_id"] = cells
    cap = gnb.set_index(["time_s", "gnb_id"])["available_capacity_mbps"]
    rs["gnb_available_capacity_mbps"] = [cap.get((t, g), np.nan) if g is not None else np.nan
                                         for t, g in zip(rs.time_s, rs.gnb_id)]
    rs.insert(0, "scenario_id", scenario_id)

    # ---- candidate links: in-range pairs with an active request ----
    keys = act[["time_s", "request_id", "cue_id", "required_rate_mbps"]]
    cl = channel.merge(keys, on=["time_s", "cue_id"], how="inner")
    cl = cl[cl.distance_m <= discovery_range_m]
    cl = cl.merge(reliability[["time_s", "pue_id", "cue_id", "lstm_window_available", "p_survive"]],
                  on=["time_s", "pue_id", "cue_id"], how="left")
    pstat = pue.assign(provider_status=provider_status(pue, wd))[
        ["time_s", "pue_id", "provider_status", "link_peak_mbps", "access_capacity_mbps", "backhaul_capacity_mbps"]]
    cl = cl.merge(pstat, on=["time_s", "pue_id"], how="left")

    c3 = np.minimum(cl["link_peak_mbps"].to_numpy(float), cl["realized_radio_capacity_mbps"].to_numpy(float))
    cl["realized_link_capacity_mbps"] = c3
    stack = np.vstack([c3, cl["access_capacity_mbps"].to_numpy(float), cl["backhaul_capacity_mbps"].to_numpy(float)])
    with np.errstate(invalid="ignore"):
        cl["pue_max_rate_mbps"] = np.min(stack, axis=0)
    names = np.array(["C3_link", "C4_access", "C5_backhaul"], dtype=object)
    finite = np.all(np.isfinite(stack), axis=0)
    cl["binding_constraint"] = np.where(finite, names[np.argmin(np.where(np.isfinite(stack), stack, np.inf), axis=0)], None)
    cl["covers_required_rate"] = finite & (cl["pue_max_rate_mbps"].to_numpy() >= cl["required_rate_mbps"].to_numpy())

    wireless = wireless_cause(cl)
    base = {"provider_unavailable": (cl["provider_status"] == "available").to_numpy(),
            "insufficient_history": cl["lstm_window_available"].fillna(False).to_numpy(bool),
            "wireless_outage": cl["realized_radio_capacity_mbps"].to_numpy(float) > 0.0,
            "resource_zero": finite & (np.nan_to_num(cl["pue_max_rate_mbps"].to_numpy(float)) > 0.0)}
    p = cl["p_survive"].to_numpy(float)
    for tag, theta in thresholds.items():
        with np.errstate(invalid="ignore"):
            rel = np.nan_to_num(p, nan=-1.0) >= theta
        cl[f"reliable_{tag}"] = rel
        flags = dict(base, reliability_below_threshold=rel)
        ok = np.logical_and.reduce([flags[s] for s in STAGE_ORDER])
        cl[f"feasible_{tag}"] = ok
        cl[f"first_failure_{tag}"] = first_failure(flags, wireless)
    cl["provider_ok"] = base["provider_unavailable"]
    cl["wireless_ok"] = base["wireless_outage"]
    cl["resource_ok"] = base["resource_zero"]
    cl.insert(0, "scenario_id", scenario_id)
    cols = ["scenario_id", "channel_scenario", "time_s", "request_id", "cue_id", "pue_id", "distance_m",
            "required_rate_mbps", "provider_status", "provider_ok", "lstm_window_available", "p_survive"] + \
           [f"reliable_{t}" for t in thresholds] + \
           ["classification_source", "fallback", "fallback_reason", "primary_eligible", "building_blocked",
            "n_buildings_crossed", "p_los", "p_nlosv", "p_nlos", "episode_id", "realized_state",
            "realized_snr_db", "realized_radio_capacity_mbps", "wireless_ok", "expected_radio_capacity_mbps",
            "expected_outage_probability", "link_peak_mbps", "realized_link_capacity_mbps",
            "access_capacity_mbps", "backhaul_capacity_mbps", "pue_max_rate_mbps", "binding_constraint",
            "resource_ok", "covers_required_rate"] + \
           [c for t in thresholds for c in (f"feasible_{t}", f"first_failure_{t}")]
    cl = cl[cols].sort_values(["channel_scenario", "time_s", "cue_id", "pue_id"]).reset_index(drop=True)
    return rs.reset_index(drop=True), cl

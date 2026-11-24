"""
Descriptive statistics for generated scenarios (moved from scripts/generate_marketplace_data.py).

Capacity-accounting descriptors only (no optimisation, no LSTM). The link model is passed in
through the common LinkModel interface, so statistics work with any wireless profile.
"""
import numpy as np
import pandas as pd

from src.datagen.metrics import (active_request_seconds, gnb_demand_coverage, pue_supply_demand,
                                 serving_cell)


def scenario_stats(run, params, files, conn, link_model, sites):
    pue, cue, gnb = files["pue_resources"], files["cue_requests"], files["gnodeb_resources"]
    act = active_request_seconds(cue, run)
    cells = serving_cell(run, act, sites, params["gnb"]["coverage_radius_m"])
    gcov = gnb_demand_coverage(act, cells, gnb)
    sdr = pue_supply_demand(act, pue, conn)
    conc = act.groupby("time_s").size().reindex(run.times, fill_value=0) if len(act) else pd.Series([0])
    # candidate rows: active CUE, in-range, AVAILABLE PUE (geographic + registration only)
    cand = conn.merge(act[["time_s", "cue_id"]].drop_duplicates(), on=["time_s", "cue_id"]).merge(
        pue[pue.available][["time_s", "pue_id", "link_peak_mbps"]], on=["time_s", "pue_id"])
    act_ids = act[["time_s", "cue_id"]].drop_duplicates()
    with_cand = cand[["time_s", "cue_id"]].drop_duplicates()
    with_hist = cand[cand.history_ok][["time_s", "cue_id"]].drop_duplicates()
    b = link_model.link_capacity_mbps(cand.distance_m.values, cand.link_peak_mbps.values) if len(cand) else np.array([])
    snr = link_model.snr_db(cand.distance_m.values) if len(cand) else np.array([])
    reg = pue[pue.registered].drop_duplicates("pue_id")
    return {
        "requests": int(len(cue)), "requesting_cues": int(cue.cue_id.nunique()),
        "consumers_in_run": len(run.consumers), "providers_in_run": len(run.providers),
        "requests_by_class": cue.qos_class.value_counts().to_dict(),
        "mean_required_rate_mbps": float(cue.required_rate_mbps.mean()) if len(cue) else None,
        "total_volume_gb": float(cue.data_volume_gb.sum()),
        "concurrent_requests_mean": float(conc.mean()), "concurrent_requests_max": int(conc.max()),
        "registered_pues": int(reg.shape[0]),
        "available_pue_seconds_share": float(pue.available.mean()),
        "withdrawals": int(len(files["pue_withdrawals"])),
        "pue_price_per_gb_mean": float(reg.price_per_gb.mean()) if len(reg) else None,
        "share_pue_cheaper_than_gnb": float((reg.price_per_gb < params["gnb_price_per_gb"]).mean()) if len(reg) else None,
        "share_requests_cannot_afford_gnb": float((cue.max_price_per_gb < params["gnb_price_per_gb"]).mean()) if len(cue) else None,
        "active_request_seconds": int(len(act_ids)),
        "share_active_seconds_with_geo_candidate": float(len(with_cand) / len(act_ids)) if len(act_ids) else None,
        "share_active_seconds_with_history_ok_candidate": float(len(with_hist) / len(act_ids)) if len(act_ids) else None,
        "gnb_covered_demand_share": gcov["covered_share"], "gnb_carry_share": gcov["gnb_carry_share"],
        "pue_supply_demand_ratio": sdr["supply_demand_ratio"],
        "candidate_rows": int(len(cand)),
        "candidate_link_capacity_mbps_pctl": ({p: float(np.percentile(b, p)) for p in (5, 25, 50, 75, 95)}
                                             if len(b) else None),
        "candidate_share_distance_gt_500m": float((cand.distance_m > 500).mean()) if len(cand) else None,
        "candidate_share_snr_below_reporting_floor": float((snr < link_model.reporting_snr_floor_db).mean())
        if len(snr) and link_model.reporting_snr_floor_db is not None else None,
    }


def aggregate_by_configuration(stats, manifest):
    df = pd.DataFrame([{**{"config_id": m["config_id"]}, **{k: v for k, v in stats[m["scenario_id"]].items()
                                                         if not isinstance(v, dict)}} for m in manifest])
    num = df.select_dtypes("number").columns
    agg = df.groupby("config_id")[list(num)].agg(["mean", "std"])
    out = {}
    for cid in agg.index:
        out[cid] = {c: {"mean": round(float(agg.loc[cid, (c, "mean")]), 4),
                        "std": round(float(agg.loc[cid, (c, "std")]), 4)} for c in num}
    return out

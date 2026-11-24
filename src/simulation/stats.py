"""Descriptive statistics for Phase 3 outputs (no tuning; reported per split)."""
import numpy as np
import pandas as pd


def link_stats(cl: pd.DataFrame, tags):
    out = {}
    for (scen), g in cl.groupby("channel_scenario"):
        prim = g[g.primary_eligible.astype(bool)] if scen == "G" else g
        d = {"candidate_link_seconds": int(len(g)),
             "primary_eligible_link_seconds": int(len(prim)),
             "fallback_link_seconds": int(g.fallback.astype(bool).sum()),
             "realized_state_share": prim.realized_state.value_counts(normalize=True).round(4).to_dict(),
             "realized_outage_share": round(float((~prim.wireless_ok).mean()), 4) if len(prim) else None,
             "expected_outage_probability_mean": round(float(prim.expected_outage_probability.mean()), 4) if len(prim) else None,
             "expected_radio_capacity_mbps_mean": round(float(prim.expected_radio_capacity_mbps.mean()), 3) if len(prim) else None,
             "realized_radio_capacity_mbps_mean": round(float(prim.realized_radio_capacity_mbps.mean()), 3) if len(prim) else None,
             "realized_link_capacity_mbps_median_available_pue":
                 round(float(prim.loc[prim.provider_ok, "realized_link_capacity_mbps"].median()), 3) if prim.provider_ok.any() else None,
             "binding_constraint_share_available_pue":
                 prim.loc[prim.provider_ok, "binding_constraint"].value_counts(normalize=True).round(4).to_dict()}
        for t in tags:
            d[f"feasible_share_{t}"] = round(float(prim[f"feasible_{t}"].mean()), 4) if len(prim) else None
            d[f"first_failure_{t}"] = prim[f"first_failure_{t}"].value_counts(normalize=True).round(4).to_dict()
        out[scen] = d
    return out


def request_second_stats(rs: pd.DataFrame, cl: pd.DataFrame, tags):
    res = {"request_seconds": int(len(rs)),
           "share_with_any_in_range_pue": round(float((rs.n_in_range_pue > 0).mean()), 4) if len(rs) else None,
           "mean_out_of_range_pue": round(float(rs.n_out_of_range_pue.mean()), 3) if len(rs) else None,
           "share_gnb_covered": round(float(rs.gnb_covered.mean()), 4) if len(rs) else None}
    for scen, g in cl.groupby("channel_scenario"):
        prim = g[g.primary_eligible.astype(bool)] if scen == "G" else g
        for t in tags:
            f = prim[prim[f"feasible_{t}"]]
            anyf = f.groupby(["time_s", "request_id"]).size()
            full = f[f.covers_required_rate].groupby(["time_s", "request_id"]).size()
            res[f"{scen}_{t}_share_request_seconds_with_feasible_pue"] = round(len(anyf) / len(rs), 4) if len(rs) else None
            res[f"{scen}_{t}_share_request_seconds_with_full_rate_pue"] = round(len(full) / len(rs), 4) if len(rs) else None
    return res

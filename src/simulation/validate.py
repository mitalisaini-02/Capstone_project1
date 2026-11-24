"""Consistency checks for the Phase 3 per-run caches and per-scenario outputs."""
import numpy as np
import pandas as pd

from src.datagen.validate import Checks
from src.simulation.feasibility import STAGE_ORDER


def validate_run_cache(channel, reliability, episodes, conn, link_model, scenarios, discovery_range_m):
    ck = Checks()
    n = len(conn)
    ck.add("channel.rows_per_scenario", all((channel.channel_scenario == s).sum() == n for s in scenarios),
           f"{n} rows x {len(scenarios)} scenarios")
    ck.add("channel.in_range", (channel.distance_m <= discovery_range_m).all(), "")
    ck.add("channel.prob_sum", np.allclose(channel[["p_los", "p_nlosv", "p_nlos"]].sum(1), 1.0), "")
    g = channel[channel.channel_scenario == "G"]
    blk = g.building_blocked.fillna(False).astype(bool)
    ck.add("channel.G_blocked_is_NLOS", (g.loc[blk, "realized_state"] == "NLOS").all(), "")
    ck.add("channel.G_no_urban_plos_on_blocked", (g.loc[blk, ["p_los", "p_nlosv"]] == 0).all().all(), "")
    u, c1 = channel.episode_u.to_numpy(), channel.p_los.to_numpy()
    c2 = c1 + channel.p_nlosv.to_numpy()
    exp_state = np.where(u < c1, "LOS", np.where(u < c2, "NLOSv", "NLOS"))
    ck.add("channel.state_rule", (exp_state == channel.realized_state.to_numpy()).all(), "")
    per_ep = channel.groupby("episode_id")[["episode_u", "z_shadow", "z_blockage"]].nunique()
    ck.add("channel.draws_fixed_per_episode", (per_ep <= 1).all().all(), "")
    ck.add("channel.draws_shared_across_scenarios",
           channel.groupby(["time_s", "pue_id", "cue_id"])[["episode_u", "z_shadow"]].nunique().max().max() == 1, "")
    cap = link_model.capacity_from_snr_mbps(channel.realized_snr_db.to_numpy())
    ck.add("channel.realized_capacity_from_realized_snr",
           np.array_equal(cap, channel.realized_radio_capacity_mbps.to_numpy()), "")
    ck.add("channel.outage_iff_snr_below_min",
           ((channel.realized_snr_db < link_model.snr_min_db) == (channel.realized_radio_capacity_mbps <= 0)).all(), "")
    cmax = link_model.bw_hz * link_model.thr_max / 1e6
    ck.add("channel.expected_in_range", channel.expected_radio_capacity_mbps.between(0, cmax + 1e-9).all()
           and channel.expected_outage_probability.between(0, 1 + 1e-12).all(), "")
    ck.add("reliability.rows", len(reliability) == n, "")
    ck.add("reliability.window_iff_history_ok",
           (reliability.lstm_window_available.to_numpy() == conn.history_ok.to_numpy()).all(), "")
    p = reliability.p_survive
    ck.add("reliability.p_range", p[reliability.lstm_window_available].between(0, 1).all()
           and p[~reliability.lstm_window_available].isna().all(), "")
    ck.add("episodes.end_cause_consistent", not (episodes.end_cause == "inconsistent").any(),
           str(episodes.end_cause.value_counts().to_dict()))
    return ck


def validate_scenario_outputs(rs, cl, thresholds):
    ck = Checks()
    ck.add("rs.unique", not rs.duplicated(["time_s", "request_id"]).any(), "")
    ck.add("rs.counts", ((rs.n_in_range_pue + rs.n_out_of_range_pue == rs.n_present_pue)
                         & (rs.n_out_of_range_pue >= 0)).all(), "")
    ck.add("rs.gnb_capacity_when_covered", rs.loc[rs.gnb_covered, "gnb_available_capacity_mbps"].notna().all()
           and rs.loc[~rs.gnb_covered, "gnb_available_capacity_mbps"].isna().all(), "")
    if len(cl) == 0:
        ck.add("cl.non_empty", True, "no candidate links")
        return ck
    per = cl.groupby(["channel_scenario"]).size()
    ck.add("cl.same_rows_each_channel_scenario", per.nunique() == 1, str(per.to_dict()))
    inr = rs.set_index(["time_s", "request_id"]).n_in_range_pue
    cnt = cl[cl.channel_scenario == cl.channel_scenario.iloc[0]].groupby(["time_s", "request_id"]).size()
    ck.add("cl.rows_match_in_range_counts", np.array_equal(cnt.reindex(inr.index, fill_value=0).to_numpy(), inr.to_numpy()), "")
    ok = cl.provider_status == "available"
    c3 = cl.realized_link_capacity_mbps
    ck.add("cl.C3_le_peak_and_radio", ((c3 <= cl.link_peak_mbps + 1e-12) & (c3 <= cl.realized_radio_capacity_mbps + 1e-12))[ok].all(), "")
    m = np.minimum(np.minimum(c3, cl.access_capacity_mbps), cl.backhaul_capacity_mbps)
    ck.add("cl.pue_max_rate_is_min", np.allclose(m[ok], cl.pue_max_rate_mbps[ok]), "")
    ck.add("cl.wireless_uses_realized", (cl.wireless_ok == (cl.realized_radio_capacity_mbps > 0)).all(), "")
    for tag, theta in thresholds.items():
        rel = cl[f"reliable_{tag}"]
        ck.add(f"cl.reliable_{tag}_rule", (rel == (cl.p_survive.fillna(-1) >= theta)).all(), f"theta={theta}")
        allok = cl.provider_ok & cl.lstm_window_available.fillna(False).astype(bool) & rel & cl.wireless_ok & cl.resource_ok
        ck.add(f"cl.feasible_{tag}_iff_all_stages", (allok == cl[f"feasible_{tag}"]).all(), "")
        ck.add(f"cl.first_failure_{tag}_consistent",
               ((cl[f"first_failure_{tag}"] == "feasible") == cl[f"feasible_{tag}"]).all(), "")
    for tag in thresholds:
        wb = cl[f"first_failure_{tag}"] == "wireless_outage_building_blocked"
        geo_blocked = (cl.classification_source == "geometry") & (cl.building_blocked.astype(str) == "True") \
            & (cl.realized_state == "NLOS")
        ck.add(f"cl.building_reason_{tag}_only_on_geometry_blocked", bool((~wb | geo_blocked).all()), "")
    ck.add("cl.G_fallback_not_primary", (~cl.loc[cl.fallback.astype(bool), "primary_eligible"].astype(bool)).all(), "")
    return ck

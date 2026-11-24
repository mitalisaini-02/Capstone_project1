"""
Tiny synthetic Phase 2/3 scenario + SUMO trace for Phase 5 unit tests (SYNTHETIC test fixture only).

Trace: t = 0..20 (trace end 20).
  providers  1: 0..20 (censored), always available     2: 0..10 (departs at 11)
             3: 0..20, withdrawn at t = 8              4: 0..20, unavailable at t = 5, 6
             5: 3..20 (appears at 3), not registered at t = 3, 4, backhaul 0 at t = 10
  consumers  100: 0..20   101: 0..12 (departs)   102: 0..20
Requests (time_window): R0 cue100 t0 dur10 -> window_end at 9
                        R1 cue101 t2 dur20 -> cue_departed after 12
                        R2 cue102 t5 dur30 -> trace_end (censored) at 20
                        R3 cue100 t11 dur10 -> nominal end 20 = trace end -> window_end
Candidate rows: every active request-second x present provider, except out-of-range pairs
  (R0, 1) at t = 4; (R2, 4) always; (R2, 3) at t <= 7.
Link specifics (channel G, F1, F3): p_survive 0.9 except (R0, 1, t=2) 0.5 (below both thetas, link fine)
  and (R0, 1, t=6) 0.63 (reliable at 0.62, not at 0.65); (R0, 1, t=3) realized radio capacity 0 (outage);
  lstm window missing at t = 0. Channel F2: every link in outage.
"""
import gzip
import os

import numpy as np
import pandas as pd

from src.mobility.trace import RunMobility

T_END = 20
PROVIDERS = {1: (0, 20), 2: (0, 10), 3: (0, 20), 4: (0, 20), 5: (3, 20)}
CONSUMERS = {100: (0, 20), 101: (0, 12), 102: (0, 20)}
REQUESTS = [("R0", 100, 0, 10), ("R1", 101, 2, 20), ("R2", 102, 5, 30), ("R3", 100, 11, 10)]
WITHDRAW = {3: 8.0}
THETA = {"primary": 0.62, "sensitivity": 0.65}
CHANNELS = ("G", "F1", "F2", "F3")
RATE, PRICE_PUE, PRICE_GNB, ACCESS, BACKHAUL, PEAK, RADIO = 5.0, 0.5, 1.0, 40.0, 25.0, 50.0, 30.0


def mobility() -> RunMobility:
    rows = [(float(t), v, "provider", 10.0 * v, 0.0, 10.0) for v, (a, b) in PROVIDERS.items() for t in range(a, b + 1)]
    rows += [(float(t), v, "consumer", 5.0 * v, 1.0, 12.0) for v, (a, b) in CONSUMERS.items() for t in range(a, b + 1)]
    df = pd.DataFrame(rows, columns=["time_s", "vehicle_id", "role", "x", "y", "speed_mps"]).sort_values(
        ["time_s", "vehicle_id"]).reset_index(drop=True)
    roles = {v: "provider" for v in PROVIDERS} | {v: "consumer" for v in CONSUMERS}
    se = PROVIDERS | CONSUMERS
    return RunMobility("fixture_run", df, roles, {v: float(a) for v, (a, _) in se.items()},
                       {v: float(b) for v, (_, b) in se.items()}, np.arange(0.0, T_END + 1.0))


OVR = {}     # active overrides for the current write() call (see write())


def pue_state(p, t):
    """(registered, available, backhaul)."""
    registered = not (p == 5 and t in (3, 4))
    available = registered and not (p == 3 and t >= WITHDRAW[3]) and not (p == 4 and t in (5, 6))
    backhaul = 0.0 if (p == 5 and t == 10) else BACKHAUL
    backhaul = OVR.get("backhaul", {}).get((p, t), backhaul)
    return registered, available, backhaul


def status(p, t):
    reg, av, _ = pue_state(p, t)
    if av:
        return "available"
    if p in WITHDRAW and t >= WITHDRAW[p]:
        return "withdrawn"
    return "unavailable" if reg else "not_registered"


def in_range(r, p, t):
    if "range" in OVR:
        return OVR["range"](r, p, t)
    return not ((r == "R0" and p == 1 and t == 4) or (r == "R2" and p == 4) or (r == "R2" and p == 3 and t <= 7))


def last_active(rt, dur, cue):
    return min(rt + dur - 1, CONSUMERS[cue][1])


def write(root: str, expected_value: float = 7.0, tamper=None, range_fn=None, backhaul=None, uncovered=(),
          cell_cap=None, p_survive=None, outage=()) -> tuple:
    """Writes phase2/ and phase3/ dirs under root. Returns (phase3_dir, phase2_dir).
    Optional overrides (synthetic scenario variants): range_fn(r, p, t) -> in range; backhaul {(p, t): Mbps};
    uncovered {(r, t)}: request-second without gNodeB coverage; cell_cap {t: Mbps} for cell g1;
    p_survive {(r, p, t): value}; outage {(r, p, t)}: realized radio capacity 0."""
    OVR.clear()
    OVR.update({k: v for k, v in (("range", range_fn), ("backhaul", backhaul), ("p", p_survive), ("outage", set(outage))) if v})
    cap = lambda t: (cell_cap or {}).get(t, 100.0)
    p2, p3 = os.path.join(root, "phase2"), os.path.join(root, "phase3")
    os.makedirs(p2, exist_ok=True)
    os.makedirs(p3, exist_ok=True)
    sid = "FIX__fixture_run__r0"
    cue = pd.DataFrame([{"scenario_id": sid, "request_id": r, "cue_id": c, "request_time_s": float(rt),
                         "service_duration_s": float(d), "required_rate_mbps": RATE, "max_price_per_gb": 2.0,
                         "priority_weight": 1.0} for r, c, rt, d in REQUESTS])
    pue = pd.DataFrame([{"scenario_id": sid, "time_s": float(t), "pue_id": p, "registered": pue_state(p, t)[0],
                         "available": pue_state(p, t)[1], "backhaul_capacity_mbps": pue_state(p, t)[2],
                         "access_capacity_mbps": ACCESS, "link_peak_mbps": PEAK, "price_per_gb": PRICE_PUE}
                        for p, (a, b) in PROVIDERS.items() for t in range(a, b + 1)])
    gnb = pd.DataFrame([{"scenario_id": sid, "time_s": float(t), "gnb_id": "g1", "available_capacity_mbps": cap(t),
                         "price_per_gb": PRICE_GNB} for t in range(0, T_END + 1)])
    wd = pd.DataFrame([{"scenario_id": sid, "pue_id": p, "withdraw_time_s": w, "reason": "voluntary_withdrawal"}
                       for p, w in WITHDRAW.items()])
    rs, cl = [], []
    for r, c, rt, d in REQUESTS:
        for t in range(rt, last_active(rt, d, c) + 1):
            cov = (r, t) not in set(uncovered)
            rs.append({"scenario_id": sid, "time_s": float(t), "request_id": r, "cue_id": c, "required_rate_mbps": RATE,
                       "gnb_covered": cov, "gnb_id": "g1" if cov else None,
                       "gnb_available_capacity_mbps": cap(t) if cov else float("nan")})
            for p, (a, b) in PROVIDERS.items():
                if not (a <= t <= b) or not in_range(r, p, t):
                    continue
                for ch in CHANNELS:
                    cl.append(link_row(sid, ch, r, c, p, t, expected_value))
    rs, cl = pd.DataFrame(rs), pd.DataFrame(cl)
    if tamper == "provider_status":
        cl.loc[0, "provider_status"] = "withdrawn"
    if tamper == "request_window":
        rs = rs[~((rs.request_id == "R0") & (rs.time_s == 9.0))]
    for name, df in (("cue_requests", cue), ("pue_resources", pue), ("gnodeb_resources", gnb)):
        with gzip.GzipFile(os.path.join(p2, f"{name}.csv.gz"), "wb", mtime=0) as f:
            f.write(df.to_csv(index=False).encode())
    wd.to_csv(os.path.join(p2, "pue_withdrawals.csv"), index=False)
    for name, df in (("request_seconds", rs), ("candidate_links", cl)):
        with gzip.GzipFile(os.path.join(p3, f"{name}.csv.gz"), "wb", mtime=0) as f:
            f.write(df.to_csv(index=False).encode())
    return p3, p2


def link_row(sid, ch, r, c, p, t, expected_value):
    st = status(p, t)
    _, _, backhaul = pue_state(p, t)
    p_s = 0.5 if (r, p, t) == ("R0", 1, 2) else 0.63 if (r, p, t) == ("R0", 1, 6) else 0.9
    p_s = OVR.get("p", {}).get((r, p, t), p_s)
    lstm = t >= 1
    outage = ch == "F2" or (r, p, t) == ("R0", 1, 3) or (r, p, t) in OVR.get("outage", ())
    radio = 0.0 if outage else RADIO
    c3 = min(PEAK, radio)
    prov_ok, wl_ok = st == "available", radio > 0
    res_ok = min(c3, ACCESS, backhaul) > 0
    row = {"scenario_id": sid, "channel_scenario": ch, "time_s": float(t), "request_id": r, "cue_id": c, "pue_id": p,
           "distance_m": 100.0, "required_rate_mbps": RATE, "provider_status": st, "provider_ok": prov_ok,
           "lstm_window_available": lstm, "p_survive": p_s if lstm else np.nan,
           "classification_source": "geometry", "fallback": False, "realized_state": "NLOS" if outage else "LOS",
           "building_blocked": outage, "realized_radio_capacity_mbps": radio, "wireless_ok": wl_ok,
           "expected_radio_capacity_mbps": expected_value, "expected_outage_probability": expected_value / 100,
           "link_peak_mbps": PEAK, "realized_link_capacity_mbps": c3, "access_capacity_mbps": ACCESS,
           "backhaul_capacity_mbps": backhaul, "resource_ok": res_ok}
    for tag, th in THETA.items():
        rel = lstm and p_s >= th
        row[f"reliable_{tag}"] = rel
        stages = [("provider_unavailable", prov_ok), ("insufficient_history", lstm),
                  ("reliability_below_threshold", rel),
                  ("wireless_outage_building_blocked" if outage else "wireless_outage_los", wl_ok),
                  ("resource_zero", res_ok)]
        ff = next((s for s, ok in stages if not ok), "feasible")
        row[f"feasible_{tag}"] = ff == "feasible"
        row[f"first_failure_{tag}"] = ff
    return row

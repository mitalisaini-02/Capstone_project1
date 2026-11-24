"""
Allocation evaluation and comparison.

* result_rows / epoch_row: per-request and per-epoch records (PUE recorded only with positive traffic).
* scenario_metrics: per (method, channel, theta) metrics from those records.
* dominance checks: properties the exact lexicographic MILP must satisfy against any other
  feasible method on the SAME epoch (sanity checks, expected to be zero violations).
* paired_comparison: per-scenario paired differences with a bootstrap CI over scenarios.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.units import MBIT_PER_GB
from src.marketplace.admission import rejection_reason
from src.marketplace.allocation import audit

TOL_W, TOL_C = 1e-9, 1e-6


def result_rows(p, a, method, channel, theta_tag, theta, pricing, rate_eps) -> List[Dict]:
    rows = []
    pue_price = np.array([p.pue_price_per_gb[i] if i >= 0 else np.nan for i in a.pue_index])
    pue_rev, gnb_rev, pay = pricing.payments(a.pue_rate_mbps, pue_price, a.gnb_rate_mbps, p.gnb_price_per_gb, p.dt_s)
    for j in range(p.n_requests):
        adm = bool(a.admitted[j])
        has_pue = a.pue_index[j] >= 0
        if adm:
            served = "pue_only" if has_pue and a.gnb_rate_mbps[j] <= rate_eps else \
                "pue_plus_gnb" if has_pue else "gnb_only"
            reason, detail = "", ""
        else:
            served = "rejected"
            reason, detail = rejection_reason(p, j, rate_eps)
        l = a.link_index[j]
        rows.append({
            "scenario_id": p.scenario_id, "method": method, "channel_scenario": channel, "theta_tag": theta_tag,
            "theta": theta, "time_s": p.time_s, "request_id": p.request_ids[j], "cue_id": int(p.cue_ids[j]),
            "required_rate_mbps": float(p.demand_mbps[j]), "admitted": adm,
            "pue_id": int(p.pue_ids[a.pue_index[j]]) if has_pue else None,
            "pue_rate_mbps": float(a.pue_rate_mbps[j]),
            "link_realized_capacity_mbps": float(p.link_capacity_mbps[l]) if l >= 0 else None,
            "link_p_survive": float(p.link_p_survive[l]) if l >= 0 else None,
            "gnb_id": p.cell_ids[p.cell_index[j]] if (adm and a.gnb_rate_mbps[j] > rate_eps) else None,
            "gnb_rate_mbps": float(a.gnb_rate_mbps[j]),
            "payment": float(pay[j]), "pue_revenue": float(pue_rev[j]), "gnb_revenue": float(gnb_rev[j]),
            "served_by": served, "rejection_reason": reason, "rejection_detail": detail})
    return rows


def epoch_row(p, a, method, channel, theta_tag, audit_v) -> Dict:
    pue_cap = float(np.minimum(p.access_mbps, p.backhaul_mbps).sum())
    return {"scenario_id": p.scenario_id, "method": method, "channel_scenario": channel, "theta_tag": theta_tag,
            "time_s": p.time_s, "n_requests": p.n_requests, "n_links": p.n_links, "n_pue": len(p.pue_ids),
            "n_cells": len(p.cell_ids), "admitted_weight": float((p.weight * a.admitted).sum()),
            "scaled_cost": float(a.stage2_objective), "status": a.status, "runtime_s": a.runtime_s,
            "n_vars": a.notes.get("n_vars"), "n_cons": a.notes.get("n_cons"),
            "phantom_x": a.notes.get("phantom_x", 0),
            "pue_capacity_mbps": pue_cap, "pue_used_mbps": float(a.pue_rate_mbps.sum()),
            "cell_capacity_mbps": float(p.cell_capacity_mbps.sum()), "gnb_used_mbps": float(a.gnb_rate_mbps.sum()),
            "audit_violations": int(sum(len(v) for v in audit_v.values())),
            "audit_families": ";".join(sorted(audit_v))}


def dominance_violations(milp, other, label) -> Dict[str, int]:
    """Exact lexicographic optimum must (a) admit >= any feasible method, (b) cost <= at equal admission."""
    return {f"milp_admits_less_than_{label}": int(milp.stage1_objective < other.stage1_objective - TOL_W),
            f"milp_costs_more_than_{label}_at_equal_admission":
                int(abs(milp.stage1_objective - other.stage1_objective) <= TOL_W
                    and milp.stage2_objective > other.stage2_objective + TOL_C)}


def scenario_metrics(res: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (sid, m, ch, tt), r in res.groupby(["scenario_id", "method", "channel_scenario", "theta_tag"], sort=True):
        e = ep[(ep.scenario_id == sid) & (ep.method == m) & (ep.channel_scenario == ch) & (ep.theta_tag == tt)]
        adm = r[r.admitted]
        dt_gb = 1.0 / MBIT_PER_GB     # dt = 1 s
        pue_gb, gnb_gb = adm.pue_rate_mbps.sum() * dt_gb, adm.gnb_rate_mbps.sum() * dt_gb
        d = {"scenario_id": sid, "method": m, "channel_scenario": ch, "theta_tag": tt,
             "request_seconds": len(r), "admitted": int(r.admitted.sum()),
             "admitted_share": r.admitted.mean(), "payment_total": adm.payment.sum(),
             "delivered_gb": pue_gb + gnb_gb, "pue_gb": pue_gb, "gnb_gb": gnb_gb,
             "pue_volume_share": pue_gb / (pue_gb + gnb_gb) if pue_gb + gnb_gb > 0 else np.nan,
             "payment_per_gb": adm.payment.sum() / (pue_gb + gnb_gb) if pue_gb + gnb_gb > 0 else np.nan,
             "pue_utilisation": e.pue_used_mbps.sum() / e.pue_capacity_mbps.sum() if e.pue_capacity_mbps.sum() > 0 else np.nan,
             "gnb_utilisation": e.gnb_used_mbps.sum() / e.cell_capacity_mbps.sum() if e.cell_capacity_mbps.sum() > 0 else np.nan,
             "runtime_total_s": e.runtime_s.sum(), "runtime_p95_ms": 1000 * e.runtime_s.quantile(0.95),
             "audit_violations": int(e.audit_violations.sum()), "phantom_x": int(e.phantom_x.fillna(0).sum())}
        for s in ("pue_only", "pue_plus_gnb", "gnb_only", "rejected"):
            d[f"served_{s}"] = float((r.served_by == s).mean())
        for reason in ("no_service_option", "insufficient_capacity_standalone", "budget_exceeded", "capacity_contention"):
            d[f"reject_{reason}"] = int((r.rejection_reason == reason).sum())
        out.append(d)
    return pd.DataFrame(out)


def paired_comparison(metrics: pd.DataFrame, a: str, b: str, by_config: pd.Series, B=2000, seed=42) -> Dict:
    """Mean paired difference (a - b) per (channel, theta) over scenarios, with bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    out = {}
    for (ch, tt), g in metrics[metrics.method == a].groupby(["channel_scenario", "theta_tag"]):
        ref = metrics[(metrics.method == b) & (((metrics.channel_scenario == ch) & (metrics.theta_tag == tt))
                                               | (metrics.channel_scenario == "none"))]
        mg = g.merge(ref, on="scenario_id", suffixes=("_a", "_b"))
        res = {"n_scenarios": len(mg)}
        for col in ("admitted_share", "payment_total", "payment_per_gb", "pue_volume_share"):
            diff = (mg[f"{col}_a"] - mg[f"{col}_b"]).to_numpy(float)
            diff = diff[np.isfinite(diff)]
            if len(diff) == 0:
                continue
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(B)]
            res[col] = {"mean_diff": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                                   float(np.percentile(boots, 97.5))]}
        res["scenarios_where_a_admits_more"] = int((mg.admitted_a > mg.admitted_b).sum())
        res["scenarios_where_a_admits_less"] = int((mg.admitted_a < mg.admitted_b).sum())
        res["scenarios_identical_admission_and_payment"] = int(
            ((mg.admitted_a == mg.admitted_b) & np.isclose(mg.payment_total_a, mg.payment_total_b, rtol=0, atol=1e-9)).sum())
        out[f"{ch}|{tt}"] = res
    return out

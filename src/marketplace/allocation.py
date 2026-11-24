"""
Allocation result and the INDEPENDENT constraint audit.

Recording rule (phantom-PUE correction): a request's `pue_index` is set ONLY when the request
receives strictly positive PUE traffic (> rate_eps); solver binaries are never used to derive it.
A request served entirely by the gNodeB has pue_index = -1 (written as null).
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.marketplace.problem import EpochProblem


@dataclass
class Allocation:
    method: str
    admitted: np.ndarray          # bool per request
    pue_index: np.ndarray         # int per request, -1 = no PUE
    link_index: np.ndarray        # int per request, -1 = no PUE link
    pue_rate_mbps: np.ndarray     # per request
    gnb_rate_mbps: np.ndarray     # per request
    status: str = "ok"
    stage1_objective: float = float("nan")
    stage2_objective: float = float("nan")
    runtime_s: float = 0.0
    notes: Dict = field(default_factory=dict)


def from_link_rates(problem: EpochProblem, method: str, admitted, link_rates, gnb_rates,
                    rate_eps: float, **kw) -> Allocation:
    """Builds an Allocation from per-link rates. PUE recorded only for rate > rate_eps; sub-eps
    rates are treated as zero (they are numerical noise) and must still satisfy C6 in the audit."""
    n = problem.n_requests
    pue_idx = np.full(n, -1, dtype=int)
    link_idx = np.full(n, -1, dtype=int)
    pue_rate = np.zeros(n)
    for j in range(n):
        ls = problem.links_of_request(j)
        pos = [l for l in ls if link_rates[l] > rate_eps]
        if len(pos) > 1:
            raise ValueError(f"request {problem.request_ids[j]} uses {len(pos)} PUEs (violates C1)")
        if pos:
            l = pos[0]
            link_idx[j], pue_idx[j], pue_rate[j] = l, problem.link_pue[l], float(link_rates[l])
    g = np.where(np.asarray(gnb_rates) > rate_eps, np.asarray(gnb_rates, float), 0.0)
    return Allocation(method, np.asarray(admitted, bool), pue_idx, link_idx, pue_rate, g, **kw)


FAMILIES = ["C1_one_pue", "C2_eligible_link", "C3_link", "C4_access", "C5_backhaul", "C6_full_service",
            "C7_cell", "C8_coverage", "C9_budget", "phantom_pue", "nonnegative"]


def audit(problem: EpochProblem, a: Allocation, tol: float = 1e-6) -> Dict[str, List[str]]:
    """Re-checks every constraint family from the allocation alone. Returns {family: [violations]}."""
    v = {f: [] for f in FAMILIES}
    n = problem.n_requests
    rid = problem.request_ids
    for j in range(n):
        p, l, r, g = a.pue_index[j], a.link_index[j], a.pue_rate_mbps[j], a.gnb_rate_mbps[j]
        if r < -tol or g < -tol:
            v["nonnegative"].append(rid[j])
        # phantom checks: a recorded PUE must carry traffic over an eligible link of THIS request
        if (p >= 0) != (l >= 0):
            v["phantom_pue"].append(f"{rid[j]}: pue/link index mismatch")
        if p >= 0 and r <= tol:
            v["phantom_pue"].append(f"{rid[j]}: PUE {problem.pue_ids[p]} recorded with zero traffic")
        if p < 0 and r > tol:
            v["phantom_pue"].append(f"{rid[j]}: PUE traffic without a recorded PUE")
        if l >= 0:
            if problem.link_request[l] != j or problem.link_pue[l] != p:
                v["C2_eligible_link"].append(f"{rid[j]}: link {l} does not belong to request/PUE")
            elif r > problem.link_capacity_mbps[l] + tol:
                v["C3_link"].append(f"{rid[j]}: {r:.6f} > B {problem.link_capacity_mbps[l]:.6f}")
        if not a.admitted[j]:
            if r > tol or g > tol or p >= 0:
                v["C1_one_pue"].append(f"{rid[j]}: service to a non-admitted request")
            continue
        if abs(r + g - problem.demand_mbps[j]) > tol:
            v["C6_full_service"].append(f"{rid[j]}: {r + g:.6f} != d {problem.demand_mbps[j]:.6f}")
        if g > tol and not problem.covered[j]:
            v["C8_coverage"].append(rid[j])
        pay = (r * (problem.pue_price_per_gb[p] if p >= 0 else 0.0) + g * problem.gnb_price_per_gb)
        if pay > problem.max_price_per_gb[j] * problem.demand_mbps[j] * (1 + 1e-9) + tol:
            v["C9_budget"].append(rid[j])
    for i in range(len(problem.pue_ids)):
        load = a.pue_rate_mbps[a.pue_index == i].sum()
        if load > problem.access_mbps[i] + tol:
            v["C4_access"].append(f"PUE {problem.pue_ids[i]}: {load:.6f} > A {problem.access_mbps[i]:.6f}")
        if load > problem.backhaul_mbps[i] + tol:
            v["C5_backhaul"].append(f"PUE {problem.pue_ids[i]}: {load:.6f} > C {problem.backhaul_mbps[i]:.6f}")
    for k in range(len(problem.cell_ids)):
        load = a.gnb_rate_mbps[(problem.cell_index == k) & a.admitted].sum()
        if load > problem.cell_capacity_mbps[k] + tol:
            v["C7_cell"].append(f"cell {problem.cell_ids[k]}: {load:.6f} > G {problem.cell_capacity_mbps[k]:.6f}")
    return {f: x for f, x in v.items() if x}

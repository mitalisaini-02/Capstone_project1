"""
Request admission: single-request options and rejection reasons (shared by all methods).

best_option(problem, j, A_rem, C_rem, G_rem)
    Exact cost-minimal way to serve request j ALONE with the given remaining resources, under
    C1-C9: either gNodeB only, or ONE eligible PUE i plus the gNodeB remainder.
    For PUE i the PUE rate r ranges over [r_lo, r_hi] with
        r_hi = min(d, B_l, A_i, C_i)                 (C3, C4, C5)
        r_lo = max(0, d - G_k*cov)                   (C6, C7, C8)
    and cost (price/GB x Mbps) = pi_i r + pi_g (d - r) is linear in r, so the minimum is at
    r_hi if pi_i < pi_g, at r_lo if pi_i > pi_g (ties: r_hi, i.e. prefer the PUE share).
    An option with r <= rate_eps is a gNodeB-only option (no PUE recorded).
    C9 (budget): cost <= pi_j^max * d; since we pick the cheapest point, budget infeasibility at that
    point means the option is budget-infeasible.

Rejection reasons, in this fixed order:
    no_service_option                  not covered and no eligible PUE link
    insufficient_capacity_standalone   no capacity-feasible option even with all resources free
    budget_exceeded                    capacity-feasible options exist but all exceed the budget
    capacity_contention                servable alone, but not admitted alongside the others
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.marketplace.problem import EpochProblem

REASONS = ["no_service_option", "insufficient_capacity_standalone", "budget_exceeded", "capacity_contention"]


@dataclass(frozen=True)
class Option:
    link: int            # -1 = gNodeB only
    pue: int             # -1 = gNodeB only
    pue_rate: float
    gnb_rate: float
    scaled_cost: float
    budget_ok: bool


def options(problem: EpochProblem, j: int, A_rem, C_rem, G_rem, rate_eps: float, tol: float = 1e-9) -> List[Option]:
    """All capacity-feasible single-request options (each at its cost-minimal PUE rate)."""
    d = float(problem.demand_mbps[j])
    k = int(problem.cell_index[j])
    g_cap = float(G_rem[k]) if (problem.covered[j] and k >= 0) else 0.0
    pg = problem.gnb_price_per_gb
    budget = float(problem.max_price_per_gb[j]) * d
    out = []
    if g_cap >= d - tol:
        out.append(Option(-1, -1, 0.0, d, pg * d, pg * d <= budget * (1 + 1e-12) + tol))
    for l in problem.links_of_request(j):
        i = int(problem.link_pue[l])
        r_hi = min(d, float(problem.link_capacity_mbps[l]), float(A_rem[i]), float(C_rem[i]))
        r_lo = max(0.0, d - g_cap)
        if r_hi < r_lo - tol or r_hi <= rate_eps:
            continue
        pi = float(problem.pue_price_per_gb[i])
        r = r_hi if pi <= pg else max(r_lo, 0.0)
        if r <= rate_eps:          # cheapest point uses no PUE traffic -> it is the gNodeB-only option
            continue
        r = min(r, d)
        g = d - r
        cost = pi * r + pg * g
        out.append(Option(int(l), i, r, g, cost, cost <= budget * (1 + 1e-12) + tol))
    return out


def best_option(problem, j, A_rem, C_rem, G_rem, rate_eps) -> Optional[Option]:
    """Cheapest budget-feasible option; ties -> larger PUE rate, then lower PUE id."""
    feas = [o for o in options(problem, j, A_rem, C_rem, G_rem, rate_eps) if o.budget_ok]
    if not feas:
        return None
    pid = lambda o: int(problem.pue_ids[o.pue]) if o.pue >= 0 else 10 ** 12
    return min(feas, key=lambda o: (round(o.scaled_cost, 12), -o.pue_rate, pid(o)))


def full_resources(problem):
    return problem.access_mbps.copy(), problem.backhaul_mbps.copy(), problem.cell_capacity_mbps.copy()


def rejection_reason(problem: EpochProblem, j: int, rate_eps: float):
    """(reason, detail) for a request that was not admitted."""
    if (not problem.covered[j]) and len(problem.links_of_request(j)) == 0:
        summ = problem.ineligible_summary.get(j, {})
        detail = ";".join(f"{k}={v}" for k, v in sorted(summ.items())) if summ else "no_pue_in_range"
        return "no_service_option", detail
    A, C, G = full_resources(problem)
    opts = options(problem, j, A, C, G, rate_eps)
    if not opts:
        return "insufficient_capacity_standalone", ""
    if not any(o.budget_ok for o in opts):
        cheapest = min(o.scaled_cost for o in opts) / problem.demand_mbps[j]
        return "budget_exceeded", f"cheapest_price_per_gb={cheapest:.4f}>max={problem.max_price_per_gb[j]:.4f}"
    return "capacity_contention", ""

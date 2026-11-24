"""
Greedy allocation (deterministic heuristic).

Requests are processed in priority order: weight (desc), arrival time (asc), request_id (asc).
Each request takes the CHEAPEST budget-feasible option (admission.best_option) given the
resources still remaining (C4/C5 per PUE, C7 per cell); remaining resources are then reduced.
A request with no feasible option is rejected. Same inputs, constraints and audit as the MILP.
"""
import time

import numpy as np

from src.marketplace.admission import best_option, full_resources
from src.marketplace.allocation import Allocation


class GreedyAllocator:
    name = "greedy"

    def __init__(self, rate_eps: float = 1e-6):
        self.rate_eps = rate_eps

    def order(self, problem):
        return sorted(range(problem.n_requests),
                      key=lambda j: (-problem.weight[j], problem.arrival_s[j], problem.request_ids[j]))

    def solve(self, problem) -> Allocation:
        t0 = time.perf_counter()
        n = problem.n_requests
        A, C, G = full_resources(problem)
        admitted = np.zeros(n, bool)
        pue_idx = np.full(n, -1, int)
        link_idx = np.full(n, -1, int)
        pr, gr = np.zeros(n), np.zeros(n)
        for j in self.order(problem):
            o = best_option(problem, j, A, C, G, self.rate_eps)
            if o is None:
                continue
            admitted[j] = True
            if o.pue >= 0:
                pue_idx[j], link_idx[j], pr[j] = o.pue, o.link, o.pue_rate
                A[o.pue] -= o.pue_rate
                C[o.pue] -= o.pue_rate
            gr[j] = o.gnb_rate
            if o.gnb_rate > 0:
                G[problem.cell_index[j]] -= o.gnb_rate
        pue_price = np.array([problem.pue_price_per_gb[i] if i >= 0 else 0.0 for i in pue_idx])
        cost = float((pr * pue_price).sum() + gr.sum() * problem.gnb_price_per_gb)
        return Allocation(self.name, admitted, pue_idx, link_idx, pr, gr, "ok",
                          float((problem.weight * admitted).sum()), cost, time.perf_counter() - t0)

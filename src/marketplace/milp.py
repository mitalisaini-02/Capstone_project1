"""
Two-stage lexicographic MILP (HiGHS via scipy.optimize.milp).

Variables per epoch: x_l (binary), r_l >= 0, g_j >= 0, y_j (binary).
  (C1) sum_{l in L(j)} x_l - y_j <= 0         (C3) r_l - B_l x_l <= 0
  (C4) sum_{l in L(i)} r_l <= A_i               (C5) sum_{l in L(i)} r_l <= C_i
  (C6) sum_{l in L(j)} r_l + g_j - d_j y_j = 0  (C7) sum_{j in J(k)} g_j <= G_k
  (C8) g_j <= d_j cov_j (variable bound)        (C9) sum pi_i r_l + pi_g g_j - pi_j^max d_j y_j <= 0
  (C2) only Phase-3-eligible links have variables.
Stage 1: max sum w_j y_j.  Stage 2: min sum pi_i r_l + pi_g g_j  s.t. sum w_j y_j >= W*.
Output PUE assignment comes from the RATES (Allocation.from_link_rates), never from x.
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from src.marketplace.allocation import from_link_rates


class MilpAllocator:
    name = "milp"

    def __init__(self, rate_eps=1e-6, mip_rel_gap=0.0, time_limit_s=30.0, presolve=True, name=None):
        self.rate_eps = rate_eps
        self.opts = {"mip_rel_gap": mip_rel_gap, "time_limit": time_limit_s, "presolve": presolve, "disp": False}
        if name:
            self.name = name

    def _model(self, p):
        nL, nJ = p.n_links, p.n_requests
        X, R, Gv, Y = 0, nL, 2 * nL, 2 * nL + nJ
        nv = 2 * nL + 2 * nJ
        rows, lo, hi = [], [], []

        def add(coefs, l_, h_):
            rows.append(coefs); lo.append(l_); hi.append(h_)

        for j in range(nJ):
            ls = p.links_of_request(j)
            add({**{X + l: 1.0 for l in ls}, Y + j: -1.0}, -np.inf, 0.0)                       # C1
            add({**{R + l: 1.0 for l in ls}, Gv + j: 1.0, Y + j: -float(p.demand_mbps[j])}, 0.0, 0.0)   # C6
            add({**{R + l: float(p.pue_price_per_gb[p.link_pue[l]]) for l in ls},
                 Gv + j: float(p.gnb_price_per_gb),
                 Y + j: -float(p.max_price_per_gb[j] * p.demand_mbps[j])}, -np.inf, 0.0)        # C9
        for l in range(nL):
            add({R + l: 1.0, X + l: -float(p.link_capacity_mbps[l])}, -np.inf, 0.0)               # C3
        for i in range(len(p.pue_ids)):
            ls = p.links_of_pue(i)
            if len(ls):
                add({R + l: 1.0 for l in ls}, -np.inf, float(p.access_mbps[i]))                   # C4
                add({R + l: 1.0 for l in ls}, -np.inf, float(p.backhaul_mbps[i]))                 # C5
        for k in range(len(p.cell_ids)):
            js = np.where(p.cell_index == k)[0]
            if len(js):
                add({Gv + j: 1.0 for j in js}, -np.inf, float(p.cell_capacity_mbps[k]))          # C7
        A = lil_matrix((len(rows), nv))
        for r_, coefs in enumerate(rows):
            for c, v in coefs.items():
                A[r_, c] = v
        lb = np.zeros(nv)
        ub = np.concatenate([np.ones(nL), p.link_capacity_mbps.astype(float),
                             np.where(p.covered, p.demand_mbps, 0.0).astype(float), np.ones(nJ)])     # C8 bound
        # requests with no service option cannot be admitted
        no_opt = np.array([(not p.covered[j]) and len(p.links_of_request(j)) == 0 for j in range(nJ)], bool)
        ub[Y:Y + nJ][no_opt] = 0.0
        integ = np.concatenate([np.ones(nL), np.zeros(nL), np.zeros(nJ), np.ones(nJ)])
        return A.tocsr(), np.array(lo), np.array(hi), Bounds(lb, ub), integ, (X, R, Gv, Y, nv)

    def solve(self, p):
        t0 = time.perf_counter()
        A, lo, hi, bounds, integ, (X, R, Gv, Y, nv) = self._model(p)
        nJ, nL = p.n_requests, p.n_links
        c1 = np.zeros(nv)
        c1[Y:Y + nJ] = -p.weight
        cons = [LinearConstraint(A, lo, hi)] if A.shape[0] else []
        s1 = milp(c1, constraints=cons, integrality=integ, bounds=bounds, options=self.opts)
        if s1.status != 0:
            raise RuntimeError(f"stage 1 failed at t={p.time_s}: {s1.message}")
        W = -float(s1.fun)
        c2 = np.zeros(nv)
        c2[R:R + nL] = p.pue_price_per_gb[p.link_pue] if nL else 0.0
        c2[Gv:Gv + nJ] = p.gnb_price_per_gb
        wrow = np.zeros((1, nv))
        wrow[0, Y:Y + nJ] = p.weight
        cons2 = cons + [LinearConstraint(wrow, W - 1e-6, np.inf)]
        s2 = milp(c2, constraints=cons2, integrality=integ, bounds=bounds, options=self.opts)
        if s2.status != 0:
            raise RuntimeError(f"stage 2 failed at t={p.time_s}: {s2.message}")
        x = s2.x
        admitted = x[Y:Y + nJ] > 0.5
        alloc = from_link_rates(p, self.name, admitted, x[R:R + nL], x[Gv:Gv + nJ], self.rate_eps,
                                status="optimal", stage1_objective=W, stage2_objective=float(s2.fun),
                                runtime_s=time.perf_counter() - t0,
                                notes={"n_vars": nv, "n_cons": int(A.shape[0]),
                                       "phantom_x": int(sum(1 for l in range(nL)
                                                            if x[X + l] > 0.5 and x[R + l] <= self.rate_eps))})
        return alloc

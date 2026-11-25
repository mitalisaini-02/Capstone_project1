"""Common snapshot evaluation and four allocation methods."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


@dataclass
class Snapshot:
    problem: object
    link_se: np.ndarray
    cell_se: np.ndarray
    pc5_mhz: float
    gnb_mhz: float
    eta: float
    epsilon: float
    minimum: float
    threshold: float

    def __post_init__(self):
        p = self.problem
        self.factor = p.dt_s / 8000
        self.prices = p.pue_price_per_gb[p.link_pue]
        self.links = [np.flatnonzero(p.link_request == j) for j in range(p.n_requests)]
        self.provider_cap = np.minimum(p.access_mbps, p.backhaul_mbps)
        self.budget = p.demand_mbps * p.max_price_per_gb * self.factor

    def payment(self, link, relay, cell):
        return self.factor * (relay * self.prices[link] + cell * self.problem.gnb_price_per_gb) + self.eta

    def empty(self):
        p = self.problem
        return tuple(np.zeros(n) for n in (p.n_links, p.n_requests, p.n_links, p.n_requests))

    def evaluate(self, allocation, tol=1e-7):
        p = self.problem
        relay, cell, rb, gb = allocation
        selected = relay > self.minimum / 2
        delivered = np.bincount(p.link_request, weights=relay, minlength=p.n_requests) + cell
        link_rate = np.minimum(p.link_capacity_mbps, rb * self.link_se)
        cell_rate = gb * self.cell_se
        pue_cost = relay * self.prices * self.factor
        gnb_cost = cell * p.gnb_price_per_gb * self.factor
        fees = selected * self.eta
        request_cost = np.bincount(p.link_request, weights=pue_cost + fees, minlength=p.n_requests) + gnb_cost
        valid = p.link_p_survive >= self.threshold
        violations = {
            "demand": np.count_nonzero(abs(delivered - p.demand_mbps) > tol),
            "selection": np.count_nonzero(np.bincount(p.link_request, weights=selected, minlength=p.n_requests) > 1),
            "link_capacity": np.count_nonzero(relay > link_rate + tol),
            "activity": np.count_nonzero((~selected & (rb > tol)) | (selected & (relay < self.minimum - tol))),
            "reliability": np.count_nonzero(selected & ~valid),
            "coverage": np.count_nonzero(((cell > tol) | (gb > tol)) & ~p.covered),
            "gnb_radio": np.count_nonzero(cell > cell_rate + tol),
            "budget": np.count_nonzero(request_cost > self.budget + tol * self.factor),
            "invalid": sum(np.count_nonzero(~np.isfinite(a) | (a < -tol)) for a in allocation),
        }
        for name, values, groups, limits in [
            ("access", relay, p.link_pue, p.access_mbps),
            ("backhaul", relay, p.link_pue, p.backhaul_mbps),
            ("pc5_bandwidth", rb, p.link_pue, np.full(len(p.pue_ids), self.pc5_mhz)),
            ("cell_capacity", cell[p.covered], p.cell_index[p.covered], p.cell_capacity_mbps),
            ("gnb_bandwidth", gb[p.covered], p.cell_index[p.covered], np.full(len(p.cell_ids), self.gnb_mhz)),
        ]:
            usage = np.bincount(groups, weights=values, minlength=len(limits))
            violations[name] = np.count_nonzero(usage > limits + tol)
        served = delivered >= p.demand_mbps - tol
        total_violations = sum(violations.values())
        return dict(
            cost=float(request_cost.sum()) if not total_violations else np.nan,
            supplied_traffic_payment=float(request_cost.sum()),
            pue_cost=pue_cost.sum(), gnb_cost=gnb_cost.sum(), mec_fee=fees.sum(),
            rate=link_rate.sum() + cell_rate.sum(), pue_rate=link_rate.sum(), gnb_rate=cell_rate.sum(),
            delivered_mbps=delivered.sum(), demand_mbps=p.demand_mbps.sum(),
            pue_gb=relay.sum() * self.factor, gnb_gb=cell.sum() * self.factor,
            served=served.sum(), partial=((~served) & (delivered > tol)).sum(),
            unserved=((~served) & (delivered <= tol)).sum(),
            requests=p.n_requests, providers=len(p.pue_ids), selected=selected.sum(),
            reliable_selected=(selected & valid).sum(), violations=total_violations,
            **{f"violations_{key}": n for key, n in violations.items()},
        )

    def constructive(self, mode, rng=None):
        """Sequential full-demand commitments with a feasible interval for each split."""
        p = self.problem
        plan = self.empty()
        relay, cell, rb, gb = plan
        provider = self.provider_cap.copy()
        cells = p.cell_capacity_mbps.copy()
        pb = np.full(len(provider), self.pc5_mhz)
        cb = np.full(len(cells), self.gnb_mhz)
        order = np.lexsort((np.arange(p.n_requests), p.arrival_s, -p.weight))
        if mode == "Random":
            order = rng.permutation(order)
        for j in order:
            k = p.cell_index[j]
            demand = p.demand_mbps[j]
            fallback = min(cells[k], cb[k] * self.cell_se[j]) if k >= 0 else 0.
            options = []
            g_cost = self.factor * p.gnb_price_per_gb * demand
            if fallback >= demand and g_cost <= self.budget[j]:
                options.append((-1, 0., g_cost, -np.inf))
            for link in self.links[j]:
                i = p.link_pue[link]
                hi = min(demand, provider[i], pb[i] * self.link_se[link], p.link_capacity_mbps[link])
                lo = max(self.minimum, demand - fallback)
                fixed = g_cost + self.eta
                slope = self.factor * (self.prices[link] - p.gnb_price_per_gb)
                if slope > 0:
                    hi = min(hi, (self.budget[j] - fixed) / slope)
                elif slope < 0:
                    lo = max(lo, (self.budget[j] - fixed) / slope)
                elif fixed > self.budget[j]:
                    continue
                if lo <= hi:
                    r = rng.uniform(lo, hi) if mode == "Random" else hi if slope <= 0 or mode == "Best-SINR" else lo
                    options.append((link, r, self.payment(link, r, demand - r), self.link_se[link]))
            if not options:
                continue
            if mode == "Random":
                link, r, _, _ = options[rng.integers(len(options))]
            elif mode == "Best-SINR":
                link, r, _, _ = max(options, key=lambda x: x[3])
            else:
                best = min(x[2] for x in options)
                link, r, _, _ = max((x for x in options if x[2] <= best + self.epsilon),
                                    key=lambda x: (x[3], -x[2]))
            if link >= 0:
                i = p.link_pue[link]
                relay[link] = r
                rb[link] = r / self.link_se[link]
                provider[i] -= r
                pb[i] -= rb[link]
            cell[j] = max(0., demand - r)
            if cell[j] > 0:
                gb[j] = cell[j] / self.cell_se[j]
                cells[k] -= cell[j]
                cb[k] -= gb[j]

        # Assign spare bandwidth to the highest-efficiency active links first.
        for link in np.argsort(-self.link_se):
            if relay[link] > 0:
                i = p.link_pue[link]
                added = min(max(0., pb[i]), max(0., p.link_capacity_mbps[link] / self.link_se[link] - rb[link]))
                rb[link] += added
                pb[i] -= added
        for j in np.argsort(-self.cell_se):
            if cell[j] > 0:
                k = p.cell_index[j]
                gb[j] += max(0., cb[k])
                cb[k] = 0.
        return plan

    def exact(self, time_limit, gap):
        """Two-stage MILP with fixed snapshot efficiencies and a bandwidth cap at saturation."""
        p = self.problem
        l, n = p.n_links, p.n_requests
        relay, cell, rb, gb, x, y = np.split(np.arange(3*l + 3*n), np.cumsum([l, n, l, n, l]))
        cost = np.zeros(3*l + 3*n)
        cost[relay], cost[cell], cost[x] = self.prices * self.factor, p.gnb_price_per_gb * self.factor, self.eta
        rate = np.zeros_like(cost)
        rate[rb], rate[gb] = self.link_se, self.cell_se
        ub = np.r_[p.link_capacity_mbps, p.demand_mbps * p.covered,
                   np.minimum(self.pc5_mhz, p.link_capacity_mbps / self.link_se),
                   p.covered * self.gnb_mhz, np.ones(l), p.covered]
        integrality = np.zeros(len(cost), dtype=int)
        integrality[np.r_[x, y]] = 1
        rows, lows, highs = [], [], []

        def add(indices, coeffs, high, low=-np.inf):
            row = np.zeros_like(cost)
            row[indices] = coeffs
            rows.append(row)
            lows.append(low)
            highs.append(high)

        for j, links in enumerate(self.links):
            add(x[links], 1, 1)
            delivery = np.r_[relay[links], cell[j]]
            add(delivery, 1, p.demand_mbps[j], p.demand_mbps[j])
            bill = np.r_[delivery, x[links]]
            add(bill, cost[bill] / self.factor, self.budget[j] / self.factor)
            add([cell[j], gb[j]], [1, -self.cell_se[j]], 0)
            add([gb[j], y[j]], [1, -self.gnb_mhz], 0)
            add([cell[j], y[j]], [1, -self.minimum], np.inf, low=0)
        for i, cap in enumerate(self.provider_cap):
            links = p.link_pue == i
            add(relay[links], 1, cap)
            add(rb[links], 1, self.pc5_mhz)
        for k, cap in enumerate(p.cell_capacity_mbps):
            requests = p.cell_index == k
            add(cell[requests], 1, cap)
            add(gb[requests], 1, self.gnb_mhz)
        for h in range(l):
            add([relay[h], rb[h]], [1, -self.link_se[h]], 0)
            add([rb[h], x[h]], [1, -ub[rb[h]]], 0)
            add([relay[h], x[h]], [1, -self.minimum], np.inf, low=0)

        def solve(objective):
            return milp(objective, integrality=integrality, bounds=Bounds(0, ub),
                        constraints=LinearConstraint(csc_matrix(rows), lows, highs),
                        options={"time_limit": time_limit, "mip_rel_gap": gap})

        first = solve(cost / self.factor)
        info = {"stage1_status": first.status, "stage1_gap": getattr(first, "mip_gap", None),
                "stage1_message": first.message, "stage2_status": None, "stage2_gap": None,
                "proven_optimal": False}
        if first.x is None:
            return self.empty(), info
        optimum = float(cost @ first.x)
        info["stage1_cost"] = optimum
        add(np.arange(len(cost)), cost / self.factor, (optimum + self.epsilon) / self.factor)
        second = solve(-rate)
        chosen = second.x if second.x is not None else first.x
        info.update(stage2_status=second.status, stage2_gap=getattr(second, "mip_gap", None),
                    stage2_message=second.message, stage2_rate=float(rate @ chosen),
                    proven_optimal=first.status == second.status == 0 and
                    getattr(first, "mip_gap", 0) == getattr(second, "mip_gap", 0) == 0)
        return tuple(np.maximum(0., chosen[index]) for index in (relay, cell, rb, gb)), info


def choose(evaluations, epsilon):
    feasible = [i for i, m in enumerate(evaluations) if not m["violations"]]
    if not feasible:
        return max(range(len(evaluations)), key=lambda i: (evaluations[i]["served"],
                                                           evaluations[i]["delivered_mbps"]))
    minimum = min(evaluations[i]["cost"] for i in feasible)
    return max((i for i in feasible if evaluations[i]["cost"] <= minimum + epsilon),
               key=lambda i: (evaluations[i]["rate"], -evaluations[i]["cost"]))

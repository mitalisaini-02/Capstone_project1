"""Common lifecycle for Experiment B. Only arrivals reach an allocation policy."""
from dataclasses import dataclass, replace
import hashlib
from time import perf_counter

import numpy as np
import pandas as pd

from objective_comparison.allocation import choose
from objective_comparison.persistent_policy import ResidualSnapshot

TOL = 1e-7


@dataclass
class Connection:
    started: float
    mode: str = "blocked"
    pue: int | None = None
    site: str | None = None
    relay: float = 0.
    cell: float = 0.
    rb: float = 0.
    gb: float = 0.
    failure_s: float | None = None
    pue_seconds: int = 0
    gnb_seconds: int = 0
    outage_seconds: int = 0
    requested_seconds: int = 0
    last_s: float = 0.
    payment: float = 0.
    delivered_gb: float = 0.


def residual(s, plan):
    p = s.problem
    relay, cell, rb, gb = plan
    covered = p.covered
    counts = (len(p.pue_ids), len(p.cell_ids))
    used = [
        np.bincount(p.link_pue, weights=relay, minlength=counts[0]),
        np.bincount(p.cell_index[covered], weights=cell[covered], minlength=counts[1]),
        np.bincount(p.link_pue, weights=rb, minlength=counts[0]),
        np.bincount(p.cell_index[covered], weights=gb[covered], minlength=counts[1]),
    ]
    return used


def pending_snapshot(s, plan, requests):
    p = s.problem
    requests = np.asarray(requests, dtype=int)
    jmap = np.full(p.n_requests, -1)
    jmap[requests] = np.arange(len(requests))
    links = np.flatnonzero(jmap[p.link_request] >= 0)
    used_p, used_g, used_pb, used_gb = residual(s, plan)
    pending = replace(p, request_ids=[p.request_ids[j] for j in requests],
                      **{key: getattr(p, key)[requests] for key in
                         ("cue_ids", "demand_mbps", "weight", "max_price_per_gb", "arrival_s", "covered", "cell_index")},
                      access_mbps=np.maximum(0., p.access_mbps - used_p),
                      backhaul_mbps=np.maximum(0., p.backhaul_mbps - used_p),
                      cell_capacity_mbps=np.maximum(0., p.cell_capacity_mbps - used_g),
                      link_request=jmap[p.link_request[links]],
                      **{key: getattr(p, key)[links] for key in
                         ("link_pue", "link_capacity_mbps", "link_p_survive")}, ineligible_summary={})
    sub = ResidualSnapshot(pending, s.link_se[links], s.cell_se[requests], s.pc5_mhz,
                           s.gnb_mhz, s.eta, s.epsilon, s.minimum, s.threshold)
    sub.pc5_remaining = np.maximum(0., s.pc5_mhz - used_pb)
    sub.gnb_remaining = np.maximum(0., s.gnb_mhz - used_gb)
    return sub, links


def initial(s, method, cfg):
    if method == "Exact":
        return s.exact(cfg["solver_time_limit_s"], cfg["solver_relative_gap"])
    if method == "Random":
        p = s.problem
        digest = hashlib.sha256(f"{cfg['seed']}:{p.scenario_id}:{p.time_s}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        plans = [s.constructive(method, rng) for _ in range(cfg["random_trials"])]
        return plans[choose([s.evaluate(plan) for plan in plans], s.epsilon)], {}
    return s.constructive(method), {}


def valid(s, plan):
    result = s.evaluate(plan)
    return result["violations"] == result["violations_demand"]


class PolicyState:
    def __init__(self, method, cfg):
        self.method = method
        self.cfg = cfg
        self.connections = {}
        self.active = {}
        self.events = []

    def step(self, s):
        start = perf_counter()
        p = s.problem
        plan = s.empty()
        relay, cell, rb, gb = plan
        index = {rid: j for j, rid in enumerate(p.request_ids)}
        sites = [p.cell_ids[k] if k >= 0 else None for k in p.cell_index]
        previous = self.active
        self.active = {rid: c for rid, c in previous.items() if rid in index}
        arrivals = [j for rid, j in index.items() if rid not in self.connections]
        fallback = []
        failures = kept = 0
        for rid in sorted(self.active, key=lambda rid: (self.active[rid].started, rid)):
            c = self.active[rid]
            j = index[rid]
            if c.mode == "blocked":
                continue
            h = next((h for h in s.links[j] if p.pue_ids[p.link_pue[h]] == c.pue), None)
            link_ok = c.mode != "pue" or h is not None
            cell_ok = c.cell <= TOL or sites[j] == c.site
            if link_ok and cell_ok:
                if h is not None:
                    relay[h], rb[h] = c.relay, c.rb
                cell[j], gb[j] = c.cell, c.gb
                if abs(c.relay + c.cell - p.demand_mbps[j]) <= TOL and valid(s, plan):
                    kept += c.mode == "pue"
                    continue
                if h is not None:
                    relay[h] = rb[h] = 0.
                cell[j] = gb[j] = 0.
            if c.mode == "pue":
                failures += 1
                c.failure_s = p.time_s
                self.events.append(dict(request_id=rid, time_s=p.time_s, event="pue_to_gnb",
                                        pue_id=c.pue, pue_service_seconds=c.pue_seconds))
            c.mode, c.pue, c.site = "gnb", None, None
            c.relay = c.cell = c.rb = c.gb = 0.
            fallback.append(j)

        # Apply the original gNodeB-only feasibility and spare-bandwidth rules
        # directly. No initial allocation policy is involved in fallback.
        _, used_g, _, used_gb = residual(s, plan)
        cells = p.cell_capacity_mbps - used_g
        widths = s.gnb_mhz - used_gb
        served_fallback = []
        for j in fallback:
            k = p.cell_index[j]
            demand = p.demand_mbps[j]
            if (k >= 0 and min(cells[k], widths[k] * s.cell_se[j]) >= demand
                    and s.factor * p.gnb_price_per_gb * demand <= s.budget[j]):
                cell[j], gb[j] = demand, demand / s.cell_se[j]
                cells[k] -= cell[j]
                widths[k] -= gb[j]
                served_fallback.append(j)
        for j in sorted(served_fallback, key=lambda j: -s.cell_se[j]):
            k = p.cell_index[j]
            gb[j] += max(0., widths[k])
            widths[k] = 0.

        notes = {}
        if arrivals:
            sub, links = pending_snapshot(s, plan, arrivals)
            added, notes = initial(sub, self.method, self.cfg)
            relay[links], cell[arrivals], rb[links], gb[arrivals] = added
        admitted = initial_pue = initial_gnb = 0
        for j in arrivals:
            rid = p.request_ids[j]
            c = Connection(p.time_s)
            self.active[rid] = self.connections[rid] = c
            h = next((h for h in s.links[j] if relay[h] > s.minimum / 2), None)
            if h is not None:
                c.mode, c.pue = "pue", int(p.pue_ids[p.link_pue[h]])
                c.relay, c.rb = relay[h], rb[h]
                initial_pue += 1
            elif cell[j] > TOL:
                c.mode = "gnb"
                initial_gnb += 1
            admitted += c.mode != "blocked"

        for j, rid in enumerate(p.request_ids):
            c = self.active[rid]
            c.last_s = p.time_s
            c.requested_seconds += 1
            c.cell, c.gb = float(cell[j]), float(gb[j])
            c.site = sites[j] if cell[j] > TOL else None
            has_pue = c.mode == "pue"
            c.pue_seconds += has_pue
            c.gnb_seconds += c.mode == "gnb" and cell[j] > TOL
            c.outage_seconds += not has_pue and cell[j] <= TOL
            if has_pue:
                h = next(h for h in s.links[j] if p.pue_ids[p.link_pue[h]] == c.pue)
                c.payment += s.payment(h, c.relay, c.cell)
            else:
                c.payment += s.factor * p.gnb_price_per_gb * c.cell
            c.delivered_gb += s.factor * (c.relay + c.cell)

        metrics = s.evaluate(plan)
        if metrics["violations"] != metrics["violations_demand"]:
            raise RuntimeError(f"{p.scenario_id} {p.time_s} {self.method}: invalid reservation {metrics}")
        up, ug, ubp, ubg = residual(s, plan)
        caps = (s.provider_cap.sum(), p.cell_capacity_mbps.sum(),
                len(up) * s.pc5_mhz, len(ug) * s.gnb_mhz)
        util = {f"{name}_utilization": values.sum() / cap if cap else 0.
                for name, values, cap in zip(("pue_mbps", "gnb_mbps", "pue_mhz", "gnb_mhz"),
                                              (up, ug, ubp, ubg), caps)}
        return dict(**metrics, **notes, **util, runtime_s=perf_counter() - start,
                    arrivals=len(arrivals), admitted=admitted, initial_pue=initial_pue,
                    initial_gnb=initial_gnb, failures=failures, retained_pue=kept,
                    immediate_fallbacks=sum(j in served_fallback for j in fallback
                                            if self.active[p.request_ids[j]].failure_s == p.time_s))


class PersistentScenario:
    def __init__(self, cfg):
        self.states = [PolicyState(method, cfg) for method in cfg["algorithms"]]

    def step(self, snapshot):
        return [(state.method, state.step(snapshot)) for state in self.states]

    def write(self, scenario_id, out, raw):
        pd.DataFrame(raw).to_csv(out / f"{scenario_id}.epochs.csv.gz", index=False)
        sessions = []
        events = []
        for state in self.states:
            for rid, c in state.connections.items():
                sessions.append(dict(algorithm=state.method, request_id=rid, **vars(c)))
            events.extend(dict(algorithm=state.method, **event) for event in state.events)
        pd.DataFrame(sessions).to_csv(out / f"{scenario_id}.sessions.csv.gz", index=False)
        pd.DataFrame(events, columns=["algorithm", "request_id", "time_s", "event", "pue_id",
                                      "pue_service_seconds"]).to_csv(out / f"{scenario_id}.events.csv.gz", index=False)

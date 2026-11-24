"""
AllocationService: calls the FROZEN Phase 4 allocator (GreedyAllocator / MilpAllocator, unchanged) on a
sub-problem of one second: a subset of requests, a filtered link set and RESIDUAL capacities after the
loads reserved by sessions that keep their assignment.

The EpochProblem is assembled exactly like Phase 4's ScenarioInputs._build (same ordering, same sources):
requests sorted by id; cells = sorted covered serving cells; providers = sorted ids with a link; links
sorted by (request, provider); C3 = realized_link_capacity_mbps; C4/C5/prices from Phase 2; gNodeB
price constant. Reservations subtract from C4/C5/C7. With all active requests, the policy-eligible
links and no reservation the result is the Phase 4 problem itself (tested), so the stateless mode
reproduces Phase 4. Every solution is checked with the Phase 4 audit before it is returned.
Only `ineligible_summary` (a Phase 4 reporting diagnostic) is left empty.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.marketplace.allocation import Allocation, audit
from src.marketplace.problem import EpochProblem
from src.mec.inputs import Phase5ScenarioInputs
from src.mec.session import Assignment, Session, SessionState


class AllocationServiceError(RuntimeError):
    pass


@dataclass
class Reservation:
    pue_load: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    cell_load: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @classmethod
    def of(cls, assignments: Iterable[Assignment]) -> "Reservation":
        r = cls()
        for a in assignments:
            if a is None:
                continue
            if a.pue_id is not None:
                r.pue_load[a.pue_id] += a.pue_rate_mbps
            if a.cell_id is not None:
                r.cell_load[a.cell_id] += a.gnb_rate_mbps
        return r


class AllocationService:
    def __init__(self, inputs: Phase5ScenarioInputs, allocator, rate_eps: float = 1e-6, tol: float = 1e-6,
                 continuity_weight: float = 1.0):
        self.inputs, self.allocator, self.eps, self.tol, self.cw = inputs, allocator, rate_eps, tol, continuity_weight
        self.pg = float(inputs.gnb_price.iloc[0])
        self._rs = {float(t): g.set_index("request_id") for t, g in inputs.rs.groupby("time_s")}
        self.calls = 0

    def build_problem(self, t: float, request_ids: List[str], links: pd.DataFrame,
                      reserved: Optional[Reservation] = None, continuing: Iterable[str] = ()) -> EpochProblem:
        t = float(t)
        reserved = reserved or Reservation()
        cont = set(continuing)
        rids = sorted(request_ids)
        jmap = {r: j for j, r in enumerate(rids)}
        rs = self._rs[t].loc[rids]
        req = self.inputs.req.loc[rids]
        covered = rs.gnb_covered.to_numpy(bool)
        cells = sorted(rs.loc[rs.gnb_covered.astype(bool), "gnb_id"].unique().tolist())
        kmap = {c: k for k, c in enumerate(cells)}
        cap = rs[rs.gnb_covered.astype(bool)].groupby("gnb_id").gnb_available_capacity_mbps.first()
        cell_cap = np.array([float(cap[c]) - reserved.cell_load.get(c, 0.0) for c in cells])
        if np.any(cell_cap < -self.tol):
            raise AllocationServiceError(f"t={t}: reserved gNodeB load exceeds cell capacity")
        cell_index = np.array([kmap[c] if cv else -1 for c, cv in zip(rs.gnb_id, covered)], dtype=int)
        weight = req.priority_weight.to_numpy(float) * np.array([self.cw if r in cont else 1.0 for r in rids])
        el = links[links.request_id.isin(jmap)].sort_values(["request_id", "pue_id"]) if len(links) else links
        if len(el):
            if el.duplicated(["request_id", "pue_id"]).any():
                raise AllocationServiceError(f"t={t}: duplicate links")
            pids = np.array(sorted(el.pue_id.astype(int).unique()))
            imap = {p: i for i, p in enumerate(pids)}
            ph2 = self.inputs.pue.loc[[(t, p) for p in pids]]
            A = ph2.access_capacity_mbps.to_numpy(float) - np.array([reserved.pue_load.get(p, 0.0) for p in pids])
            C = ph2.backhaul_capacity_mbps.to_numpy(float) - np.array([reserved.pue_load.get(p, 0.0) for p in pids])
            if np.any(A < -self.tol) or np.any(C < -self.tol):
                raise AllocationServiceError(f"t={t}: reserved provider load exceeds C4/C5")
            price = ph2.price_per_gb.to_numpy(float)
            B = el.realized_link_capacity_mbps.to_numpy(float)
            if not np.all(B > 0):
                raise AllocationServiceError(f"t={t}: link with non-positive realized capacity")
            lr = np.array([jmap[r] for r in el.request_id])
            lp = np.array([imap[int(p)] for p in el.pue_id])
            ps = el.p_survive.to_numpy(float)
            A, C = np.maximum(A, 0.0), np.maximum(C, 0.0)
        else:
            pids, A, C, price = np.empty(0, int), np.empty(0), np.empty(0), np.empty(0)
            lr = lp = np.empty(0, int)
            B = ps = np.empty(0)
        return EpochProblem(self.inputs.scenario_id, t, self.inputs.dt_s, rids, rs.cue_id.to_numpy(),
                            rs.required_rate_mbps.to_numpy(float), weight, req.max_price_per_gb.to_numpy(float),
                            req.request_time_s.to_numpy(float), covered, cell_index, cells,
                            np.maximum(cell_cap, 0.0), self.pg, pids, A, C, price, lr, lp, B, ps, {})

    def to_assignments(self, p: EpochProblem, a: Allocation) -> Dict[str, Optional[Assignment]]:
        out = {}
        for j, rid in enumerate(p.request_ids):
            if not a.admitted[j]:
                out[rid] = None
                continue
            pue = int(p.pue_ids[a.pue_index[j]]) if a.pue_index[j] >= 0 else None
            g = float(a.gnb_rate_mbps[j])
            cell = p.cell_ids[p.cell_index[j]] if g > self.eps else None
            out[rid] = Assignment(pue, float(a.pue_rate_mbps[j]) if pue is not None else 0.0, g if cell else 0.0, cell)
        return out

    def solve(self, t, request_ids, links, reserved=None, continuing=()) -> Tuple[Dict[str, Optional[Assignment]],
                                                                                   Optional[EpochProblem],
                                                                                   Optional[Allocation]]:
        if not request_ids:
            return {}, None, None
        p = self.build_problem(t, request_ids, links, reserved, continuing)
        a = self.allocator.solve(p)
        self.calls += 1
        v = audit(p, a, self.tol)
        if v:
            raise AllocationServiceError(f"t={t}: Phase 4 audit failed {v}")
        return self.to_assignments(p, a), p, a

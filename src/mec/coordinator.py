"""
MecCoordinator: the per-second MEC control loop (Design Revision 2, 6.1), primary configuration:
proactive switching OFF, reallocation delay 0, incremental re-optimisation, arrival policy `block`.

Per second t:
  1 snapshot          StateManager.snapshot (presence, status, E_t, O_t, Phase 4 problem)
  2 detect            EventMonitor.detect (observable-at-t only)
  3 decide            FailureDetector.decide (one action per session + rebalance groups)
  4 terminate         window end -> COMPLETED; consumer gone -> CUE_DEPARTED
  5 release           hard failures release the whole assignment -> reallocation set R
  6 joint rebalance   one Phase 4 call for all capacity-affected sessions (current provider only)
  7 keep              unaffected sessions keep provider and rates (gNodeB share follows handovers)
  8 allocate          one Phase 4 call for R + interrupted + new arrivals over E_t, residual capacity
                      admitted -> ACTIVE; R/interrupted not admitted -> INTERRUPTED; arrivals -> BLOCKED
  9 ledger            one row per non-final session
 10 interruption      INTERRUPTED longer than max_interruption_s -> DROPPED
 11 audit             full Phase 4 audit of ALL assignments at t + session invariants
After the last trace second: open sessions -> COMPLETED (window ended exactly at trace end) or CENSORED.

Failure safety: allocation results are computed first and applied only after the Phase 4 audit of
that sub-problem passed; if a solve fails, released sessions become INTERRUPTED and arrivals BLOCKED,
so the state stays consistent (tested with a failing allocator).
"""
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from src.marketplace.allocation import FAMILIES, audit, from_link_rates
from src.mec.accounting import Ledger
from src.mec.allocation_service import AllocationService, AllocationServiceError, Reservation
from src.mec.events import EventMonitor
from src.mec.failure import Action, FailureDetector
from src.mec.inputs import Phase5ScenarioInputs
from src.mec.reallocation import ReallocationEngine
from src.mec.session import SessionManager, SessionState
from src.mec.state import StateManager

INVARIANTS = ["continued_link_not_operational", "target_link_not_policy_eligible",
              "rebalanced_link_not_operational", "consumer_multiple_sessions", "active_without_request_second",
              "request_second_without_session", "ledger_rows_mismatch"]


class MecCoordinator:
    def __init__(self, inputs: Phase5ScenarioInputs, state: StateManager, channel: str, policy_tag: str,
                 allocator, max_interruption_s: int, rate_eps: float = 1e-6, tol: float = 1e-6,
                 continuity_weight: float = 1.0, proactive: bool = False, reallocation_delay_s: int = 0):
        if proactive:
            raise NotImplementedError("proactive switching is not implemented (disabled in the primary system)")
        if reallocation_delay_s != 0:
            raise NotImplementedError("only reallocation_delay_s = 0 is implemented")
        self.inputs, self.state, self.channel, self.tag = inputs, state, channel, policy_tag
        self.eps, self.tol = rate_eps, tol
        self.sm = SessionManager(max_interruption_s, rate_eps, tol)
        self.monitor = EventMonitor(state, channel, policy_tag, rate_eps, tol)
        self.fd = FailureDetector()
        self.service = AllocationService(inputs, allocator, rate_eps, tol, continuity_weight)
        self.engine = ReallocationEngine(self.service, policy_tag)
        price = inputs.pue["price_per_gb"]
        self.ledger = Ledger(self.service.pg, lambda t, p: float(price.loc[(t, p)]), inputs.dt_s, tol)
        self.arrivals = defaultdict(list)
        for rid, w in state.windows.items():
            if w.last_active_s >= w.start_s:
                self.arrivals[w.start_s].append(rid)
        self.req = inputs.req
        self.event_rows: List[Dict] = []
        self.audit_rows: List[Dict] = []
        self.errors: List[str] = []
        self.counters = Counter()

    # ------------------------------------------------------------------ control loop
    def run(self):
        for t in self.state.mob.times:
            self.step(float(t))
        self.finish()
        return self

    def step(self, t: float):
        snap = self.state.snapshot(self.channel, self.tag, t)
        sm = self.sm
        nonfinal = sm.nonfinal()
        det = self.monitor.detect(t, nonfinal)
        self.event_rows += [{"time_s": e.time_s, "event": e.type.value, "severity": e.severity.name.lower(),
                             "request_id": e.request_id, "pue_id": e.pue_id, "cell_id": e.cell_id, "detail": e.detail}
                            for e in det.events]
        dec, groups = self.fd.decide(det, [s.request_id for s in nonfinal])
        targets, rebalanced = [], []
        # 4 terminate, 5 release
        for rid, d in dec.items():
            if d.action is Action.TERMINATE:
                sm.terminate(rid, t, d.terminal_cause)
        R = []
        for rid, d in dec.items():
            if d.action is Action.RELEASE:
                sm.release(rid, t, d.cause.type.value)
                R.append(rid)
        waiting = [s.request_id for s in sm.nonfinal() if s.state is SessionState.INTERRUPTED]
        # 7 keep: gNodeB share follows the consumer's serving cell
        for rid, d in dec.items():
            s = sm.sessions[rid]
            if d.action is Action.KEEP and s.state is SessionState.ACTIVE and s.assignment.cell_id is not None:
                _, cell, _ = self.monitor.coverage.of(t, rid)
                if cell != s.assignment.cell_id:
                    sm.handover(rid, t, cell)
        # 6 joint rebalance (one call for all groups)
        gids = [r for g in groups for r in g]
        if gids:
            self.counters["rebalance_calls"] += 1
            outside = Reservation.of(s.assignment for s in sm.active() if s.request_id not in set(gids))
            try:
                prop = self.engine.joint_rebalance(snap, [sm.sessions[r] for r in gids], outside)
            except AllocationServiceError as e:
                self.errors.append(f"t={t} rebalance: {e}")
                prop = {}
            for r in gids:
                a, s = prop.get(r), sm.sessions[r]
                if a is not None and a.pue_id == s.assignment.pue_id:
                    sm.rebalance(r, t, a)
                    if a.pue_id is not None:
                        rebalanced.append((r, a.pue_id))
                elif a is not None:                       # moved from its provider to gNodeB only
                    sm.release(r, t, "rebalance_to_gnb")
                    sm.reassign(r, t, a)
                else:
                    sm.release(r, t, "rebalance_failed")
                    R.append(r)
        # 8 allocate: released + waiting + arrivals over E_t, residual after every kept assignment
        new = []
        for rid in self.arrivals.get(t, []):
            w = self.state.windows[rid]
            q = self.req.loc[rid]
            sm.open(t, rid, w.cue_id, q.required_rate_mbps, q.priority_weight, q.max_price_per_gb, w.start_s,
                    w.last_active_s)
            new.append(rid)
        cand = R + waiting + new
        if cand:
            self.counters["allocate_calls"] += 1
            reserved = Reservation.of(s.assignment for s in sm.active())
            try:
                prop = self.engine.reallocate(snap, cand, continuing=R + waiting, reserved=reserved)
            except AllocationServiceError as e:
                self.errors.append(f"t={t} allocate: {e}")
                prop = {}
            for r in cand:
                a, s = prop.get(r), sm.sessions[r]
                if s.state is SessionState.REQUESTED:
                    if a is not None:
                        sm.admit(r, t, a)
                    else:
                        sm.block(r, t, "no_feasible_assignment")
                elif a is not None:
                    sm.reassign(r, t, a)
                else:
                    sm.interrupt(r, t, "no_feasible_assignment")
                if a is not None and a.pue_id is not None:
                    targets.append((r, a.pue_id))
        # 9 ledger, 10 interruption tolerance, 11 audit
        n_rows_before = len(self.ledger.rows)
        self.ledger.record_second(t, sm.nonfinal())
        n_rows = len(self.ledger.rows) - n_rows_before
        n_nonfinal = len(sm.nonfinal())
        dropped = sm.end_of_second(t)
        self.counters["dropped"] += len(dropped)
        self._audit(t, snap, targets, rebalanced, n_rows, n_nonfinal)

    def finish(self):
        for e in self.monitor.end_of_trace_events(self.sm.nonfinal()):
            self.event_rows.append({"time_s": e.time_s, "event": e.type.value, "severity": "terminal",
                                    "request_id": e.request_id, "pue_id": None, "cell_id": None, "detail": e.detail})
            cause = "window_end" if e.type.value == "request_window_end" else e.type.value
            self.sm.terminate(e.request_id, e.time_s, cause)

    # ------------------------------------------------------------------ audit
    def _audit(self, t, snap, targets, rebalanced, n_rows, n_nonfinal):
        sm = self.sm
        active_rs = snap.active_requests
        act = {s.request_id: s for s in sm.active()}
        inv = dict.fromkeys(INVARIANTS, 0)
        cur = {(r, s.assignment.pue_id) for r, s in act.items() if s.assignment.pue_id is not None}
        inv["continued_link_not_operational"] = len(cur - set(snap.operational))
        inv["target_link_not_policy_eligible"] = len(set(targets) - set(snap.policy_eligible))
        inv["rebalanced_link_not_operational"] = len(set(rebalanced) - set(snap.operational))
        cues = Counter(s.cue_id for s in sm.nonfinal())
        inv["consumer_multiple_sessions"] = sum(1 for c in cues.values() if c > 1)
        inv["active_without_request_second"] = len(set(act) - set(active_rs))
        inv["request_second_without_session"] = len(set(active_rs) - set(sm.sessions))
        inv["ledger_rows_mismatch"] = int(n_rows != n_nonfinal)
        fam = dict.fromkeys(FAMILIES, 0)
        if active_rs:
            L = snap.links
            mask = L[f"feasible_{self.tag}"].astype(bool).to_numpy() | np.array(
                [k in cur for k in zip(L.request_id, L.pue_id.astype(int))], dtype=bool) if len(L) else []
            p = self.service.build_problem(t, active_rs, L[mask] if len(L) else L)
            link_rates, gnb = np.zeros(p.n_links), np.zeros(p.n_requests)
            admitted = np.zeros(p.n_requests, bool)
            for j, r in enumerate(p.request_ids):
                s = act.get(r)
                if s is None:
                    continue
                admitted[j] = True
                gnb[j] = s.assignment.gnb_rate_mbps
                if s.assignment.pue_id is not None:
                    ls = [l for l in p.links_of_request(j) if int(p.pue_ids[p.link_pue[l]]) == s.assignment.pue_id]
                    if len(ls) != 1:
                        inv["continued_link_not_operational"] += 1
                        continue
                    link_rates[ls[0]] = s.assignment.pue_rate_mbps
            a = from_link_rates(p, "phase5", admitted, link_rates, gnb, self.eps)
            for f, v in audit(p, a, self.tol).items():
                fam[f] = len(v)
        self.audit_rows.append({"time_s": t, "active_requests": len(active_rs), "active_sessions": len(act),
                                "interrupted_sessions": sum(s.state is SessionState.INTERRUPTED for s in sm.nonfinal()),
                                **fam, **inv, "violations": sum(fam.values()) + sum(inv.values())})

    # ------------------------------------------------------------------ results
    def results(self) -> Dict[str, pd.DataFrame]:
        sessions = self.sm.sessions
        summary = self.ledger.session_summary(sessions)
        return {"ledger": self.ledger.frame(), "sessions": summary, "events": pd.DataFrame(self.event_rows),
                "audit": pd.DataFrame(self.audit_rows),
                "timeline": pd.DataFrame([{"request_id": r, "time_s": t, "action": a, "detail": d}
                                          for r, s in sorted(sessions.items()) for t, a, d in s.timeline])}

    def report(self) -> Dict:
        res = self.results()
        s, au = res["sessions"], res["audit"]
        ledger_check = self.ledger.verify(self.sm.sessions)
        return {"scenario_id": self.inputs.scenario_id, "channel": self.channel, "policy_tag": self.tag,
                "allocator": getattr(self.service.allocator, "name", type(self.service.allocator).__name__),
                "max_interruption_s": self.sm.max_interruption_s, "sessions": len(s),
                "final_states": s.final_state.value_counts().to_dict() if len(s) else {},
                "reallocations": int(s.n_reallocations.sum()) if len(s) else 0,
                "rebalances": int(s.n_rebalances.sum()) if len(s) else 0,
                "recoveries": int(s.n_recoveries.sum()) if len(s) else 0,
                "interruptions": int(s.n_interruptions.sum()) if len(s) else 0,
                "events": res["events"].event.value_counts().to_dict() if len(res["events"]) else {},
                "allocator_calls": self.service.calls, "errors": list(self.errors),
                "audit_seconds": len(au), "audit_violations": int(au.violations.sum()) if len(au) else 0,
                "ledger_check": ledger_check,
                "payment_total": float(res["ledger"].payment.sum()) if len(res["ledger"]) else 0.0}

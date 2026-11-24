"""
Session ledger: per-second service and payment records, with conservation checks.

One row per non-final session per second (after the second's decisions): ACTIVE rows carry the
assignment (pue + gNodeB = demand, C6), INTERRUPTED rows carry zero rates. Payments use the frozen
Phase 4 PostedPricePolicy (src.common.units conversion), never a new formula.
"""
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.common.units import MBIT_PER_GB
from src.marketplace.pricing import PostedPricePolicy
from src.mec.session import FINAL, Session, SessionState


class LedgerError(RuntimeError):
    pass


class Ledger:
    def __init__(self, gnb_price_per_gb: float, pue_price: Callable[[float, int], float], dt_s: float = 1.0,
                 tol: float = 1e-6, pricing: PostedPricePolicy = PostedPricePolicy()):
        self.pg, self.pue_price, self.dt, self.tol, self.pricing = gnb_price_per_gb, pue_price, dt_s, tol, pricing
        self.rows: List[Dict] = []
        self._seen = set()

    def record_second(self, t: float, sessions: List[Session]):
        for s in sessions:
            if s.is_final:
                continue
            if s.pending:
                raise LedgerError(f"{s.request_id} pending at t={t}")
            key = (float(t), s.request_id)
            if key in self._seen:
                raise LedgerError(f"duplicate ledger row {key}")
            self._seen.add(key)
            if s.state is SessionState.ACTIVE:
                a = s.assignment
                if a is None or abs(a.pue_rate_mbps + a.gnb_rate_mbps - s.demand_mbps) > self.tol:
                    raise LedgerError(f"{s.request_id} active at t={t} without full service")
                r, g, p, cell = a.pue_rate_mbps, a.gnb_rate_mbps, a.pue_id, a.cell_id
            elif s.state is SessionState.INTERRUPTED:
                if s.assignment is not None:
                    raise LedgerError(f"{s.request_id} interrupted with an assignment")
                r, g, p, cell = 0.0, 0.0, None, None
            else:
                raise LedgerError(f"{s.request_id}: unexpected state {s.state} in ledger")
            price = self.pue_price(float(t), p) if p is not None else np.nan
            pue_rev, gnb_rev, pay = self.pricing.payments(r, price, g, self.pg, self.dt)
            self.rows.append({"time_s": float(t), "request_id": s.request_id, "state": s.state.value,
                              "pue_id": p, "pue_rate_mbps": r, "gnb_id": cell, "gnb_rate_mbps": g,
                              "required_rate_mbps": s.demand_mbps, "pue_price_per_gb": price,
                              "pue_revenue": float(pue_rev), "gnb_revenue": float(gnb_rev), "payment": float(pay)})

    def frame(self) -> pd.DataFrame:
        cols = ["time_s", "request_id", "state", "pue_id", "pue_rate_mbps", "gnb_id", "gnb_rate_mbps",
                "required_rate_mbps", "pue_price_per_gb", "pue_revenue", "gnb_revenue", "payment"]
        return pd.DataFrame(self.rows, columns=cols)

    def session_summary(self, sessions: Dict[str, Session]) -> pd.DataFrame:
        led = self.frame()
        g = led.groupby("request_id") if len(led) else None
        out = []
        for rid, s in sorted(sessions.items()):
            if not s.is_final:
                raise LedgerError(f"{rid} not final at summary time")
            r = led[led.request_id == rid] if g is not None and rid in g.groups else led.iloc[0:0]
            delivered = float((r.pue_rate_mbps + r.gnb_rate_mbps).sum() * self.dt)
            required = float(s.demand_mbps * len(r) * self.dt)
            out.append({"request_id": rid, "cue_id": s.cue_id, "final_state": s.state.value,
                        "start_s": s.start_s, "final_s": s.final_s, "ledger_seconds": len(r),
                        "active_seconds": int((r.state == "active").sum()),
                        "interrupted_seconds": int((r.state == "interrupted").sum()),
                        "delivered_mbit": delivered, "required_mbit": required,
                        "delivered_ratio": delivered / required if required > 0 else np.nan,
                        "delivered_gb": delivered / MBIT_PER_GB,
                        "pue_mbit": float(r.pue_rate_mbps.sum() * self.dt),
                        "payment": float(r.payment.sum()), "pue_revenue": float(r.pue_revenue.sum()),
                        "gnb_revenue": float(r.gnb_revenue.sum()),
                        "n_reallocations": s.n_reallocations, "n_rebalances": s.n_rebalances,
                        "n_recoveries": s.n_recoveries, "n_interruptions": s.n_interruptions,
                        "n_proactive_switches": s.n_proactive_switches,
                        "n_providers_used": len({p for _, p in s.providers if p is not None})})
        return pd.DataFrame(out)

    def verify(self, sessions: Dict[str, Session]) -> Dict[str, int]:
        """Conservation: payment = pue + gNodeB revenue; interrupted rows carry nothing; rows unique;
        no rows for a session after its final second; interrupted seconds match the session counter."""
        led = self.frame()
        problems = {}
        if len(led):
            problems["payment_split"] = int((abs(led.payment - led.pue_revenue - led.gnb_revenue) > 1e-12).sum())
            itr = led[led.state == "interrupted"]
            problems["interrupted_with_service"] = int(((itr.pue_rate_mbps + itr.gnb_rate_mbps) > 0).sum())
            act = led[led.state == "active"]
            problems["active_not_full"] = int((abs(act.pue_rate_mbps + act.gnb_rate_mbps - act.required_rate_mbps)
                                               > self.tol).sum())
            problems["duplicates"] = int(led.duplicated(["time_s", "request_id"]).sum())
            fin = {r: s.final_s for r, s in sessions.items() if s.final_s is not None}
            # terminated at t: no row at t; dropped at the end of t: its (interrupted) row at t is the last
            last_ok = {r: (f if sessions[r].state is SessionState.DROPPED else f - self.dt) for r, f in fin.items()}
            problems["rows_after_final"] = int(sum(t > last_ok[r] for t, r in zip(led.time_s, led.request_id)
                                                   if r in last_ok))
            cnt = led[led.state == "interrupted"].groupby("request_id").size()
            problems["interrupted_count_mismatch"] = int(sum(
                cnt.get(r, 0) != s.interrupted_total_s for r, s in sessions.items()))
        return problems

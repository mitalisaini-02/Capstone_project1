"""
Session state machine and manager (Design Revision 2, section 7).

States: REQUESTED -> ACTIVE | BLOCKED(final)
        ACTIVE <-> INTERRUPTED ; INTERRUPTED -> DROPPED(final) after > max_interruption_s seconds
        ACTIVE | INTERRUPTED -> COMPLETED | CUE_DEPARTED | CENSORED (final)
Within one second a hard failure first RELEASES the assignment (transient flag `pending`), then the
session is either REASSIGNED (ACTIVE, counted as a reallocation) or INTERRUPTED. Same-provider rate
changes are REBALANCES (counted separately). Illegal transitions raise SessionError.

Assignment recording follows Phase 4's phantom-PUE rule: a provider is recorded only if its rate > eps.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SessionError(RuntimeError):
    pass


class SessionState(str, Enum):
    REQUESTED = "requested"
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"
    DROPPED = "dropped"
    COMPLETED = "completed"
    CUE_DEPARTED = "cue_departed"
    CENSORED = "censored"


S = SessionState
FINAL = frozenset({S.BLOCKED, S.DROPPED, S.COMPLETED, S.CUE_DEPARTED, S.CENSORED})
TRANSITIONS = {
    S.REQUESTED: {S.ACTIVE, S.BLOCKED},
    S.ACTIVE: {S.ACTIVE, S.INTERRUPTED, S.COMPLETED, S.CUE_DEPARTED, S.CENSORED},
    S.INTERRUPTED: {S.ACTIVE, S.INTERRUPTED, S.DROPPED, S.COMPLETED, S.CUE_DEPARTED, S.CENSORED},
}
TERMINAL_CAUSE = {"window_end": S.COMPLETED, "cue_departed": S.CUE_DEPARTED, "trace_end": S.CENSORED}


@dataclass(frozen=True)
class Assignment:
    pue_id: Optional[int]
    pue_rate_mbps: float
    gnb_rate_mbps: float
    cell_id: Optional[str]

    def check(self, demand: float, eps: float, tol: float = 1e-6):
        if self.pue_rate_mbps < -tol or self.gnb_rate_mbps < -tol:
            raise SessionError(f"negative rate in {self}")
        if (self.pue_id is not None) != (self.pue_rate_mbps > eps):
            raise SessionError(f"phantom provider in {self}")
        if (self.cell_id is not None) != (self.gnb_rate_mbps > eps):
            raise SessionError(f"gNodeB share without a cell (or vice versa) in {self}")
        if abs(self.pue_rate_mbps + self.gnb_rate_mbps - demand) > tol:
            raise SessionError(f"assignment {self} does not serve demand {demand} (C6)")


@dataclass
class Session:
    request_id: str
    cue_id: int
    demand_mbps: float
    weight: float
    max_price_per_gb: float
    start_s: float
    last_active_s: float
    state: SessionState = S.REQUESTED
    assignment: Optional[Assignment] = None
    pending: bool = False                       # released this second, awaiting reassignment
    n_reallocations: int = 0
    n_rebalances: int = 0
    n_recoveries: int = 0
    n_interruptions: int = 0
    n_proactive_switches: int = 0
    interrupted_run_s: int = 0
    interrupted_total_s: int = 0
    final_s: Optional[float] = None
    providers: List[Tuple[float, Optional[int]]] = field(default_factory=list)   # (t, provider) on change
    timeline: List[Tuple[float, str, str]] = field(default_factory=list)          # (t, action, detail)

    @property
    def is_final(self) -> bool:
        return self.state in FINAL


class SessionManager:
    def __init__(self, max_interruption_s: int, rate_eps: float = 1e-6, tol: float = 1e-6):
        if not (isinstance(max_interruption_s, int) and max_interruption_s >= 0):
            raise SessionError("max_interruption_s must be a non-negative int")
        self.max_interruption_s, self.eps, self.tol = max_interruption_s, rate_eps, tol
        self.sessions: Dict[str, Session] = {}

    # ---------------- helpers ----------------
    def _get(self, rid) -> Session:
        if rid not in self.sessions:
            raise SessionError(f"unknown session {rid}")
        return self.sessions[rid]

    def _move(self, s: Session, to: SessionState, t: float, action: str, detail: str = ""):
        if s.state in FINAL or to not in TRANSITIONS[s.state]:
            raise SessionError(f"{s.request_id}: illegal transition {s.state.value} -> {to.value} ({action})")
        s.state = to
        s.timeline.append((t, action, detail))
        if to in FINAL:
            s.final_s = t

    def _set(self, s: Session, a: Optional[Assignment], t: float):
        if a is not None:
            a.check(s.demand_mbps, self.eps, self.tol)
        new = a.pue_id if a else None
        if not s.providers or s.providers[-1][1] != new:
            s.providers.append((t, new))
        s.assignment = a

    def nonfinal(self) -> List[Session]:
        return [s for s in self.sessions.values() if not s.is_final]

    def active(self) -> List[Session]:
        return [s for s in self.sessions.values() if s.state is S.ACTIVE and not s.pending]

    # ---------------- lifecycle ----------------
    def open(self, t, request_id, cue_id, demand, weight, max_price, start_s, last_active_s) -> Session:
        if request_id in self.sessions:
            raise SessionError(f"session {request_id} already exists")
        if any(s.cue_id == cue_id for s in self.nonfinal()):
            raise SessionError(f"consumer {cue_id} already has a non-final session")
        if t != start_s:
            raise SessionError(f"{request_id} opened at {t}, request time is {start_s}")
        s = Session(request_id, int(cue_id), float(demand), float(weight), float(max_price), float(start_s),
                    float(last_active_s))
        s.timeline.append((t, "requested", ""))
        self.sessions[request_id] = s
        return s

    def admit(self, rid, t, a: Assignment):
        s = self._get(rid)
        if s.state is not S.REQUESTED:
            raise SessionError(f"{rid}: admit from {s.state.value}")
        self._set(s, a, t)
        self._move(s, S.ACTIVE, t, "admitted", f"pue={a.pue_id}")

    def block(self, rid, t, reason=""):
        self._move(self._get(rid), S.BLOCKED, t, "blocked", reason)

    def release(self, rid, t, reason: str):
        s = self._get(rid)
        if s.state is not S.ACTIVE or s.pending:
            raise SessionError(f"{rid}: release from {s.state.value} (pending={s.pending})")
        s.assignment, s.pending = None, True
        s.timeline.append((t, "released", reason))

    def rebalance(self, rid, t, a: Assignment):
        """Same provider (or still gNodeB-only), possibly new rates."""
        s = self._get(rid)
        if s.state is not S.ACTIVE or s.pending:
            raise SessionError(f"{rid}: rebalance from {s.state.value}")
        if a.pue_id != s.assignment.pue_id:
            raise SessionError(f"{rid}: rebalance must keep provider {s.assignment.pue_id}, got {a.pue_id}")
        changed = (abs(a.pue_rate_mbps - s.assignment.pue_rate_mbps) > self.tol
                   or abs(a.gnb_rate_mbps - s.assignment.gnb_rate_mbps) > self.tol or a.cell_id != s.assignment.cell_id)
        self._set(s, a, t)
        if changed:
            s.n_rebalances += 1
            s.timeline.append((t, "rebalanced", f"pue={a.pue_id} r={a.pue_rate_mbps:.4f} g={a.gnb_rate_mbps:.4f}"))

    def handover(self, rid, t, cell_id: str):
        """Consumer moved to another serving cell; the gNodeB share follows it (not a reallocation)."""
        s = self._get(rid)
        if s.state is not S.ACTIVE or s.pending or s.assignment is None or s.assignment.cell_id is None:
            raise SessionError(f"{rid}: handover without an active gNodeB share")
        a = s.assignment
        self._set(s, Assignment(a.pue_id, a.pue_rate_mbps, a.gnb_rate_mbps, cell_id), t)
        s.timeline.append((t, "handover", f"{a.cell_id}->{cell_id}"))

    def reassign(self, rid, t, a: Assignment):
        """After a release (reallocation, no interruption) or from INTERRUPTED (recovery)."""
        s = self._get(rid)
        if s.state is S.ACTIVE and s.pending:
            self._set(s, a, t)
            s.pending = False
            s.n_reallocations += 1
            self._move(s, S.ACTIVE, t, "reallocated", f"pue={a.pue_id}")
        elif s.state is S.INTERRUPTED:
            self._set(s, a, t)
            s.n_recoveries += 1
            s.interrupted_run_s = 0
            self._move(s, S.ACTIVE, t, "recovered", f"pue={a.pue_id}")
        else:
            raise SessionError(f"{rid}: reassign from {s.state.value} (pending={s.pending})")

    def interrupt(self, rid, t, reason=""):
        s = self._get(rid)
        if s.state is S.ACTIVE and s.pending:
            s.pending = False
            s.n_interruptions += 1
            self._set(s, None, t)
            self._move(s, S.INTERRUPTED, t, "interrupted", reason)
        elif s.state is S.INTERRUPTED:
            pass                                     # still waiting; counted in end_of_second
        else:
            raise SessionError(f"{rid}: interrupt from {s.state.value} (pending={s.pending})")

    def terminate(self, rid, t, cause: str):
        s = self._get(rid)
        if cause not in TERMINAL_CAUSE:
            raise SessionError(f"unknown terminal cause {cause}")
        s.assignment, s.pending = None, False
        self._move(s, TERMINAL_CAUSE[cause], t, "terminated", cause)

    def end_of_second(self, t) -> List[str]:
        """Interruption accounting; returns ids dropped this second. No session may still be pending."""
        dropped = []
        for s in self.nonfinal():
            if s.pending:
                raise SessionError(f"{s.request_id} left pending at the end of t={t}")
            if s.state is S.INTERRUPTED:
                s.interrupted_run_s += 1
                s.interrupted_total_s += 1
                if s.interrupted_run_s > self.max_interruption_s:
                    self._move(s, S.DROPPED, t, "dropped", f"interrupted {s.interrupted_run_s}s")
                    dropped.append(s.request_id)
            elif s.state is S.ACTIVE:
                s.interrupted_run_s = 0
        return dropped

"""
Event detection (Design Revision 2, section 4). Uses only information observable at second t
(StateManager + recorded Phase 2/3 rows); Phase 3 episode ground truth is never read.

Terminal events are raised at the first second AFTER the service window (time_window semantics: the
last active second is served). Sessions still open after the final trace second are closed by
`end_of_trace_events` (TRACE_END = censored if the window continues, else REQUEST_WINDOW_END).

Loads for overload checks count only sessions that keep their assignment this second (sessions with
terminal or hard-failure events are released and do not load any resource).
"""
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.mec.session import Session, SessionState
from src.mec.state import ProviderCondition as PC, RequestCondition as RC, StateManager


class Severity(IntEnum):
    TERMINAL = 0
    HARD = 1
    CAPACITY = 2
    ADVISORY = 3


class EventType(str, Enum):
    REQUEST_WINDOW_END = "request_window_end"
    CUE_DEPARTED = "cue_departed"
    TRACE_END = "trace_end"
    PUE_DEPARTED = "pue_departed"
    PUE_WITHDRAWN = "pue_withdrawn"
    PUE_UNAVAILABLE = "pue_unavailable"
    PUE_OUT_OF_RANGE = "pue_out_of_range"
    LINK_OUTAGE = "link_outage"
    LINK_CAPACITY_DROP = "link_capacity_drop"
    PUE_ACCESS_OVERLOAD = "pue_access_overload"
    PUE_BACKHAUL_OVERLOAD = "pue_backhaul_overload"
    CUE_OUT_OF_COVERAGE = "cue_out_of_coverage"
    GNB_CELL_OVERLOAD = "gnb_cell_overload"
    RELIABILITY_LOW = "reliability_low"


E = EventType
SEVERITY = {E.REQUEST_WINDOW_END: Severity.TERMINAL, E.CUE_DEPARTED: Severity.TERMINAL,
            E.TRACE_END: Severity.TERMINAL,
            E.PUE_DEPARTED: Severity.HARD, E.PUE_WITHDRAWN: Severity.HARD, E.PUE_UNAVAILABLE: Severity.HARD,
            E.PUE_OUT_OF_RANGE: Severity.HARD, E.LINK_OUTAGE: Severity.HARD,
            E.LINK_CAPACITY_DROP: Severity.CAPACITY, E.PUE_ACCESS_OVERLOAD: Severity.CAPACITY,
            E.PUE_BACKHAUL_OVERLOAD: Severity.CAPACITY, E.CUE_OUT_OF_COVERAGE: Severity.CAPACITY,
            E.GNB_CELL_OVERLOAD: Severity.CAPACITY, E.RELIABILITY_LOW: Severity.ADVISORY}
ORDER = {e: i for i, e in enumerate(EventType)}          # deterministic tie-break within a severity
TERMINAL_CAUSE = {E.REQUEST_WINDOW_END: "window_end", E.CUE_DEPARTED: "cue_departed", E.TRACE_END: "trace_end"}
_PROVIDER_EVENT = {PC.PUE_DEPARTED: E.PUE_DEPARTED, PC.PUE_WITHDRAWN: E.PUE_WITHDRAWN,
                   PC.PUE_UNAVAILABLE: E.PUE_UNAVAILABLE, PC.PUE_OUT_OF_RANGE: E.PUE_OUT_OF_RANGE}
_REQUEST_EVENT = {RC.WINDOW_END: E.REQUEST_WINDOW_END, RC.CUE_DEPARTED: E.CUE_DEPARTED, RC.TRACE_END: E.TRACE_END}


@dataclass(frozen=True)
class Event:
    time_s: float
    type: EventType
    request_id: str
    pue_id: Optional[int] = None
    cell_id: Optional[str] = None
    detail: str = ""

    @property
    def severity(self) -> Severity:
        return SEVERITY[self.type]

    @property
    def rank(self) -> Tuple[int, int]:
        return int(self.severity), ORDER[self.type]


@dataclass
class Detection:
    time_s: float
    events: List[Event]
    resources: Dict[str, Set[Tuple[str, object]]]   # session -> {("pue", id), ("cell", id)} it loads at t

    def by_session(self) -> Dict[str, List[Event]]:
        out = defaultdict(list)
        for e in self.events:
            out[e.request_id].append(e)
        return {k: sorted(v, key=lambda e: e.rank) for k, v in out.items()}


class Coverage:
    """gNodeB coverage / serving cell / available cell capacity per (t, request), from Phase 3 request_seconds."""

    def __init__(self, rs):
        self.req = {(float(t), r): (bool(c), (g if bool(c) else None), float(k) if bool(c) else 0.0)
                    for t, r, c, g, k in zip(rs.time_s, rs.request_id, rs.gnb_covered, rs.gnb_id,
                                             rs.gnb_available_capacity_mbps)}
        self.cell = {}
        for (t, _), (c, g, k) in self.req.items():
            if c:
                self.cell[(t, g)] = k

    def of(self, t: float, request_id: str) -> Tuple[bool, Optional[str], float]:
        return self.req.get((float(t), request_id), (False, None, 0.0))

    def cell_capacity(self, t: float, cell_id: str) -> float:
        return self.cell.get((float(t), cell_id), 0.0)


class EventMonitor:
    def __init__(self, state: StateManager, channel: str, policy_tag: str, rate_eps: float = 1e-6,
                 tol: float = 1e-6):
        self.state, self.channel, self.tag, self.eps, self.tol = state, channel, policy_tag, rate_eps, tol
        self.aware = policy_tag in ("primary", "sensitivity")
        self.coverage = Coverage(state.inputs.rs)

    def detect(self, t: float, sessions: Iterable[Session]) -> Detection:
        t = float(t)
        ev: List[Event] = []
        res: Dict[str, Set] = {}
        pue_load, pue_caps, pue_members = defaultdict(float), {}, defaultdict(list)
        cell_load, cell_members = defaultdict(float), defaultdict(list)
        for s in sessions:
            if s.is_final or s.state is SessionState.REQUESTED:
                continue
            rid = s.request_id
            cond = self.state.request_condition(rid, t)
            if cond is not RC.ACTIVE:
                ev.append(Event(t, _REQUEST_EVENT[cond], rid))
                continue
            if s.state is SessionState.INTERRUPTED or s.assignment is None:
                continue
            a, hard, keys = s.assignment, False, set()
            if a.pue_id is not None:
                pc, obs = self.state.classify_provider(self.channel, rid, a.pue_id, t)
                if pc is not PC.LINK_OBSERVED:
                    ev.append(Event(t, _PROVIDER_EVENT[pc], rid, a.pue_id))
                    hard = True
                elif not obs.wireless_ok:
                    ev.append(Event(t, E.LINK_OUTAGE, rid, a.pue_id, detail=obs.first_failure["unaware"]))
                    hard = True
                else:
                    if obs.realized_link_capacity_mbps < a.pue_rate_mbps - self.tol:
                        ev.append(Event(t, E.LINK_CAPACITY_DROP, rid, a.pue_id,
                                        detail=f"C3={obs.realized_link_capacity_mbps:.4f}<r={a.pue_rate_mbps:.4f}"))
                    if self.aware and not (obs.lstm_window_available and obs.reliable[self.tag]):
                        ev.append(Event(t, E.RELIABILITY_LOW, rid, a.pue_id,
                                        detail="insufficient_history" if not obs.lstm_window_available
                                        else f"p_survive={obs.p_survive:.4f}"))
                    pue_load[a.pue_id] += a.pue_rate_mbps
                    pue_caps[a.pue_id] = (obs.access_capacity_mbps, obs.backhaul_capacity_mbps)
                    pue_members[a.pue_id].append(rid)
                    keys.add(("pue", a.pue_id))
            if hard:
                continue
            if a.gnb_rate_mbps > self.eps:
                covered, cell, _ = self.coverage.of(t, rid)
                if not covered:
                    ev.append(Event(t, E.CUE_OUT_OF_COVERAGE, rid, a.pue_id))
                else:
                    cell_load[cell] += a.gnb_rate_mbps
                    cell_members[cell].append(rid)
                    keys.add(("cell", cell))
            res[rid] = keys
        for p, load in pue_load.items():
            A, C = pue_caps[p]
            for kind, cap in ((E.PUE_ACCESS_OVERLOAD, A), (E.PUE_BACKHAUL_OVERLOAD, C)):
                if load > cap + self.tol:
                    ev += [Event(t, kind, r, p, detail=f"load={load:.4f}>cap={cap:.4f}") for r in pue_members[p]]
        for k, load in cell_load.items():
            cap = self.coverage.cell_capacity(t, k)
            if load > cap + self.tol:
                ev += [Event(t, E.GNB_CELL_OVERLOAD, r, cell_id=k, detail=f"load={load:.4f}>cap={cap:.4f}")
                       for r in cell_members[k]]
        return Detection(t, ev, res)

    def end_of_trace_events(self, sessions: Iterable[Session]) -> List[Event]:
        """Closes sessions still open after the last trace second (event time = trace_end + 1 s,
        i.e. the first unobserved second, so the last trace second is served and accounted)."""
        t = self.state.trace_end + 1.0
        out = []
        for s in sessions:
            if s.is_final:
                continue
            w = self.state.windows[s.request_id]
            typ = E.REQUEST_WINDOW_END if w.end_cause is RC.WINDOW_END else _REQUEST_EVENT[w.end_cause]
            out.append(Event(t, typ, s.request_id, detail="end_of_trace"))
        return out

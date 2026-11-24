"""
FailureDetector: one action per session per second, plus joint-rebalance groups (Revision 2, 6.1-6.2).

Action from the most severe event (terminal > hard > capacity > advisory):
  TERMINATE   terminal event              RELEASE    hard failure (whole assignment released)
  REBALANCE   capacity event, or member of an affected group
  KEEP        no event, or advisory only (proactive switching is off in the primary system)

Affected groups: seeds are sessions with a capacity event; a group is closed transitively over shared
resources (provider, serving cell) among sessions that keep their assignment this second. Released and
terminated sessions do not load any resource, so they never join a group.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.mec.events import Detection, Event, Severity, TERMINAL_CAUSE


class Action(str, Enum):
    TERMINATE = "terminate"
    RELEASE = "release"
    REBALANCE = "rebalance"
    KEEP = "keep"


@dataclass
class Decision:
    request_id: str
    action: Action
    cause: Optional[Event] = None
    events: List[Event] = field(default_factory=list)

    @property
    def terminal_cause(self) -> str:
        return TERMINAL_CAUSE[self.cause.type]


class FailureDetector:
    def decide(self, det: Detection, session_ids: List[str]):
        """Returns (decisions for every given session, list of rebalance groups as sorted id lists)."""
        evs = det.by_session()
        dec: Dict[str, Decision] = {}
        for rid in session_ids:
            e = evs.get(rid, [])
            worst = e[0] if e else None
            if worst is None or worst.severity is Severity.ADVISORY:
                a = Action.KEEP
            else:
                a = {Severity.TERMINAL: Action.TERMINATE, Severity.HARD: Action.RELEASE,
                     Severity.CAPACITY: Action.REBALANCE}[worst.severity]
            dec[rid] = Decision(rid, a, worst, e)
        # union-find over resources of sessions that keep their assignment (KEEP or REBALANCE)
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        holders = [r for r in session_ids if dec[r].action in (Action.KEEP, Action.REBALANCE)]
        for r in holders:
            node = ("session", r)
            find(node)
            for k in det.resources.get(r, ()):
                parent[find(node)] = find(k)
        seeds = {find(("session", r)) for r in holders if dec[r].action is Action.REBALANCE}
        groups: Dict[object, List[str]] = {}
        for r in holders:
            root = find(("session", r))
            if root in seeds:
                groups.setdefault(root, []).append(r)
                if dec[r].action is Action.KEEP:
                    dec[r].action = Action.REBALANCE        # pulled in through a shared resource
        return dec, [sorted(g) for g in sorted(groups.values(), key=lambda g: sorted(g)[0])]

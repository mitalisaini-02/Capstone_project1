"""
ReallocationEngine (Revision 2, 6.2 and step 8 of 6.1). Pure computation: returns proposed assignments
and never mutates sessions (the coordinator applies results only after they pass the audit).

joint_rebalance  ONE Phase 4 call for all capacity-affected sessions; each session may use only its
                 CURRENT provider's link, and only if that link is operational (O_t); gNodeB covers the
                 remainder. Capacities are residual after every session outside the group.
reallocate       ONE Phase 4 call for released/interrupted sessions + new arrivals over the
                 policy-eligible links E_t, residual after all sessions that keep an assignment.
Proactive switching is not implemented (disabled in the primary system; Revision 2 section 13 pending).
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.mec.allocation_service import AllocationService, Reservation
from src.mec.session import Assignment, Session
from src.mec.state import NetworkSnapshot


class ReallocationEngine:
    def __init__(self, service: AllocationService, policy_tag: str):
        self.service, self.tag = service, policy_tag

    def joint_rebalance(self, snap: NetworkSnapshot, group: List[Session], reserved: Reservation
                        ) -> Dict[str, Optional[Assignment]]:
        if not group:
            return {}
        cur = {(s.request_id, s.assignment.pue_id) for s in group if s.assignment and s.assignment.pue_id is not None}
        keep = cur & set(snap.operational)
        L = snap.links
        links = L[[k in keep for k in zip(L.request_id, L.pue_id.astype(int))]] if len(L) else L
        res, _, _ = self.service.solve(snap.time_s, [s.request_id for s in group], links, reserved,
                                       continuing=[s.request_id for s in group])
        return res

    def reallocate(self, snap: NetworkSnapshot, request_ids: List[str], continuing: List[str],
                   reserved: Reservation) -> Dict[str, Optional[Assignment]]:
        if not request_ids:
            return {}
        L = snap.links
        links = L[L[f"feasible_{self.tag}"].astype(bool)] if len(L) else L
        res, _, _ = self.service.solve(snap.time_s, request_ids, links, reserved, continuing=continuing)
        return res

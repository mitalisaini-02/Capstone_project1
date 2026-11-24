"""Plain-data description of one allocation epoch (one second)."""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpochProblem:
    scenario_id: str
    time_s: float
    dt_s: float
    # requests (index j)
    request_ids: List[str]
    cue_ids: np.ndarray
    demand_mbps: np.ndarray          # d_j
    weight: np.ndarray               # w_j
    max_price_per_gb: np.ndarray     # pi_j^max
    arrival_s: np.ndarray            # request_time_s (greedy ordering)
    covered: np.ndarray              # cov_j
    cell_index: np.ndarray           # index into cell_ids, -1 if uncovered
    # cells (index k)
    cell_ids: List[str]
    cell_capacity_mbps: np.ndarray   # G_k (available)
    gnb_price_per_gb: float
    # providers (index i)
    pue_ids: np.ndarray
    access_mbps: np.ndarray          # A_i  (C4)
    backhaul_mbps: np.ndarray        # C_i  (C5)
    pue_price_per_gb: np.ndarray     # pi_i
    # eligible links (index l): (request j, provider i)
    link_request: np.ndarray
    link_pue: np.ndarray
    link_capacity_mbps: np.ndarray   # B_l = REALIZED link capacity (C3)
    link_p_survive: np.ndarray       # audit only (never used by solvers)
    # diagnostics from Phase 3 for requests without eligible links: j -> {first_failure: count}
    ineligible_summary: Dict[int, Dict[str, int]] = field(default_factory=dict)

    @property
    def n_requests(self) -> int:
        return len(self.request_ids)

    @property
    def n_links(self) -> int:
        return len(self.link_request)

    def links_of_request(self, j: int) -> np.ndarray:
        return np.where(self.link_request == j)[0]

    def links_of_pue(self, i: int) -> np.ndarray:
        return np.where(self.link_pue == i)[0]

    def without_links(self) -> "EpochProblem":
        """Same epoch with no provider links (gNodeB-only baseline)."""
        e = np.empty(0, dtype=int)
        return EpochProblem(self.scenario_id, self.time_s, self.dt_s, self.request_ids, self.cue_ids,
                            self.demand_mbps, self.weight, self.max_price_per_gb, self.arrival_s,
                            self.covered, self.cell_index, self.cell_ids, self.cell_capacity_mbps,
                            self.gnb_price_per_gb, self.pue_ids[:0], self.access_mbps[:0],
                            self.backhaul_mbps[:0], self.pue_price_per_gb[:0], e, e,
                            np.empty(0), np.empty(0), {})

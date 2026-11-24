"""
Provider selection and eligibility: builds one EpochProblem per active second from FILES.

Eligibility is taken verbatim from Phase 3 (`feasible_<theta>` for the chosen channel scenario);
nothing about reliability or the radio is recomputed. Link capacity is the REALIZED C3
(`realized_link_capacity_mbps`); `expected_*` columns are never read (they are not in READ_COLUMNS).
Prices/budgets/weights come from Phase 2. Consistency with Phase 2 is asserted.
"""
import os
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from src.marketplace.problem import EpochProblem

CANDIDATE_COLUMNS = ["channel_scenario", "time_s", "request_id", "cue_id", "pue_id",
                     "realized_link_capacity_mbps", "access_capacity_mbps", "backhaul_capacity_mbps",
                     "p_survive", "fallback", "feasible_primary", "first_failure_primary",
                     "feasible_sensitivity", "first_failure_sensitivity"]
REQUEST_SECOND_COLUMNS = ["time_s", "request_id", "cue_id", "required_rate_mbps", "gnb_covered", "gnb_id",
                          "gnb_available_capacity_mbps"]


class EligibilityError(ValueError):
    pass


class ScenarioInputs:
    """All inputs of one scenario, loaded once and shared by every method / channel / theta."""

    def __init__(self, scenario_id: str, phase3_dir: str, phase2_dir: str, dt_s: float = 1.0):
        self.scenario_id, self.dt_s = scenario_id, float(dt_s)
        self.rs = pd.read_csv(os.path.join(phase3_dir, "request_seconds.csv.gz"), usecols=REQUEST_SECOND_COLUMNS)
        self.cl = pd.read_csv(os.path.join(phase3_dir, "candidate_links.csv.gz"), usecols=CANDIDATE_COLUMNS,
                              low_memory=False)
        req = pd.read_csv(os.path.join(phase2_dir, "cue_requests.csv.gz"),
                          usecols=["request_id", "request_time_s", "max_price_per_gb", "priority_weight", "required_rate_mbps"])
        self.req = req.set_index("request_id")
        pue = pd.read_csv(os.path.join(phase2_dir, "pue_resources.csv.gz"),
                          usecols=["time_s", "pue_id", "price_per_gb", "access_capacity_mbps", "backhaul_capacity_mbps"])
        self.pue = pue.set_index(["time_s", "pue_id"])
        gnb = pd.read_csv(os.path.join(phase2_dir, "gnodeb_resources.csv.gz"), usecols=["time_s", "gnb_id", "price_per_gb"])
        self.gnb_price = gnb.set_index(["time_s", "gnb_id"])["price_per_gb"]
        self._check()

    def _check(self):
        if self.rs.duplicated(["time_s", "request_id"]).any():
            raise EligibilityError("duplicate request-seconds")
        rates = self.req.loc[self.rs.request_id, "required_rate_mbps"].to_numpy()
        if not np.array_equal(rates, self.rs.required_rate_mbps.to_numpy()):
            raise EligibilityError("request rates differ between Phase 2 and Phase 3")
        if self.gnb_price.nunique() != 1:
            raise EligibilityError("gNodeB price is not a single constant (not supported)")

    def problems(self, channel: str, theta_tag: str) -> Iterator[EpochProblem]:
        feas_col, ff_col = f"feasible_{theta_tag}", f"first_failure_{theta_tag}"
        cl = self.cl[self.cl.channel_scenario == channel]
        if cl.empty and len(self.cl):
            raise EligibilityError(f"no rows for channel scenario {channel}")
        elig = cl[cl[feas_col].astype(bool)]
        inel = cl[~cl[feas_col].astype(bool)]
        e_by_t = {t: g for t, g in elig.groupby("time_s")}
        i_by_t = {t: g for t, g in inel.groupby("time_s")}
        pg = float(self.gnb_price.iloc[0])
        for t, rs_t in self.rs.groupby("time_s", sort=True):
            yield self._build(t, rs_t.sort_values("request_id"), e_by_t.get(t), i_by_t.get(t), ff_col, pg)

    def _build(self, t, rs_t, el, inel, ff_col, pg) -> EpochProblem:
        rids = rs_t.request_id.tolist()
        jmap = {r: j for j, r in enumerate(rids)}
        req = self.req.loc[rids]
        covered = rs_t.gnb_covered.to_numpy(bool)
        cells = sorted(rs_t.loc[rs_t.gnb_covered, "gnb_id"].unique().tolist())
        kmap = {c: k for k, c in enumerate(cells)}
        cap = rs_t[rs_t.gnb_covered].groupby("gnb_id").gnb_available_capacity_mbps
        if (cap.nunique() > 1).any():
            raise EligibilityError(f"t={t}: inconsistent gNodeB capacity within a cell")
        cell_cap = np.array([float(cap.first()[c]) for c in cells])
        cell_index = np.array([kmap[c] if cov else -1 for c, cov in zip(rs_t.gnb_id, covered)], dtype=int)
        if el is not None and len(el):
            el = el.sort_values(["request_id", "pue_id"])
            pids = np.array(sorted(el.pue_id.unique()))
            imap = {p: i for i, p in enumerate(pids)}
            per = el.groupby("pue_id")[["access_capacity_mbps", "backhaul_capacity_mbps"]]
            if (per.nunique() > 1).any().any():
                raise EligibilityError(f"t={t}: C4/C5 not unique per PUE")
            first = per.first().loc[pids]
            ph2 = self.pue.loc[[(t, p) for p in pids]]
            if not (np.array_equal(first.access_capacity_mbps.to_numpy(), ph2.access_capacity_mbps.to_numpy())
                    and np.array_equal(first.backhaul_capacity_mbps.to_numpy(), ph2.backhaul_capacity_mbps.to_numpy())):
                raise EligibilityError(f"t={t}: C4/C5 differ from Phase 2")
            price = ph2.price_per_gb.to_numpy(float)
            B = el.realized_link_capacity_mbps.to_numpy(float)
            if not (np.all(np.isfinite(B)) and np.all(B > 0) and np.all(np.isfinite(price)) and np.all(price > 0)):
                raise EligibilityError(f"t={t}: eligible link with non-positive capacity or missing price")
            lr = np.array([jmap[r] for r in el.request_id])
            lp = np.array([imap[p] for p in el.pue_id])
            A, C = first.access_capacity_mbps.to_numpy(float), first.backhaul_capacity_mbps.to_numpy(float)
            ps = el.p_survive.to_numpy(float)
        else:
            pids, A, C, price = np.empty(0, int), np.empty(0), np.empty(0), np.empty(0)
            lr = lp = np.empty(0, int)
            B = ps = np.empty(0)
        summary: Dict[int, Dict[str, int]] = {}
        if inel is not None and len(inel):
            for r, g in inel.groupby("request_id"):
                j = jmap[r]
                if not np.any(lr == j):
                    summary[j] = g[ff_col].value_counts().to_dict()
        return EpochProblem(self.scenario_id, float(t), self.dt_s, rids, rs_t.cue_id.to_numpy(),
                            rs_t.required_rate_mbps.to_numpy(float), req.priority_weight.to_numpy(float),
                            req.max_price_per_gb.to_numpy(float), req.request_time_s.to_numpy(float),
                            covered, cell_index, cells, cell_cap, pg, pids, A, C, price, lr, lp, B, ps, summary)

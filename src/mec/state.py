"""
StateManager: what the MEC coordinator can observe at second t (Design Revision 2, sections 3 and 5).

Sources (all recorded files, nothing regenerated):
  * SUMO trace (src.mobility.trace.RunMobility): vehicle presence, trace end.
  * Phase 2 pue_resources + pue_withdrawals: provider status (same rule as Phase 3, re-derived here
    from Phase 2 and cross-checked against Phase 3's recorded provider_status on every candidate row).
  * Phase 2 cue_requests: request windows. Phase 3 request_seconds: checked against those windows.
  * Phase5ScenarioInputs: candidate rows, policy-eligible set E_t, operational set O_t, Phase 4 problems.

Phase 3 episode ground truth (episodes.csv.gz, end_cause, first_break_*) is NOT read here: it is for
evaluation only. Every decision below uses information observable at t.

Provider decision tree for (request, provider, t); the first match wins:
  t outside [0, trace_end]                 -> error (no such second)
  provider not yet in the trace            -> PUE_NOT_YET_PRESENT
  provider no longer in the trace          -> PUE_DEPARTED           (only possible if t > its last second
                                                                     and that second < trace_end)
  status withdrawn                         -> PUE_WITHDRAWN          (permanent)
  status unavailable / not_registered      -> PUE_UNAVAILABLE        (temporary)
  no candidate row for (t, request, PUE)   -> PUE_OUT_OF_RANGE       (provider present, so out of range)
  row exists                               -> LINK_OBSERVED (+ operational / policy-eligible flags)
A provider still present at trace end is never "departed" (censoring).

Request window (time_window semantics): active on [request_time, last_active] with
  nominal_end = request_time + service_duration - 1, last_active = min(nominal_end, consumer last second).
  End cause: WINDOW_END if nominal_end <= consumer last second (also when it equals trace end);
             otherwise TRACE_END (censored) if the consumer is still present at trace end,
             else CUE_DEPARTED.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.marketplace.problem import EpochProblem
from src.mec.inputs import POLICY_TAGS, Phase5ScenarioInputs
from src.mobility.trace import RunMobility


class StateError(ValueError):
    pass


class ProviderCondition(str, Enum):
    LINK_OBSERVED = "link_observed"
    PUE_NOT_YET_PRESENT = "pue_not_yet_present"
    PUE_DEPARTED = "pue_departed"
    PUE_WITHDRAWN = "pue_withdrawn"
    PUE_UNAVAILABLE = "pue_unavailable"
    PUE_OUT_OF_RANGE = "pue_out_of_range"


class RequestCondition(str, Enum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    WINDOW_END = "window_end"        # service window completed
    CUE_DEPARTED = "cue_departed"    # consumer left the trace before the window ended
    TRACE_END = "trace_end"          # censored: trace ended while the window continued


PROVIDER_STATUSES = ("available", "withdrawn", "unavailable", "not_registered")


@dataclass(frozen=True)
class LinkObservation:
    time_s: float
    request_id: str
    pue_id: int
    provider_status: str
    realized_link_capacity_mbps: float      # C3 (REALIZED)
    access_capacity_mbps: float             # C4
    backhaul_capacity_mbps: float           # C5
    p_survive: float                        # LSTM output recorded by Phase 3 (NaN without a window)
    lstm_window_available: bool
    reliable: Dict[str, bool]               # per aware tag
    wireless_ok: bool
    resource_ok: bool
    operational: bool                       # O_t membership
    policy_eligible: Dict[str, bool]        # E_t membership per policy tag
    first_failure: Dict[str, str]           # per policy tag


@dataclass(frozen=True)
class RequestWindow:
    request_id: str
    cue_id: int
    start_s: float
    nominal_end_s: float
    last_active_s: float
    end_cause: RequestCondition


@dataclass
class NetworkSnapshot:
    time_s: float
    channel: str
    policy_tag: str
    problem: Optional[EpochProblem]                 # Phase 4 problem over E_t (None: no active request)
    active_requests: List[str]
    present_providers: FrozenSet[int]
    provider_status: Dict[int, str]
    policy_eligible: FrozenSet[Tuple[str, int]]     # E_t as (request_id, pue_id)
    operational: FrozenSet[Tuple[str, int]]         # O_t as (request_id, pue_id)
    links: pd.DataFrame                             # every in-range candidate row at t (no expected_*)


def derive_provider_status(pue: pd.DataFrame, withdrawals: pd.DataFrame) -> np.ndarray:
    """available > withdrawn (permanent, from the withdrawal time) > unavailable (registered) > not_registered."""
    wd = dict(zip(withdrawals.pue_id, withdrawals.withdraw_time_s)) if len(withdrawals) else {}
    w_t = pue["pue_id"].map(wd)
    withdrawn = (w_t.notna() & (pue["time_s"] >= w_t)).to_numpy(bool)
    return np.select([pue.available.to_numpy(bool), withdrawn, pue.registered.to_numpy(bool)],
                     ["available", "withdrawn", "unavailable"], default="not_registered")


class StateManager:
    def __init__(self, inputs: Phase5ScenarioInputs, mobility: RunMobility):
        self.inputs, self.mob = inputs, mobility
        self.trace_end = float(np.max(mobility.times))
        self.trace_start = float(np.min(mobility.times))
        p2 = inputs.phase2_dir
        pue = pd.read_csv(f"{p2}/pue_resources.csv.gz", usecols=["time_s", "pue_id", "registered", "available"])
        wd = pd.read_csv(f"{p2}/pue_withdrawals.csv")
        cue = pd.read_csv(f"{p2}/cue_requests.csv.gz",
                          usecols=["request_id", "cue_id", "request_time_s", "service_duration_s"])
        self._status = dict(zip(zip(pue.time_s.astype(float), pue.pue_id.astype(int)),
                                derive_provider_status(pue, wd)))
        self.withdraw_time = {int(p): float(t) for p, t in zip(wd.pue_id, wd.withdraw_time_s)} if len(wd) else {}
        self.windows = {r.request_id: self._window(r) for r in cue.itertuples(index=False)}
        self._check_inputs()
        self._row_index: Dict[str, Dict[Tuple[float, str, int], int]] = {}
        self._by_time: Dict[str, Dict[float, pd.DataFrame]] = {}
        self._problems: Dict[Tuple[str, str], Dict[float, EpochProblem]] = {}

    # ---------------- construction checks ----------------
    def _window(self, r) -> RequestWindow:
        cue_end = self.mob.end_t[int(r.cue_id)]
        nominal = float(r.request_time_s + r.service_duration_s - 1)
        last = min(nominal, cue_end)
        if nominal <= cue_end:
            cause = RequestCondition.WINDOW_END
        elif cue_end >= self.trace_end:
            cause = RequestCondition.TRACE_END
        else:
            cause = RequestCondition.CUE_DEPARTED
        return RequestWindow(r.request_id, int(r.cue_id), float(r.request_time_s), nominal, float(last), cause)

    def _check_inputs(self):
        """Input compatibility: Phase 2 status/presence and Phase 3 rows/windows must agree."""
        prov = set(self.mob.providers)
        for (t, p) in self._status:
            if p not in prov or not self.is_present(p, t):
                raise StateError(f"pue_resources row ({t}, {p}) for a vehicle absent from the SUMO trace")
        cl = self.inputs.cl
        got = np.array([self._status.get((float(t), int(p)), "absent") for t, p in zip(cl.time_s, cl.pue_id)],
                       dtype=object)
        if not np.array_equal(got, cl.provider_status.to_numpy(object)):
            raise StateError("provider status derived from Phase 2 differs from Phase 3 provider_status")
        rs = self.inputs.rs
        span = rs.groupby("request_id").time_s.agg(["min", "max", "size"])
        for rid, s in span.iterrows():
            w = self.windows.get(rid)
            if w is None or (s["min"], s["max"], s["size"]) != (w.start_s, w.last_active_s,
                                                                 w.last_active_s - w.start_s + 1):
                raise StateError(f"request_seconds of {rid} differ from its window")
        nonempty = {r for r, w in self.windows.items() if w.last_active_s >= w.start_s}
        if nonempty != set(span.index):
            raise StateError("requests with a non-empty window are missing from request_seconds (or extra)")

    # ---------------- presence / status ----------------
    def _check_t(self, t: float):
        if not (self.trace_start <= t <= self.trace_end):
            raise StateError(f"t={t} outside the trace [{self.trace_start}, {self.trace_end}]")

    def is_present(self, vehicle_id: int, t: float) -> bool:
        v = int(vehicle_id)
        return v in self.mob.start_t and self.mob.start_t[v] <= t <= self.mob.end_t[v]

    def is_censored(self, vehicle_id: int) -> bool:
        """Still present at trace end: its departure is not observed."""
        return self.mob.end_t[int(vehicle_id)] >= self.trace_end

    def provider_status(self, pue_id: int, t: float) -> str:
        self._check_t(t)
        if not self.is_present(pue_id, t):
            return "absent"
        return self._status[(float(t), int(pue_id))]

    def present_providers(self, t: float) -> FrozenSet[int]:
        self._check_t(t)
        return frozenset(p for p in self.mob.providers if self.is_present(p, t))

    # ---------------- requests ----------------
    def request_condition(self, request_id: str, t: float) -> RequestCondition:
        self._check_t(t)
        w = self.windows[request_id]
        if t < w.start_s:
            return RequestCondition.NOT_STARTED
        if t <= w.last_active_s:
            return RequestCondition.ACTIVE
        return w.end_cause

    # ---------------- links ----------------
    def _index(self, channel: str):
        if channel not in self._row_index:
            rows = self.inputs.channel_rows(channel)
            self._row_index[channel] = {(float(t), r, int(p)): i for i, t, r, p in
                                        zip(rows.index, rows.time_s, rows.request_id, rows.pue_id)}
            self._by_time[channel] = {float(t): g for t, g in rows.groupby("time_s")}
        return self._row_index[channel]

    def link_row(self, channel: str, request_id: str, pue_id: int, t: float) -> Optional[LinkObservation]:
        i = self._index(channel).get((float(t), request_id, int(pue_id)))
        if i is None:
            return None
        r = self.inputs.cl.loc[i]
        return LinkObservation(
            float(t), request_id, int(pue_id), r.provider_status, float(r.realized_link_capacity_mbps),
            float(r.access_capacity_mbps), float(r.backhaul_capacity_mbps), float(r.p_survive),
            bool(r.lstm_window_available), {g: bool(r[f"reliable_{g}"]) for g in ("primary", "sensitivity")},
            bool(r.wireless_ok), bool(r.resource_ok), bool(r.operational),
            {g: bool(r[f"feasible_{g}"]) for g in POLICY_TAGS},
            {g: str(r[f"first_failure_{g}"]) for g in POLICY_TAGS})

    def classify_provider(self, channel: str, request_id: str, pue_id: int, t: float
                          ) -> Tuple[ProviderCondition, Optional[LinkObservation]]:
        """Presence decision tree (module docstring). Observable-at-t information only."""
        self._check_t(t)
        p = int(pue_id)
        if p not in self.mob.start_t or self.mob.roles.get(p) != "provider":
            raise StateError(f"{p} is not a provider in the SUMO trace")
        if t < self.mob.start_t[p]:
            return ProviderCondition.PUE_NOT_YET_PRESENT, None
        if t > self.mob.end_t[p]:
            return ProviderCondition.PUE_DEPARTED, None
        status = self._status[(float(t), p)]
        if status == "withdrawn":
            return ProviderCondition.PUE_WITHDRAWN, None
        if status in ("unavailable", "not_registered"):
            return ProviderCondition.PUE_UNAVAILABLE, None
        if self.request_condition(request_id, t) is not RequestCondition.ACTIVE:
            raise StateError(f"request {request_id} is not active at t={t}")
        obs = self.link_row(channel, request_id, p, t)
        if obs is None:
            return ProviderCondition.PUE_OUT_OF_RANGE, None
        return ProviderCondition.LINK_OBSERVED, obs

    # ---------------- snapshots ----------------
    def _problems_for(self, channel: str, tag: str) -> Dict[float, EpochProblem]:
        key = (channel, tag)
        if key not in self._problems:
            self._problems[key] = {float(p.time_s): p for p in self.inputs.problems(channel, tag)}
        return self._problems[key]

    def snapshot(self, channel: str, policy_tag: str, t: float) -> NetworkSnapshot:
        self._check_t(t)
        if policy_tag not in POLICY_TAGS:
            raise StateError(f"unknown policy tag {policy_tag}")
        self._index(channel)
        rows = self._by_time[channel].get(float(t))
        if rows is None:
            rows = self.inputs.cl.iloc[0:0]
        problem = self._problems_for(channel, policy_tag).get(float(t))
        active = sorted(r for r, w in self.windows.items() if w.start_s <= t <= w.last_active_s)
        if (problem is None and active) or (problem is not None and problem.request_ids != active):
            raise StateError(f"t={t}: Phase 4 problem requests differ from the active request windows")
        pres = self.present_providers(t)
        keys = lambda m: frozenset(zip(rows.request_id[m], rows.pue_id[m].astype(int)))
        return NetworkSnapshot(
            float(t), channel, policy_tag, problem, active, pres,
            {p: self._status[(float(t), p)] for p in pres},
            keys(rows[f"feasible_{policy_tag}"].astype(bool)), keys(rows["operational"].astype(bool)), rows)

    def iter_snapshots(self, channel: str, policy_tag: str) -> Iterator[NetworkSnapshot]:
        for t in self.mob.times:
            yield self.snapshot(channel, policy_tag, float(t))

"""
Phase5ScenarioInputs: the frozen Phase 4 `ScenarioInputs` plus the link flags Phase 5 needs.

Nothing is recomputed from models or radio: every flag is a recorded Phase 3 column. The subclass
reads a few extra columns from the SAME candidate_links file, checks that the rows line up key by key
with the parent's rows, and adds three IN-MEMORY columns (no file is written):

  feasible_unaware / first_failure_unaware   policy-eligible set of the reliability-UNAWARE policy:
        provider_ok AND wireless_ok AND resource_ok (geographic range = the row exists).
        It ignores ONLY the LSTM stages (insufficient_history, reliability_below_threshold).
        First-failure order (Phase 3 order with the two LSTM stages removed):
        provider_unavailable -> wireless_outage_<cause> -> resource_zero -> feasible
  operational                                links on which an EXISTING session may continue:
        provider_ok AND wireless_ok AND resource_ok, for every policy (aware or not).

So for aware tags the parent's `feasible_<theta>` is the policy-eligible set E_t (identical to Phase 4),
and `operational` is O_t with E_t subset of O_t. A working link whose p_survive falls below theta leaves
E_t but stays in O_t. The parent `problems(channel, tag)` works unchanged for tag = "unaware" because
Phase 4 reads `feasible_{tag}` / `first_failure_{tag}`.

Capacity is always the REALIZED C3 (`realized_link_capacity_mbps`); expected_* columns are never read.
"""
import os
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from src.marketplace.eligibility import CANDIDATE_COLUMNS, EligibilityError, ScenarioInputs
from src.marketplace.problem import EpochProblem

KEY_COLUMNS = ["channel_scenario", "time_s", "request_id", "pue_id"]
EXTRA_COLUMNS = ["provider_status", "provider_ok", "lstm_window_available", "reliable_primary",
                 "reliable_sensitivity", "realized_state", "realized_radio_capacity_mbps", "wireless_ok",
                 "building_blocked", "classification_source", "resource_ok"]
AWARE_TAGS = ("primary", "sensitivity")
POLICY_TAGS = AWARE_TAGS + ("unaware",)
assert not any(c.startswith("expected_") for c in CANDIDATE_COLUMNS + EXTRA_COLUMNS + KEY_COLUMNS)


def wireless_cause(fallback, realized_state, building_blocked, classification_source) -> np.ndarray:
    """Phase 3's wireless-outage cause label, from recorded columns (same precedence as Phase 3)."""
    fb = np.asarray(fallback, bool)
    st = np.asarray(realized_state, object)
    blocked = pd.Series(building_blocked).fillna(False).to_numpy(bool)
    geo = np.asarray(classification_source, object) == "geometry"
    return np.select(
        [fb, (st == "NLOS") & blocked & geo, st == "NLOS", st == "NLOSv", st == "LOS"],
        ["wireless_outage_fallback", "wireless_outage_building_blocked", "wireless_outage_nlos_assumed",
         "wireless_outage_vehicle_blocked", "wireless_outage_los"], default="wireless_outage_unknown")


class Phase5ScenarioInputs(ScenarioInputs):
    """Phase 4 inputs + policy-eligible (per tag) and operational link flags. Read-only over files."""

    def __init__(self, scenario_id: str, phase3_dir: str, phase2_dir: str, dt_s: float = 1.0):
        super().__init__(scenario_id, phase3_dir, phase2_dir, dt_s)
        self.phase3_dir, self.phase2_dir = phase3_dir, phase2_dir
        extra = pd.read_csv(os.path.join(phase3_dir, "candidate_links.csv.gz"),
                            usecols=KEY_COLUMNS + EXTRA_COLUMNS, low_memory=False)
        extra = extra[KEY_COLUMNS + EXTRA_COLUMNS]
        if len(extra) != len(self.cl) or not all(
                np.array_equal(extra[c].to_numpy(), self.cl[c].to_numpy()) for c in KEY_COLUMNS):
            raise EligibilityError(f"{scenario_id}: extra candidate columns do not align with Phase 4 rows")
        for c in EXTRA_COLUMNS:
            self.cl[c] = extra[c].to_numpy()
        self._check_and_derive()

    def _check_and_derive(self):
        cl = self.cl
        b = {c: cl[c].astype(bool).to_numpy() for c in
             ("provider_ok", "lstm_window_available", "reliable_primary", "reliable_sensitivity",
              "wireless_ok", "resource_ok")}
        # the recorded stage flags must reproduce the recorded Phase 3/4 eligibility exactly
        if not np.array_equal(b["provider_ok"], (cl.provider_status == "available").to_numpy()):
            raise EligibilityError("provider_ok inconsistent with provider_status")
        if not np.array_equal(b["wireless_ok"], cl.realized_radio_capacity_mbps.to_numpy(float) > 0):
            raise EligibilityError("wireless_ok inconsistent with realized radio capacity")
        physical = b["provider_ok"] & b["wireless_ok"] & b["resource_ok"]
        for tag in AWARE_TAGS:
            rec = physical & b["lstm_window_available"] & b[f"reliable_{tag}"]
            if not np.array_equal(rec, cl[f"feasible_{tag}"].astype(bool).to_numpy()):
                raise EligibilityError(f"feasible_{tag} is not the conjunction of its recorded stage flags")
        cause = wireless_cause(cl.fallback, cl.realized_state, cl.building_blocked, cl.classification_source)
        ff = np.full(len(cl), "feasible", dtype=object)
        ff[~b["resource_ok"]] = "resource_zero"
        ff[~b["wireless_ok"]] = cause[~b["wireless_ok"]]
        ff[~b["provider_ok"]] = "provider_unavailable"
        cl["feasible_unaware"] = physical
        cl["first_failure_unaware"] = ff
        cl["operational"] = physical

    # ---------------- policy-facing views ----------------
    def problems(self, channel: str, theta_tag: str) -> Iterator[EpochProblem]:
        if theta_tag not in POLICY_TAGS:
            raise EligibilityError(f"unknown policy tag {theta_tag}")
        return super().problems(channel, theta_tag)

    def channel_rows(self, channel: str) -> pd.DataFrame:
        if channel not in set(self.cl.channel_scenario):
            if len(self.cl):
                raise EligibilityError(f"no rows for channel scenario {channel}")
        return self.cl[self.cl.channel_scenario == channel]

    def link_sets(self, channel: str, theta_tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """(policy-eligible rows E, operational rows O) for one channel scenario and policy tag."""
        if theta_tag not in POLICY_TAGS:
            raise EligibilityError(f"unknown policy tag {theta_tag}")
        rows = self.channel_rows(channel)
        return rows[rows[f"feasible_{theta_tag}"].astype(bool)], rows[rows["operational"].astype(bool)]

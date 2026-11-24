"""
Phase 5 M1 tests: configuration, Phase5ScenarioInputs (Phase 4 equivalence, unaware / operational
link sets, no expected_* use), StateManager (presence decision tree, request windows, censoring,
snapshots) and package import rules. Unit tests use the SYNTHETIC fixture in tests/mec_fixtures.py;
the real-data tests read one validation scenario and are skipped if the data are absent.
"""
import ast
import copy
import glob
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.marketplace.eligibility import ScenarioInputs
from src.mec.config import (ConfigError, PendingDecisionError, Phase4DependencyError, Phase5Config)
from src.mec.inputs import EXTRA_COLUMNS, KEY_COLUMNS, Phase5ScenarioInputs, wireless_cause
from src.mec.state import (ProviderCondition as PC, RequestCondition as RC, StateError, StateManager,
                           derive_provider_status)
from tests import mec_fixtures as fx

CONFIG = "config/mec/phase5.json"
REAL_SID, REAL_RUN = "C09_withdrawals_some__run_006__r0", "run_006"


def assert_same_problem(tc, a, b):
    tc.assertEqual(set(vars(a)), set(vars(b)))
    for k, va in vars(a).items():
        vb = getattr(b, k)
        if isinstance(va, np.ndarray):
            tc.assertEqual(va.dtype.kind, np.asarray(vb).dtype.kind, k)
            tc.assertTrue(np.array_equal(va, vb, equal_nan=va.dtype.kind == "f"), f"{a.time_s} {k}")
        else:
            tc.assertEqual(va, vb, f"{a.time_s} {k}")


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.p3, cls.p2 = fx.write(cls.tmp.name)
        cls.inp = Phase5ScenarioInputs("FIX__fixture_run__r0", cls.p3, cls.p2)
        cls.sm = StateManager(cls.inp, fx.mobility())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


# ------------------------------------------------------------------ config
class TestConfig(unittest.TestCase):
    def setUp(self):
        self.raw = json.load(open(CONFIG))

    def test_loads_and_phase4_dependency_verified(self):
        cfg = Phase5Config.load(CONFIG)
        r = cfg.verify_phase4_dependency()
        self.assertEqual(r["n_frozen_files"], 12)
        self.assertEqual(cfg.channel_scenarios, ["G", "F1", "F2", "F3"])
        self.assertEqual({p.theta_tag for p in cfg.policies.values()}, {"primary", "sensitivity", "unaware"})

    def test_tampered_pin_is_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["phase4_dependency"]["interface_sha256"]["src/marketplace/milp.py"] = "0" * 64
        with self.assertRaises(Phase4DependencyError):
            Phase5Config(raw).verify_phase4_dependency()
        raw = copy.deepcopy(self.raw)
        raw["phase4_dependency"]["frozen_manifest_sha256"] = "0" * 64
        with self.assertRaises(Phase4DependencyError):
            Phase5Config(raw).verify_phase4_dependency()

    def test_pending_decisions_cannot_be_used(self):
        cfg = Phase5Config(self.raw)
        self.assertIn("proactive_reliability.k_consecutive_s", cfg.pending())
        with self.assertRaises(PendingDecisionError):
            cfg.require("interruption_tolerance", "selected_s")
        self.assertEqual(cfg.require("interruption_tolerance", "candidates_s"), [0, 5, 10, 30, 60])

    def test_strict_validation(self):
        for mutate in (lambda r: r.update(extra_key=1), lambda r: r.pop("dt_s"),
                       lambda r: r.update(channel_scenarios=["G"]),
                       lambda r: r["proactive_reliability"].update(primary=True),
                       lambda r: r["interruption_tolerance"].update(selection_split="test"),
                       lambda r: r["policies"].update(bad={"kind": "aware", "theta_tag": "unaware"}),
                       lambda r: r["reallocation_delay_s"].update(primary=1)):
            raw = copy.deepcopy(self.raw)
            mutate(raw)
            with self.assertRaises(ConfigError):
                Phase5Config(raw)

    def test_thresholds_come_from_phase3_records(self):
        if not os.path.exists("data/simulation/phase3/val/experiment_config.json"):
            self.skipTest("Phase 3 outputs absent")
        self.assertEqual(Phase5Config(self.raw).thresholds("val"), {"primary": 0.62, "sensitivity": 0.65})


# ------------------------------------------------------------------ inputs / link sets
class TestInputs(Fixture):
    def test_aware_problems_identical_to_phase4(self):
        base = ScenarioInputs("FIX__fixture_run__r0", self.p3, self.p2)
        for ch in fx.CHANNELS:
            for tag in ("primary", "sensitivity"):
                a, b = list(self.inp.problems(ch, tag)), list(base.problems(ch, tag))
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    assert_same_problem(self, x, y)

    def test_unaware_ignores_only_lstm_stages(self):
        cl = self.inp.cl
        for tag in ("primary", "sensitivity"):
            self.assertTrue((cl.feasible_unaware | ~cl[f"feasible_{tag}"]).all(), "unaware must be a superset")
            extra = cl[cl.feasible_unaware & ~cl[f"feasible_{tag}"]]
            self.assertTrue(len(extra) > 0)
            self.assertTrue(extra[f"first_failure_{tag}"].isin(["insufficient_history",
                                                                 "reliability_below_threshold"]).all())
            # rows failing a non-LSTM stage first fail identically under both policies
            non_lstm = ~cl[f"first_failure_{tag}"].isin(["insufficient_history", "reliability_below_threshold"])
            self.assertTrue((cl.first_failure_unaware[non_lstm] == cl[f"first_failure_{tag}"][non_lstm]).all())
        self.assertTrue((cl.feasible_unaware == (cl.provider_ok & cl.wireless_ok & cl.resource_ok)).all())

    def test_unaware_still_enforces_provider_wireless_resource(self):
        cl = self.inp.cl
        bad = cl[cl.feasible_unaware & ~(cl.provider_status.eq("available") & (cl.realized_link_capacity_mbps > 0)
                                         & (cl.backhaul_capacity_mbps > 0))]
        self.assertEqual(len(bad), 0)
        g = cl[cl.channel_scenario == "G"]
        ff = dict(zip(zip(g.request_id, g.pue_id, g.time_s), g.first_failure_unaware))
        self.assertEqual(ff[("R0", 1, 3.0)], "wireless_outage_building_blocked")
        self.assertEqual(ff[("R0", 3, 8.0)], "provider_unavailable")
        self.assertEqual(ff[("R1", 5, 10.0)], "resource_zero")
        self.assertEqual(ff[("R0", 1, 0.0)], "feasible")        # no LSTM window: unaware ignores it
        self.assertFalse(self.inp.cl[self.inp.cl.channel_scenario == "F2"].feasible_unaware.any())

    def test_unaware_problems_are_valid_phase4_problems(self):
        for p in self.inp.problems("G", "unaware"):
            self.assertTrue(np.all(p.link_capacity_mbps > 0))
        elig, oper = self.inp.link_sets("G", "unaware")
        self.assertEqual(len(elig), len(oper))

    def test_operational_superset_of_policy_eligible(self):
        for ch in fx.CHANNELS:
            for tag in ("primary", "sensitivity", "unaware"):
                e, o = self.inp.link_sets(ch, tag)
                self.assertTrue(set(e.index) <= set(o.index))

    def test_expected_columns_never_affect_results(self):
        with tempfile.TemporaryDirectory() as d:
            p3, p2 = fx.write(d, expected_value=-999.0)
            other = Phase5ScenarioInputs("FIX__fixture_run__r0", p3, p2)
        self.assertFalse(any(c.startswith("expected_") for c in self.inp.cl.columns))
        self.assertFalse(any(c.startswith("expected_") for c in EXTRA_COLUMNS + KEY_COLUMNS))
        for tag in ("primary", "sensitivity", "unaware"):
            for x, y in zip(self.inp.problems("G", tag), other.problems("G", tag)):
                assert_same_problem(self, x, y)

    def test_wireless_cause_matches_phase3_label(self):
        cl = self.inp.cl
        m = cl.first_failure_primary.str.startswith("wireless_outage")
        got = wireless_cause(cl.fallback[m], cl.realized_state[m], cl.building_blocked[m], cl.classification_source[m])
        self.assertTrue((got == cl.first_failure_primary[m].to_numpy()).all())

    def test_unknown_policy_tag_rejected(self):
        from src.marketplace.eligibility import EligibilityError
        with self.assertRaises(EligibilityError):
            list(self.inp.problems("G", "optimistic"))


# ------------------------------------------------------------------ state / presence tree
class TestPresenceTree(Fixture):
    def c(self, r, p, t, ch="G"):
        return self.sm.classify_provider(ch, r, p, t)[0]

    def test_departure(self):
        self.assertEqual(self.c("R1", 2, 10.0), PC.LINK_OBSERVED)
        self.assertEqual(self.c("R1", 2, 11.0), PC.PUE_DEPARTED)
        self.assertFalse(self.sm.is_censored(2))
        self.assertNotIn(2, self.sm.present_providers(11.0))

    def test_withdrawal_is_permanent_and_precedes_range(self):
        self.assertEqual(self.c("R0", 3, 7.0), PC.LINK_OBSERVED)
        for t in (8.0, 9.0):
            self.assertEqual(self.c("R0", 3, t), PC.PUE_WITHDRAWN)
        self.assertEqual(self.c("R2", 3, 6.0), PC.PUE_OUT_OF_RANGE)     # before withdrawal: out of range
        self.assertEqual(self.c("R2", 3, 7.0), PC.PUE_OUT_OF_RANGE)
        self.assertEqual(self.c("R2", 3, 15.0), PC.PUE_WITHDRAWN)
        self.assertEqual(self.sm.provider_status(3, 20.0), "withdrawn")

    def test_temporary_unavailability_recovers(self):
        self.assertEqual(self.c("R0", 4, 4.0), PC.LINK_OBSERVED)
        self.assertEqual(self.c("R0", 4, 5.0), PC.PUE_UNAVAILABLE)
        self.assertEqual(self.c("R0", 4, 6.0), PC.PUE_UNAVAILABLE)
        self.assertEqual(self.c("R0", 4, 7.0), PC.LINK_OBSERVED)
        self.assertEqual(self.sm.provider_status(4, 5.0), "unavailable")

    def test_not_yet_present_and_not_registered(self):
        self.assertEqual(self.c("R0", 5, 2.0), PC.PUE_NOT_YET_PRESENT)
        self.assertEqual(self.c("R0", 5, 3.0), PC.PUE_UNAVAILABLE)
        self.assertEqual(self.sm.provider_status(5, 3.0), "not_registered")
        self.assertEqual(self.c("R0", 5, 5.0), PC.LINK_OBSERVED)

    def test_out_of_range(self):
        self.assertEqual(self.c("R0", 1, 4.0), PC.PUE_OUT_OF_RANGE)
        self.assertEqual(self.c("R0", 1, 5.0), PC.LINK_OBSERVED)

    def test_trace_end_is_censoring_not_departure(self):
        self.assertTrue(self.sm.is_censored(1))
        cond, obs = self.sm.classify_provider("G", "R2", 1, 20.0)
        self.assertEqual(cond, PC.LINK_OBSERVED)
        with self.assertRaises(StateError):
            self.sm.classify_provider("G", "R2", 1, 21.0)
        for p in fx.PROVIDERS:
            for t in range(0, fx.T_END + 1):
                if self.sm.is_censored(p):
                    self.assertTrue(self.sm.is_present(p, float(t)) or t < fx.PROVIDERS[p][0])

    def test_inactive_request_rejected(self):
        with self.assertRaises(StateError):
            self.sm.classify_provider("G", "R0", 1, 10.0)

    def test_operational_link_continues_below_threshold(self):
        cond, o = self.sm.classify_provider("G", "R0", 1, 2.0)
        self.assertEqual(cond, PC.LINK_OBSERVED)
        self.assertTrue(o.operational and o.wireless_ok and o.resource_ok)
        self.assertEqual(o.policy_eligible, {"primary": False, "sensitivity": False, "unaware": True})
        self.assertEqual(o.first_failure["primary"], "reliability_below_threshold")
        self.assertGreater(o.realized_link_capacity_mbps, 0)
        _, o6 = self.sm.classify_provider("G", "R0", 1, 6.0)
        self.assertEqual(o6.policy_eligible, {"primary": True, "sensitivity": False, "unaware": True})
        # an outage is not operational for anyone
        _, o3 = self.sm.classify_provider("G", "R0", 1, 3.0)
        self.assertFalse(o3.operational)
        self.assertEqual(o3.policy_eligible, {"primary": False, "sensitivity": False, "unaware": False})

    def test_channel_scenarios_supported(self):
        for ch in fx.CHANNELS:
            cond, o = self.sm.classify_provider(ch, "R0", 1, 5.0)
            self.assertEqual(cond, PC.LINK_OBSERVED)
            self.assertEqual(o.operational, ch != "F2")


class TestRequestWindows(Fixture):
    def test_end_causes(self):
        w = self.sm.windows
        self.assertEqual((w["R0"].last_active_s, w["R0"].end_cause), (9.0, RC.WINDOW_END))
        self.assertEqual((w["R1"].last_active_s, w["R1"].end_cause), (12.0, RC.CUE_DEPARTED))
        self.assertEqual((w["R2"].last_active_s, w["R2"].end_cause), (20.0, RC.TRACE_END))
        self.assertEqual((w["R3"].last_active_s, w["R3"].end_cause), (20.0, RC.WINDOW_END))

    def test_conditions_over_time(self):
        f = self.sm.request_condition
        self.assertEqual(f("R2", 4.0), RC.NOT_STARTED)
        self.assertEqual(f("R0", 9.0), RC.ACTIVE)
        self.assertEqual(f("R0", 10.0), RC.WINDOW_END)
        self.assertEqual(f("R1", 13.0), RC.CUE_DEPARTED)
        self.assertEqual(f("R2", 20.0), RC.ACTIVE)


class TestSnapshots(Fixture):
    def test_snapshot_sets(self):
        for tag in ("primary", "sensitivity", "unaware"):
            for s in self.sm.iter_snapshots("G", tag):
                self.assertTrue(s.policy_eligible <= s.operational)
                if tag == "unaware":
                    self.assertEqual(s.policy_eligible, s.operational)
                self.assertTrue({p for _, p in s.operational} <= s.present_providers)
                if s.problem is not None:
                    self.assertEqual(s.problem.request_ids, s.active_requests)
                    elig = {(s.problem.request_ids[j], int(s.problem.pue_ids[i]))
                            for j, i in zip(s.problem.link_request, s.problem.link_pue)}
                    self.assertEqual(elig, set(s.policy_eligible))
                self.assertFalse(any(c.startswith("expected_") for c in s.links.columns))

    def test_snapshot_example(self):
        s = self.sm.snapshot("G", "primary", 2.0)
        self.assertIn(("R0", 1), s.operational)
        self.assertNotIn(("R0", 1), s.policy_eligible)
        self.assertIn(("R0", 1), self.sm.snapshot("G", "unaware", 2.0).policy_eligible)


class TestInputCompatibility(unittest.TestCase):
    def test_status_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p3, p2 = fx.write(d, tamper="provider_status")
            from src.marketplace.eligibility import EligibilityError
            with self.assertRaises((StateError, EligibilityError)):
                StateManager(Phase5ScenarioInputs("FIX__fixture_run__r0", p3, p2), fx.mobility())

    def test_window_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p3, p2 = fx.write(d, tamper="request_window")
            with self.assertRaises(StateError):
                StateManager(Phase5ScenarioInputs("FIX__fixture_run__r0", p3, p2), fx.mobility())

    def test_no_ground_truth_read(self):
        reads = []
        real = pd.read_csv

        def spy(path, *a, **k):
            reads.append(str(path))
            return real(path, *a, **k)
        with tempfile.TemporaryDirectory() as d, mock.patch("pandas.read_csv", spy):
            p3, p2 = fx.write(d)
            sm = StateManager(Phase5ScenarioInputs("FIX__fixture_run__r0", p3, p2), fx.mobility())
            list(sm.iter_snapshots("G", "primary"))
        self.assertTrue(reads)
        self.assertFalse(any("episodes" in r or "phase3_cache" in r for r in reads))

    def test_derive_provider_status_precedence(self):
        pue = pd.DataFrame({"time_s": [1.0, 2.0, 3.0, 4.0], "pue_id": [7, 7, 7, 8],
                            "registered": [True, True, True, False], "available": [True, False, False, False]})
        wd = pd.DataFrame({"pue_id": [7], "withdraw_time_s": [3.0]})
        self.assertEqual(list(derive_provider_status(pue, wd)), ["available", "unavailable", "withdrawn", "not_registered"])


class TestImportRules(unittest.TestCase):
    FORBIDDEN = ("src.reliability", "src.models", "src.network", "src.features", "src.simulation",
                 "src.datagen", "src.training", "torch")
    ALLOWED = ("src.marketplace", "src.common", "src.mobility.trace", "src.utils", "src.mec")

    def test_mec_imports(self):
        for f in glob.glob("src/mec/*.py"):
            for node in ast.walk(ast.parse(open(f).read())):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                for m in mods:
                    self.assertFalse(m.startswith(self.FORBIDDEN), f"{f} imports {m}")
                    if m.startswith("src."):
                        self.assertTrue(m.startswith(self.ALLOWED), f"{f} imports {m}")

    def test_no_expected_or_ground_truth_identifiers_in_code(self):
        for f in glob.glob("src/mec/*.py"):
            tree = ast.parse(open(f).read())
            code =[n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and "\n" not in n.value]
            self.assertFalse([c for c in code if c.startswith("expected_radio") or c.startswith("expected_outage")
                              or c in ("end_cause", "first_break_s", "episodes.csv.gz")], f)


# ------------------------------------------------------------------ real data (validation split)
@unittest.skipUnless(os.path.exists(f"data/simulation/phase3/val/{REAL_SID}/candidate_links.csv.gz")
                     and os.path.exists(f"data/raw/sumo/{REAL_RUN}/mobility.csv"), "validation data absent")
class TestRealScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.mobility.trace import load_run
        cfg = Phase5Config.load(CONFIG)
        cls.p3, cls.p2 = cfg.phase3_dir("val", REAL_SID), cfg.phase2_dir("val", REAL_SID)
        cls.inp = Phase5ScenarioInputs(REAL_SID, cls.p3, cls.p2)
        cls.sm = StateManager(cls.inp, load_run(REAL_RUN, cfg["inputs"]["raw_mobility_dir"]))

    def test_phase4_equivalence_all_channels(self):
        base = ScenarioInputs(REAL_SID, self.p3, self.p2)
        for ch in ("G", "F1", "F2", "F3"):
            for tag in ("primary", "sensitivity"):
                a, b = list(self.inp.problems(ch, tag)), list(base.problems(ch, tag))
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    assert_same_problem(self, x, y)

    def test_link_set_relations(self):
        cl = self.inp.cl
        for tag in ("primary", "sensitivity"):
            self.assertTrue((cl.feasible_unaware | ~cl[f"feasible_{tag}"]).all())
            ex = cl[cl.feasible_unaware & ~cl[f"feasible_{tag}"]]
            self.assertGreater(len(ex), 0)
            self.assertTrue(ex[f"first_failure_{tag}"].isin(["insufficient_history",
                                                             "reliability_below_threshold"]).all())
        below = cl[cl.operational & cl.lstm_window_available & ~cl.reliable_primary]
        self.assertGreater(len(below), 0, "expected operational links below theta in real data")

    def test_withdrawn_and_departed_classification(self):
        cl = self.inp.cl[self.inp.cl.channel_scenario == "G"]
        w = cl[cl.provider_status == "withdrawn"].iloc[0]
        self.assertEqual(self.sm.classify_provider("G", w.request_id, w.pue_id, w.time_s)[0], PC.PUE_WITHDRAWN)
        mob = self.sm.mob
        departed = [p for p in mob.providers if mob.end_t[p] < self.sm.trace_end]
        self.assertTrue(departed)
        for p in departed:
            self.assertFalse(self.sm.is_censored(p))
        for p in mob.providers:
            if mob.end_t[p] >= self.sm.trace_end:
                self.assertTrue(self.sm.is_censored(p))


if __name__ == "__main__":
    unittest.main()

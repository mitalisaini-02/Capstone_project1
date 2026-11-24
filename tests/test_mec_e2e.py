"""
Phase 5 M3-M5 tests: allocation service (Phase 4 equivalence), joint rebalance, reactive reallocation,
coordinator end-to-end lifecycles on SYNTHETIC scenario variants (tests/mec_fixtures.py), failure safety,
and real validation-data integration (skipped if the data are absent).

Every coordinator run is checked for: 0 per-second audit violations (Phase 4 C1-C9 + phantom + session
invariants), 0 ledger conservation problems, and no allocator errors unless the test injects one.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.marketplace.eligibility import ScenarioInputs
from src.marketplace.greedy import GreedyAllocator
from src.marketplace.milp import MilpAllocator
from src.mec.allocation_service import AllocationService, Reservation
from src.mec.coordinator import MecCoordinator
from src.mec.inputs import Phase5ScenarioInputs
from src.mec.state import StateManager
from tests import mec_fixtures as fx
from tests.test_mec_state import assert_same_problem

SID = "FIX__fixture_run__r0"


def build(**kw):
    d = tempfile.mkdtemp()
    p3, p2 = fx.write(d, **kw)
    inp = Phase5ScenarioInputs(SID, p3, p2)
    return inp, StateManager(inp, fx.mobility())


def run(kw, tag="primary", allocator=None, max_i=5, channel="G"):
    inp, st = build(**kw)
    c = MecCoordinator(inp, st, channel, tag, allocator or MilpAllocator(), max_interruption_s=max_i).run()
    return c


def only(m):
    """range_fn from {request: predicate(p, t)}; requests not listed have no provider in range."""
    return lambda r, p, t: m[r](p, t) if r in m else False


class E2E(unittest.TestCase):
    def check_clean(self, c, errors=0):
        rep = c.report()
        self.assertEqual(rep["audit_violations"], 0, c.results()["audit"].query("violations > 0").to_string())
        self.assertEqual(set(rep["ledger_check"].values()) - {0}, set(), rep["ledger_check"])
        self.assertEqual(len(rep["errors"]), errors, rep["errors"])
        self.assertTrue(all(s.is_final for s in c.sm.sessions.values()))
        return rep

    def led(self, c, rid):
        L = c.results()["ledger"]
        return L[L.request_id == rid].set_index("time_s")

    def actions(self, c, rid):
        return [(t, a) for t, a, _ in c.sm.sessions[rid].timeline]


# ------------------------------------------------------------------ M3: allocation service
class TestAllocationService(unittest.TestCase):
    def test_stateless_problem_equals_phase4(self):
        inp, st = build()
        svc = AllocationService(inp, MilpAllocator())
        for ch in fx.CHANNELS:
            for tag in ("primary", "sensitivity", "unaware"):
                for t in st.mob.times:
                    s = st.snapshot(ch, tag, float(t))
                    if s.problem is None:
                        continue
                    E = s.links[s.links[f"feasible_{tag}"].astype(bool)]
                    p = svc.build_problem(t, s.active_requests, E)
                    p.ineligible_summary = s.problem.ineligible_summary
                    assert_same_problem(self, p, s.problem)

    def test_residual_capacity_and_reservation_guard(self):
        inp, st = build()
        svc = AllocationService(inp, MilpAllocator())
        s = st.snapshot("G", "primary", 7.0)
        E = s.links[s.links.feasible_primary.astype(bool)]
        r = Reservation()
        r.pue_load[1] += 22.0
        r.cell_load["g1"] += 99.0
        p = svc.build_problem(7.0, s.active_requests, E, r)
        i = list(p.pue_ids).index(1)
        self.assertAlmostEqual(p.backhaul_mbps[i], fx.BACKHAUL - 22.0)
        self.assertAlmostEqual(p.cell_capacity_mbps[0], 1.0)
        r.pue_load[1] += 10.0
        from src.mec.allocation_service import AllocationServiceError
        with self.assertRaises(AllocationServiceError):
            svc.build_problem(7.0, s.active_requests, E, r)

    def test_greedy_and_milp_results_pass_audit(self):
        inp, st = build()
        for alloc in (GreedyAllocator(), MilpAllocator()):
            svc = AllocationService(inp, alloc)
            s = st.snapshot("G", "unaware", 7.0)
            res, p, a = svc.solve(7.0, s.active_requests, s.links[s.links.feasible_unaware.astype(bool)])
            self.assertTrue(all(v is not None for v in res.values()))


# ------------------------------------------------------------------ M4/M5: lifecycles
class TestLifecycles(E2E):
    def test_baseline_no_failures(self):
        rep = self.check_clean(run({}))
        self.assertEqual(rep["final_states"], {"completed": 2, "cue_departed": 1, "censored": 1})

    def test_pue_departure_replacement_pue(self):
        c = run({"range_fn": only({"R1": lambda p, t: p == 2 or (p == 4 and t >= 11)})})
        self.check_clean(c)
        L = self.led(c, "R1")
        self.assertEqual(L.loc[10.0, "pue_id"], 2)
        self.assertEqual(L.loc[11.0, "pue_id"], 4)                     # replaced in the same second
        s = c.sm.sessions["R1"]
        self.assertEqual((s.n_reallocations, s.n_interruptions, s.state.value), (1, 0, "cue_departed"))
        ev = c.results()["events"]
        self.assertIn("pue_departed", set(ev[(ev.request_id == "R1") & (ev.time_s == 11.0)].event))

    def test_link_outage_gnb_fallback(self):
        c = run({"range_fn": only({"R1": lambda p, t: p == 1}), "outage": [("R1", 1, 6)]})
        self.check_clean(c)
        L = self.led(c, "R1")
        self.assertEqual(L.loc[5.0, "pue_id"], 1)
        self.assertTrue(pd.isna(L.loc[6.0, "pue_id"]))
        self.assertAlmostEqual(L.loc[6.0, "gnb_rate_mbps"], 5.0)
        self.assertGreater(L.loc[6.0, "payment"], L.loc[5.0, "payment"])     # gNodeB is dearer
        self.assertEqual(c.sm.sessions["R1"].n_reallocations, 1)

    def test_withdrawal_unavailable_out_of_range(self):
        for rng, t_fail, kind in ((lambda p, t: p == 3, 8.0, "pue_withdrawn"),
                                  (lambda p, t: p == 4, 5.0, "pue_unavailable"),
                                  (lambda p, t: p == 1 and t <= 7, 8.0, "pue_out_of_range")):
            c = run({"range_fn": only({"R1": rng})})
            self.check_clean(c)
            ev = c.results()["events"]
            self.assertIn(kind, set(ev[(ev.request_id == "R1") & (ev.time_s == t_fail)].event))
            L = self.led(c, "R1")
            self.assertFalse(pd.isna(L.loc[t_fail - 1, "pue_id"]))
            self.assertTrue(pd.isna(L.loc[t_fail, "pue_id"]))
            self.assertEqual(L.loc[t_fail, "state"], "active")

    def test_interrupted_then_recovered(self):
        kw = {"range_fn": only({"R2": lambda p, t: (p == 2) or (p == 4 and t >= 13)}),
              "uncovered": [("R2", 11), ("R2", 12)]}
        c = run(kw, max_i=5)
        self.check_clean(c)
        L = self.led(c, "R2")
        self.assertEqual(L.loc[10.0, "pue_id"], 2)
        self.assertEqual(list(L.loc[[11.0, 12.0], "state"]), ["interrupted", "interrupted"])
        self.assertEqual(float(L.loc[[11.0, 12.0], "payment"].sum()), 0.0)
        self.assertEqual(L.loc[13.0, "pue_id"], 4)
        s = c.sm.sessions["R2"]
        self.assertEqual((s.n_interruptions, s.n_recoveries, s.interrupted_total_s), (1, 1, 2))

    def test_interrupted_then_dropped(self):
        kw = {"range_fn": only({"R2": lambda p, t: (p == 2) or (p == 4 and t >= 13)}),
              "uncovered": [("R2", 11), ("R2", 12)]}
        c = run(kw, max_i=1)
        self.check_clean(c)
        s = c.sm.sessions["R2"]
        self.assertEqual((s.state.value, s.final_s), ("dropped", 12.0))
        self.assertEqual(self.led(c, "R2").index.max(), 12.0)

    def test_low_reliability_kept_when_link_works(self):
        kw = {"range_fn": only({"R1": lambda p, t: p == 1}), "p_survive": {("R1", 1, 6): 0.5}}
        c = run(kw)
        self.check_clean(c)
        self.assertEqual(self.led(c, "R1").loc[6.0, "pue_id"], 1)
        self.assertEqual(c.sm.sessions["R1"].n_reallocations, 0)
        ev = c.results()["events"]
        self.assertIn("reliability_low", set(ev[ev.time_s == 6.0].event))
        cu = run(kw, tag="unaware")
        self.assertNotIn("reliability_low", set(cu.results()["events"].event))

    def test_new_session_respects_reliability_filter(self):
        kw = {"range_fn": only({"R1": lambda p, t: p == 1}), "p_survive": {("R1", 1, 2): 0.5}}
        aware, unaware = run(kw), run(kw, tag="unaware")
        self.check_clean(aware), self.check_clean(unaware)
        self.assertTrue(pd.isna(self.led(aware, "R1").loc[2.0, "pue_id"]))     # p 0.5 < 0.62: gNodeB
        self.assertEqual(self.led(unaware, "R1").loc[2.0, "pue_id"], 1)          # unaware ignores the LSTM

    def test_joint_rebalance_shared_provider_backhaul_drop(self):
        kw = {"range_fn": only({r: (lambda p, t: p == 1) for r in ("R1", "R2", "R3")}),
              "backhaul": {(1, 12): 12.0}}
        c = run(kw)
        self.check_clean(c)
        L = c.results()["ledger"]
        at = L[(L.time_s == 12.0) & (L.pue_id == 1)]
        self.assertEqual(set(at.request_id), {"R1", "R2", "R3"})           # all kept on their provider
        self.assertLessEqual(at.pue_rate_mbps.sum(), 12.0 + 1e-6)            # no oversubscription of C5
        self.assertTrue(np.allclose(at.pue_rate_mbps + at.gnb_rate_mbps, 5.0))
        self.assertGreaterEqual(sum(c.sm.sessions[r].n_rebalances for r in ("R1", "R2", "R3")), 1)
        self.assertEqual(sum(c.sm.sessions[r].n_reallocations for r in ("R1", "R2", "R3")), 0)

    def test_cell_overload_reallocation_interruption_and_blocking(self):
        kw = {"range_fn": only({"R0": lambda p, t: p == 1 and t >= 4}), "cell_cap": {5.0: 3.0}}
        c = run(kw)
        self.check_clean(c)
        self.assertEqual(self.led(c, "R0").loc[5.0, "pue_id"], 1)           # cell overload -> moved to PUE
        self.assertEqual(self.led(c, "R1").loc[5.0, "state"], "interrupted")  # no option this second
        self.assertEqual(self.led(c, "R1").loc[6.0, "state"], "active")
        self.assertEqual(c.sm.sessions["R2"].state.value, "blocked")          # arrival with no capacity
        L = c.results()["ledger"]
        self.assertLessEqual(L[L.time_s == 5.0].gnb_rate_mbps.sum(), 3.0 + 1e-6)

    def test_simultaneous_departure_and_cell_overload(self):
        kw = {"range_fn": only({"R1": lambda p, t: p == 2 or (p == 4 and t >= 11)}), "cell_cap": {11.0: 2.0}}
        c = run(kw)
        self.check_clean(c)
        self.assertEqual(self.led(c, "R1").loc[11.0, "pue_id"], 4)
        self.assertEqual(self.led(c, "R2").loc[11.0, "state"], "interrupted")
        self.assertEqual(self.led(c, "R2").loc[12.0, "state"], "active")
        self.assertEqual(c.sm.sessions["R3"].state.value, "blocked")

    def test_channel_f2_all_blocked_uses_gnb(self):
        c = run({}, channel="F2")
        self.check_clean(c)
        self.assertTrue(c.results()["ledger"].pue_id.isna().all())

    def test_greedy_allocator_end_to_end(self):
        c = run({"range_fn": only({"R1": lambda p, t: p == 2 or (p == 4 and t >= 11)})}, allocator=GreedyAllocator())
        self.check_clean(c)
        self.assertEqual(self.led(c, "R1").loc[11.0, "pue_id"], 4)


class FailingAllocator:
    """Wraps a real allocator; raises on chosen seconds to test failure safety."""
    name = "failing"

    def __init__(self, fail_at):
        self.inner, self.fail_at = MilpAllocator(), set(fail_at)

    def solve(self, p):
        if p.time_s in self.fail_at:
            from src.mec.allocation_service import AllocationServiceError
            raise AllocationServiceError(f"injected failure at {p.time_s}")
        return self.inner.solve(p)


class TestFailureSafety(E2E):
    def test_failed_reallocation_does_not_corrupt_state(self):
        kw = {"range_fn": only({"R1": lambda p, t: p == 2 or (p == 4 and t >= 11)})}
        inp, st = build(**kw)
        c = MecCoordinator(inp, st, "G", "primary", FailingAllocator({11.0}), max_interruption_s=5).run()
        rep = self.check_clean(c, errors=1)
        L = self.led(c, "R1")
        self.assertEqual(L.loc[11.0, "state"], "interrupted")                # safe fallback, no half state
        self.assertEqual(L.loc[12.0, "pue_id"], 4)                           # recovers next second
        self.assertEqual(c.sm.sessions["R1"].n_recoveries, 1)

    def test_proactive_and_delay_not_silently_enabled(self):
        inp, st = build()
        with self.assertRaises(NotImplementedError):
            MecCoordinator(inp, st, "G", "primary", MilpAllocator(), 5, proactive=True)
        with self.assertRaises(NotImplementedError):
            MecCoordinator(inp, st, "G", "primary", MilpAllocator(), 5, reallocation_delay_s=1)


# ------------------------------------------------------------------ real validation data
REAL_SID, REAL_RUN = "C09_withdrawals_some__run_006__r0", "run_006"
P4_VAL = f"data/allocation/phase4/val/{REAL_SID}/allocation_results.csv.gz"


@unittest.skipUnless(os.path.exists(f"data/simulation/phase3/val/{REAL_SID}/candidate_links.csv.gz")
                     and os.path.exists(P4_VAL), "validation data absent")
class TestRealScenario(E2E):
    @classmethod
    def setUpClass(cls):
        from src.mobility.trace import load_run
        cls.inp = Phase5ScenarioInputs(REAL_SID, f"data/simulation/phase3/val/{REAL_SID}",
                                       f"data/synthetic/marketplace/val/{REAL_SID}")
        cls.st = StateManager(cls.inp, load_run(REAL_RUN))

    def test_stateless_reproduces_recorded_phase4_validation(self):
        """Allocation service on all active requests + E_t, no reservation == recorded Phase 4 MILP (G, primary)."""
        rec = pd.read_csv(P4_VAL, low_memory=False)
        rec = rec[(rec.method == "milp") & (rec.channel_scenario == "G") & (rec.theta_tag == "primary")]
        svc = AllocationService(self.inp, MilpAllocator())
        rows = []
        for t in sorted(rec.time_s.unique()):
            s = self.st.snapshot("G", "primary", float(t))
            res, _, _ = svc.solve(t, s.active_requests, s.links[s.links.feasible_primary.astype(bool)])
            rows += [(float(t), r, a is not None, a.pue_id if a else None,
                      round(a.pue_rate_mbps, 9) if a else 0.0, round(a.gnb_rate_mbps, 9) if a else 0.0)
                     for r, a in res.items()]
        got = pd.DataFrame(rows, columns=["time_s", "request_id", "admitted", "pue_id", "r", "g"]).sort_values(
            ["time_s", "request_id"]).reset_index(drop=True)
        exp = rec.sort_values(["time_s", "request_id"]).reset_index(drop=True)
        self.assertEqual(len(got), len(exp))
        self.assertTrue((got.admitted.to_numpy() == exp.admitted.to_numpy()).all())
        self.assertTrue((got.pue_id.fillna(-1).to_numpy(float) == exp.pue_id.fillna(-1).to_numpy(float)).all())
        self.assertTrue(np.allclose(got.r, exp.pue_rate_mbps, atol=1e-9))
        self.assertTrue(np.allclose(got.g, exp.gnb_rate_mbps, atol=1e-9))

    def test_full_coordinator_run_milp(self):
        c = MecCoordinator(self.inp, self.st, "G", "primary", MilpAllocator(), max_interruption_s=10).run()
        rep = self.check_clean(c)
        self.assertGreater(rep["sessions"], 0)
        self.assertEqual(rep["audit_seconds"], len(self.st.mob.times))
        ev = c.results()["events"]
        self.assertTrue({"request_window_end", "cue_departed", "trace_end"} <= set(ev.event), set(ev.event))
        self.assertGreater(int((ev.severity == "hard").sum()), 0)            # real failures happened ...
        self.assertGreater(rep["reallocations"] + rep["recoveries"], 0)      # ... and service resumed


if __name__ == "__main__":
    unittest.main()

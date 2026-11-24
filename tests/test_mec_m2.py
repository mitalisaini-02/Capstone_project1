"""
Phase 5 M2 unit tests: session lifecycle (state machine, counters, interruption boundary), event
detection and ordering, failure classification and rebalance groups, ledger accounting.
Uses the SYNTHETIC fixture in tests/mec_fixtures.py.
"""
import tempfile
import unittest

import numpy as np

from src.common.units import cost
from src.mec.accounting import Ledger, LedgerError
from src.mec.events import Detection, Event, EventMonitor, EventType as E, Severity
from src.mec.failure import Action, FailureDetector
from src.mec.inputs import Phase5ScenarioInputs
from src.mec.session import Assignment, SessionError, SessionManager, SessionState as S
from src.mec.state import StateManager
from tests import mec_fixtures as fx


def mgr(max_i=2):
    m = SessionManager(max_i)
    m.open(0.0, "A", 1, 5.0, 1.0, 2.0, 0.0, 10.0)
    return m


PUE = Assignment(7, 5.0, 0.0, None)
GNB = Assignment(None, 0.0, 5.0, "g1")
SPLIT = Assignment(7, 3.0, 2.0, "g1")


class TestSessionLifecycle(unittest.TestCase):
    def test_admit_block_and_final(self):
        m = mgr()
        m.admit("A", 0.0, PUE)
        self.assertIs(m.sessions["A"].state, S.ACTIVE)
        m.open(1.0, "B", 2, 5.0, 1.0, 2.0, 1.0, 5.0)
        m.block("B", 1.0)
        self.assertTrue(m.sessions["B"].is_final)
        with self.assertRaises(SessionError):
            m.admit("B", 2.0, PUE)

    def test_illegal_transitions(self):
        m = mgr()
        with self.assertRaises(SessionError):
            m.release("A", 0.0, "x")                  # not active yet
        with self.assertRaises(SessionError):
            m.reassign("A", 0.0, PUE)
        m.admit("A", 0.0, PUE)
        with self.assertRaises(SessionError):
            m.admit("A", 1.0, PUE)
        with self.assertRaises(SessionError):
            m.interrupt("A", 1.0)                     # must be released first
        m.terminate("A", 5.0, "window_end")
        for f in (lambda: m.release("A", 6.0, "x"), lambda: m.terminate("A", 6.0, "window_end")):
            with self.assertRaises(SessionError):
                f()

    def test_assignment_validation(self):
        m = mgr()
        for bad in (Assignment(7, 0.0, 5.0, "g1"),          # phantom provider
                    Assignment(None, 1e-3, 4.999, "g1"),     # traffic without provider
                    Assignment(7, 3.0, 1.0, "g1"),           # does not serve demand (C6)
                    Assignment(7, 3.0, 2.0, None)):          # gNodeB share without a cell
            with self.assertRaises(SessionError):
                m.admit("A", 0.0, bad)
        self.assertIs(m.sessions["A"].state, S.REQUESTED)    # failed admission left state intact

    def test_one_session_per_consumer(self):
        m = mgr()
        with self.assertRaises(SessionError):
            m.open(1.0, "A2", 1, 5.0, 1.0, 2.0, 1.0, 4.0)
        with self.assertRaises(SessionError):
            m.open(1.0, "A", 9, 5.0, 1.0, 2.0, 1.0, 4.0)     # duplicate id
        with self.assertRaises(SessionError):
            m.open(1.0, "C", 9, 5.0, 1.0, 2.0, 2.0, 4.0)     # opened before its request time

    def test_reallocation_rebalance_recovery_counters(self):
        m = mgr(max_i=3)
        m.admit("A", 0.0, PUE)
        m.rebalance("A", 1.0, PUE)                            # unchanged rates: not counted
        m.rebalance("A", 2.0, SPLIT)
        self.assertEqual(m.sessions["A"].n_rebalances, 1)
        with self.assertRaises(SessionError):
            m.rebalance("A", 3.0, Assignment(8, 5.0, 0.0, None))   # another provider is not a rebalance
        m.release("A", 3.0, "pue_departed")
        m.reassign("A", 3.0, GNB)
        s = m.sessions["A"]
        self.assertEqual((s.n_reallocations, s.n_interruptions, s.state), (1, 0, S.ACTIVE))
        m.release("A", 4.0, "link_outage")
        m.interrupt("A", 4.0)
        m.end_of_second(4.0)
        m.interrupt("A", 5.0)
        m.end_of_second(5.0)
        m.reassign("A", 6.0, PUE)
        self.assertEqual((s.n_recoveries, s.n_interruptions, s.interrupted_total_s, s.interrupted_run_s),
                         (1, 1, 2, 0))
        self.assertEqual([p for _, p in s.providers], [7, None, 7])

    def test_interruption_boundary(self):
        for max_i, drop_at in ((0, 1), (2, 3)):
            m = mgr(max_i)
            m.admit("A", 0.0, PUE)
            m.release("A", 1.0, "x")
            m.interrupt("A", 1.0)
            t, dropped = 1.0, []
            while not dropped:
                dropped = m.end_of_second(t)
                if not dropped:
                    t += 1
                    m.interrupt("A", t)
            self.assertEqual(m.sessions["A"].interrupted_total_s, drop_at)
            self.assertIs(m.sessions["A"].state, S.DROPPED)

    def test_pending_must_be_resolved(self):
        m = mgr()
        m.admit("A", 0.0, PUE)
        m.release("A", 1.0, "x")
        with self.assertRaises(SessionError):
            m.end_of_second(1.0)

    def test_terminal_causes(self):
        for cause, st in (("window_end", S.COMPLETED), ("cue_departed", S.CUE_DEPARTED), ("trace_end", S.CENSORED)):
            m = mgr()
            m.admit("A", 0.0, PUE)
            m.terminate("A", 3.0, cause)
            self.assertIs(m.sessions["A"].state, st)
            self.assertIsNone(m.sessions["A"].assignment)


class FixtureCase(unittest.TestCase):
    kw = {}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        p3, p2 = fx.write(cls.tmp.name, **cls.kw)
        cls.inp = Phase5ScenarioInputs("FIX__fixture_run__r0", p3, p2)
        cls.st = StateManager(cls.inp, fx.mobility())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestEvents(FixtureCase):
    def sess(self, rid, a, max_i=5):
        m = SessionManager(max_i)
        w = self.st.windows[rid]
        m.open(w.start_s, rid, w.cue_id, 5.0, 1.0, 2.0, w.start_s, w.last_active_s)
        m.admit(rid, w.start_s, a)
        return m

    def ev(self, rid, a, t, tag="primary", ch="G"):
        m = self.sess(rid, a)
        return [e.type for e in EventMonitor(self.st, ch, tag).detect(t, m.nonfinal()).events]

    def test_hard_failures(self):
        self.assertEqual(self.ev("R1", Assignment(2, 5.0, 0.0, None), 11.0), [E.PUE_DEPARTED])
        self.assertEqual(self.ev("R0", Assignment(3, 5.0, 0.0, None), 8.0), [E.PUE_WITHDRAWN])
        self.assertEqual(self.ev("R0", Assignment(4, 5.0, 0.0, None), 5.0), [E.PUE_UNAVAILABLE])
        self.assertEqual(self.ev("R0", Assignment(1, 5.0, 0.0, None), 4.0), [E.PUE_OUT_OF_RANGE])
        self.assertEqual(self.ev("R0", Assignment(1, 5.0, 0.0, None), 3.0), [E.LINK_OUTAGE])
        self.assertEqual(self.ev("R0", Assignment(1, 5.0, 0.0, None), 5.0, ch="F2"), [E.LINK_OUTAGE])

    def test_capacity_and_advisory(self):
        self.assertEqual(self.ev("R0", Assignment(1, 5.0, 0.0, None), 2.0), [E.RELIABILITY_LOW])
        self.assertEqual(self.ev("R0", Assignment(1, 5.0, 0.0, None), 2.0, tag="unaware"), [])
        self.assertEqual(self.ev("R1", Assignment(5, 5.0, 0.0, None), 10.0), [E.PUE_BACKHAUL_OVERLOAD])
        self.assertEqual(self.ev("R0", Assignment(1, 5.0, 0.0, None), 5.0), [])

    def test_terminal_after_window_and_precedence(self):
        # R0's window ends at 9: at 10 the terminal event wins even though its provider withdrew
        self.assertEqual(self.ev("R0", Assignment(3, 5.0, 0.0, None), 10.0), [E.REQUEST_WINDOW_END])
        self.assertEqual(self.ev("R1", Assignment(2, 5.0, 0.0, None), 13.0), [E.CUE_DEPARTED])
        self.assertEqual(self.ev("R0", Assignment(3, 5.0, 0.0, None), 9.0), [E.PUE_WITHDRAWN])

    def test_end_of_trace(self):
        m = SessionManager(5)
        for rid in ("R2", "R3"):
            w = self.st.windows[rid]
            m.open(w.start_s, rid, w.cue_id, 5.0, 1.0, 2.0, w.start_s, w.last_active_s)
            m.admit(rid, w.start_s, GNB)
        evs = {e.request_id: (e.type, e.time_s) for e in EventMonitor(self.st, "G", "primary").end_of_trace_events(m.nonfinal())}
        self.assertEqual(evs, {"R2": (E.TRACE_END, 21.0), "R3": (E.REQUEST_WINDOW_END, 21.0)})

    def test_severity_order(self):
        ranks = sorted([Event(1.0, E.RELIABILITY_LOW, "x"), Event(1.0, E.GNB_CELL_OVERLOAD, "x"),
                        Event(1.0, E.PUE_DEPARTED, "x"), Event(1.0, E.CUE_DEPARTED, "x")], key=lambda e: e.rank)
        self.assertEqual([e.severity for e in ranks],
                         [Severity.TERMINAL, Severity.HARD, Severity.CAPACITY, Severity.ADVISORY])


class TestCellEvents(FixtureCase):
    kw = {"cell_cap": {7.0: 6.0}, "uncovered": [("R2", 8)]}

    def test_cell_overload_and_coverage(self):
        m = SessionManager(5)
        for rid in ("R0", "R1"):
            w = self.st.windows[rid]
            m.open(w.start_s, rid, w.cue_id, 5.0, 1.0, 2.0, w.start_s, w.last_active_s)
            m.admit(rid, w.start_s, GNB)
        mon = EventMonitor(self.st, "G", "primary")
        self.assertEqual(sorted(e.type for e in mon.detect(7.0, m.nonfinal()).events), [E.GNB_CELL_OVERLOAD] * 2)
        self.assertEqual(mon.detect(6.0, m.nonfinal()).events, [])
        w = self.st.windows["R2"]
        m.open(w.start_s, "R2", w.cue_id, 5.0, 1.0, 2.0, w.start_s, w.last_active_s)
        m.admit("R2", w.start_s, Assignment(1, 3.0, 2.0, "g1"))
        self.assertIn(E.CUE_OUT_OF_COVERAGE, [e.type for e in mon.detect(8.0, m.nonfinal()).events])


class TestFailureDetector(unittest.TestCase):
    def test_actions_and_transitive_groups(self):
        evs = [Event(1.0, E.PUE_BACKHAUL_OVERLOAD, "a", 1), Event(1.0, E.RELIABILITY_LOW, "a", 1),
               Event(1.0, E.PUE_DEPARTED, "d", 3), Event(1.0, E.REQUEST_WINDOW_END, "e"),
               Event(1.0, E.RELIABILITY_LOW, "f", 4), Event(1.0, E.PUE_OUT_OF_RANGE, "e", 9)]
        res = {"a": {("pue", 1), ("cell", "g1")}, "b": {("pue", 2), ("cell", "g1")}, "c": {("pue", 2)},
               "d": set(), "f": {("pue", 4)}, "h": {("cell", "g2")}}
        dec, groups = FailureDetector().decide(Detection(1.0, evs, res), list("abcdefh"))
        self.assertEqual({k: v.action for k, v in dec.items()},
                         {"a": Action.REBALANCE, "b": Action.REBALANCE, "c": Action.REBALANCE, "d": Action.RELEASE,
                          "e": Action.TERMINATE, "f": Action.KEEP, "h": Action.KEEP})
        self.assertEqual(groups, [["a", "b", "c"]])                 # a-b share g1, b-c share pue 2
        self.assertEqual(dec["e"].terminal_cause, "window_end")     # terminal beats hard
        self.assertIs(dec["a"].cause.type, E.PUE_BACKHAUL_OVERLOAD)  # capacity beats advisory

    def test_disjoint_groups(self):
        evs = [Event(1.0, E.GNB_CELL_OVERLOAD, "a", cell_id="g1"), Event(1.0, E.LINK_CAPACITY_DROP, "x", 5)]
        res = {"a": {("cell", "g1")}, "b": {("cell", "g1")}, "x": {("pue", 5)}, "y": {("pue", 6)}}
        dec, groups = FailureDetector().decide(Detection(1.0, evs, res), ["a", "b", "x", "y"])
        self.assertEqual(groups, [["a", "b"], ["x"]])
        self.assertIs(dec["y"].action, Action.KEEP)


class TestLedger(unittest.TestCase):
    def test_rows_payments_and_conservation(self):
        m = mgr(max_i=5)
        L = Ledger(1.0, lambda t, p: 0.5)
        m.admit("A", 0.0, SPLIT)
        L.record_second(0.0, m.nonfinal())
        m.release("A", 1.0, "x")
        m.interrupt("A", 1.0)
        L.record_second(1.0, m.nonfinal())
        m.end_of_second(1.0)
        m.reassign("A", 2.0, GNB)
        L.record_second(2.0, m.nonfinal())
        m.terminate("A", 3.0, "window_end")
        f = L.frame()
        self.assertEqual(list(f.state), ["active", "interrupted", "active"])
        self.assertAlmostEqual(f.payment[0], cost(3.0, 1.0, 0.5) + cost(2.0, 1.0, 1.0), places=15)
        self.assertEqual(f.payment[1], 0.0)
        self.assertEqual(set(L.verify(m.sessions).values()), {0})
        s = L.session_summary(m.sessions).iloc[0]
        self.assertEqual((s.ledger_seconds, s.active_seconds, s.interrupted_seconds), (3, 2, 1))
        self.assertAlmostEqual(s.delivered_ratio, 10.0 / 15.0)
        self.assertAlmostEqual(s.payment, s.pue_revenue + s.gnb_revenue, places=15)

    def test_duplicates_and_pending_rejected(self):
        m = mgr()
        L = Ledger(1.0, lambda t, p: 0.5)
        m.admit("A", 0.0, PUE)
        L.record_second(0.0, m.nonfinal())
        with self.assertRaises(LedgerError):
            L.record_second(0.0, m.nonfinal())
        m.release("A", 1.0, "x")
        with self.assertRaises(LedgerError):
            L.record_second(1.0, m.nonfinal())

    def test_summary_requires_final(self):
        m = mgr()
        m.admit("A", 0.0, PUE)
        with self.assertRaises(LedgerError):
            Ledger(1.0, lambda t, p: 0.5).session_summary(m.sessions)


if __name__ == "__main__":
    unittest.main()

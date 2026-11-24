"""
Tests for Phase 4 marketplace allocation on crafted epochs with known optimal answers,
plus audit, phantom-PUE handling, rejection reasons, pricing, isolation and a validation integration.
"""
import ast
import os
import unittest

import numpy as np

from src.marketplace.admission import best_option, full_resources, rejection_reason
from src.marketplace.allocation import Allocation, audit, from_link_rates
from src.marketplace.eligibility import CANDIDATE_COLUMNS
from src.marketplace.greedy import GreedyAllocator
from src.marketplace.milp import MilpAllocator
from src.marketplace.pricing import PostedPricePolicy
from src.marketplace.problem import EpochProblem

EPS = 1e-6


def P(requests, links=(), pues=(), cells=(("g1", 100.0),), pg=1.0):
    """requests: (id, d, max_price, covered, cell, weight, arrival); links: (req_idx, pue_idx, B);
    pues: (id, A, C, price)."""
    cells = list(cells)
    kmap = {c: k for k, (c, _) in enumerate(cells)}
    r = list(requests)
    return EpochProblem(
        "s", 1.0, 1.0, [x[0] for x in r], np.arange(len(r)), np.array([x[1] for x in r], float),
        np.array([x[5] for x in r], float), np.array([x[2] for x in r], float), np.array([x[6] for x in r], float),
        np.array([x[3] for x in r], bool), np.array([kmap[x[4]] if x[3] else -1 for x in r], int),
        [c for c, _ in cells], np.array([g for _, g in cells], float), pg,
        np.array([p[0] for p in pues], int), np.array([p[1] for p in pues], float),
        np.array([p[2] for p in pues], float), np.array([p[3] for p in pues], float),
        np.array([l[0] for l in links], int), np.array([l[1] for l in links], int),
        np.array([l[2] for l in links], float), np.full(len(links), 0.9))


def req(i, d=5.0, mp=2.0, cov=True, w=1.0, t=0.0):
    return (f"r{i}", d, mp, cov, "g1", w, t)


class TestCorePolicies(unittest.TestCase):
    def setUp(self):
        self.g, self.m = GreedyAllocator(), MilpAllocator()

    def both(self, p):
        a, b = self.g.solve(p), self.m.solve(p)
        self.assertEqual(audit(p, a), {})
        self.assertEqual(audit(p, b), {})
        return a, b

    def test_cheapest_option_pue_cheaper(self):
        p = P([req(0)], links=[(0, 0, 50.0)], pues=[(7, 50, 50, 0.5)])
        a, b = self.both(p)
        for x in (a, b):
            self.assertEqual((x.pue_index[0], x.pue_rate_mbps[0], x.gnb_rate_mbps[0]), (0, 5.0, 0.0))

    def test_cheapest_option_gnb_cheaper_gives_null_pue(self):
        p = P([req(0)], links=[(0, 0, 50.0)], pues=[(7, 50, 50, 1.5)])
        for x in self.both(p):
            self.assertTrue(x.admitted[0])
            self.assertEqual(x.pue_index[0], -1)          # served entirely by gNodeB -> pue null
            self.assertEqual(x.link_index[0], -1)
            self.assertEqual(x.gnb_rate_mbps[0], 5.0)

    def test_pue_plus_gnb_remainder_limited_by_C3(self):
        p = P([req(0, d=10)], links=[(0, 0, 4.0)], pues=[(7, 50, 50, 0.5)])
        for x in self.both(p):
            self.assertEqual((x.pue_rate_mbps[0], x.gnb_rate_mbps[0]), (4.0, 6.0))

    def test_C4_and_C5_independently_binding(self):
        # A (access) = 6 binds for one PUE, C (backhaul) = 3 binds for the other
        p = P([req(0, d=10), req(1, d=10)], links=[(0, 0, 50.0), (1, 1, 50.0)],
              pues=[(7, 6, 50, 0.5), (8, 50, 3, 0.5)])
        for x in self.both(p):
            self.assertEqual(x.pue_rate_mbps.tolist(), [6.0, 3.0])

    def test_one_pue_serves_multiple_cues_within_limits(self):
        p = P([req(0, d=4), req(1, d=4)], links=[(0, 0, 50.0), (1, 0, 50.0)], pues=[(7, 6, 50, 0.5)])
        for x in self.both(p):
            self.assertTrue(x.admitted.all())
            self.assertAlmostEqual(x.pue_rate_mbps.sum(), 6.0)      # shared C4 limit respected
            self.assertTrue(all(i in (0, -1) for i in x.pue_index))

    def test_one_pue_per_cue(self):
        p = P([req(0, d=10)], links=[(0, 0, 6.0), (0, 1, 6.0)], pues=[(7, 50, 50, 0.5), (8, 50, 50, 0.4)])
        for x in self.both(p):
            self.assertEqual(x.pue_index[0], 1)                     # cheaper PUE only, not both
            self.assertEqual((x.pue_rate_mbps[0], x.gnb_rate_mbps[0]), (6.0, 4.0))

    def test_gnb_fallback_when_no_pue(self):
        p = P([req(0)])
        for x in self.both(p):
            self.assertTrue(x.admitted[0])
            self.assertEqual(x.pue_index[0], -1)

    def test_admission_before_cost_milp_beats_greedy(self):
        # cell capacity 5: greedy gives r0 the cheaper gNodeB and then cannot admit r1 (no PUE);
        # MILP admits both by moving r0 to the (more expensive) PUE.
        p = P([req(0, t=0), req(1, t=1)], links=[(0, 0, 50.0)], pues=[(7, 50, 50, 1.5)],
              cells=[("g1", 5.0)])
        a, b = self.both(p)
        self.assertEqual(a.admitted.sum(), 1)
        self.assertEqual(b.admitted.sum(), 2)
        self.assertEqual(b.pue_index[0], 0)
        self.assertEqual(rejection_reason(p, 1, EPS)[0], "capacity_contention")

    def test_milp_minimises_cost_at_max_admission(self):
        p = P([req(0, d=4), req(1, d=4)], links=[(0, 0, 50.0), (1, 0, 50.0)], pues=[(7, 4, 50, 0.5)],
              cells=[("g1", 100.0)])
        b = self.m.solve(p)
        self.assertEqual(b.admitted.sum(), 2)
        self.assertAlmostEqual(b.stage2_objective, 4 * 0.5 + 4 * 1.0)

    def test_priority_order_in_greedy(self):
        p = P([req(0, w=1.0, t=0), req(1, w=2.0, t=5)], cells=[("g1", 5.0)])
        a = self.g.solve(p)
        self.assertEqual(a.admitted.tolist(), [False, True])       # higher weight first

    def test_budget_and_reasons(self):
        p = P([req(0, mp=0.5), req(1, d=5, cov=False), req(2, d=50, cov=False)],
              links=[(2, 0, 10.0)], pues=[(7, 50, 50, 0.4)])
        for x in self.both(p):
            self.assertFalse(x.admitted.any())
        self.assertEqual(rejection_reason(p, 0, EPS)[0], "budget_exceeded")
        self.assertEqual(rejection_reason(p, 1, EPS), ("no_service_option", "no_pue_in_range"))
        self.assertEqual(rejection_reason(p, 2, EPS)[0], "insufficient_capacity_standalone")

    def test_budget_satisfied_by_cheap_pue(self):
        p = P([req(0, mp=0.6)], links=[(0, 0, 50.0)], pues=[(7, 50, 50, 0.5)])
        for x in self.both(p):
            self.assertTrue(x.admitted[0])
            self.assertEqual(x.pue_index[0], 0)

    def test_best_option_respects_remaining_resources(self):
        p = P([req(0, d=10)], links=[(0, 0, 50.0)], pues=[(7, 50, 50, 0.5)], cells=[("g1", 100.0)])
        A, C, G = full_resources(p)
        A[0] = 3.0
        o = best_option(p, 0, A, C, G, EPS)
        self.assertEqual((o.pue_rate, o.gnb_rate), (3.0, 7.0))


class TestAuditAndPhantom(unittest.TestCase):
    def setUp(self):
        self.p = P([req(0, d=5), req(1, d=5, cov=False)], links=[(0, 0, 4.0), (1, 0, 4.0)],
                   pues=[(7, 6, 6, 0.5)], cells=[("g1", 3.0)])

    def alloc(self, **kw):
        base = dict(method="t", admitted=np.array([True, False]), pue_index=np.array([0, -1]),
                    link_index=np.array([0, -1]), pue_rate_mbps=np.array([4.0, 0.0]),
                    gnb_rate_mbps=np.array([1.0, 0.0]))
        base.update(kw)
        return Allocation(**base)

    def test_valid(self):
        self.assertEqual(audit(self.p, self.alloc()), {})

    def test_phantom_detected(self):
        v = audit(self.p, self.alloc(pue_rate_mbps=np.array([0.0, 0.0]), gnb_rate_mbps=np.array([5.0, 0.0])))
        self.assertIn("phantom_pue", v)
        v = audit(self.p, self.alloc(pue_index=np.array([-1, -1]), link_index=np.array([-1, -1])))
        self.assertIn("phantom_pue", v)

    def test_from_link_rates_never_records_zero_rate_pue(self):
        # cell large enough for a full gNodeB service (the shared fixture's 3 Mbps cell is not)
        p = P([req(0, d=5), req(1, d=5, cov=False)], links=[(0, 0, 4.0), (1, 0, 4.0)],
              pues=[(7, 6, 6, 0.5)], cells=[("g1", 10.0)])
        a = from_link_rates(p, "m", [True, False], np.array([1e-9, 0.0]), np.array([5.0, 0.0]), EPS)
        self.assertEqual(a.pue_index.tolist(), [-1, -1])
        self.assertEqual(audit(p, a), {})
        self.assertIn("C7_cell", audit(self.p, a))     # and the audit still catches the 3 Mbps cell

    def test_from_link_rates_rejects_two_pues(self):
        p = P([req(0, d=10)], links=[(0, 0, 6.0), (0, 1, 6.0)], pues=[(7, 50, 50, 0.5), (8, 50, 50, 0.4)])
        with self.assertRaises(ValueError):
            from_link_rates(p, "m", [True], np.array([5.0, 5.0]), np.array([0.0]), EPS)

    def test_each_family_detected_independently(self):
        cases = {
            "C3_link": dict(pue_rate_mbps=np.array([4.5, 0.0]), gnb_rate_mbps=np.array([0.5, 0.0])),
            "C6_full_service": dict(gnb_rate_mbps=np.array([0.5, 0.0])),
            "C7_cell": dict(pue_rate_mbps=np.array([1.0, 0.0]), gnb_rate_mbps=np.array([4.0, 0.0])),
            "C8_coverage": dict(admitted=np.array([True, True]), pue_index=np.array([0, -1]), link_index=np.array([0, -1]),
                                pue_rate_mbps=np.array([4.0, 0.0]), gnb_rate_mbps=np.array([1.0, 5.0])),
            "C2_eligible_link": dict(link_index=np.array([1, -1])),
            "C1_one_pue": dict(gnb_rate_mbps=np.array([1.0, 2.0])),
        }
        for fam, kw in cases.items():
            self.assertIn(fam, audit(self.p, self.alloc(**kw)), fam)
        p4 = P([req(0, d=5), req(1, d=5)], links=[(0, 0, 5.0), (1, 0, 5.0)], pues=[(7, 6, 50, 0.5)])
        a = Allocation("t", np.array([True, True]), np.array([0, 0]), np.array([0, 1]),
                       np.array([5.0, 5.0]), np.array([0.0, 0.0]))
        self.assertIn("C4_access", audit(p4, a))
        p5 = P([req(0, d=5), req(1, d=5)], links=[(0, 0, 5.0), (1, 0, 5.0)], pues=[(7, 50, 6, 0.5)])
        v5 = audit(p5, a)
        self.assertIn("C5_backhaul", v5)
        self.assertNotIn("C4_access", v5)
        p9 = P([req(0, d=5, mp=0.4)], links=[(0, 0, 5.0)], pues=[(7, 50, 50, 0.5)])
        a9 = Allocation("t", np.array([True]), np.array([0]), np.array([0]), np.array([5.0]), np.array([0.0]))
        self.assertIn("C9_budget", audit(p9, a9))


class TestPricingAndIsolation(unittest.TestCase):
    def test_payments(self):
        pue, gnb, tot = PostedPricePolicy().payments(8.0, 0.5, 8.0, 1.0, 1000.0)
        self.assertAlmostEqual(pue, 0.5)
        self.assertAlmostEqual(gnb, 1.0)
        self.assertAlmostEqual(tot, 1.5)

    def test_expected_capacity_never_read(self):
        self.assertFalse([c for c in CANDIDATE_COLUMNS if c.startswith("expected")])
        self.assertIn("realized_link_capacity_mbps", CANDIDATE_COLUMNS)

    def test_marketplace_imports_no_reliability_or_radio_code(self):
        forbidden = ("src.reliability", "src.models", "src.network", "src.features", "src.simulation")
        for f in os.listdir("src/marketplace"):
            if f.endswith(".py"):
                tree = ast.parse(open(os.path.join("src/marketplace", f)).read())
                for n in ast.walk(tree):
                    mods = [n.module] if isinstance(n, ast.ImportFrom) and n.module else \
                        [a.name for a in n.names] if isinstance(n, ast.Import) else []
                    for m in mods:
                        self.assertFalse(m.startswith(forbidden), f"{f} imports {m}")


SID = "C04_demand_high__run_006__r0"


@unittest.skipUnless(os.path.exists(f"data/simulation/phase3/val/{SID}/candidate_links.csv.gz"), "Phase 3 data missing")
class TestValidationIntegration(unittest.TestCase):
    def test_identical_inputs_audit_and_dominance(self):
        from src.marketplace.eligibility import ScenarioInputs
        S = ScenarioInputs(SID, f"data/simulation/phase3/val/{SID}", f"data/synthetic/marketplace/val/{SID}")
        a = list(S.problems("G", "primary"))[:250]
        b = list(S.problems("G", "primary"))[:250]
        g, m = GreedyAllocator(), MilpAllocator()
        for p, q in zip(a, b):
            np.testing.assert_array_equal(p.link_capacity_mbps, q.link_capacity_mbps)   # same eligible links
            x, y, z = g.solve(p), m.solve(p), m.solve(p.without_links())
            for al in (x, y, z):
                self.assertEqual(audit(p, al), {})
            self.assertGreaterEqual(y.stage1_objective, x.stage1_objective - 1e-9)
            self.assertGreaterEqual(y.stage1_objective, z.stage1_objective - 1e-9)
            if abs(y.stage1_objective - x.stage1_objective) < 1e-9:
                self.assertLessEqual(y.stage2_objective, x.stage2_objective + 1e-6)


if __name__ == "__main__":
    unittest.main()

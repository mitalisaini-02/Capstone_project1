import unittest

import numpy as np

from objective_comparison.allocation import Snapshot, choose
from objective_comparison.sinr import dbm_to_mw, snapshot
from src.marketplace.problem import EpochProblem
from src.network.tr37885 import TR37885SidelinkModel, load_tr37885_profile


class RadioTest(unittest.TestCase):
    def test_linear_noise_interference_and_movement(self):
        model = TR37885SidelinkModel(load_tr37885_profile(
            "config/network/link_profiles_nr.json", "nr_sidelink_tr37885_v1"))
        tx = [[0, 0], [30, 0], [40, 0], [50, 0]]
        states = [["LOS"] * 4]
        result = snapshot(model, tx, [[10, 0]], [0, 0, 0, 1], states)
        distance, path, power, interference, noise, linear, db = result
        np.testing.assert_allclose(distance[0], [10, 20, 30, 40])
        expected_path = 38.77 + 16.7 * np.log10(distance) + 18.2 * np.log10(6)
        np.testing.assert_allclose(path, expected_path)
        np.testing.assert_allclose(power, 10 ** ((29 - expected_path) / 10))
        self.assertAlmostEqual(noise, 10 ** ((-174 + 10 * np.log10(20e6) + 9) / 10), places=23)
        np.testing.assert_allclose(interference[0], [power[0, 1] + power[0, 2],
                                                     power[0, 0] + power[0, 2],
                                                     power[0, 0] + power[0, 1], 0])
        np.testing.assert_allclose(linear, power / (interference + noise))
        np.testing.assert_allclose(db, 10 * np.log10(linear))
        isolated = snapshot(model, tx, [[10, 0]], range(4), states)
        np.testing.assert_allclose(isolated[5], power / noise)
        self.assertFalse(np.allclose(linear, snapshot(model, tx, [[5, 0]], [0, 0, 0, 1], states)[5]))
        np.testing.assert_array_equal(linear, snapshot(model, tx, [[10, 0]], [0, 0, 0, 1], states)[5])
        np.testing.assert_allclose(dbm_to_mw([-10, 0, 10]), [0.1, 1, 10])
        with self.assertRaises(ValueError):
            snapshot(model, tx, [[np.nan, 0]], range(4), states)


def example():
    a = np.array
    p = EpochProblem("manual", 10, 1, ["r"], a([1]), a([8.]), a([1.]), a([2.]), a([0.]),
                     a([True]), a([0]), ["g"], a([10.]), 2., a([2]), a([6.]), a([6.]),
                     a([1.]), a([0]), a([0]), a([6.]), a([.85]))
    return Snapshot(p, a([2.]), a([1.]), 5., 8., .00005, .000001, .00001, .62)


class AllocationTest(unittest.TestCase):
    def test_manual_cost_rate_and_both_solver_stages(self):
        s = example()
        manual = tuple(np.array([v]) for v in (6., 2., 3., 8.))
        result = s.evaluate(manual)
        self.assertEqual(result["violations"], 0)
        self.assertAlmostEqual(result["pue_cost"], 6 / 8000)
        self.assertAlmostEqual(result["gnb_cost"], 4 / 8000)
        self.assertAlmostEqual(result["mec_fee"], .00005)
        self.assertAlmostEqual(result["cost"], .00130)
        self.assertAlmostEqual(result["rate"], 14.)
        solution, info = s.exact(5, 0)
        self.assertTrue(info["proven_optimal"])
        self.assertAlmostEqual(info["stage1_cost"], result["cost"])
        self.assertAlmostEqual(s.evaluate(solution)["rate"], result["rate"])
        for method in ("Greedy", "Best-SINR"):
            self.assertEqual(s.evaluate(s.constructive(method))["violations"], 0)
        rng = np.random.default_rng(42)
        attempts = [s.evaluate(s.constructive("Random", rng)) for _ in range(16)]
        best = attempts[choose(attempts, s.epsilon)]
        self.assertEqual(best["violations"], 0)
        self.assertLessEqual(best["cost"], min(a["cost"] for a in attempts) + s.epsilon)

    def test_manual_fallback_only_and_cost_band(self):
        s = example()
        plan = tuple(np.array([v]) for v in (0., 8., 0., 8.))
        result = s.evaluate(plan)
        self.assertEqual(result["violations"], 0)
        self.assertAlmostEqual(result["cost"], 2 * 8 / 8000)
        self.assertEqual(result["mec_fee"], 0)
        self.assertEqual(result["pue_gb"], 0)
        self.assertEqual(result["rate"], 8)
        attempts = [
            dict(violations=0, cost=.001, rate=8),
            dict(violations=0, cost=.0010005, rate=9),
            dict(violations=0, cost=.0010015, rate=10),
            dict(violations=1, cost=float("nan"), rate=100),
        ]
        self.assertEqual(choose(attempts, s.epsilon), 1)

    def test_constraint_detection(self):
        s = example()
        s.problem.link_p_survive[:] = .61
        invalid = tuple(np.array([v]) for v in (7., 2., 6., 9.))
        result = s.evaluate(invalid)
        for key in ("demand", "link_capacity", "access", "backhaul", "pc5_bandwidth",
                    "gnb_bandwidth", "reliability"):
            self.assertEqual(result[f"violations_{key}"], 1)
        self.assertTrue(np.isnan(result["cost"]))

    def test_exact_reports_infeasible(self):
        s = example()
        s.budget[:] = 0
        plan, info = s.exact(5, 0)
        self.assertEqual(info["stage1_status"], 2)
        self.assertFalse(info["proven_optimal"])
        self.assertEqual(s.evaluate(plan)["violations_demand"], 1)


if __name__ == "__main__":
    unittest.main()

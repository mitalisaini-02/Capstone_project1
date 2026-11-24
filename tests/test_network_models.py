"""
Tests for the link-model interface, both profiles, the TR 37.885 equations (hand-computed
from TR 37.885 V15.3.0 §6.2) and the TR 36.942 Annex A throughput mapping (Table A.2 rows).
"""
import math
import unittest

import numpy as np

from src.network.base import LinkModel
from src.network.link_capacity import LinkCapacityModel, load_link_profile
from src.network.registry import MODEL_TYPES, load_link_model, register_model_type
from src.network.technology import load_technology
from src.network.tr37885 import TR37885SidelinkModel

V0 = ("config/marketplace/link_profiles.json", "abstract_sidelink_v0")
NR = ("config/network/link_profiles_nr.json", "nr_sidelink_tr37885_v1")
NRH = ("config/network/link_profiles_nr.json", "nr_sidelink_tr37885_v1_highway")


class TestProfilesLoadIndependently(unittest.TestCase):
    def test_each_profile_loads_alone(self):
        v0 = load_link_model(*V0)
        nr = load_link_model(*NR)
        hw = load_link_model(*NRH)
        self.assertIsInstance(v0, LinkCapacityModel)
        self.assertIsInstance(nr, TR37885SidelinkModel)
        self.assertEqual((nr.scenario, hw.scenario), ("urban", "highway"))
        for m in (v0, nr, hw):
            self.assertIsInstance(m, LinkModel)
            self.assertEqual(len(m.capacity_table()), 5)

    def test_registry_matches_legacy_loader_for_v0(self):
        a = LinkCapacityModel(load_link_profile(*V0))
        b = load_link_model(*V0)
        d = np.linspace(1, 1500, 500)
        np.testing.assert_array_equal(a.capacity_mbps(d), b.capacity_mbps(d))
        self.assertEqual(a.capacity_table(), b.capacity_table())

    def test_v0_ignores_state_and_keeps_values(self):
        m = load_link_model(*V0)
        np.testing.assert_array_equal(m.capacity_mbps(500.0), m.capacity_mbps(500.0, state="NLOS"))
        self.assertAlmostEqual(float(m.capacity_mbps(1000.0)), 0.695, places=3)
        self.assertEqual(m.reporting_snr_floor_db, -5.0)

    def test_technologies(self):
        nr = load_technology("config/network/technologies.json", "nr_sidelink_u2n_relay")
        ab = load_technology("config/network/technologies.json", "abstract_sidelink")
        self.assertEqual(nr.discovery_range_m, 1000.0)
        self.assertIsInstance(nr.link_model(), TR37885SidelinkModel)
        self.assertIsInstance(ab.link_model(), LinkCapacityModel)
        self.assertIn("smartphone", nr.extra["device_assumptions"].lower())

    def test_unknown_model_type_rejected(self):
        with self.assertRaises(KeyError):
            load_link_model("config/network/link_profiles_nr.json", "does_not_exist")


class TestTR37885Equations(unittest.TestCase):
    def setUp(self):
        self.u = load_link_model(*NR)
        self.h = load_link_model(*NRH)

    def test_pathloss_hand_computed(self):
        fc, d = 6.0, 1000.0
        self.assertAlmostEqual(float(self.u.path_loss_db(d, "LOS")),
                               38.77 + 16.7 * 3 + 18.2 * math.log10(fc), places=9)
        self.assertAlmostEqual(float(self.u.path_loss_db(d, "NLOS")),
                               36.85 + 30 * 3 + 18.9 * math.log10(fc), places=9)
        self.assertAlmostEqual(float(self.h.path_loss_db(d, "LOS")),
                               32.4 + 20 * 3 + 20 * math.log10(fc), places=9)
        self.assertEqual(float(self.u.path_loss_db(d, "LOS")), float(self.u.path_loss_db(d, "NLOSv")))
        with self.assertRaises(ValueError):
            self.h.path_loss_db(d, "NLOS")

    def test_los_probability(self):
        self.assertAlmostEqual(float(self.u.p_los(100.0)), min(1, 1.05 * math.exp(-1.14)), places=12)
        self.assertEqual(float(self.u.p_los(0.0)), 1.0)
        a, b, c = 2.1013e-6, -0.002, 1.0193
        self.assertAlmostEqual(float(self.h.p_los(300.0)), min(1, a * 9e4 + b * 300 + c), places=12)
        self.assertAlmostEqual(float(self.h.p_los(600.0)), 0.54 - 0.001 * 125, places=12)
        self.assertEqual(float(self.h.p_los(2000.0)), 0.0)
        p = self.u.state_probabilities(np.array([50.0, 500.0]))
        np.testing.assert_allclose(p["LOS"] + p["NLOSv"], 1.0)
        self.assertEqual(float(self.u.state_probabilities(50.0, same_street=False)["NLOS"]), 1.0)

    def test_shadow_and_blockage(self):
        self.assertEqual(self.u.shadow_std_db("LOS"), 3.0)
        self.assertEqual(self.u.shadow_std_db("NLOSv"), 3.0)
        self.assertEqual(self.u.shadow_std_db("NLOS"), 4.0)
        mu, sd = self.u.blockage_params_db(np.array([100.0, 1000.0]))   # Case 3 (all heights 1.6 m)
        np.testing.assert_allclose(mu, [5.0, 5.0 + (45 - 41)])
        np.testing.assert_allclose(sd, [4.0, 4.0])
        prof = dict(self.u.p, tx_antenna_height_m=0.75, rx_antenna_height_m=0.75)   # Case 2
        mu2, sd2 = TR37885SidelinkModel(prof).blockage_params_db(np.array([1000.0]))
        self.assertAlmostEqual(float(mu2[0]), 13.0)
        self.assertAlmostEqual(float(sd2[0]), 4.5)
        prof3 = dict(self.u.p, tx_antenna_height_m=3.0, rx_antenna_height_m=3.0)    # Case 1
        self.assertEqual(float(TR37885SidelinkModel(prof3).blockage_params_db(np.array([500.0]))[0][0]), 0.0)

    def test_snr_link_budget(self):
        noise = -174 + 10 * math.log10(20e6) + 9
        pl = 38.77 + 16.7 * 3 + 18.2 * math.log10(6.0)
        self.assertAlmostEqual(float(self.u.snr_db(1000.0, "LOS")), 23 + 3 + 3 - pl - noise, places=9)
        self.assertAlmostEqual(float(self.u.snr_db(1000.0, "NLOSv")), 23 + 6 - pl - noise - 9.0, places=9)

    def test_tr36942_table_a2(self):
        # Table A.2 (DL column): SNIR dB -> bps/Hz
        rows = {-11: 0.0, -10: 0.08, -5: 0.24, -1: 0.51, 6: 1.39, 10: 2.08, 15: 3.02, 19: 3.80}
        for snr, se in rows.items():
            self.assertAlmostEqual(round(float(self.u.spectral_efficiency(snr)), 2), se, places=9, msg=str(snr))
        self.assertAlmostEqual(float(self.u.spectral_efficiency(40)), 4.4)

    def _monte_carlo(self, m, d, same_street, n=400_000, seed=7):
        """Independent sampler written directly from the TR 37.885 text (state, shadowing, blockage)."""
        rng = np.random.default_rng(seed)
        if same_street:
            los = rng.random(n) < float(m.p_los(d))
            pl = np.where(los, float(m.path_loss_db(d, "LOS")), float(m.path_loss_db(d, "NLOSv")))
            mu, sd = m.blockage_params_db(np.array([d]))
            blk = np.where(los, 0.0, np.maximum(0.0, rng.normal(mu[0], sd[0], n)))
            sf = rng.normal(0.0, 3.0, n)
        else:
            pl = np.full(n, float(m.path_loss_db(d, "NLOS")))
            blk = 0.0
            sf = rng.normal(0.0, 4.0, n)
        return m._snr_from_pl(pl) - blk - sf

    def test_outage_and_expected_capacity_match_monte_carlo(self):
        for d, same in ((150.0, True), (1000.0, True), (2500.0, True), (200.0, False), (350.0, False)):
            snr = self._monte_carlo(self.u, d, same)
            ctx = {"same_street": same}
            for thr in (-10.0, 0.0, 10.0):
                self.assertAlmostEqual(float(self.u.outage_probability(d, thr, ctx)[0]),
                                       float((snr < thr).mean()), delta=4e-3, msg=f"d={d} same={same} thr={thr}")
            self.assertAlmostEqual(float(self.u.expected_capacity_mbps(d, ctx)[0]),
                                   float(self.u.capacity_from_snr_mbps(snr).mean()), delta=0.15,
                                   msg=f"d={d} same={same}")

    def test_los_outage_closed_form(self):
        mu = float(self.u._snr_from_pl(self.u.path_loss_db(1500.0, "LOS")))
        from scipy.stats import norm
        self.assertAlmostEqual(float(self.u._state_outage(1500.0, "LOS", 0.0)[0]),
                               float(norm.cdf((0.0 - mu) / 3.0)), places=12)

    def test_expected_capacity_bounds(self):
        d = np.array([50.0, 300.0, 1000.0])
        e = self.u.expected_capacity_mbps(d)
        self.assertTrue(np.all(e >= 0) and np.all(e <= 20 * 4.4 + 1e-9))
        self.assertTrue(np.all(np.diff(e) <= 1e-9))
        nlos = self.u.expected_capacity_mbps(d, {"same_street": False})
        self.assertTrue(np.all(nlos <= e + 1e-9))


class _ConstantLink(LinkModel):
    """Dummy technology used only to prove replaceability."""
    def __init__(self, profile):
        self.name = profile["name"]
        self.c = float(profile["capacity"])

    def snr_db(self, d_m, state=None):
        return np.full(np.shape(d_m), 10.0)

    def capacity_mbps(self, d_m, state=None):
        return np.full(np.shape(d_m), self.c)

    def capacity_table(self, distances_m=(50,)):
        return [{"distance_m": d, "capacity": self.c} for d in distances_m]


class TestReplaceability(unittest.TestCase):
    def test_new_technology_plugs_in_without_touching_other_modules(self):
        import json, os, tempfile
        if "constant_test" not in MODEL_TYPES:
            register_model_type("constant_test", _ConstantLink)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "profiles.json")
            json.dump({"flat": {"model": "constant_test", "capacity": 7.0}}, open(p, "w"))
            m = load_link_model(p, "flat")
            np.testing.assert_array_equal(m.link_capacity_mbps(np.array([10.0, 900.0]), np.array([5.0, 50.0])),
                                          [5.0, 7.0])
            self.assertIsNone(m.reporting_snr_floor_db)


if __name__ == "__main__":
    unittest.main()

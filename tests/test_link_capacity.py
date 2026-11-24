"""Tests for the provisional abstract sidelink capacity model (C3)."""
import unittest

import numpy as np

from src.network.link_capacity import LinkCapacityModel, load_link_profile


class TestLinkCapacity(unittest.TestCase):
    def setUp(self):
        self.m = LinkCapacityModel(load_link_profile("config/marketplace/link_profiles.json",
                                                     "abstract_sidelink_v0"))

    def test_monotone_decreasing(self):
        d = np.array([10, 50, 100, 200, 500, 1000], dtype=float)
        c = self.m.capacity_mbps(d)
        self.assertTrue(np.all(np.diff(c) <= 1e-12))
        self.assertTrue(np.all(np.diff(self.m.snr_db(d)) < 0))

    def test_spectral_efficiency_cap(self):
        p = self.m.p
        cap = p["implementation_efficiency"] * p["bandwidth_mhz"] * p["max_spectral_efficiency_bps_per_hz"]
        self.assertAlmostEqual(float(self.m.capacity_mbps(1.0)), cap, places=6)

    def test_individual_link_capacity_is_min_with_peak(self):
        b = self.m.link_capacity_mbps(np.array([50.0, 500.0]), np.array([20.0, 20.0]))
        self.assertAlmostEqual(b[0], 20.0)
        self.assertAlmostEqual(b[1], float(self.m.capacity_mbps(500.0)))

    def test_capacity_table_and_snr_flag(self):
        t = {r["distance_m"]: r for r in self.m.capacity_table()}
        self.assertEqual(sorted(t), [50, 100, 200, 500, 1000])
        self.assertFalse(t[50]["below_reporting_snr_floor"])
        self.assertTrue(t[1000]["below_reporting_snr_floor"])
        self.assertGreater(t[1000]["capacity_mbps_uncapped_by_pue"], 0)   # >0 but flagged, not "usable"


if __name__ == "__main__":
    unittest.main()

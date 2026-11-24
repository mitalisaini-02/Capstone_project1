"""
Tests for the Phase 3 simulation: expected vs realized capacity separation, realized channel,
online LSTM features, reliability layer, feasibility stages and reasons, and cache integrity.
"""
import json
import os
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.network.registry import load_link_model
from src.simulation.channel_layer import episode_normals
from src.simulation.feasibility import (STAGE_ORDER, evaluate_scenario, first_failure,
                                        provider_status, wireless_cause)

NR = ("config/network/link_profiles_nr.json", "nr_sidelink_tr37885_v1")
V0 = ("config/marketplace/link_profiles.json", "abstract_sidelink_v0")
THR = {"primary": 0.62, "sensitivity": 0.65}


class TestRealizedChannel(unittest.TestCase):
    def setUp(self):
        self.m = load_link_model(*NR)

    def test_zero_draws_equal_median_state_snr(self):
        d = np.array([80.0, 400.0, 950.0])
        for st in ("LOS", "NLOSv", "NLOS"):
            np.testing.assert_array_equal(self.m.realized_snr_db(d, st, 0.0, 0.0), self.m.snr_db(d, st))

    def test_draws_shift_snr_as_specified(self):
        d = np.array([300.0])
        base = float(self.m.realized_snr_db(d, "LOS", 0.0, 0.0)[0])
        self.assertAlmostEqual(float(self.m.realized_snr_db(d, "LOS", 1.5, 0.0)[0]), base - 3.0 * 1.5, places=12)
        self.assertAlmostEqual(float(self.m.realized_snr_db(d, "NLOS", 1.0, 0.0)[0]),
                               float(self.m.snr_db(d, "NLOS")[0]) - 4.0, places=12)
        mu, sd = self.m.blockage_params_db(d)
        v_lo = float(self.m.realized_snr_db(d, "NLOSv", 0.0, -10.0)[0])     # blockage clipped at 0 dB
        self.assertAlmostEqual(v_lo, base, places=12)
        v_hi = float(self.m.realized_snr_db(d, "NLOSv", 0.0, 1.0)[0])
        self.assertAlmostEqual(v_hi, base - (mu[0] + sd[0]), places=12)

    def test_unknown_state_rejected(self):
        with self.assertRaises(ValueError):
            self.m.realized_snr_db([100.0], "BLOCKED", 0.0, 0.0)

    def test_realized_capacity_is_mapping_of_realized_snr(self):
        d = np.array([100.0, 700.0])
        snr = self.m.realized_snr_db(d, ["LOS", "NLOSv"], [0.3, -1.0], [0.2, 0.5])
        np.testing.assert_array_equal(self.m.realized_capacity_mbps(d, ["LOS", "NLOSv"], [0.3, -1.0], [0.2, 0.5]),
                                      self.m.capacity_from_snr_mbps(snr))

    def test_deterministic_model_has_no_random_effects(self):
        v0 = load_link_model(*V0)
        d = np.array([100.0, 900.0])
        np.testing.assert_array_equal(v0.realized_capacity_mbps(d, "NLOS", 3.0, 3.0), v0.capacity_mbps(d))

    def test_episode_normals_reproducible_and_separate_stream(self):
        a = episode_normals(1, "run_x", 3, 4, 10.0)
        self.assertEqual(a, episode_normals(1, "run_x", 3, 4, 10.0))
        self.assertNotEqual(a, episode_normals(1, "run_x", 3, 4, 11.0))


class TestFeasibilityLogic(unittest.TestCase):
    def test_first_failure_order(self):
        n = 6
        flags = {s: np.ones(n, bool) for s in STAGE_ORDER}
        flags["provider_unavailable"][0] = False
        flags["insufficient_history"][[0, 1]] = False
        flags["reliability_below_threshold"][[1, 2]] = False
        flags["wireless_outage"][[2, 3]] = False
        flags["resource_zero"][[3, 4]] = False
        w = np.array(["wireless_outage_los"] * n, dtype=object)
        self.assertEqual(first_failure(flags, w).tolist(),
                         ["provider_unavailable", "insufficient_history", "reliability_below_threshold",
                          "wireless_outage_los", "resource_zero", "feasible"])

    def test_provider_status(self):
        pue = pd.DataFrame({"time_s": [5.0, 5.0, 5.0, 5.0, 9.0], "pue_id": [1, 2, 3, 4, 4],
                            "registered": [True, True, False, True, False],
                            "available": [True, False, False, False, False]})
        wd = pd.DataFrame({"pue_id": [4], "withdraw_time_s": [9.0]})
        self.assertEqual(provider_status(pue, wd).tolist(),
                         ["available", "unavailable", "not_registered", "unavailable", "withdrawn"])

    def test_wireless_cause(self):
        df = pd.DataFrame({"fallback": [False, False, False, False, True, False],
                           "realized_state": ["NLOS", "NLOS", "NLOSv", "LOS", "NLOS", "NLOS"],
                           "building_blocked": pd.array([True, pd.NA, False, False, pd.NA, True], dtype="boolean"),
                           "classification_source": ["geometry", "assumed_all_building_blocked", "geometry",
                                                     "geometry", "fallback:F2_all_building_blocked", "road_topology"]})
        self.assertEqual(wireless_cause(df).tolist(),
                         ["wireless_outage_building_blocked", "wireless_outage_nlos_assumed",
                          "wireless_outage_vehicle_blocked", "wireless_outage_los", "wireless_outage_fallback",
                          "wireless_outage_nlos_assumed"])


def _crafted():
    """One CUE (id 1) requesting at t=10 (5 Mbps); four in-range PUEs with different conditions."""
    run = SimpleNamespace(end_t={1: 100.0}, state=lambda t, c: (0.0, 0.0, 0.0))
    pue = pd.DataFrame({
        "time_s": [10.0] * 5, "pue_id": [5, 6, 7, 8, 9],
        "registered": [True, True, True, False, True], "available": [True, True, True, False, True],
        "link_peak_mbps": [30.0, 30.0, 30.0, np.nan, 4.0], "access_capacity_mbps": [50.0, 50.0, 2.0, np.nan, 50.0],
        "backhaul_capacity_mbps": [40.0, 40.0, 40.0, np.nan, 3.0]})
    files = {"pue_resources": pue,
             "cue_requests": pd.DataFrame({"request_id": ["r1"], "cue_id": [1], "request_time_s": [10.0],
                                           "service_duration_s": [1], "required_rate_mbps": [5.0]}),
             "gnodeb_resources": pd.DataFrame({"time_s": [10.0], "gnb_id": ["g1"], "available_capacity_mbps": [21.0]}),
             "gnodeb_sites": pd.DataFrame({"coverage_radius_m": [2000.0]}),
             "pue_withdrawals": pd.DataFrame(columns=["pue_id", "withdraw_time_s"])}
    ids = [5, 6, 7, 8, 9]
    conn = pd.DataFrame({"time_s": [10.0] * 5, "pue_id": ids, "cue_id": [1] * 5,
                         "distance_m": [100.0, 900.0, 200.0, 300.0, 150.0]})
    ch = pd.DataFrame({
        "time_s": [10.0] * 5, "pue_id": ids, "cue_id": [1] * 5, "distance_m": conn.distance_m,
        "channel_scenario": ["G"] * 5, "classification_source": ["geometry"] * 5, "fallback": [False] * 5,
        "fallback_reason": [""] * 5, "primary_eligible": [True] * 5,
        "building_blocked": pd.array([False, True, False, False, False], dtype="boolean"),
        "n_buildings_crossed": pd.array([0, 2, 0, 0, 0], dtype="Int64"),
        "p_los": [0.3, 0.0, 0.2, 0.1, 0.2], "p_nlosv": [0.7, 0.0, 0.8, 0.9, 0.8], "p_nlos": [0.0, 1.0, 0.0, 0.0, 0.0],
        "episode_id": [f"e{i}" for i in ids], "realized_state": ["NLOSv", "NLOS", "LOS", "LOS", "LOS"],
        "realized_snr_db": [20.0, -20.0, 25.0, 25.0, 25.0],
        "realized_radio_capacity_mbps": [60.0, 0.0, 80.0, 80.0, 80.0],
        "expected_radio_capacity_mbps": [1.0, 88.0, 1.0, 1.0, 1.0],     # deliberately contradictory
        "expected_outage_probability": [0.9, 0.0, 0.9, 0.9, 0.9]})
    rel = pd.DataFrame({"time_s": [10.0] * 5, "pue_id": ids, "cue_id": [1] * 5,
                        "lstm_window_available": [True, True, True, True, True],
                        "p_survive": [0.9, 0.95, 0.63, 0.9, 0.9]})
    sites = [("g1", 0.0, 0.0)]
    return run, files, sites, ch, rel, conn


class TestEvaluateScenario(unittest.TestCase):
    def setUp(self):
        self.run, self.files, self.sites, self.ch, self.rel, self.conn = _crafted()

    def evaluate(self, ch=None):
        return evaluate_scenario("s", self.run, self.files, self.sites, self.ch if ch is None else ch,
                                 self.rel, self.conn, THR, 1000.0)

    def test_stages_reasons_and_capacities(self):
        rs, cl = self.evaluate()
        r = cl.set_index("pue_id")
        self.assertEqual(r.loc[5, "first_failure_primary"], "feasible")
        self.assertEqual(r.loc[6, "first_failure_primary"], "wireless_outage_building_blocked")
        self.assertEqual(r.loc[7, "first_failure_primary"], "feasible")           # 0.63 >= 0.62
        self.assertEqual(r.loc[7, "first_failure_sensitivity"], "reliability_below_threshold")   # 0.63 < 0.65
        self.assertEqual(r.loc[8, "first_failure_primary"], "provider_unavailable")
        self.assertEqual(r.loc[5, "realized_link_capacity_mbps"], 30.0)           # C3 = min(peak 30, radio 60)
        self.assertEqual(r.loc[5, "binding_constraint"], "C3_link")
        self.assertEqual(r.loc[7, "binding_constraint"], "C4_access")
        self.assertEqual(r.loc[9, "binding_constraint"], "C5_backhaul")
        self.assertEqual(r.loc[9, "pue_max_rate_mbps"], 3.0)
        self.assertFalse(bool(r.loc[9, "covers_required_rate"]))
        self.assertTrue(bool(r.loc[5, "covers_required_rate"]))
        self.assertEqual(rs.iloc[0].n_in_range_pue, 5)
        self.assertTrue(rs.iloc[0].gnb_covered)
        self.assertEqual(rs.iloc[0].gnb_available_capacity_mbps, 21.0)

    def test_expected_capacity_never_drives_feasibility(self):
        _, a = self.evaluate()
        ch = self.ch.copy()
        ch["expected_radio_capacity_mbps"] = ch["expected_radio_capacity_mbps"][::-1].to_numpy() * 100
        ch["expected_outage_probability"] = 1.0 - ch["expected_outage_probability"]
        _, b = self.evaluate(ch)
        for c in ("feasible_primary", "feasible_sensitivity", "first_failure_primary", "wireless_ok",
                  "realized_link_capacity_mbps", "pue_max_rate_mbps"):
            pd.testing.assert_series_equal(a[c], b[c])

    def test_realized_capacity_drives_feasibility(self):
        ch = self.ch.copy()
        ch.loc[ch.pue_id == 5, "realized_radio_capacity_mbps"] = 0.0
        _, cl = self.evaluate(ch)
        self.assertEqual(cl.set_index("pue_id").loc[5, "first_failure_primary"], "wireless_outage_vehicle_blocked")


CACHE = "data/derived/phase3_cache/run_000"


@unittest.skipUnless(os.path.exists(f"{CACHE}/channel.csv.gz"), "Phase 3 cache not built")
class TestCacheIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = pd.read_csv(f"{CACHE}/channel.csv.gz")
        cls.rel = pd.read_csv(f"{CACHE}/reliability.csv.gz")
        cls.m = load_link_model(*NR)

    def test_realized_recomputes_exactly(self):
        c = self.ch.sample(500, random_state=0)
        snr = self.m.realized_snr_db(c.distance_m.to_numpy(), c.realized_state.to_numpy(object),
                                     c.z_shadow.to_numpy(), c.z_blockage.to_numpy())
        np.testing.assert_allclose(snr, c.realized_snr_db.to_numpy(), rtol=0, atol=1e-9)

    def test_expected_and_realized_are_distinct_columns(self):
        self.assertIn("expected_radio_capacity_mbps", self.ch.columns)
        self.assertIn("realized_radio_capacity_mbps", self.ch.columns)
        self.assertNotIn("capacity_mbps", self.ch.columns)
        self.assertFalse(np.allclose(self.ch.expected_radio_capacity_mbps, self.ch.realized_radio_capacity_mbps))

    def test_lstm_probabilities_match_adapter_on_training_windows(self):
        from src.reliability.lstm_distance import LSTMDistanceSurvivalModel
        ck = "outputs/large_dataset_experiment/checkpoints/"
        model = LSTMDistanceSurvivalModel(ck + "best_model.pt", ck + "model_config.json", ck + "scaler.json",
                                          ck + "checkpoint_metadata.json")
        meta = json.load(open("data/processed/large_dataset/metadata.json"))
        X = np.load("data/processed/large_dataset/pairwise_sequences.npz")["X"]
        idx = [i for i, m in enumerate(meta) if m["run_id"] == "run_000"][:300]
        key = pd.DataFrame({"time_s": [meta[i]["time_s"] for i in idx],
                            "pue_id": [int(meta[i]["provider_id"].split("__")[1]) for i in idx],
                            "cue_id": [int(meta[i]["consumer_id"].split("__")[1]) for i in idx]})
        got = key.merge(self.rel, on=["time_s", "pue_id", "cue_id"], how="left").p_survive.to_numpy()
        np.testing.assert_array_equal(got, model.predict_proba(X[idx].astype(np.float32)))


class TestOnlineFeatures(unittest.TestCase):
    @unittest.skipUnless(os.path.exists("data/raw/sumo/run_000/mobility.csv"), "trace missing")
    def test_bitwise_equal_to_training_windows(self):
        from src.mobility import load_run
        from src.reliability.online_features import OnlineFeatureBuilder
        meta = json.load(open("data/processed/large_dataset/metadata.json"))
        X = np.load("data/processed/large_dataset/pairwise_sequences.npz")["X"]
        idx = [i for i, m in enumerate(meta) if m["run_id"] == "run_000"]
        fb = OnlineFeatureBuilder(load_run("run_000").df)
        W, mask = fb.windows([meta[i]["time_s"] for i in idx],
                             [int(meta[i]["provider_id"].split("__")[1]) for i in idx],
                             [int(meta[i]["consumer_id"].split("__")[1]) for i in idx])
        self.assertTrue(mask.all())
        self.assertTrue(np.array_equal(W, X[idx]))
        self.assertIsNone(fb.window(1.0, 0, 1))          # no history at the first second


if __name__ == "__main__":
    unittest.main()

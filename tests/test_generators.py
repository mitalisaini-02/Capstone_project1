"""Tests for synthetic PUE / CUE / gNodeB generators, seeding and common random numbers."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.datagen.generators import generate_cues, generate_gnbs, generate_pues, load_gnb_sites
from src.datagen.io_utils import write_csv_gz
from src.datagen.mobility_source import load_run
from src.datagen.scenarios import (check_lstm_consistency, expand_configurations, load_json,
                                   resolve_parameters)
from src.datagen.validate import validate_scenario
from src.datagen.connectivity import build_connectivity

BASE = load_json("config/marketplace/base.json")
FROZEN = {"p_register": {"low": 0.3, "med": 0.6, "high": 0.9},
          "gnb_total_capacity_mbps": {"sufficient": 150.0, "insufficient": 60.0}}
REF = {"availability": "med", "demand": "high", "price": "mixed", "gnb_capacity": "sufficient",
       "coverage": "full", "withdrawals": "none"}
RUN = "run_000"   # validation run (never a test run in unit tests)


@unittest.skipUnless(os.path.exists(f"data/raw/sumo/{RUN}/mobility.csv"), "SUMO run not available")
class TestGenerators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mob = load_run(RUN)
        cls.sites = load_gnb_sites("infrastructure.add.xml")

    def params(self, **kw):
        return resolve_parameters(BASE, dict(REF, **kw), FROZEN)

    def test_identical_seed_identical_output(self):
        a = generate_pues(self.mob, self.params(), "s", 7, 0)[0]
        b = generate_pues(self.mob, self.params(), "s", 7, 0)[0]
        pd.testing.assert_frame_equal(a, b)
        c = generate_pues(self.mob, self.params(), "s", 7, 1)[0]
        self.assertFalse(a.equals(c))

    def test_common_random_numbers_across_price_regimes(self):
        a = generate_pues(self.mob, self.params(price="cheap"), "s", 7, 0)[0]
        b = generate_pues(self.mob, self.params(price="expensive"), "s", 7, 0)[0]
        cols = ["time_s", "pue_id", "registered", "available", "backhaul_capacity_mbps", "access_capacity_mbps"]
        pd.testing.assert_frame_equal(a[cols], b[cols])
        reg = a.registered
        self.assertTrue((b.price_per_gb[reg] > a.price_per_gb[reg]).all())

    def test_registration_monotone_in_p_register(self):
        lo = generate_pues(self.mob, resolve_parameters(BASE, REF, {**FROZEN, "p_register": {"low": .3, "med": .2, "high": .9}}), "s", 7, 0)[2]
        hi = generate_pues(self.mob, resolve_parameters(BASE, REF, {**FROZEN, "p_register": {"low": .3, "med": .8, "high": .9}}), "s", 7, 0)[2]
        self.assertTrue(set(lo[lo.registers].pue_id) <= set(hi[hi.registers].pue_id))

    def test_cue_requests_independent_of_pue_params(self):
        a = generate_cues(self.mob, self.params(availability="low"), "s", 7, 0)
        b = generate_cues(self.mob, self.params(availability="high", price="cheap"), "s", 7, 0)
        pd.testing.assert_frame_equal(a, b)

    def test_scenario_passes_validation_including_withdrawals(self):
        p = self.params(withdrawals="some")
        pue, wd, _ = generate_pues(self.mob, p, "s", 7, 0)
        cue = generate_cues(self.mob, p, "s", 7, 0)
        gnb, sites = generate_gnbs(self.mob, p, "s", 7, 0, self.sites)
        conn, _, _ = build_connectivity(self.mob, 1000.0, 10, 30)
        files = dict(pue_resources=pue, cue_requests=cue, gnodeb_resources=gnb,
                     gnodeb_sites=sites, pue_withdrawals=wd)
        ck = validate_scenario(self.mob, p, files, conn)
        self.assertTrue(ck.passed, ck.failures())

    def test_validation_detects_corruption(self):
        p = self.params()
        pue, wd, _ = generate_pues(self.mob, p, "s", 7, 0)
        cue = generate_cues(self.mob, p, "s", 7, 0)
        gnb, sites = generate_gnbs(self.mob, p, "s", 7, 0, self.sites)
        conn, _, _ = build_connectivity(self.mob, 1000.0, 10, 30)
        bad_pue = pue.copy(); bad_pue.loc[0, "x_m"] += 1.0                      # position drift
        bad_cue = cue.copy(); bad_cue.loc[0, "max_budget"] *= 1.01                # budget inconsistency
        ck = validate_scenario(self.mob, p, dict(pue_resources=bad_pue, cue_requests=bad_cue,
                                                 gnodeb_resources=gnb, gnodeb_sites=sites,
                                                 pue_withdrawals=wd), conn)
        failed = {c["check"] for c in ck.failures()}
        self.assertIn("pue.positions_match_sumo", failed)
        self.assertIn("cue.budget_consistent", failed)


class TestScenarioConfig(unittest.TestCase):
    def test_thirteen_configurations(self):
        cfgs = expand_configurations(load_json("config/marketplace/scenarios.json"))
        self.assertEqual(len(cfgs), 13)
        base = [c for c in cfgs if c["config_id"] == "C00_baseline"][0]["levels"]
        for c in cfgs:
            n_diff = sum(c["levels"][k] != base[k] for k in base)
            self.assertLessEqual(n_diff, 2)

    def test_uncalibrated_parameters_rejected(self):
        with self.assertRaises(ValueError):
            resolve_parameters(BASE, REF, {"p_register": {"med": "CALIBRATE"},
                                           "gnb_total_capacity_mbps": {"sufficient": 1.0}})

    def test_link_definition_matches_lstm(self):
        check_lstm_consistency(BASE, "config/stage3_large_dataset.json")
        bad = {**BASE, "link": {**BASE["link"], "comm_range_m": 200.0}}
        with self.assertRaises(ValueError):
            check_lstm_consistency(bad, "config/stage3_large_dataset.json")


class TestIO(unittest.TestCase):
    def test_gzip_deterministic_and_no_overwrite(self):
        df = pd.DataFrame({"a": [1.5, 2.25], "b": ["x", "y"]})
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = os.path.join(d, "1.csv.gz"), os.path.join(d, "2.csv.gz")
            write_csv_gz(df, p1)
            write_csv_gz(df, p2)
            self.assertEqual(open(p1, "rb").read(), open(p2, "rb").read())
            with self.assertRaises(FileExistsError):
                write_csv_gz(df, p1)


if __name__ == "__main__":
    unittest.main()

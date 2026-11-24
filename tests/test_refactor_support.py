"""Tests for coverage, units, strict config loading and the mobility package move."""
import json
import os
import tempfile
import unittest

import numpy as np

from src.common import units
from src.network.coverage import load_gnb_sites, nearest_site
from src.utils.config_loader import ConfigNotFoundError, load_config

SITES = [("A", 0.0, 0.0), ("B", 1000.0, 0.0)]


class TestCoverage(unittest.TestCase):
    def test_nearest_within_radius(self):
        ids = nearest_site([10.0, 990.0, 500.0, 5000.0], [0.0, 0.0, 0.0, 0.0], SITES, 600.0)
        self.assertEqual(list(ids), ["A", "B", "A", None])   # tie at 500 m -> first site

    def test_sites_from_infrastructure_file(self):
        sites = load_gnb_sites("infrastructure.add.xml")
        self.assertEqual([s[0] for s in sites], [f"gNB-{i}" for i in range(1, 7)])

    def test_datagen_wrapper_delegates(self):
        from src.datagen.generators import load_gnb_sites as legacy
        self.assertEqual(legacy("infrastructure.add.xml"), load_gnb_sites("infrastructure.add.xml"))


class TestUnits(unittest.TestCase):
    def test_conversions(self):
        self.assertEqual(units.MBIT_PER_GB, 8000.0)
        self.assertAlmostEqual(units.volume_gb(8.0, 1000), 1.0)
        self.assertAlmostEqual(units.cost(8.0, 1000, 2.5), 2.5)
        self.assertAlmostEqual(units.rate_mbps_for_volume(1.0, 1000), 8.0)
        from src.datagen.generators import MBIT_PER_GB
        self.assertIs(MBIT_PER_GB, units.MBIT_PER_GB)


class TestStrictConfigLoader(unittest.TestCase):
    def test_missing_raises_and_never_substitutes(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"x": 1}, open(os.path.join(d, "c.json"), "w"))
            self.assertEqual(load_config(os.path.join(d, "c.json")), {"x": 1})
            with self.assertRaises(ConfigNotFoundError):
                load_config(os.path.join(d, "c.yaml"))          # json sibling exists: still an error
            with self.assertRaises(ConfigNotFoundError):
                load_config(os.path.join(d, "missing.json"))

    def test_existing_project_configs_load(self):
        self.assertIn("training", load_config("config/stage3_large_dataset.json"))


class TestMobilityPackage(unittest.TestCase):
    def test_old_and_new_import_paths(self):
        from src.datagen.mobility_source import load_run as old
        from src.mobility import load_run as new
        from src.mobility.trace import RunMobility
        self.assertIs(old, new)
        self.assertTrue(callable(RunMobility))


if __name__ == "__main__":
    unittest.main()

"""Tests that roles and lifetimes come from the trace, never from ID ranges."""
import tempfile
import unittest

from src.data.validator import MobilityValidationError
from src.datagen.mobility_source import load_run
from tests.datagen_fixtures import two_vehicle_rows, write_trace


class TestMobilitySource(unittest.TestCase):
    def test_roles_from_data_not_id(self):
        with tempfile.TemporaryDirectory() as d:
            write_trace(d, "r", two_vehicle_rows(p_id=40, c_id=3))
            run = load_run("r", raw_dir=d)
            self.assertEqual(run.providers, [40])
            self.assertEqual(run.consumers, [3])
            self.assertEqual(run.start_t[3], 1.0)
            self.assertEqual(run.state(5.0, 3), (100.0, 0.0, 20.0))

    def test_conflicting_roles_rejected(self):
        rows = two_vehicle_rows()
        rows.append((81.0, 3, "provider", 0.0, 0.0, 0.0, "e"))
        with tempfile.TemporaryDirectory() as d:
            write_trace(d, "r", rows)
            with self.assertRaises(MobilityValidationError):
                load_run("r", raw_dir=d)

    def test_gaps_rejected(self):
        rows = [r for r in two_vehicle_rows() if not (r[1] == 3 and r[0] == 10.0)]
        with tempfile.TemporaryDirectory() as d:
            write_trace(d, "r", rows)
            with self.assertRaises(MobilityValidationError):
                load_run("r", raw_dir=d)


if __name__ == "__main__":
    unittest.main()

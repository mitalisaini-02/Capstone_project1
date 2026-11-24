"""Tests for derived connectivity and evaluation-only ground truth."""
import tempfile
import unittest

from src.data.validator import MobilityValidator
from src.datagen.connectivity import FUTURE_ONLY_COLUMNS, build_connectivity
from src.datagen.mobility_source import load_run
from src.features.labels import compute_link_survival_label
from tests.datagen_fixtures import two_vehicle_rows, write_trace


def _build(rows):
    d = tempfile.TemporaryDirectory()
    write_trace(d.name, "r", rows)
    run = load_run("r", raw_dir=d.name)
    vis, gt, meta = build_connectivity(run, 1000.0, 10, 30)
    return d, run, vis, gt


class TestConnectivity(unittest.TestCase):
    def test_range_history_and_out_of_range_break(self):
        # consumer at 20 t metres: in range for t <= 50, out of range from t = 51
        d, run, vis, gt = _build(two_vehicle_rows(t_end=80))
        self.assertEqual(vis.time_s.max(), 50.0)
        self.assertTrue(set(FUTURE_ONLY_COLUMNS).isdisjoint(vis.columns))
        r = gt[gt.time_s == 20.0].iloc[0]
        self.assertEqual(r.time_to_out_of_range_s, 31.0)
        self.assertEqual(r.first_break_cause, "out_of_range")
        self.assertEqual(int(r.survives_30s_lstm_def), 1)   # leaves range at 51 = t+31 > horizon
        self.assertEqual(int(gt[gt.time_s == 25.0].iloc[0].survives_30s_lstm_def), 0)   # 51 = t+26
        self.assertFalse(vis[vis.time_s == 9.0].history_ok.iloc[0])
        self.assertTrue(vis[vis.time_s == 10.0].history_ok.iloc[0])
        d.cleanup()

    def test_labels_match_original_label_function(self):
        d, run, vis, gt = _build(two_vehicle_rows(t_end=80, c_end=70))
        lc = MobilityValidator.validate_timestamps_and_lifecycles(run.df)
        traj = {(t, v): (x, y) for (t, v), (x, y, s) in run.lookup.items()}
        for r in gt.itertuples():
            ref = compute_link_survival_label(r.pue_id, r.cue_id, r.time_s, 30.0, 1000.0, lc, traj, 1.0)
            mine = None if str(r.survives_30s_lstm_def) == "<NA>" else int(r.survives_30s_lstm_def)
            self.assertEqual(ref, mine, f"t={r.time_s}")
        d.cleanup()

    def test_provider_vs_consumer_departure_vs_trace_end(self):
        # slow consumer stays in range; provider leaves at t=40
        d, run, vis, gt = _build(two_vehicle_rows(t_end=80, p_end=40, speed=1.0))
        r = gt[gt.time_s == 30.0].iloc[0]
        self.assertEqual(r.first_break_cause, "pue_departure")
        self.assertEqual(r.first_break_s, 11.0)
        self.assertFalse(r.pue_end_is_trace_end)
        d.cleanup()
        d, run, vis, gt = _build(two_vehicle_rows(t_end=80, c_end=40, speed=1.0))
        self.assertEqual(gt[gt.time_s == 30.0].iloc[0].first_break_cause, "cue_departure")
        d.cleanup()
        # both present until the end of the trace -> censored, not a departure
        d, run, vis, gt = _build(two_vehicle_rows(t_end=80, speed=1.0))
        r = gt[gt.time_s == 30.0].iloc[0]
        self.assertEqual(r.first_break_cause, "censored_trace_end")
        self.assertTrue(r.pue_end_is_trace_end and r.cue_end_is_trace_end)
        d.cleanup()


if __name__ == "__main__":
    unittest.main()

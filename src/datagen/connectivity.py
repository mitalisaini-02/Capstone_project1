"""
Pairwise PUE–CUE connectivity derived from REAL SUMO mobility (no synthetic values).

Two outputs, kept strictly separate:

1. MEC-visible candidate-discovery table (`pairwise_connectivity`)
   Uses only past and current observations at time t:
     distance_m, geo_in_range (distance <= comm_range_m),
     pue_observed_s / cue_observed_s (seconds observed up to and including t),
     history_ok (both >= history_window_s, i.e. an LSTM window [t-9, t] exists).
   "geo_in_range" is GEOGRAPHIC candidacy only. It is not a radio-quality claim,
   not an LSTM prediction, and not allocation feasibility.

2. Evaluation-only ground truth (`pairwise_ground_truth`) — uses FUTURE positions.
   Must never be read by the MEC pipeline.
     survives_30s_lstm_def : identical definition to the LSTM labels
                             (1 / 0 / null when censored by a departure or trace end)
     time_to_out_of_range_s: seconds until the pair first exceeds comm_range_m while
                             both vehicles are still present (null if it never does)
     pue_remaining_s / cue_remaining_s: seconds until that vehicle's last observation
     pue_end_is_trace_end / cue_end_is_trace_end: True if the last observation is the
                             end of the SUMO run (censored), i.e. NOT a real departure
     first_break_s, first_break_cause: out_of_range | pue_departure | cue_departure |
                             both_departure | censored_trace_end
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.mobility.trace import RunMobility

MEC_VISIBLE_COLUMNS = ["run_id", "time_s", "pue_id", "cue_id", "distance_m", "geo_in_range",
                       "pue_observed_s", "cue_observed_s", "history_ok"]
GROUND_TRUTH_COLUMNS = ["run_id", "time_s", "pue_id", "cue_id", "survives_30s_lstm_def",
                        "time_to_out_of_range_s", "pue_remaining_s", "cue_remaining_s",
                        "pue_end_is_trace_end", "cue_end_is_trace_end",
                        "first_break_s", "first_break_cause"]
# Columns that reveal the future; asserted absent from every MEC-visible file.
FUTURE_ONLY_COLUMNS = set(GROUND_TRUTH_COLUMNS) - {"run_id", "time_s", "pue_id", "cue_id"}


def _series(run: RunMobility, vid: int, t0: float, t1: float) -> np.ndarray:
    ts = np.arange(t0, t1 + 1.0)
    return np.array([run.lookup[(t, vid)][:2] for t in ts])


def build_connectivity(run: RunMobility, comm_range_m: float, history_window_s: int,
                       horizon_s: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    trace_end = float(run.times.max())
    pos_cache = {v: _series(run, v, run.start_t[v], run.end_t[v]) for v in run.roles}

    vis, gt = [], []
    for p in run.providers:
        for c in run.consumers:
            t0 = max(run.start_t[p], run.start_t[c])
            t1 = min(run.end_t[p], run.end_t[c])
            if t1 < t0:
                continue
            ts = np.arange(t0, t1 + 1.0)
            pp = pos_cache[p][int(t0 - run.start_t[p]): int(t1 - run.start_t[p]) + 1]
            cc = pos_cache[c][int(t0 - run.start_t[c]): int(t1 - run.start_t[c]) + 1]
            d = np.hypot(pp[:, 0] - cc[:, 0], pp[:, 1] - cc[:, 1])
            inr = d <= comm_range_m
            if not inr.any():
                continue
            n = len(ts)
            idx = np.arange(n)
            # next index strictly after i where the pair is out of range
            exceed_idx = np.where(~inr, idx, n + 10 ** 6)
            nxt = np.minimum.accumulate(exceed_idx[::-1])[::-1]
            next_after = np.concatenate([nxt[1:], [n + 10 ** 6]])

            sel = np.where(inr)[0]
            t_sel = ts[sel]
            tto = next_after[sel] - sel
            tto_valid = next_after[sel] < n
            # LSTM-definition label: needs both vehicles observed through t + horizon
            complete = (t_sel + horizon_s) <= t1
            survives = np.where(complete, (tto > horizon_s) | ~tto_valid, np.nan)

            p_rem = run.end_t[p] - t_sel
            c_rem = run.end_t[c] - t_sel
            p_trace_end = run.end_t[p] >= trace_end
            c_trace_end = run.end_t[c] >= trace_end
            p_obs = t_sel - run.start_t[p] + 1
            c_obs = t_sel - run.start_t[c] + 1

            for k, i in enumerate(sel):
                t = float(t_sel[k])
                vis.append((run.run_id, t, p, c, float(d[i]), True,
                            float(p_obs[k]), float(c_obs[k]),
                            bool(p_obs[k] >= history_window_s and c_obs[k] >= history_window_s)))
                # first break: out of range (while both present) vs departure (+1 s after last obs)
                cands = []
                if tto_valid[k]:
                    cands.append((float(tto[k]), "out_of_range"))
                p_dep = None if p_trace_end else float(p_rem[k]) + 1.0
                c_dep = None if c_trace_end else float(c_rem[k]) + 1.0
                if p_dep is not None and c_dep is not None and p_dep == c_dep:
                    cands.append((p_dep, "both_departure"))
                else:
                    if p_dep is not None:
                        cands.append((p_dep, "pue_departure"))
                    if c_dep is not None:
                        cands.append((c_dep, "cue_departure"))
                if cands:
                    fb, cause = min(cands, key=lambda z: z[0])
                else:
                    fb, cause = float("nan"), "censored_trace_end"
                gt.append((run.run_id, t, p, c,
                           None if np.isnan(survives[k]) else int(survives[k]),
                           float(tto[k]) if tto_valid[k] else float("nan"),
                           float(p_rem[k]), float(c_rem[k]),
                           bool(p_trace_end), bool(c_trace_end), fb, cause))

    vis_df = pd.DataFrame(vis, columns=MEC_VISIBLE_COLUMNS).sort_values(
        ["time_s", "cue_id", "pue_id"]).reset_index(drop=True)
    gt_df = pd.DataFrame(gt, columns=GROUND_TRUTH_COLUMNS).sort_values(
        ["time_s", "cue_id", "pue_id"]).reset_index(drop=True)
    gt_df["survives_30s_lstm_def"] = gt_df["survives_30s_lstm_def"].astype("Int64")

    meta = {
        "run_id": run.run_id, "comm_range_m": comm_range_m, "history_window_s": history_window_s,
        "prediction_horizon_s": horizon_s, "trace_end_s": trace_end,
        "rows_in_range": int(len(vis_df)),
        "rows_history_ok": int(vis_df["history_ok"].sum()) if len(vis_df) else 0,
        "unique_pairs": int(vis_df[["pue_id", "cue_id"]].drop_duplicates().shape[0]) if len(vis_df) else 0,
        "first_break_cause_counts": gt_df["first_break_cause"].value_counts().to_dict(),
        "survives_30s_counts": {str(k): int(v) for k, v in
                                gt_df["survives_30s_lstm_def"].value_counts(dropna=False).items()},
        "origin": "Derived from SUMO mobility; no synthetic values.",
    }
    return vis_df, gt_df, meta

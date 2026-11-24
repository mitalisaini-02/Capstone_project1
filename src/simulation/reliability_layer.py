"""
Per-run reliability layer: LSTM survival probability for every geographic candidate row.
Uses the unchanged LinkSurvivalPredictor through the ReliabilityModel adapter and the unchanged
training preprocessing (OnlineFeatureBuilder). No threshold is applied here (policy is separate).
"""
import numpy as np
import pandas as pd

from src.reliability.online_features import OnlineFeatureBuilder


def build_reliability_table(conn: pd.DataFrame, mobility_df: pd.DataFrame, model) -> pd.DataFrame:
    fb = OnlineFeatureBuilder(mobility_df, history_window_s=model.history_window_s)
    W, mask = fb.windows(conn.time_s.to_numpy(), conn.pue_id.to_numpy(), conn.cue_id.to_numpy())
    p = np.full(len(conn), np.nan)
    if len(W):
        p[mask] = model.predict_proba(W)
    out = conn[["time_s", "pue_id", "cue_id"]].copy()
    out["lstm_window_available"] = mask
    out["p_survive"] = p
    out["reliability_model"] = model.name
    out["model_label_range_m"] = model.label_range_m
    return out

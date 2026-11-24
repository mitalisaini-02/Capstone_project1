"""
Per-run channel layer (scenario-independent: depends only on real mobility, building geometry,
the NR link model and the channel seed). Computed once per run and cached, so the expensive
geometry test is never repeated per marketplace scenario. Outputs are identical to calling the
classifiers directly (caching only).

Two capacity quantities are produced and must never be confused:
  expected_radio_capacity_mbps  probability-weighted over channel states, shadowing and blockage.
                                REPORTING / statistical comparison only.
  realized_radio_capacity_mbps  capacity of the ONE sampled channel of this link episode
                                (realised state + fixed per-episode shadowing/blockage draws).
                                The ONLY capacity used for simulated feasibility and allocation inputs.
"""
import zlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.network.channel_state import (AllBuildingBlockedClassifier, AllBuildingClearClassifier,
                                       BuildingGeometryClassifier, RoadTopologyClassifier,
                                       assign_episode_states, build_pairs)

REALIZED_STREAM = 0x5EED   # separate random stream from the episode state uniform


def episode_normals(seed: int, run_id: str, pue: int, cue: int, start_t: float) -> Tuple[float, float]:
    """(z_shadow, z_blockage): standard-normal draws fixed for one link episode.
    Keyed like the episode state uniform (scenario-independent), on a separate stream."""
    ss = np.random.SeedSequence([int(seed), zlib.crc32(run_id.encode()) & 0xFFFFFFFF,
                                 int(pue), int(cue), int(round(start_t)), REALIZED_STREAM])
    z = np.random.default_rng(ss).standard_normal(2)
    return float(z[0]), float(z[1])


def make_classifiers(layer, link_model, map_scenario: str, fallback_label: str) -> Dict[str, object]:
    f2 = AllBuildingBlockedClassifier()
    f3 = AllBuildingClearClassifier(link_model, map_scenario)
    fallback = {"F2": f2, "F3": f3}.get(fallback_label)
    if fallback is None:
        raise ValueError(f"unsupported G fallback scenario '{fallback_label}'")
    return {"G": BuildingGeometryClassifier(layer, link_model, map_scenario, fallback=fallback),
            "F1": RoadTopologyClassifier(link_model, map_scenario), "F2": f2, "F3": f3}


def build_channel_table(run_id: str, conn: pd.DataFrame, positions, road_ids, classifiers: Dict,
                        scenarios, link_model, seed: int) -> pd.DataFrame:
    pairs = build_pairs(conn, positions, road_ids)
    frames = []
    for label in scenarios:
        cls = classifiers[label].classify(pairs)
        st = assign_episode_states(cls, run_id, seed)
        frames.append(st)
    df = pd.concat(frames, ignore_index=True)
    # per-episode realised draws (identical for every channel scenario)
    keys = df[["episode_id", "pue_id", "cue_id", "episode_start_s"]].drop_duplicates("episode_id")
    draws = {r.episode_id: episode_normals(seed, run_id, r.pue_id, r.cue_id, r.episode_start_s)
             for r in keys.itertuples(index=False)}
    df["z_shadow"] = df["episode_id"].map(lambda e: draws[e][0])
    df["z_blockage"] = df["episode_id"].map(lambda e: draws[e][1])
    d = df["distance_m"].to_numpy(float)
    df["realized_snr_db"] = link_model.realized_snr_db(d, df["realized_state"].to_numpy(object),
                                                       df["z_shadow"].to_numpy(), df["z_blockage"].to_numpy())
    df["realized_radio_capacity_mbps"] = link_model.capacity_from_snr_mbps(df["realized_snr_db"].to_numpy())
    df["realized_outage"] = df["realized_radio_capacity_mbps"] <= 0.0
    pl, pv, pn = (df[c].to_numpy(float) for c in ("p_los", "p_nlosv", "p_nlos"))
    df["expected_radio_capacity_mbps"] = link_model.expected_capacity_given_states(d, pl, pv, pn)
    df["expected_outage_probability"] = link_model.outage_probability_given_states(
        d, link_model.snr_min_db, pl, pv, pn)
    df = df.rename(columns={"scenario_label": "channel_scenario"})
    return df.sort_values(["channel_scenario", "time_s", "cue_id", "pue_id"]).reset_index(drop=True)


def build_episode_table(run_id: str, channel_g: pd.DataFrame, gt: pd.DataFrame, trace_end: float) -> pd.DataFrame:
    """Physical link episodes (continuous geographic candidacy) and why each ended (ground truth)."""
    ep = channel_g.groupby("episode_id").agg(pue_id=("pue_id", "first"), cue_id=("cue_id", "first"),
                                             start_s=("time_s", "min"), end_s=("time_s", "max"),
                                             n_seconds=("time_s", "size")).reset_index()
    last = ep.merge(gt[["time_s", "pue_id", "cue_id", "first_break_s", "first_break_cause"]],
                    left_on=["end_s", "pue_id", "cue_id"], right_on=["time_s", "pue_id", "cue_id"],
                    how="left").drop(columns=["time_s"])
    cause = last["first_break_cause"].astype(object)
    # an episode that ends while the pair stays in range one second later cannot happen; flag it
    cause = np.where(last["end_s"] >= trace_end, "trace_end",
                     np.where(last["first_break_s"] == 1.0, cause, "inconsistent"))
    last["end_cause"] = cause
    last.insert(0, "run_id", run_id)
    return last.drop(columns=["first_break_cause"])

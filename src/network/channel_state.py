"""
Building-aware channel-state classification for PUE–CUE sidelink pairs.

Order (conceptually following TR 37.885 §6.2):
  1. Building geometry: if the direct 2-D path crosses a mapped building interior -> NLOS
     (deterministic, recalculated from the current positions every second).
  2. Building-clear path: urban LOS/NLOSv probability model, P(LOS) from TR 37.885 Table 6.2-1,
     P(NLOSv) = 1 - P(LOS). Vehicle blockage exists ONLY inside the NLOSv state; it is never
     detected geometrically from simulated vehicles (simulated traffic is too sparse).
  3. Link quality is then computed by the link model for each state (not in this module).

IMPORTANT
  * Applying the urban P(LOS) to building-clear paths between DIFFERENT road edges is a
    geometry-based EXTENSION of TR 37.885, whose model is defined for vehicles in the same street.
  * OSM buildings are incomplete and have no heights: scenario G is an OPTIMISTIC estimate.

Four labelled scenarios (never mixed; every row records its source and fallback status):
  G   BuildingGeometryClassifier   primary, geometry-based
  F1  RoadTopologyClassifier       same OSM way -> LOS/NLOSv model, otherwise NLOS
  F2  AllBuildingBlockedClassifier every pair NLOS (conservative bound)
  F3  AllBuildingClearClassifier   every pair LOS/NLOSv model (optimistic bound)
Rows that G cannot classify from geometry (endpoint outside the building-data coverage, or an
endpoint inside a footprint) are flagged and given the probabilities of a labelled fallback
scenario (F2 by default). They are excluded from primary geometry-based results.
"""
import zlib
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.network.buildings import BuildingLayer

REQUIRED_COLUMNS = ["time_s", "pue_id", "cue_id", "distance_m", "pue_x", "pue_y", "cue_x", "cue_y"]
OUTPUT_COLUMNS = ["time_s", "pue_id", "cue_id", "distance_m", "scenario_label", "classifier",
                  "classification_source", "fallback", "fallback_reason", "primary_eligible",
                  "building_blocked", "n_buildings_crossed", "p_los", "p_nlosv", "p_nlos"]

SRC_GEOMETRY = "geometry"
SRC_ROAD = "road_topology"
SRC_ROAD_INTERNAL = "road_topology_internal_edge"
SRC_ALL_BLOCKED = "assumed_all_building_blocked"
SRC_ALL_CLEAR = "assumed_all_building_clear"
REASON_OUTSIDE = "outside_building_coverage"
REASON_ENDPOINT = "endpoint_inside_building"


class UrbanLosProbability:
    """P(LOS) provider bound to a link model's scenario. Refuses scenario mismatches."""

    def __init__(self, link_model, map_scenario: str):
        scen = getattr(link_model, "scenario", None)
        if scen is None or not hasattr(link_model, "p_los"):
            raise TypeError("link model provides no LOS-probability model (p_los/scenario)")
        if scen != map_scenario:
            raise ValueError(f"link model scenario '{scen}' does not match map scenario '{map_scenario}': "
                             f"{scen} LOS probabilities must not be applied to this map")
        if map_scenario != "urban":
            raise ValueError("building-aware classification is defined for the urban scenario only "
                             "(TR 37.885 has no building/NLOS state on highways)")
        self.scenario = scen
        self._f = link_model.p_los

    def __call__(self, d_m):
        return np.asarray(self._f(np.asarray(d_m, dtype=float)), dtype=float)


def _check_input(pairs: pd.DataFrame, extra=()):
    missing = [c for c in list(REQUIRED_COLUMNS) + list(extra) if c not in pairs.columns]
    if missing:
        raise ValueError(f"pairs missing columns: {missing}")
    if not np.all(np.isfinite(pairs[["distance_m", "pue_x", "pue_y", "cue_x", "cue_y"]].to_numpy(float))):
        raise ValueError("non-finite positions or distances in pairs")


def _frame(pairs, label, name, source, fallback, reason, eligible, blocked, ncross, pl, pv, pn):
    out = pd.DataFrame({
        "time_s": pairs["time_s"].to_numpy(), "pue_id": pairs["pue_id"].to_numpy(),
        "cue_id": pairs["cue_id"].to_numpy(), "distance_m": pairs["distance_m"].to_numpy(float),
        "scenario_label": label, "classifier": name, "classification_source": source,
        "fallback": fallback, "fallback_reason": reason, "primary_eligible": eligible,
        "building_blocked": pd.array(blocked, dtype="boolean"),
        "n_buildings_crossed": pd.array(ncross, dtype="Int64"),
        "p_los": pl, "p_nlosv": pv, "p_nlos": pn})
    return out[OUTPUT_COLUMNS]


class ChannelStateClassifier(ABC):
    label: str
    name: str

    @abstractmethod
    def classify(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """pairs (REQUIRED_COLUMNS) -> OUTPUT_COLUMNS, one row per input row, same order."""


class AllBuildingBlockedClassifier(ChannelStateClassifier):
    label, name = "F2", "all_building_blocked"

    def classify(self, pairs):
        _check_input(pairs)
        n = len(pairs)
        z, o = np.zeros(n), np.ones(n)
        return _frame(pairs, self.label, self.name, SRC_ALL_BLOCKED, False, "", True,
                      [True] * n, [None] * n, z, z, o)


class AllBuildingClearClassifier(ChannelStateClassifier):
    label, name = "F3", "all_building_clear"

    def __init__(self, link_model, map_scenario: str = "urban"):
        self.p_los = UrbanLosProbability(link_model, map_scenario)

    def classify(self, pairs):
        _check_input(pairs)
        n = len(pairs)
        pl = self.p_los(pairs["distance_m"].to_numpy(float))
        return _frame(pairs, self.label, self.name, SRC_ALL_CLEAR, False, "", True,
                      [False] * n, [None] * n, pl, 1.0 - pl, np.zeros(n))


def osm_way_of(edge_id: str) -> Optional[str]:
    """OSM way id of a SUMO edge id ('-123#4' -> '123'); None for internal junction edges (':...')."""
    e = str(edge_id)
    if e.startswith(":") or e in ("", "nan"):
        return None
    return e.lstrip("-").split("#")[0]


class RoadTopologyClassifier(ChannelStateClassifier):
    """F1: same OSM way -> urban LOS/NLOSv model; different way or internal junction edge -> NLOS.
    Approximation only: road identity is not proof of an unobstructed radio path."""
    label, name = "F1", "road_topology"

    def __init__(self, link_model, map_scenario: str = "urban"):
        self.p_los = UrbanLosProbability(link_model, map_scenario)

    def classify(self, pairs):
        _check_input(pairs, extra=("pue_edge", "cue_edge"))
        wp = [osm_way_of(e) for e in pairs["pue_edge"]]
        wc = [osm_way_of(e) for e in pairs["cue_edge"]]
        internal = np.array([a is None or b is None for a, b in zip(wp, wc)])
        same = np.array([(a is not None) and a == b for a, b in zip(wp, wc)])
        pl = np.where(same, self.p_los(pairs["distance_m"].to_numpy(float)), 0.0)
        pv = np.where(same, 1.0 - pl, 0.0)
        pn = np.where(same, 0.0, 1.0)
        src = np.where(internal, SRC_ROAD_INTERNAL, SRC_ROAD)
        return _frame(pairs, self.label, self.name, src, False, "", True,
                      (~same).tolist(), [None] * len(pairs), pl, pv, pn)


class BuildingGeometryClassifier(ChannelStateClassifier):
    """G: building-footprint intersection (primary)."""
    label, name = "G", "building_geometry"

    def __init__(self, layer: BuildingLayer, link_model, map_scenario: str = "urban",
                 fallback: Optional[ChannelStateClassifier] = None):
        if not isinstance(layer, BuildingLayer):
            raise TypeError("layer must be a BuildingLayer")
        self.layer = layer
        self.p_los = UrbanLosProbability(link_model, map_scenario)
        self.fallback = fallback or AllBuildingBlockedClassifier()
        if isinstance(self.fallback, BuildingGeometryClassifier):
            raise ValueError("fallback must be a non-geometry scenario")

    def classify(self, pairs):
        _check_input(pairs)
        n = len(pairs)
        px, py = pairs["pue_x"].to_numpy(float), pairs["pue_y"].to_numpy(float)
        qx, qy = pairs["cue_x"].to_numpy(float), pairs["cue_y"].to_numpy(float)
        d = pairs["distance_m"].to_numpy(float)
        reason = np.array([""] * n, dtype=object)
        blocked = [None] * n
        ncross = [None] * n
        for i in range(n):
            if not (self.layer.in_coverage(px[i], py[i]) and self.layer.in_coverage(qx[i], qy[i])):
                reason[i] = REASON_OUTSIDE
                continue
            if self.layer.endpoint_building(px[i], py[i]) >= 0 or self.layer.endpoint_building(qx[i], qy[i]) >= 0:
                reason[i] = REASON_ENDPOINT
                continue
            k = len(self.layer.blocking_buildings(px[i], py[i], qx[i], qy[i]))
            blocked[i], ncross[i] = k > 0, k
        geo = reason == ""
        geo_blocked = np.array([bool(b) if b is not None else False for b in blocked])
        pl_clear = self.p_los(d)
        pl = np.where(geo & ~geo_blocked, pl_clear, 0.0)
        pv = np.where(geo & ~geo_blocked, 1.0 - pl_clear, 0.0)
        pn = np.where(geo & geo_blocked, 1.0, 0.0)
        src = np.array([SRC_GEOMETRY] * n, dtype=object)
        if (~geo).any():
            fb = self.fallback.classify(pairs.loc[~geo])
            pl[~geo], pv[~geo], pn[~geo] = fb.p_los.to_numpy(), fb.p_nlosv.to_numpy(), fb.p_nlos.to_numpy()
            src[~geo] = [f"fallback:{self.fallback.label}_{self.fallback.name}"] * int((~geo).sum())
        return _frame(pairs, self.label, self.name, src, ~geo, reason, geo, blocked, ncross, pl, pv, pn)


# --------------------------------------------------------------------------- episode state rule
def _episode_uniform(seed: int, run_id: str, pue: int, cue: int, start_t: float) -> float:
    ss = np.random.SeedSequence([int(seed), zlib.crc32(run_id.encode()) & 0xFFFFFFFF,
                                 int(pue), int(cue), int(round(start_t))])
    return float(np.random.default_rng(ss).random())


def assign_episode_states(classified: pd.DataFrame, run_id: str, seed: int) -> pd.DataFrame:
    """
    Reproducible per-link random state within a continuous link episode.

    Episode: maximal run of consecutive seconds (step 1 s) in which a (PUE, CUE) pair is a
    candidate. One uniform U per episode, keyed by (seed, run, PUE, CUE, episode start) and
    independent of the scenario label, so G/F1/F2/F3 share the same randomness.
    Realised state each second, from that second's probabilities (order LOS, NLOSv, NLOS):
        LOS if U < p_los;  NLOSv if U < p_los + p_nlosv;  else NLOS.
    Building blockage (p_nlos = 1) is therefore deterministic from geometry; on building-clear
    seconds the state changes only when P(LOS)(d) crosses the fixed U, so it is NOT redrawn every
    second. Shadowing realisations are not produced here (no spatially correlated shadowing).
    """
    df = classified.sort_values(["pue_id", "cue_id", "time_s"]).copy()
    gap = (df["time_s"].diff() != 1.0) | (df["pue_id"] != df["pue_id"].shift()) | \
          (df["cue_id"] != df["cue_id"].shift())
    df["episode_index"] = gap.cumsum()
    starts = df.groupby("episode_index")["time_s"].transform("min")
    df["episode_start_s"] = starts
    keys = df[["episode_index", "pue_id", "cue_id", "episode_start_s"]].drop_duplicates("episode_index")
    u_map = {r.episode_index: _episode_uniform(seed, run_id, r.pue_id, r.cue_id, r.episode_start_s)
             for r in keys.itertuples(index=False)}
    df["episode_u"] = df["episode_index"].map(u_map)
    u = df["episode_u"].to_numpy()
    c1 = df["p_los"].to_numpy()
    c2 = c1 + df["p_nlosv"].to_numpy()
    df["realized_state"] = np.where(u < c1, "LOS", np.where(u < c2, "NLOSv", "NLOS"))
    df["episode_id"] = (run_id + ":" + df["pue_id"].astype(str) + "-" + df["cue_id"].astype(str)
                        + "@" + df["episode_start_s"].astype(int).astype(str))
    return df.drop(columns=["episode_index"]).sort_index()


def build_pairs(conn: pd.DataFrame, positions: Dict[Tuple[float, int], Tuple[float, float]],
                road_ids: Optional[Dict[Tuple[float, int], str]] = None) -> pd.DataFrame:
    """Joins candidate rows (time_s, pue_id, cue_id, distance_m) with positions (and edges)."""
    p = [positions[(t, v)] for t, v in zip(conn.time_s, conn.pue_id)]
    c = [positions[(t, v)] for t, v in zip(conn.time_s, conn.cue_id)]
    out = pd.DataFrame({"time_s": conn.time_s.to_numpy(), "pue_id": conn.pue_id.to_numpy(),
                        "cue_id": conn.cue_id.to_numpy(), "distance_m": conn.distance_m.to_numpy(float),
                        "pue_x": [a[0] for a in p], "pue_y": [a[1] for a in p],
                        "cue_x": [a[0] for a in c], "cue_y": [a[1] for a in c]})
    if road_ids is not None:
        out["pue_edge"] = [road_ids[(t, v)] for t, v in zip(conn.time_s, conn.pue_id)]
        out["cue_edge"] = [road_ids[(t, v)] for t, v in zip(conn.time_s, conn.cue_id)]
    return out

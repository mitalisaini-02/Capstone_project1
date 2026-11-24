"""
Configuration loading, scenario expansion and reproducible seeding.

Seeding / common random numbers
  Every random stream is keyed by (base_seed, run_id, replicate, stream, entity_id)
  and is INDEPENDENT of the scenario configuration. Generators draw a fixed set of
  uniforms per entity and only transform them with the configuration parameters.
  Therefore (a) identical seeds give identical files, and (b) changing one factor
  (e.g. price regime) leaves all other draws unchanged, so configurations and
  marketplace methods are compared on matched randomness.
"""
import copy
import hashlib
import json
import zlib
from typing import Any, Dict, List

import numpy as np

STREAMS = {"pue": 1, "cue": 2, "gnb": 3}


def stable_int(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def entity_rng(base_seed: int, run_id: str, replicate: int, stream: str, entity: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(base_seed), stable_int(run_id), int(replicate),
                                 STREAMS[stream], int(entity)])
    return np.random.default_rng(ss)


def seed_key(base_seed: int, run_id: str, replicate: int) -> Dict[str, Any]:
    return {"base_seed": base_seed, "run_id": run_id, "run_key": stable_int(run_id),
            "replicate": replicate, "streams": STREAMS,
            "entity_seed": "SeedSequence([base_seed, run_key, replicate, stream, entity_id])"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def check_lstm_consistency(base: Dict[str, Any], lstm_config_path: str) -> None:
    """The marketplace link definition must match the LSTM training labels exactly."""
    lstm = load_json(lstm_config_path)["link"]
    for k in ("comm_range_m", "history_window_s", "prediction_horizon_s"):
        if float(base["link"][k]) != float(lstm[k]):
            raise ValueError(f"link.{k}={base['link'][k]} differs from LSTM config ({lstm[k]}). "
                             f"Changing it would invalidate the trained LSTM labels.")


def expand_configurations(scenarios: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in scenarios["configurations"]:
        levels = dict(scenarios["baseline"])
        for f in scenarios["factors"]:
            if f in c:
                levels[f] = c[f]
        if c["config_id"] in seen:
            raise ValueError(f"duplicate config_id {c['config_id']}")
        seen.add(c["config_id"])
        out.append({"config_id": c["config_id"], "levels": levels})
    return out


def resolve_parameters(base: Dict[str, Any], levels: Dict[str, str],
                       frozen: Dict[str, Any]) -> Dict[str, Any]:
    """Concrete generator parameters for one configuration, using frozen calibrated values."""
    pue_lvl = copy.deepcopy(base["pue"]["availability_levels"][levels["availability"]])
    pue_lvl["p_register"] = frozen["p_register"][levels["availability"]]
    gnb_lvl = copy.deepcopy(base["gnb"]["capacity_levels"][levels["gnb_capacity"]])
    gnb_lvl["total_capacity_mbps"] = frozen["gnb_total_capacity_mbps"][levels["gnb_capacity"]]
    for name, v in (("p_register", pue_lvl["p_register"]),
                    ("total_capacity_mbps", gnb_lvl["total_capacity_mbps"])):
        if isinstance(v, str):
            raise ValueError(f"{name} is not calibrated ({v}); run calibration and freeze first.")
    return {
        "levels": dict(levels),
        "step_s": base["time"]["step_s"],
        "gnb_price_per_gb": base["pricing"]["gnb_price_per_gb"],
        "pue": {
            "registration_delay_s": base["pue"]["registration_delay_s"],
            "backhaul_base_mbps": base["pue"]["backhaul_base_mbps"],
            "backhaul_ar1_rho": base["pue"]["backhaul_ar1_rho"],
            "backhaul_log_sigma": base["pue"]["backhaul_log_sigma"],
            "backhaul_multiplier_clip": base["pue"]["backhaul_multiplier_clip"],
            "access_capacity_mbps": base["pue"]["access_capacity_mbps"],
            "link_peak_mbps": base["pue"]["link_peak_mbps"],
            "p_register": float(pue_lvl["p_register"]),
            "mean_on_s": pue_lvl["mean_on_s"], "mean_off_s": pue_lvl["mean_off_s"],
            "price_multiplier": base["pue"]["price_regimes"][levels["price"]],
            "withdrawal_hazard_per_s": base["pue"]["withdrawal_levels"][levels["withdrawals"]],
        },
        "cue": {
            "p_ever_requests": base["cue"]["p_ever_requests"],
            "qos_classes": base["cue"]["qos_classes"],
            "arrival_rate_per_s": base["cue"]["demand_levels"][levels["demand"]]["arrival_rate_per_s"],
            "class_mix": base["cue"]["demand_levels"][levels["demand"]]["class_mix"],
            "max_price_multiplier": base["cue"]["max_price_multiplier"],
            "priority_weight": base["cue"]["priority_weight"],
        },
        "gnb": {
            "total_capacity_mbps": float(gnb_lvl["total_capacity_mbps"]),
            "load_fraction": gnb_lvl["load_fraction"],
            "background_ar1_rho": base["gnb"]["background_ar1_rho"],
            "background_log_sigma": base["gnb"]["background_log_sigma"],
            "background_max_fraction": base["gnb"]["background_max_fraction"],
            "coverage_radius_m": base["gnb"]["coverage_levels"][levels["coverage"]],
        },
    }

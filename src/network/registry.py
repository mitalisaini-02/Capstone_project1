"""
Link-model registry: builds a LinkModel from a profile file and name.

The profile's "model" field selects the implementation. Adding a wireless technology =
add a LinkModel subclass + register it here + add a profile entry. No other module changes.
"""
import json
from typing import Callable, Dict

from src.network.base import LinkModel
from src.network.link_capacity import LinkCapacityModel
from src.network.tr37885 import TR37885SidelinkModel

MODEL_TYPES: Dict[str, Callable[[dict], LinkModel]] = {
    "log_distance_shannon": LinkCapacityModel,     # abstract_sidelink_v0
    "tr37885_v2v": TR37885SidelinkModel,           # nr_sidelink_tr37885_v1[_highway]
}


def register_model_type(model_type: str, factory: Callable[[dict], LinkModel]) -> None:
    if model_type in MODEL_TYPES:
        raise ValueError(f"model type '{model_type}' already registered")
    MODEL_TYPES[model_type] = factory


def load_link_model(profiles_path: str, name: str) -> LinkModel:
    with open(profiles_path) as f:
        profiles = json.load(f)
    if name not in profiles or name.startswith("_"):
        raise KeyError(f"Link profile '{name}' not found in {profiles_path}")
    profile = dict(profiles[name])
    profile["name"] = name
    model_type = profile.get("model")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown link model type '{model_type}' for profile '{name}'")
    return MODEL_TYPES[model_type](profile)

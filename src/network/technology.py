"""
Technology profiles: discovery range + link model for one wireless technology.
Configuration lives in config/network/technologies.json; nothing here is technology-specific.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict

from src.network.base import LinkModel
from src.network.registry import load_link_model


@dataclass(frozen=True)
class TechnologyProfile:
    name: str
    description: str
    discovery_range_m: float
    link_profiles_file: str
    link_profile: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def link_model(self) -> LinkModel:
        return load_link_model(self.link_profiles_file, self.link_profile)


def load_technology(path: str, name: str) -> TechnologyProfile:
    with open(path) as f:
        techs = json.load(f)
    if name not in techs or name.startswith("_"):
        raise KeyError(f"Technology '{name}' not found in {path}")
    t = dict(techs[name])
    core = {k: t.pop(k) for k in ("description", "discovery_range_m", "link_profiles_file", "link_profile")}
    return TechnologyProfile(name=name, extra=t, **core)

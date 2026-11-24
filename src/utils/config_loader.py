"""
Configuration loader (JSON, or YAML when PyYAML is installed).

Strict: the requested file is loaded or a clear error is raised. A different
configuration is NEVER substituted silently (previous versions fell back to
config/stage3_config.json or to a .json sibling of a missing .yaml file).
"""
import json
import os
from typing import Any, Dict


class ConfigNotFoundError(FileNotFoundError):
    """Raised when the requested configuration file does not exist."""


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads exactly `config_path`. Raises ConfigNotFoundError if it does not exist."""
    if not config_path:
        raise ValueError("config_path must be given explicitly")
    if not os.path.exists(config_path):
        raise ConfigNotFoundError(
            f"Configuration file '{config_path}' not found. "
            f"No other configuration is substituted; pass an existing path explicitly.")

    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)

    if config_path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                f"'{config_path}' is YAML but PyYAML is not installed. Install PyYAML or pass a "
                f"JSON configuration explicitly (no automatic substitution is performed).") from e
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported configuration format: '{config_path}' (expected .json/.yaml/.yml)")

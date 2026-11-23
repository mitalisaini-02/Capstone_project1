"""
Configuration loader with dual YAML/JSON support.
"""
import os
import json
from typing import Dict, Any


def load_config(config_path: str = "config/stage3_config.json") -> Dict[str, Any]:
    """Loads configuration dictionary from JSON or YAML file."""
    # If .yaml was requested but doesn't exist, check .json
    if config_path.endswith(".yaml") and not os.path.exists(config_path):
        json_alt = config_path.replace(".yaml", ".json")
        if os.path.exists(json_alt):
            config_path = json_alt

    if not os.path.exists(config_path):
        # Also try default json if path not found
        if os.path.exists("config/stage3_config.json"):
            config_path = "config/stage3_config.json"
        else:
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)

    # Try YAML if installed
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fall back to json version
        json_alt = config_path.replace(".yaml", ".json")
        if os.path.exists(json_alt):
            with open(json_alt, "r") as f:
                return json.load(f)
        raise RuntimeError(
            f"PyYAML is not installed in the environment to read '{config_path}'. "
            f"Please use JSON config '{json_alt}' or install PyYAML."
        )

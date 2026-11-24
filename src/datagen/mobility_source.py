"""
Backward-compatible shim. Mobility loading moved to `src.mobility.trace`.
Existing imports (`from src.datagen.mobility_source import load_run, RunMobility`) keep working.
"""
from src.mobility.trace import DEFAULT_RAW_DIR, RAW_DIR, RunMobility, load_run  # noqa: F401

__all__ = ["DEFAULT_RAW_DIR", "RAW_DIR", "RunMobility", "load_run"]

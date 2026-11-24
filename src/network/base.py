"""
Common interface for wireless link models (technology-independent).

A LinkModel maps PUE–CUE distance (and optionally a propagation state / context)
to SNR and capacity. It knows nothing about the LSTM, reliability thresholds,
prices or the optimiser, so link models and reliability models can be replaced
independently.

Link states follow TR 37.885 naming: "LOS", "NLOSv" (blocked by vehicles),
"NLOS" (blocked by buildings). Single-state models ignore `state`.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

LINK_STATES = ("LOS", "NLOSv", "NLOS")


class LinkModel(ABC):
    """Abstract link model. Distances in metres, SNR in dB, capacity in Mbps."""

    name: str = "unnamed"

    @abstractmethod
    def snr_db(self, d_m, state: Optional[str] = None) -> np.ndarray:
        """Median SNR (dB) at distance d for the given state (no random shadowing)."""

    @abstractmethod
    def capacity_mbps(self, d_m, state: Optional[str] = None) -> np.ndarray:
        """Capacity (Mbps) at the median SNR for the given state."""

    def link_capacity_mbps(self, d_m, link_peak_mbps, state: Optional[str] = None) -> np.ndarray:
        """Individual PUE–CUE link capacity B_ij = min(link_peak_i, capacity(d)) — constraint C3."""
        return np.minimum(np.asarray(link_peak_mbps, dtype=float), self.capacity_mbps(d_m, state=state))

    def expected_capacity_mbps(self, d_m, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Capacity averaged over the model's random effects. Deterministic models: = capacity_mbps."""
        return self.capacity_mbps(d_m)

    def outage_probability(self, d_m, snr_threshold_db: float,
                           context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """P(SNR < threshold). Deterministic models: 0/1 step."""
        return (self.snr_db(d_m) < snr_threshold_db).astype(float)

    # ---- REALIZED quantities (additive; used for simulated feasibility and allocation inputs) ----
    # Expected capacity (above) is a probability-weighted average for REPORTING only.
    # Realized capacity is the capacity of ONE sampled channel: a realised state plus fixed
    # standard-normal draws for shadowing (z_shadow) and NLOSv vehicle blockage (z_blockage).
    # Deterministic models have no random effects and return their (single-state) values.
    def realized_snr_db(self, d_m, states=None, z_shadow=0.0, z_blockage=0.0) -> np.ndarray:
        return np.asarray(self.snr_db(d_m), dtype=float)

    def realized_capacity_mbps(self, d_m, states=None, z_shadow=0.0, z_blockage=0.0) -> np.ndarray:
        return np.asarray(self.capacity_mbps(d_m), dtype=float)

    @abstractmethod
    def capacity_table(self, distances_m: Iterable[float] = (50, 100, 200, 500, 1000)) -> List[Dict]:
        """Human-readable values at reference distances (for reports)."""

    @property
    def reporting_snr_floor_db(self) -> Optional[float]:
        """Optional SNR floor used ONLY to flag low-SNR rows in statistics (None if undefined)."""
        return None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}

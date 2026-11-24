"""
Provisional, technology-agnostic PUE–CUE sidelink capacity model (constraint C3).

IMPORTANT
  * The parameters are modelling assumptions, not measurements of any radio
    technology. This is NOT a model of measured Wi-Fi performance.
  * A positive Shannon capacity is NOT evidence of a usable wireless link.
  * Link *availability* for candidate discovery is decided elsewhere by the
    1000 m geographic range (the LSTM label definition). This module only
    supplies the capacity bound B_ij used once a pair is a candidate.

Individual link capacity (C3):
    B_ij(t) = min( link_peak_i , f(d_ij(t)) )
    f(d)    = eta * W * min( log2(1 + SNR(d)), SE_max )
    SNR(d)  = P_tx + G - PL(d) - (N0 + 10 log10(W) + NF)          [dB]
    PL(d)   = FSPL(d0, fc) + 10 n log10(d / d0)                    [dB]
"""
import json
import math
from typing import Any, Dict, Optional

import numpy as np

from src.network.base import LinkModel

SPEED_OF_LIGHT = 299_792_458.0


def load_link_profile(path: str, name: str) -> Dict[str, Any]:
    with open(path) as f:
        profiles = json.load(f)
    if name not in profiles:
        raise KeyError(f"Link profile '{name}' not found in {path}")
    prof = dict(profiles[name])
    prof["name"] = name
    if prof.get("model") != "log_distance_shannon":
        raise ValueError(f"Unsupported link model '{prof.get('model')}'")
    return prof


class LinkCapacityModel(LinkModel):
    """abstract_sidelink_v0: distance-based capacity f(d) in Mbps (log-distance + Shannon).
    Single-state model: the optional `state` argument is accepted and ignored."""

    def __init__(self, profile: Dict[str, Any]):
        self.p = profile
        self.name = profile.get("name", "abstract_sidelink_v0")
        self.bandwidth_hz = profile["bandwidth_mhz"] * 1e6
        fc_hz = profile["carrier_frequency_ghz"] * 1e9
        d0 = profile["reference_distance_m"]
        # Free-space path loss at the reference distance
        self.pl_d0_db = 20 * math.log10(4 * math.pi * d0 * fc_hz / SPEED_OF_LIGHT)
        self.noise_dbm = (profile["thermal_noise_dbm_per_hz"]
                          + 10 * math.log10(self.bandwidth_hz) + profile["noise_figure_db"])

    @property
    def reporting_snr_floor_db(self):
        return self.p["reporting_snr_floor_db"]

    def path_loss_db(self, d_m):
        d = np.maximum(np.asarray(d_m, dtype=float), self.p["min_distance_m"])
        return self.pl_d0_db + 10 * self.p["path_loss_exponent"] * np.log10(d / self.p["reference_distance_m"])

    def snr_db(self, d_m, state: Optional[str] = None):
        return (self.p["tx_power_dbm"] + self.p["antenna_gains_dbi"]
                - self.path_loss_db(d_m) - self.noise_dbm)

    def capacity_mbps(self, d_m, state: Optional[str] = None):
        """f(d): distance-only capacity bound (Mbps), before the per-PUE radio peak."""
        snr_lin = 10 ** (self.snr_db(d_m) / 10.0)
        se = np.minimum(np.log2(1.0 + snr_lin), self.p["max_spectral_efficiency_bps_per_hz"])
        return self.p["implementation_efficiency"] * self.bandwidth_hz * se / 1e6

    def link_capacity_mbps(self, d_m, link_peak_mbps, state: Optional[str] = None):
        """B_ij = min(link_peak_i, f(d_ij)) — individual PUE–CUE link capacity (C3)."""
        return np.minimum(np.asarray(link_peak_mbps, dtype=float), self.capacity_mbps(d_m))

    def capacity_table(self, distances_m=(50, 100, 200, 500, 1000)):
        """Reported values: SNR and capacity at reference distances (no per-PUE cap)."""
        rows = []
        for d in distances_m:
            snr = float(self.snr_db(d))
            rows.append({
                "distance_m": d,
                "path_loss_db": round(float(self.path_loss_db(d)), 2),
                "snr_db": round(snr, 2),
                "capacity_mbps_uncapped_by_pue": round(float(self.capacity_mbps(d)), 3),
                "below_reporting_snr_floor": snr < self.p["reporting_snr_floor_db"],
            })
        return rows

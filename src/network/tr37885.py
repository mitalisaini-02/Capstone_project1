"""
nr_sidelink_tr37885_v1 — NR sidelink (PC5) V2V link model for the PUE (relay UE) → CUE (remote UE) hop.

Every equation below was verified against the official 3GPP text:
  [37.885] 3GPP TR 37.885 V15.3.0 (2019-06), §6.2 "Channel model":
           Table 6.2-1 (LOS probability), Table 6.2.1-1 (pathloss, shadow-fading std),
           §6.2.1 NLOSv vehicle-blockage loss, Table 6.1.1-1 (UE Tx power, UE noise figure,
           6 GHz carrier between UEs, 10/20 MHz sidelink simulation bandwidth),
           §6.1.2 vehicle types / antenna heights, Table 6.1.4-8 (vehicle UE element gain).
  [36.942] 3GPP TR 36.942 V19.0.0, Annex A.1 / Table A.1 (attenuated, truncated Shannon
           bound); the mapping reproduces Table A.2 exactly.

Link states:
  LOS   — same street, LOS path not blocked
  NLOSv — same street, LOS path blocked by vehicles
  NLOS  — different streets (blocked by buildings)
  For same-street links P(NLOSv) = 1 - P(LOS). WHETHER TWO VEHICLES ARE IN THE SAME STREET is
  NOT decided here: callers must pass `same_street` (or an explicit state). No street
  classifier has been chosen yet (open decision; see docs/network_models.md).

Not modelled (documented assumptions): fast fading, spatially correlated shadowing,
interference (SNR, not SINR), antenna radiation pattern (maximum element gain is used).
"""
import json
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy.special import ndtr

from src.network.base import LinkModel

# Dense trapezoidal grids over standard-normal variables (+-8 sigma). Used instead of Gauss–Hermite
# because the integrands (outage indicator, capacity with a jump at SNR_min) are discontinuous.
def _normal_grid(n: int):
    z = np.linspace(-8.0, 8.0, n)
    w = np.exp(-0.5 * z ** 2)
    w[0] *= 0.5
    w[-1] *= 0.5
    return z, w / w.sum()


_Z_SF, _W_SF = _normal_grid(401)     # shadow fading (step 0.04 sigma)
_Z_BL, _W_BL = _normal_grid(161)     # NLOSv blockage variable
_CHUNK = 64


class TR37885SidelinkModel(LinkModel):
    """TR 37.885 V2V pathloss/LOS/blockage + TR 36.942 throughput mapping."""

    def __init__(self, profile: Dict[str, Any]):
        self.p = profile
        self.name = profile.get("name", "nr_sidelink_tr37885_v1")
        self.scenario = profile["scenario"]
        if self.scenario not in ("urban", "highway"):
            raise ValueError(f"scenario must be 'urban' or 'highway', got {self.scenario}")
        self.fc = float(profile["carrier_frequency_ghz"])
        self.bw_hz = float(profile["bandwidth_mhz"]) * 1e6
        self.noise_dbm = (profile["thermal_noise_dbm_per_hz"] + 10 * np.log10(self.bw_hz)
                          + profile["ue_noise_figure_db"])
        self.eirp_gain_db = (profile["ue_tx_power_dbm"] + profile["tx_antenna_gain_dbi"]
                             + profile["rx_antenna_gain_dbi"])
        self.h_tx = float(profile["tx_antenna_height_m"])
        self.h_rx = float(profile["rx_antenna_height_m"])
        self.h_block = float(profile["blocker_height_m"])
        m = profile["throughput_mapping"]
        self.alpha, self.snr_min_db, self.thr_max = m["alpha"], m["snr_min_db"], m["thr_max_bps_per_hz"]

    # ------------------------------------------------------------------ geometry
    def d3d(self, d_m):
        d = np.maximum(np.asarray(d_m, dtype=float), self.p["min_distance_m"])
        return np.sqrt(d ** 2 + (self.h_tx - self.h_rx) ** 2)

    # ------------------------------------------------------------------ [37.885] Table 6.2-1
    def p_los(self, d_m):
        """P(LOS) for two vehicles in the SAME street; P(NLOSv) = 1 - P(LOS)."""
        d = np.asarray(d_m, dtype=float)
        if self.scenario == "highway":
            a, b, c = 2.1013e-6, -0.002, 1.0193
            near = np.minimum(1.0, a * d ** 2 + b * d + c)
            far = np.maximum(0.0, 0.54 - 0.001 * (d - 475.0))
            return np.where(d <= 475.0, near, far)
        return np.minimum(1.0, 1.05 * np.exp(-0.0114 * d))

    def state_probabilities(self, d_m, same_street: bool = True) -> Dict[str, np.ndarray]:
        d = np.asarray(d_m, dtype=float)
        if not same_street:
            if self.scenario == "highway":
                raise ValueError("TR 37.885 defines no NLOS (building) state for the highway scenario")
            z = np.zeros_like(d)
            return {"LOS": z, "NLOSv": z, "NLOS": np.ones_like(d)}
        pl = self.p_los(d)
        return {"LOS": pl, "NLOSv": 1.0 - pl, "NLOS": np.zeros_like(d)}

    # ------------------------------------------------------------------ [37.885] Table 6.2.1-1
    def path_loss_db(self, d_m, state: str = "LOS"):
        d = self.d3d(d_m)
        if state in ("LOS", "NLOSv"):
            if self.scenario == "highway":
                return 32.4 + 20 * np.log10(d) + 20 * np.log10(self.fc)
            return 38.77 + 16.7 * np.log10(d) + 18.2 * np.log10(self.fc)
        if state == "NLOS":
            if self.scenario == "highway":
                raise ValueError("TR 37.885 defines no NLOS (building) state for the highway scenario")
            return 36.85 + 30 * np.log10(d) + 18.9 * np.log10(self.fc)
        raise ValueError(f"unknown state {state}")

    @staticmethod
    def shadow_std_db(state: str) -> float:
        return 4.0 if state == "NLOS" else 3.0   # LOS and NLOSv: 3 dB; NLOS: 4 dB

    # ------------------------------------------------------------------ [37.885] §6.2.1 NLOSv blockage
    def blockage_params_db(self, d_m):
        """(mean, std) of the Gaussian-in-dB blockage variable X; applied loss = max(0, X).
        Returns (0, 0) when both antennas are above the blocker (no additional loss)."""
        d = np.asarray(d_m, dtype=float)
        excess = np.maximum(0.0, 15 * np.log10(np.maximum(d, 1e-9)) - 41)
        if min(self.h_tx, self.h_rx) > self.h_block:            # Case 1
            return np.zeros_like(d), np.zeros_like(d)
        if max(self.h_tx, self.h_rx) < self.h_block:            # Case 2
            return 9.0 + excess, np.full_like(d, 4.5)
        return 5.0 + excess, np.full_like(d, 4.0)               # Case 3

    # ------------------------------------------------------------------ SNR and capacity
    def _snr_from_pl(self, pl_db):
        return self.eirp_gain_db - pl_db - self.noise_dbm

    def snr_db(self, d_m, state: Optional[str] = "LOS"):
        """Median SNR for a state: shadowing median 0 dB; NLOSv blockage median max(0, mean)."""
        state = state or "LOS"
        snr = self._snr_from_pl(self.path_loss_db(d_m, state))
        if state == "NLOSv":
            mu, _ = self.blockage_params_db(d_m)
            snr = snr - np.maximum(0.0, mu)
        return snr

    def spectral_efficiency(self, snr_db):
        """[36.942] A.1: 0 if SNR < SNR_min, else min(alpha*log2(1+SNR), Thr_max) [bit/s/Hz]."""
        snr_db = np.asarray(snr_db, dtype=float)
        se = np.minimum(self.alpha * np.log2(1.0 + 10 ** (snr_db / 10.0)), self.thr_max)
        return np.where(snr_db < self.snr_min_db, 0.0, se)

    def capacity_from_snr_mbps(self, snr_db):
        return self.bw_hz * self.spectral_efficiency(snr_db) / 1e6

    def capacity_mbps(self, d_m, state: Optional[str] = "LOS"):
        return self.capacity_from_snr_mbps(self.snr_db(d_m, state or "LOS"))

    # ------------------------------------------------------------------ expectations over random effects
    def _expect(self, d_m, state, fn):
        """E[fn(SNR)] for one state over shadowing (and NLOSv blockage) by dense trapezoidal grids."""
        d = np.atleast_1d(np.asarray(d_m, dtype=float))
        out = np.empty(len(d))
        for s0 in range(0, len(d), _CHUNK):
            dc = d[s0:s0 + _CHUNK]
            base = self._snr_from_pl(self.path_loss_db(dc, state))[:, None]
            sf = self.shadow_std_db(state) * _Z_SF[None, :]
            if state != "NLOSv":
                out[s0:s0 + _CHUNK] = (fn(base - sf) * _W_SF[None, :]).sum(axis=1)
                continue
            mu, sd = self.blockage_params_db(dc)
            blk = np.maximum(0.0, mu[:, None] + sd[:, None] * _Z_BL[None, :])            # (n, kb)
            snr = base[:, None, :] - sf[:, None, :] - blk[:, :, None]                     # (n, kb, ks)
            w = _W_BL[:, None] * _W_SF[None, :]
            out[s0:s0 + _CHUNK] = (fn(snr) * w[None]).sum(axis=(1, 2))
        return out

    def _state_outage(self, d_m, state, thr_db):
        """P(SNR < thr | state): closed form for LOS/NLOS; smooth 1-D integral over blockage for NLOSv."""
        d = np.atleast_1d(np.asarray(d_m, dtype=float))
        mu = self._snr_from_pl(self.path_loss_db(d, state))
        sig = self.shadow_std_db(state)
        if state != "NLOSv":
            return ndtr((thr_db - mu) / sig)
        bmu, bsd = self.blockage_params_db(d)
        blk = np.maximum(0.0, bmu[:, None] + bsd[:, None] * _Z_BL[None, :])
        return (ndtr((thr_db - (mu[:, None] - blk)) / sig) * _W_BL[None, :]).sum(axis=1)

    def _mixture(self, d_m, same_street, fn):
        probs = self.state_probabilities(d_m, same_street)
        total = 0.0
        for state, p in probs.items():
            if np.any(np.asarray(p) > 0):
                total = total + np.asarray(p) * self._expect(d_m, state, fn)
        return np.asarray(total, dtype=float)

    def expected_capacity_mbps(self, d_m, context: Optional[Dict[str, Any]] = None):
        same = (context or {}).get("same_street", True)
        return self._mixture(d_m, same, self.capacity_from_snr_mbps)

    def outage_probability(self, d_m, snr_threshold_db: float, context: Optional[Dict[str, Any]] = None):
        """P(SNR < threshold) over link state, shadowing and blockage."""
        same = (context or {}).get("same_street", True)
        probs = self.state_probabilities(d_m, same)
        total = np.zeros(np.atleast_1d(np.asarray(d_m, dtype=float)).shape)
        for state, p in probs.items():
            p = np.atleast_1d(p)
            if np.any(p > 0):
                total = total + p * self._state_outage(d_m, state, snr_threshold_db)
        return total

    # ------------------------------------------------------------------ ADDITIVE (channel-state classifier)
    def _given_states(self, d_m, p_los, p_nlosv, p_nlos, per_state):
        d = np.atleast_1d(np.asarray(d_m, dtype=float))
        probs = {"LOS": np.broadcast_to(np.asarray(p_los, float), d.shape),
                 "NLOSv": np.broadcast_to(np.asarray(p_nlosv, float), d.shape),
                 "NLOS": np.broadcast_to(np.asarray(p_nlos, float), d.shape)}
        if not np.allclose(probs["LOS"] + probs["NLOSv"] + probs["NLOS"], 1.0, atol=1e-9):
            raise ValueError("state probabilities must sum to 1 for every link")
        total = np.zeros(d.shape)
        for state, p in probs.items():
            m = p > 0
            if np.any(m):
                total[m] += p[m] * per_state(d[m], state)
        return total

    def expected_capacity_given_states(self, d_m, p_los, p_nlos_v, p_nlos):
        """E[capacity] when the state probabilities come from an external classifier
        (e.g. building geometry). Uses the same verified per-state expectations."""
        return self._given_states(d_m, p_los, p_nlos_v, p_nlos,
                                  lambda d, st: self._expect(d, st, self.capacity_from_snr_mbps))

    def outage_probability_given_states(self, d_m, snr_threshold_db, p_los, p_nlos_v, p_nlos):
        """P(SNR < threshold) for externally supplied state probabilities."""
        return self._given_states(d_m, p_los, p_nlos_v, p_nlos,
                                  lambda d, st: self._state_outage(d, st, snr_threshold_db))

    # ------------------------------------------------------------------ ADDITIVE (realized channel, Phase 3)
    def realized_snr_db(self, d_m, states, z_shadow, z_blockage):
        """SNR of one sampled channel per row:
             SNR = P_tx + G - PL_state(d) - sigma_state * z_shadow - [NLOSv] max(0, mu_b(d) + sd_b(d) * z_blockage) - N
        `states` is an array of 'LOS'/'NLOSv'/'NLOS'; z_* are standard-normal draws (fixed per link
        episode by the caller). Uses the verified pathloss, shadow std and blockage parameters."""
        d = np.atleast_1d(np.asarray(d_m, dtype=float))
        st = np.broadcast_to(np.asarray(states, dtype=object), d.shape)
        zs = np.broadcast_to(np.asarray(z_shadow, dtype=float), d.shape)
        zb = np.broadcast_to(np.asarray(z_blockage, dtype=float), d.shape)
        out = np.full(d.shape, np.nan)
        for state in ("LOS", "NLOSv", "NLOS"):
            m = st == state
            if not np.any(m):
                continue
            snr = self._snr_from_pl(self.path_loss_db(d[m], state)) - self.shadow_std_db(state) * zs[m]
            if state == "NLOSv":
                mu, sd = self.blockage_params_db(d[m])
                snr = snr - np.maximum(0.0, mu + sd * zb[m])
            out[m] = snr
        if np.any(np.isnan(out)):
            raise ValueError("unknown link state in `states`")
        return out

    def realized_capacity_mbps(self, d_m, states, z_shadow, z_blockage):
        """Capacity (Mbps) of the sampled channel (TR 36.942 mapping of realized_snr_db)."""
        return self.capacity_from_snr_mbps(self.realized_snr_db(d_m, states, z_shadow, z_blockage))

    # ------------------------------------------------------------------ report
    def capacity_table(self, distances_m: Iterable[float] = (50, 100, 200, 500, 1000)) -> List[Dict]:
        rows = []
        for d in distances_m:
            r = {"distance_m": d, "p_los_same_street": round(float(self.p_los(d)), 4)}
            states = ("LOS", "NLOSv") if self.scenario == "highway" else ("LOS", "NLOSv", "NLOS")
            for st in states:
                r[f"median_snr_db_{st}"] = round(float(self.snr_db(d, st)), 2)
                r[f"capacity_mbps_{st}"] = round(float(self.capacity_mbps(d, st)), 2)
            r["expected_capacity_mbps_same_street"] = round(float(self.expected_capacity_mbps(d)[0]), 2)
            r["outage_prob_same_street_snr_below_min"] = round(
                float(self.outage_probability(d, self.snr_min_db)[0]), 4)
            rows.append(r)
        return rows

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "scenario": self.scenario,
                "carrier_frequency_ghz": self.fc, "bandwidth_mhz": self.bw_hz / 1e6,
                "noise_dbm": round(float(self.noise_dbm), 2)}


def load_tr37885_profile(path: str, name: str) -> Dict[str, Any]:
    with open(path) as f:
        profiles = json.load(f)
    if name not in profiles:
        raise KeyError(f"Link profile '{name}' not found in {path}")
    prof = dict(profiles[name])
    prof["name"] = name
    if prof.get("model") != "tr37885_v2v":
        raise ValueError(f"Profile '{name}' is not a tr37885_v2v model")
    return prof

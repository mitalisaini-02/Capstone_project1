"""
Phase 5 configuration: strict loading, validation and the pinned Phase 4 dependency.

* Unknown or missing top-level keys are errors (no silent defaults).
* Values still marked "pending" (open design decisions) can be stored but not used:
  `require(...)` raises PendingDecisionError, so a later milestone cannot run on an undecided value.
* `verify_phase4_dependency()` re-hashes the frozen Phase 4 manifest and every pinned interface file.
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from src.utils.integrity import sha256_file

PENDING = "pending"
CHANNEL_SCENARIOS = ("G", "F1", "F2", "F3")
POLICY_KINDS = {"aware": ("primary", "sensitivity"), "unaware": ("unaware",)}
REQUIRED_KEYS = {
    "_notice", "experiment_id", "phase4_dependency", "inputs", "output_root", "dt_s", "rate_eps_mbps",
    "audit_tolerance_mbps", "channel_scenarios", "primary_channel_scenario", "policies", "primary_policy",
    "service_semantics", "arrival_policy", "continuity_weight", "reoptimisation", "rebalance_counted_separately",
    "reallocation_delay_s", "interruption_tolerance", "proactive_reliability"}
INPUT_KEYS = {"phase2_root", "phase3_root", "raw_mobility_dir", "phase4_config", "evaluation_only_ground_truth_dir"}


class ConfigError(ValueError):
    pass


class PendingDecisionError(RuntimeError):
    pass


class Phase4DependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str          # "aware" | "unaware"
    theta_tag: str     # "primary" | "sensitivity" | "unaware" (Phase 3 / Phase 5 flag suffix)


class Phase5Config:
    def __init__(self, raw: Dict[str, Any], path: str = "<dict>"):
        self.raw, self.path = raw, path
        self._validate()

    # ---------------- loading / validation ----------------
    @classmethod
    def load(cls, path: str) -> "Phase5Config":
        if not os.path.exists(path):
            raise ConfigError(f"Phase 5 configuration not found: {path} (no substitute is used)")
        with open(path) as f:
            return cls(json.load(f), path)

    def _validate(self):
        r = self.raw
        missing, unknown = REQUIRED_KEYS - set(r), set(r) - REQUIRED_KEYS
        if missing or unknown:
            raise ConfigError(f"{self.path}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
        if set(r["inputs"]) != INPUT_KEYS:
            raise ConfigError(f"inputs must have exactly {sorted(INPUT_KEYS)}")
        if tuple(r["channel_scenarios"]) != CHANNEL_SCENARIOS:
            raise ConfigError(f"channel_scenarios must be {list(CHANNEL_SCENARIOS)}")
        if r["primary_channel_scenario"] not in r["channel_scenarios"]:
            raise ConfigError("primary_channel_scenario not in channel_scenarios")
        for name, p in r["policies"].items():
            if set(p) != {"kind", "theta_tag"} or p["kind"] not in POLICY_KINDS \
                    or p["theta_tag"] not in POLICY_KINDS[p["kind"]]:
                raise ConfigError(f"policy {name}: invalid {p}")
        if r["primary_policy"] not in r["policies"]:
            raise ConfigError("primary_policy not defined")
        if r["service_semantics"] != "time_window" or r["arrival_policy"] != "block" \
                or r["reoptimisation"] != "incremental":
            raise ConfigError("only time_window / block / incremental are supported (Revision 2)")
        for k in ("dt_s", "rate_eps_mbps", "audit_tolerance_mbps", "continuity_weight"):
            if not (isinstance(r[k], (int, float)) and r[k] > 0):
                raise ConfigError(f"{k} must be positive")
        if r["dt_s"] != 1.0:
            raise ConfigError("dt_s must be 1.0 (Phase 2/3 data are per second)")
        d = r["reallocation_delay_s"]
        if d.get("primary") != 0 or not all(isinstance(x, int) and x >= 0 for x in d.get("sensitivity", [])):
            raise ConfigError("reallocation_delay_s: primary must be 0, sensitivity a list of non-negative ints")
        it = r["interruption_tolerance"]
        if it.get("selection_split") != "val":
            raise ConfigError("interruption tolerance may only be selected on the validation split")
        if not it.get("candidates_s") or any((not isinstance(x, int)) or x < 0 for x in it["candidates_s"]):
            raise ConfigError("interruption_tolerance.candidates_s must be non-negative ints")
        if it["selected_s"] != PENDING and it["selected_s"] not in it["candidates_s"]:
            raise ConfigError("selected_s must be one of candidates_s")
        if r["proactive_reliability"].get("primary") is not False:
            raise ConfigError("proactive reliability switching must be off in the primary experiment")

    # ---------------- accessors ----------------
    def __getitem__(self, k):
        return self.raw[k]

    @property
    def channel_scenarios(self) -> List[str]:
        return list(self.raw["channel_scenarios"])

    @property
    def policies(self) -> Dict[str, Policy]:
        return {n: Policy(n, p["kind"], p["theta_tag"]) for n, p in self.raw["policies"].items()}

    def policy(self, name: str) -> Policy:
        if name not in self.raw["policies"]:
            raise ConfigError(f"unknown policy {name}")
        return self.policies[name]

    def require(self, section: str, key: str):
        """Value of a decision; raises if it is still pending (used by M2+ modules)."""
        v = self.raw[section][key]
        if v == PENDING:
            raise PendingDecisionError(f"{section}.{key} is still pending (Design Revision 2 section 13)")
        return v

    def pending(self) -> List[str]:
        return [f"{s}.{k}" for s in ("interruption_tolerance", "proactive_reliability")
                for k, v in self.raw[s].items() if v == PENDING]

    def phase3_dir(self, split: str, scenario_id: str) -> str:
        return os.path.join(self.raw["inputs"]["phase3_root"], split, scenario_id)

    def phase2_dir(self, split: str, scenario_id: str) -> str:
        return os.path.join(self.raw["inputs"]["phase2_root"], split, scenario_id)

    def thresholds(self, split: str) -> Dict[str, float]:
        """theta values exactly as recorded by Phase 3 for this split (never chosen here)."""
        with open(os.path.join(self.raw["inputs"]["phase3_root"], split, "experiment_config.json")) as f:
            return dict(json.load(f)["thresholds"])

    # ---------------- Phase 4 dependency ----------------
    def verify_phase4_dependency(self, root: str = ".") -> Dict[str, Any]:
        """Re-hash the frozen Phase 4 manifest and pinned interfaces; raise on any difference."""
        dep = self.raw["phase4_dependency"]
        fz = os.path.join(root, dep["frozen_manifest"])
        problems = []
        if not os.path.exists(fz):
            raise Phase4DependencyError(f"frozen Phase 4 manifest missing: {fz}")
        if sha256_file(fz) != dep["frozen_manifest_sha256"]:
            problems.append(f"{dep['frozen_manifest']}: hash differs from pin")
        frozen_code = json.load(open(fz))["code_sha256"]
        for f, h in dep["interface_sha256"].items():
            p = os.path.join(root, f)
            if not os.path.exists(p):
                problems.append(f"{f}: missing")
            elif sha256_file(p) != h:
                problems.append(f"{f}: hash differs from pin")
            if f in frozen_code and frozen_code[f] != h:
                problems.append(f"{f}: pin differs from the Phase 4 freeze manifest")
        for f, h in frozen_code.items():
            if sha256_file(os.path.join(root, f)) != h:
                problems.append(f"{f}: differs from the Phase 4 freeze manifest")
        if problems:
            raise Phase4DependencyError("; ".join(problems))
        return {"frozen_manifest_sha256": dep["frozen_manifest_sha256"],
                "n_interfaces": len(dep["interface_sha256"]), "n_frozen_files": len(frozen_code)}

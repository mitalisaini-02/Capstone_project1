#!/usr/bin/env python3
"""
Phase 2 — Synthetic marketplace dataset generation (SYNTHETIC SIMULATION DATA).

Subcommands (run in this order):
  connectivity --split val|test   Derived pairwise connectivity + ground truth from SUMO
  calibrate                       Calibrate CALIBRATE parameters on VALIDATION runs only,
                                  then freeze them (write-once frozen_config.json)
  generate --split val|test       Generate all scenarios with the FROZEN parameters
  repro-check --split val|test    Regenerate every scenario in a scratch dir and compare hashes

Safety
  * Test generation refuses to run without a frozen config, and refuses if base.json,
    scenarios.json or link_profiles.json changed after freezing.
  * Calibration refuses to touch test runs.
  * All outputs are written to new paths only; existing files are never overwritten.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.datagen.connectivity import build_connectivity  # noqa: E402
from src.datagen.generators import generate_cues, generate_gnbs, generate_pues, load_gnb_sites  # noqa: E402
from src.datagen.io_utils import read_csv_gz, write_csv, write_csv_gz, write_json, write_text  # noqa: E402
from src.datagen.metrics import (active_request_seconds, gnb_demand_coverage,  # noqa: E402
                                 pue_supply_demand, serving_cell)
from src.datagen.stats import aggregate_by_configuration, scenario_stats  # noqa: E402
from src.mobility.trace import load_run  # noqa: E402
from src.datagen.scenarios import (check_lstm_consistency, expand_configurations, load_json,  # noqa: E402
                                   resolve_parameters, seed_key, sha256_file)
from src.datagen.validate import validate_connectivity, validate_scenario  # noqa: E402
from src.network.link_capacity import LinkCapacityModel, load_link_profile  # noqa: E402
from src.utils.integrity import verify_manifest  # noqa: E402

BASE_CFG = "config/marketplace/base.json"
SCEN_CFG = "config/marketplace/scenarios.json"
LINK_CFG = "config/marketplace/link_profiles.json"
LSTM_CFG = "config/stage3_large_dataset.json"
SPLIT_INFO = "data/splits/large_dataset/split_info.json"
FROZEN_DIR = "config/marketplace/frozen"
FROZEN_CFG = f"{FROZEN_DIR}/frozen_config.json"
CALIB_REPORT = f"{FROZEN_DIR}/calibration_report.json"
CONN_DIR = "data/derived/connectivity"
OUT_ROOT = "data/synthetic/marketplace"
MANIFEST = "outputs/PROTECTED_ARTIFACTS.sha256"
SCENARIO_FILES = ["pue_resources.csv.gz", "cue_requests.csv.gz", "gnodeb_resources.csv.gz",
                  "gnodeb_sites.csv", "pue_withdrawals.csv"]

SYNTHETIC_NOTICE = """# SYNTHETIC SIMULATION DATA

Everything in this directory except vehicle positions, speeds, timestamps, roles and
lifetimes is **synthetic**. Those mobility fields are copied unchanged from the SUMO
traces in `data/raw/sumo/`.

The PUE capacities and prices, CUE requests and budgets, gNodeB capacities, background
load and coverage, and PUE withdrawals are generated from **assumed** parameter values
(`generation_config.json`). They are **not real-world measurements** and must never be
presented as such.

The gNodeB sites are hand-placed POIs from `infrastructure.add.xml`, not a real deployment.

Future ground truth (link breaks, departures) is stored separately in
`data/derived/connectivity/<run>/pairwise_ground_truth.csv.gz` and must only be used
for evaluation, never as an input to the MEC pipeline.
"""


def log(msg):
    print(msg, flush=True)


def split_runs(split):
    info = load_json(SPLIT_INFO)
    key = {"val": "val_runs", "test": "test_runs"}[split]
    return sorted(info[key])


def check_protected():
    changed, missing = verify_manifest(MANIFEST)
    if changed or missing:
        raise SystemExit(f"ABORT: protected artifacts changed={changed} missing={missing}")


def config_hashes():
    return {p: sha256_file(p) for p in (BASE_CFG, SCEN_CFG, LINK_CFG)}


# --------------------------------------------------------------------------- connectivity
def cmd_connectivity(split):
    check_protected()
    base = load_json(BASE_CFG)
    check_lstm_consistency(base, LSTM_CFG)
    L = base["link"]
    if split == "test" and not os.path.exists(FROZEN_CFG):
        raise SystemExit("ABORT: test connectivity is generated only after parameters are frozen.")
    report = {}
    for run_id in split_runs(split):
        out = f"{CONN_DIR}/{run_id}"
        if os.path.exists(out):
            raise SystemExit(f"ABORT: {out} already exists (overwrite protection).")
        run = load_run(run_id)
        vis, gt, meta = build_connectivity(run, L["comm_range_m"], L["history_window_s"],
                                           L["prediction_horizon_s"])
        ck = validate_connectivity(run, vis, gt, L["comm_range_m"], L["history_window_s"])
        if not ck.passed:
            raise SystemExit(f"ABORT: connectivity validation failed for {run_id}: {ck.failures()}")
        os.makedirs(out)
        write_csv_gz(vis, f"{out}/pairwise_connectivity.csv.gz")
        write_csv_gz(gt, f"{out}/pairwise_ground_truth.csv.gz")
        meta["validation"] = ck.items
        meta["generated"] = datetime.now().isoformat(timespec="seconds")
        meta["sha256"] = {f: sha256_file(f"{out}/{f}") for f in
                          ("pairwise_connectivity.csv.gz", "pairwise_ground_truth.csv.gz")}
        write_json(meta, f"{out}/connectivity_meta.json")
        report[run_id] = {k: meta[k] for k in ("rows_in_range", "rows_history_ok", "unique_pairs",
                                               "first_break_cause_counts", "survives_30s_counts")}
        log(f"  {run_id}: {meta['rows_in_range']:,} in-range rows, {meta['unique_pairs']} pairs, "
            f"validation {len(ck.items)}/{len(ck.items)} passed")
    check_protected()
    return report


# --------------------------------------------------------------------------- calibration
def _params(base, levels, p_reg=None, gnb_cap=None):
    frozen = {"p_register": {k: (p_reg if p_reg is not None else 0.5) for k in ("low", "med", "high")},
              "gnb_total_capacity_mbps": {k: (gnb_cap if gnb_cap is not None else 100.0)
                                          for k in ("sufficient", "insufficient")}}
    return resolve_parameters(base, levels, frozen)


def cmd_calibrate():
    check_protected()
    if os.path.exists(FROZEN_CFG):
        raise SystemExit(f"ABORT: {FROZEN_CFG} already exists; parameters are frozen.")
    base = load_json(BASE_CFG)
    check_lstm_consistency(base, LSTM_CFG)
    runs = split_runs("val")
    test_runs = set(split_runs("test"))
    assert not (set(runs) & test_runs), "calibration runs overlap test runs"
    ref = base["calibration"]["reference_conditions"]
    sites = load_gnb_sites(base["gnb"]["sites_file"])
    seed = base["base_seed"]
    reps = range(base["replicates"])
    p_grid = base["calibration"]["p_register_grid"]
    g_grid = base["calibration"]["gnb_total_capacity_grid_mbps"]

    sd = {lvl: {p: [] for p in p_grid} for lvl in ("low", "med", "high")}
    gc = {lvl: {g: [] for g in g_grid} for lvl in ("sufficient", "insufficient")}
    for run_id in runs:
        run = load_run(run_id)
        conn = read_csv_gz(f"{CONN_DIR}/{run_id}/pairwise_connectivity.csv.gz")
        for rep in reps:
            sid = f"calib__{run_id}__r{rep}"
            p0 = _params(base, ref)
            cue = generate_cues(run, p0, sid, seed, rep)
            act = active_request_seconds(cue, run)
            cells = serving_cell(run, act, sites, base["gnb"]["coverage_levels"][ref["coverage"]])
            for lvl in ("low", "med", "high"):
                for p in p_grid:
                    pp = _params(base, dict(ref, availability=lvl), p_reg=p)
                    pue, _, _ = generate_pues(run, pp, sid, seed, rep)
                    sd[lvl][p].append(pue_supply_demand(act, pue, conn)["supply_demand_ratio"])
            for lvl in ("sufficient", "insufficient"):
                for g in g_grid:
                    pg = _params(base, dict(ref, gnb_capacity=lvl), gnb_cap=g)
                    gnb, _ = generate_gnbs(run, pg, sid, seed, rep, sites)
                    gc[lvl][g].append(gnb_demand_coverage(act, cells, gnb)["gnb_carry_share"])
        log(f"  calibrated on {run_id}")

    frozen = {"p_register": {}, "gnb_total_capacity_mbps": {}}
    calib = {"runs": runs, "replicates": list(reps), "reference_conditions": ref,
             "availability": {}, "gnb_capacity": {}, "flags": []}
    for lvl in ("low", "med", "high"):
        target = base["pue"]["availability_levels"][lvl]["target_supply_demand_ratio"]
        means = {p: float(np.mean(v)) for p, v in sd[lvl].items()}
        best = min(p_grid, key=lambda p: (abs(means[p] - target), p))
        frozen["p_register"][lvl] = best
        within = abs(means[best] - target) <= 0.25 * target
        if not within:
            calib["flags"].append(f"availability '{lvl}': target {target} not reached within 25% "
                                  f"(closest {means[best]:.3f} at p_register={best})")
        calib["availability"][lvl] = {"target_supply_demand_ratio": target, "selected_p_register": best,
                                      "achieved_mean_ratio": means[best], "within_25pct": within,
                                      "grid_mean_ratio": means}
    for lvl in ("sufficient", "insufficient"):
        lo, hi = base["gnb"]["capacity_levels"][lvl]["target_demand_coverage"]
        means = {g: float(np.mean(v)) for g, v in gc[lvl].items()}
        if lvl == "sufficient":
            ok = [g for g in g_grid if means[g] >= lo]
            best = min(ok) if ok else max(g_grid)
            rule = f"smallest total capacity with gNodeB carry share >= {lo}"
        else:
            mid = (lo + hi) / 2
            best = min(g_grid, key=lambda g: (abs(means[g] - mid), g))
            rule = f"total capacity with gNodeB carry share closest to {mid} (target band [{lo}, {hi}])"
        hit = lo <= means[best] <= hi
        if not hit:
            calib["flags"].append(f"gNodeB '{lvl}': carry share {means[best]:.3f} outside [{lo}, {hi}]")
        frozen["gnb_total_capacity_mbps"][lvl] = best
        calib["gnb_capacity"][lvl] = {"rule": rule, "selected_total_capacity_mbps": best,
                                      "achieved_mean_carry_share": means[best], "within_target": hit,
                                      "load_fraction": base["gnb"]["capacity_levels"][lvl]["load_fraction"],
                                      "grid_mean_carry_share": {str(k): v for k, v in means.items()}}

    os.makedirs(FROZEN_DIR, exist_ok=True)
    write_json(calib, CALIB_REPORT)
    write_json({"frozen_at": datetime.now().isoformat(timespec="seconds"),
                "calibrated_on_runs": runs, "calibrated_on_split": "val",
                "values": frozen, "config_sha256": config_hashes(),
                "calibration_report": CALIB_REPORT,
                "rule": "Frozen before any test scenario is generated. Never re-tune using test outcomes."},
               FROZEN_CFG)
    check_protected()
    return calib, frozen


# --------------------------------------------------------------------------- generation
def generate_split(split, out_root, check_frozen_hashes=True, quiet=False):
    base = load_json(BASE_CFG)
    check_lstm_consistency(base, LSTM_CFG)
    if not os.path.exists(FROZEN_CFG):
        raise SystemExit("ABORT: no frozen config. Run `calibrate` on validation runs first.")
    frozen = load_json(FROZEN_CFG)
    if check_frozen_hashes and frozen["config_sha256"] != config_hashes():
        raise SystemExit("ABORT: config files changed after freezing; refusing to generate.")
    scen = load_json(SCEN_CFG)
    configs = expand_configurations(scen)
    link_profile = load_link_profile(LINK_CFG, base["link"]["link_profile"])
    link_model = LinkCapacityModel(link_profile)
    sites = load_gnb_sites(base["gnb"]["sites_file"])
    seed = base["base_seed"]
    runs = split_runs(split)
    split_dir = f"{out_root}/{split}"
    if os.path.exists(split_dir):
        raise SystemExit(f"ABORT: {split_dir} already exists (overwrite protection).")
    os.makedirs(split_dir)

    manifest, stats, validation = [], {}, {}
    all_ok = True
    for run_id in runs:
        run = load_run(run_id)
        conn = read_csv_gz(f"{CONN_DIR}/{run_id}/pairwise_connectivity.csv.gz")
        for cfg in configs:
            params = resolve_parameters(base, cfg["levels"], frozen["values"])
            for rep in range(base["replicates"]):
                sid = f"{cfg['config_id']}__{run_id}__r{rep}"
                sdir = f"{split_dir}/{sid}"
                os.makedirs(sdir)
                pue, wd, _ = generate_pues(run, params, sid, seed, rep)
                cue = generate_cues(run, params, sid, seed, rep)
                gnb, site_df = generate_gnbs(run, params, sid, seed, rep, sites)
                files = {"pue_resources": pue, "cue_requests": cue, "gnodeb_resources": gnb,
                         "gnodeb_sites": site_df, "pue_withdrawals": wd}
                ck = validate_scenario(run, params, files, conn)
                write_csv_gz(pue, f"{sdir}/pue_resources.csv.gz")
                write_csv_gz(cue, f"{sdir}/cue_requests.csv.gz")
                write_csv_gz(gnb, f"{sdir}/gnodeb_resources.csv.gz")
                write_csv(site_df, f"{sdir}/gnodeb_sites.csv")
                write_csv(wd, f"{sdir}/pue_withdrawals.csv")
                hashes = {f: sha256_file(f"{sdir}/{f}") for f in SCENARIO_FILES}
                write_json({"scenario_id": sid, "config_id": cfg["config_id"], "run_id": run_id,
                            "replicate": rep, "split": split, "seed": seed_key(seed, run_id, rep),
                            "parameters": params, "sha256": hashes,
                            "connectivity": f"{CONN_DIR}/{run_id}/pairwise_connectivity.csv.gz",
                            "synthetic": True}, f"{sdir}/scenario_meta.json")
                st = scenario_stats(run, params, files, conn, link_model, sites)
                stats[sid] = st
                validation[sid] = {"passed": ck.passed, "n_checks": len(ck.items),
                                   "failures": ck.failures()}
                all_ok &= ck.passed
                row = {"scenario_id": sid, "config_id": cfg["config_id"], "run_id": run_id,
                       "replicate": rep, "base_seed": seed, **{f"level_{k}": v for k, v in cfg["levels"].items()},
                       "path": sdir, "validation_passed": ck.passed}
                row.update({f"sha256_{f.split('.')[0]}": h for f, h in hashes.items()})
                manifest.append(row)
                if not ck.passed:
                    log(f"  VALIDATION FAILED: {sid}: {ck.failures()}")
        if not quiet:
            log(f"  {run_id}: {len(configs) * base['replicates']} scenarios generated")

    write_csv(pd.DataFrame(manifest), f"{split_dir}/marketplace_scenarios.csv")
    return manifest, stats, validation, all_ok, configs, base, frozen, link_model


def cmd_generate(split):
    check_protected()
    if not os.path.exists(f"{CONN_DIR}/{split_runs(split)[0]}"):
        raise SystemExit(f"ABORT: run `connectivity --split {split}` first.")
    manifest, stats, validation, ok, configs, base, frozen, link_model = generate_split(split, OUT_ROOT)
    split_dir = f"{OUT_ROOT}/{split}"
    conn_meta = {r: load_json(f"{CONN_DIR}/{r}/connectivity_meta.json") for r in split_runs(split)}
    write_json({"generated": datetime.now().isoformat(timespec="seconds"), "split": split,
                "runs": split_runs(split), "base_config": load_json(BASE_CFG),
                "scenarios_config": load_json(SCEN_CFG),
                "link_profile": load_link_profile(LINK_CFG, base["link"]["link_profile"]),
                "frozen": frozen, "configurations": configs,
                "config_sha256": config_hashes(), "synthetic": True,
                "note": "Same scenario files and seeds must be used for every marketplace method."},
               f"{split_dir}/generation_config.json")
    write_json({"split": split, "n_scenarios": len(manifest),
                "link_capacity_table": link_model.capacity_table(),
                "connectivity_per_run": {r: {k: conn_meta[r][k] for k in
                                             ("rows_in_range", "rows_history_ok", "unique_pairs",
                                              "first_break_cause_counts", "survives_30s_counts")}
                                         for r in conn_meta},
                "per_configuration": aggregate_by_configuration(stats, manifest),
                "per_scenario": stats}, f"{split_dir}/dataset_stats.json")
    n_checks = sum(v["n_checks"] for v in validation.values())
    write_json({"split": split, "all_passed": ok, "n_scenarios": len(validation),
                "n_checks_total": n_checks,
                "connectivity_validation": {r: all(c["passed"] for c in conn_meta[r]["validation"])
                                            for r in conn_meta},
                "per_scenario": validation}, f"{split_dir}/validation_report.json")
    write_text(SYNTHETIC_NOTICE, f"{split_dir}/SYNTHETIC_DATA.md")
    check_protected()
    log(f"  {len(manifest)} scenarios, {n_checks} checks, all passed = {ok}")
    if not ok:
        raise SystemExit(1)


def cmd_repro_check(split):
    """Regenerate every scenario into a scratch dir and compare file hashes with the manifest."""
    manifest = pd.read_csv(f"{OUT_ROOT}/{split}/marketplace_scenarios.csv")
    tmp = tempfile.mkdtemp(prefix="repro_")
    try:
        new_manifest, *_ = generate_split(split, tmp, quiet=True)
    finally:
        pass
    new = pd.DataFrame(new_manifest).set_index("scenario_id")
    old = manifest.set_index("scenario_id")
    cols = [c for c in old.columns if c.startswith("sha256_")]
    mism = [(sid, c) for sid in old.index for c in cols if old.loc[sid, c] != new.loc[sid, c]]
    shutil.rmtree(tmp)
    log(f"  reproducibility: {len(old) * len(cols)} file hashes compared, {len(mism)} mismatches")
    return {"split": split, "files_compared": len(old) * len(cols), "mismatches": mism}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("connectivity", "generate", "repro-check"):
        s = sub.add_parser(name)
        s.add_argument("--split", choices=["val", "test"], required=True)
    sub.add_parser("calibrate")
    a = ap.parse_args()
    if a.cmd == "connectivity":
        cmd_connectivity(a.split)
    elif a.cmd == "calibrate":
        calib, frozen = cmd_calibrate()
        log(json.dumps({"frozen": frozen, "flags": calib["flags"]}, indent=2))
    elif a.cmd == "generate":
        cmd_generate(a.split)
    elif a.cmd == "repro-check":
        res = cmd_repro_check(a.split)
        out = f"{OUT_ROOT}/{a.split}/reproducibility_report.json"
        write_json(res, out)
        if res["mismatches"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

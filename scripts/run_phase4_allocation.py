#!/usr/bin/env python3
"""
Phase 4 — marketplace allocation (per-second snapshot): greedy vs two-stage MILP vs gNodeB-only.

  benchmark --split val        MILP/greedy runtime on a representative validation subset
  run --split val|test         full grid (G/F1/F2/F3 x theta primary/sensitivity x methods)
  freeze                       record SHA-256 of code + config; REQUIRED before `run --split test`
  repro-check --split val --scenarios N   re-run the first N scenarios in scratch and compare hashes

Inputs are Phase 2/3 files only (read-only). Outputs are write-once under output_root.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.datagen.io_utils import write_csv_gz, write_json  # noqa: E402
from src.marketplace.allocation import audit  # noqa: E402
from src.marketplace.eligibility import ScenarioInputs  # noqa: E402
from src.marketplace.evaluation import (dominance_violations, epoch_row, paired_comparison,  # noqa: E402
                                        result_rows, scenario_metrics)
from src.marketplace.greedy import GreedyAllocator  # noqa: E402
from src.marketplace.milp import MilpAllocator  # noqa: E402
from src.marketplace.pricing import PostedPricePolicy  # noqa: E402
from src.utils.integrity import sha256_file, verify_manifest  # noqa: E402

CONFIG = "config/marketplace/phase4.json"
FROZEN = "config/marketplace/phase4_frozen.json"
PROTECTED = "outputs/PROTECTED_ARTIFACTS.sha256"


def log(m):
    print(m, flush=True)


def guard():
    ch, mi = verify_manifest(PROTECTED)
    if ch or mi:
        raise SystemExit(f"ABORT: protected artifacts changed={ch} missing={mi}")


def code_hashes():
    files = sorted(glob.glob("src/marketplace/*.py")) + ["scripts/run_phase4_allocation.py", CONFIG,
                                                          "src/common/units.py"]
    return {f: sha256_file(f) for f in files}


def thresholds(cfg, split):
    return json.load(open(f"{cfg['phase3_root']}/{split}/experiment_config.json"))["thresholds"]


def allocate_scenario(args):
    """Runs every (channel, theta, method) for one scenario. Returns metrics inputs + checks."""
    cfg, split, sid, out_dir = args
    eps, tol = cfg["rate_eps_mbps"], cfg["audit_tolerance_mbps"]
    thr = thresholds(cfg, split)
    S = ScenarioInputs(sid, f"{cfg['phase3_root']}/{split}/{sid}", f"{cfg['phase2_root']}/{split}/{sid}", cfg["dt_s"])
    greedy = GreedyAllocator(eps)
    milp = MilpAllocator(eps, **{k: cfg["milp"][k] for k in ("mip_rel_gap", "time_limit_s", "presolve")})
    gnb = MilpAllocator(eps, name="gnb_only", **{k: cfg["milp"][k] for k in ("mip_rel_gap", "time_limit_s", "presolve")})
    pricing = PostedPricePolicy()
    res, eps_rows, dom = [], [], {}
    gnb_by_t = {}       # gNodeB-only is independent of channel/theta: solved once per epoch
    for ch in cfg["channel_scenarios"]:
        for tt in cfg["theta_tags"]:
            for p in S.problems(ch, tt):
                ag, am = greedy.solve(p), milp.solve(p)
                for a in (ag, am):
                    v = audit(p, a, tol)
                    res += result_rows(p, a, a.method, ch, tt, thr[tt], pricing, eps)
                    eps_rows.append(epoch_row(p, a, a.method, ch, tt, v))
                if p.time_s not in gnb_by_t:
                    q = p.without_links()
                    a0 = gnb.solve(q)
                    v = audit(q, a0, tol)
                    res += result_rows(q, a0, "gnb_only", "none", "none", None, pricing, eps)
                    eps_rows.append(epoch_row(q, a0, "gnb_only", "none", "none", v))
                    gnb_by_t[p.time_s] = a0
                # the exact lexicographic MILP must dominate any feasible method on the same epoch
                for other, label in ((ag, "greedy"), (gnb_by_t[p.time_s], "gnb_only")):
                    for k, x in dominance_violations(am, other, label).items():
                        dom[k] = dom.get(k, 0) + x
    res, ep = pd.DataFrame(res), pd.DataFrame(eps_rows)
    os.makedirs(out_dir)
    write_csv_gz(res, f"{out_dir}/allocation_results.csv.gz")
    write_csv_gz(ep, f"{out_dir}/epoch_summary.csv.gz")
    return {"scenario_id": sid, "metrics": scenario_metrics(res, ep), "dominance": dom,
            "audit_violations": int(ep.audit_violations.sum()),
            "sha256": {f: sha256_file(f"{out_dir}/{f}") for f in ("allocation_results.csv.gz", "epoch_summary.csv.gz")},
            "rows": len(res), "milp_runtime_s": float(ep[ep.method == "milp"].runtime_s.sum()),
            "phantom_x": int(ep.phantom_x.fillna(0).sum())}


def run_split(cfg, split, out_root, scenario_ids, workers):
    split_dir = os.path.join(out_root, split)
    if os.path.exists(split_dir):
        raise SystemExit(f"ABORT: {split_dir} exists (overwrite protection)")
    os.makedirs(split_dir)
    jobs = [(cfg, split, sid, os.path.join(split_dir, sid)) for sid in scenario_ids]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(allocate_scenario, jobs))
    else:
        results = [allocate_scenario(j) for j in jobs]
    return split_dir, results


def cmd_run(split):
    cfg = json.load(open(CONFIG))
    guard()
    if split == "test":
        if not os.path.exists(FROZEN):
            raise SystemExit("ABORT: test split requires a frozen implementation (run `freeze` first)")
        if json.load(open(FROZEN))["code_sha256"] != code_hashes():
            raise SystemExit("ABORT: code/config changed after freezing; refusing to run the test split")
    man = pd.read_csv(f"{cfg['phase3_root']}/{split}/phase3_manifest.csv")
    t0 = time.time()
    split_dir, results = run_split(cfg, split, cfg["output_root"], man.scenario_id.tolist(), cfg["workers"])
    metrics = pd.concat([r["metrics"] for r in results], ignore_index=True)
    metrics["config_id"] = metrics.scenario_id.map(dict(zip(man.scenario_id, man.config_id)))
    write_csv_gz(metrics, f"{split_dir}/metrics.csv.gz")
    dom = {}
    for r in results:
        for k, v in r["dominance"].items():
            dom[k] = dom.get(k, 0) + v
    cmp_ = {"milp_vs_greedy": paired_comparison(metrics, "milp", "greedy", metrics.config_id, **cfg["bootstrap"]),
            "milp_vs_gnb_only": paired_comparison(metrics, "milp", "gnb_only", metrics.config_id, **cfg["bootstrap"]),
            "greedy_vs_gnb_only": paired_comparison(metrics, "greedy", "gnb_only", metrics.config_id, **cfg["bootstrap"])}
    agg_cols = ["admitted_share", "payment_total", "payment_per_gb", "pue_volume_share", "pue_utilisation",
                "gnb_utilisation", "served_pue_only", "served_pue_plus_gnb", "served_gnb_only", "served_rejected",
                "reject_no_service_option", "reject_insufficient_capacity_standalone", "reject_budget_exceeded",
                "reject_capacity_contention", "runtime_total_s"]
    by_cfg = metrics.groupby(["config_id", "method", "channel_scenario", "theta_tag"])[agg_cols].mean()
    write_json({"split": split, "n_scenarios": len(results), "comparison": cmp_,
                "per_config_mean": json.loads(by_cfg.round(6).reset_index().to_json(orient="records")),
                "overall_mean": json.loads(metrics.groupby(["method", "channel_scenario", "theta_tag"])[agg_cols]
                                           .mean().round(6).reset_index().to_json(orient="records"))},
               f"{split_dir}/comparison.json")
    total_audit = sum(r["audit_violations"] for r in results)
    write_json({"split": split, "audit_violations_total": total_audit, "dominance_violations": dom,
                "phantom_x_events_handled": sum(r["phantom_x"] for r in results),
                "all_passed": total_audit == 0 and all(v == 0 for v in dom.values())},
               f"{split_dir}/validation_report.json")
    pd.DataFrame([{"scenario_id": r["scenario_id"], "rows": r["rows"], "milp_runtime_s": r["milp_runtime_s"],
                   "audit_violations": r["audit_violations"], **{f"sha256_{k.split('.')[0]}": v for k, v in r["sha256"].items()}}
                  for r in results]).to_csv(f"{split_dir}/phase4_manifest.csv", index=False)
    write_json({"generated": datetime.now().isoformat(timespec="seconds"), "split": split, "config": cfg,
                "thresholds": thresholds(cfg, split), "code_sha256": code_hashes(),
                "phase3_manifest_sha256": sha256_file(f"{cfg['phase3_root']}/{split}/phase3_manifest.csv"),
                "phase2_manifest_sha256": sha256_file(f"{cfg['phase2_root']}/{split}/marketplace_scenarios.csv"),
                "wall_time_s": round(time.time() - t0, 1)}, f"{split_dir}/experiment_config.json")
    guard()
    log(f"  {len(results)} scenarios, audit violations={total_audit}, dominance={dom}, wall={time.time() - t0:.0f}s")


def cmd_benchmark():
    cfg = json.load(open(CONFIG))
    b = cfg["benchmark"]
    out = os.path.join(cfg["output_root"], "benchmark")
    if os.path.exists(out):
        raise SystemExit(f"ABORT: {out} exists")
    sids = [f"{c}__{r}__r{b['replicate']}" for c in b["configs"] for r in b["runs"]]
    tmp = tempfile.mkdtemp(prefix="p4bench_")
    t0 = time.time()
    try:
        _, results = run_split(cfg, b["split"], tmp, sids, workers=1)
        eps = pd.concat([pd.read_csv(f"{tmp}/{b['split']}/{s}/epoch_summary.csv.gz") for s in sids])
    finally:
        shutil.rmtree(tmp)
    wall = time.time() - t0
    rt = eps.groupby("method").runtime_s.describe(percentiles=[.5, .95, .99]).round(5)
    man = pd.read_csv(f"{cfg['phase3_root']}/{b['split']}/phase3_manifest.csv")
    full_epochs_est = len(eps[eps.method == "milp"]) / len(sids) * len(man)
    proj = full_epochs_est * eps[eps.method == "milp"].runtime_s.mean()
    os.makedirs(out)
    rep = {"scenarios": sids, "wall_time_s": round(wall, 1), "epochs_per_method": eps.method.value_counts().to_dict(),
           "runtime_s_by_method": json.loads(rt.to_json(orient="index")),
           "max_problem_size": {"n_requests": int(eps.n_requests.max()), "n_links": int(eps.n_links.max()),
                                "n_vars": int(eps.n_vars.max())},
           "audit_violations": int(eps.audit_violations.sum()),
           "projected_full_split_milp_cpu_s": round(proj, 1),
           "projected_full_split_wall_s_with_workers": round(wall / len(sids) * len(man) / cfg["workers"], 1)}
    write_json(rep, f"{out}/val_benchmark.json")
    log(json.dumps(rep, indent=2))


def cmd_freeze():
    if os.path.exists(FROZEN):
        raise SystemExit(f"ABORT: {FROZEN} exists")
    cfg = json.load(open(CONFIG))
    val = f"{cfg['output_root']}/val/validation_report.json"
    if not os.path.exists(val) or not json.load(open(val))["all_passed"]:
        raise SystemExit("ABORT: freeze requires a completed, fully passing validation run")
    write_json({"frozen_at": datetime.now().isoformat(timespec="seconds"), "code_sha256": code_hashes(),
                "validation_report_sha256": sha256_file(val),
                "rule": "Implementation and configuration frozen after validation; the test split may only run "
                        "with identical code/config hashes. Never tune on test outcomes."}, FROZEN)
    log(f"frozen: {FROZEN}")


def cmd_repro(split, n):
    cfg = json.load(open(CONFIG))
    old = pd.read_csv(f"{cfg['output_root']}/{split}/phase4_manifest.csv").head(n).set_index("scenario_id")
    tmp = tempfile.mkdtemp(prefix="p4repro_")
    try:
        _, results = run_split(cfg, split, tmp, old.index.tolist(), cfg["workers"])
    finally:
        shutil.rmtree(tmp)
    mism = [(r["scenario_id"], k) for r in results for k, v in r["sha256"].items()
            if old.loc[r["scenario_id"], f"sha256_{k.split('.')[0]}"] != v]
    rep = {"split": split, "scenarios_rerun": len(results), "files_compared": 2 * len(results), "mismatches": mism}
    write_json(rep, f"{cfg['output_root']}/{split}/reproducibility_report.json")
    log(json.dumps(rep))
    if mism:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["benchmark", "run", "freeze", "repro-check"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--scenarios", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "benchmark":
        cmd_benchmark()
    elif a.cmd == "run":
        cmd_run(a.split)
    elif a.cmd == "freeze":
        cmd_freeze()
    else:
        cmd_repro(a.split, a.scenarios or 10 ** 9)

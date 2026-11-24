#!/usr/bin/env python3
"""
Phase 3 — mobility and network simulation (no pricing / selection / allocation / MEC sessions).

  cache --split val|test        per-run channel + reliability + episode caches (scenario-independent)
  simulate --split val|test     per-scenario link feasibility using the caches and Phase 2 scenarios
  repro-check --split val|test  rebuild everything in a scratch directory and compare file hashes

All outputs are write-once. Inputs (config, thresholds, geometry, checkpoints, Phase 2 manifest)
are hashed into the manifests. Nothing is tuned: thresholds come from the recorded files.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.datagen.io_utils import read_csv_gz, write_csv_gz, write_json  # noqa: E402
from src.mobility import load_run  # noqa: E402
from src.mobility.trace import load_road_ids  # noqa: E402
from src.network.buildings import load_building_layer  # noqa: E402
from src.network.coverage import load_gnb_sites  # noqa: E402
from src.network.technology import load_technology  # noqa: E402
from src.reliability.lstm_distance import LSTMDistanceSurvivalModel  # noqa: E402
from src.reliability.policy import ThresholdPolicy  # noqa: E402
from src.simulation.channel_layer import build_channel_table, build_episode_table, make_classifiers  # noqa: E402
from src.simulation.feasibility import evaluate_scenario  # noqa: E402
from src.simulation.reliability_layer import build_reliability_table  # noqa: E402
from src.simulation.stats import link_stats, request_second_stats  # noqa: E402
from src.simulation.validate import validate_run_cache, validate_scenario_outputs  # noqa: E402
from src.utils.integrity import sha256_file, verify_manifest  # noqa: E402

CONFIG = "config/simulation/phase3.json"
PROTECTED = "outputs/PROTECTED_ARTIFACTS.sha256"


def read_cache(path):
    """Cache reader: whole-file type inference (low_memory=False) so nullable boolean / string
    columns are never inferred chunk-by-chunk. Values are identical to the default reader."""
    return pd.read_csv(path, compression="gzip", low_memory=False)


def log(m):
    print(m, flush=True)


def guard():
    ch, mi = verify_manifest(PROTECTED)
    if ch or mi:
        raise SystemExit(f"ABORT: protected artifacts changed={ch} missing={mi}")


def load_env(cfg):
    tech = load_technology(cfg["technology_file"], cfg["technology"])
    link = tech.link_model()
    g = cfg["geometry"]
    layer = load_building_layer(g["buildings"], g["coverage"], g["building_types"], g["index_cell_size_m"])
    r = cfg["reliability"]
    model = LSTMDistanceSurvivalModel(r["checkpoint"], r["model_config"], r["scaler"], r["metadata"])
    if model.range_mismatch(tech.discovery_range_m):
        raise SystemExit("ABORT: reliability label range differs from discovery range (would need a new model)")
    thresholds = {"primary": ThresholdPolicy.primary_from_metadata(r["primary_threshold_source"]).theta,
                  "sensitivity": ThresholdPolicy.sensitivity_from_file(r["sensitivity_threshold_source"]).theta}
    return tech, link, layer, model, thresholds


def runs_of(cfg, split):
    return sorted(json.load(open(cfg["split_info"]))[f"{split}_runs"])


def input_hashes(cfg):
    paths = [CONFIG, cfg["technology_file"], cfg["geometry"]["buildings"], cfg["geometry"]["coverage"],
             cfg["reliability"]["checkpoint"], cfg["reliability"]["scaler"], cfg["reliability"]["metadata"],
             cfg["reliability"]["sensitivity_threshold_source"], "config/network/link_profiles_nr.json"]
    return {p: sha256_file(p) for p in paths}


def build_cache(cfg, split, cache_root, env):
    tech, link, layer, model, _ = env
    classifiers = make_classifiers(layer, link, cfg["map_scenario"], cfg["g_fallback_scenario"])
    for run_id in runs_of(cfg, split):
        out = os.path.join(cache_root, run_id)
        if os.path.exists(out):
            raise SystemExit(f"ABORT: {out} exists (overwrite protection)")
        t0 = time.time()
        run = load_run(run_id, cfg["raw_mobility_dir"])
        conn = read_csv_gz(f"{cfg['connectivity_dir']}/{run_id}/pairwise_connectivity.csv.gz")
        gt = read_csv_gz(f"{cfg['connectivity_dir']}/{run_id}/pairwise_ground_truth.csv.gz")
        positions = {k: (v[0], v[1]) for k, v in run.lookup.items()}
        channel = build_channel_table(run_id, conn, positions, load_road_ids(run_id, cfg["raw_mobility_dir"]),
                                      classifiers, cfg["channel_scenarios"], link, cfg["channel_seed"])
        rel = build_reliability_table(conn, run.df, model)
        eps = build_episode_table(run_id, channel[channel.channel_scenario == "G"], gt, float(run.times.max()))
        ck = validate_run_cache(channel, rel, eps, conn, link, cfg["channel_scenarios"], tech.discovery_range_m)
        if not ck.passed:
            raise SystemExit(f"ABORT: cache validation failed for {run_id}: {ck.failures()}")
        os.makedirs(out)
        write_csv_gz(channel, f"{out}/channel.csv.gz")
        write_csv_gz(rel, f"{out}/reliability.csv.gz")
        write_csv_gz(eps, f"{out}/episodes.csv.gz")
        write_json({"run_id": run_id, "rows": len(conn), "episodes": len(eps), "validation": ck.items,
                    "sha256": {f: sha256_file(f"{out}/{f}") for f in ("channel.csv.gz", "reliability.csv.gz", "episodes.csv.gz")},
                    "seconds": round(time.time() - t0, 1)}, f"{out}/cache_meta.json")
        log(f"  {run_id}: {len(conn):,} candidate rows, {len(eps):,} episodes, {len(ck.items)} checks passed, {time.time() - t0:.0f}s")


def simulate(cfg, split, cache_root, out_root, env, quiet=False):
    tech, link, layer, model, thresholds = env
    sites = load_gnb_sites(cfg["gnb_sites_file"])
    manifest = pd.read_csv(f"{cfg['phase2_root']}/{split}/marketplace_scenarios.csv")
    split_dir = os.path.join(out_root, split)
    if os.path.exists(split_dir):
        raise SystemExit(f"ABORT: {split_dir} exists (overwrite protection)")
    os.makedirs(split_dir)
    rows, all_ok, val, cl_stats_acc, rs_stats = [], True, {}, [], {}
    for run_id, grp in manifest.groupby("run_id"):
        run = load_run(run_id, cfg["raw_mobility_dir"])
        conn = read_csv_gz(f"{cfg['connectivity_dir']}/{run_id}/pairwise_connectivity.csv.gz")
        channel = read_cache(f"{cache_root}/{run_id}/channel.csv.gz")
        rel = read_cache(f"{cache_root}/{run_id}/reliability.csv.gz")
        for m in grp.itertuples(index=False):
            sdir = m.path
            files = {"pue_resources": read_csv_gz(f"{sdir}/pue_resources.csv.gz"),
                     "cue_requests": read_csv_gz(f"{sdir}/cue_requests.csv.gz"),
                     "gnodeb_resources": read_csv_gz(f"{sdir}/gnodeb_resources.csv.gz"),
                     "gnodeb_sites": pd.read_csv(f"{sdir}/gnodeb_sites.csv"),
                     "pue_withdrawals": pd.read_csv(f"{sdir}/pue_withdrawals.csv")}
            rs, cl = evaluate_scenario(m.scenario_id, run, files, sites, channel, rel, conn, thresholds,
                                       tech.discovery_range_m)
            ck = validate_scenario_outputs(rs, cl, thresholds)
            all_ok &= ck.passed
            val[m.scenario_id] = {"passed": ck.passed, "n_checks": len(ck.items), "failures": ck.failures()}
            od = os.path.join(split_dir, m.scenario_id)
            os.makedirs(od)
            write_csv_gz(rs, f"{od}/request_seconds.csv.gz")
            write_csv_gz(cl, f"{od}/candidate_links.csv.gz")
            rows.append({"scenario_id": m.scenario_id, "config_id": m.config_id, "run_id": run_id,
                         "replicate": m.replicate, "phase2_path": sdir, "request_seconds": len(rs),
                         "candidate_link_rows": len(cl), "validation_passed": ck.passed,
                         "sha256_request_seconds": sha256_file(f"{od}/request_seconds.csv.gz"),
                         "sha256_candidate_links": sha256_file(f"{od}/candidate_links.csv.gz")})
            cl_stats_acc.append(cl.assign(config_id=m.config_id))
            rs_stats[m.scenario_id] = request_second_stats(rs, cl, list(thresholds))
            if not ck.passed:
                log(f"  VALIDATION FAILED {m.scenario_id}: {ck.failures()}")
        if not quiet:
            log(f"  {run_id}: {len(grp)} scenarios")
    man = pd.DataFrame(rows)
    man.to_csv(f"{split_dir}/phase3_manifest.csv", index=False)
    return man, all_ok, val, cl_stats_acc, rs_stats


def cmd_cache(split):
    cfg = json.load(open(CONFIG))
    guard()
    env = load_env(cfg)
    build_cache(cfg, split, cfg["cache_dir"], env)
    guard()


def cmd_simulate(split):
    cfg = json.load(open(CONFIG))
    guard()
    env = load_env(cfg)
    man, ok, val, cls, rss = simulate(cfg, split, cfg["cache_dir"], cfg["output_root"], env)
    split_dir = os.path.join(cfg["output_root"], split)
    allcl = pd.concat(cls, ignore_index=True)
    tags = list(env[4])
    per_config = {cid: link_stats(g, tags) for cid, g in allcl.groupby("config_id")}
    rsdf = pd.DataFrame.from_dict(rss, orient="index")
    rsdf["config_id"] = man.set_index("scenario_id").loc[rsdf.index, "config_id"]
    eps = pd.concat([read_csv_gz(f"{cfg['cache_dir']}/{r}/episodes.csv.gz") for r in runs_of(cfg, split)])
    write_json({"generated": datetime.now().isoformat(timespec="seconds"), "split": split,
                "config": cfg, "thresholds": env[4], "technology": env[0].name,
                "link_profile": env[1].name, "input_sha256": input_hashes(cfg),
                "phase2_manifest_sha256": sha256_file(f"{cfg['phase2_root']}/{split}/marketplace_scenarios.csv"),
                "cache_meta": {r: json.load(open(f"{cfg['cache_dir']}/{r}/cache_meta.json"))["sha256"] for r in runs_of(cfg, split)}},
               f"{split_dir}/experiment_config.json")
    write_json({"split": split, "n_scenarios": len(man),
                "link_stats_all_configs": link_stats(allcl, tags),
                "link_stats_per_config": per_config,
                "request_second_stats_mean_per_config": json.loads(rsdf.groupby("config_id").mean(numeric_only=True).round(4).to_json(orient="index")),
                "episode_end_causes": eps.end_cause.value_counts().to_dict(),
                "episode_duration_s": eps.n_seconds.describe().round(2).to_dict()},
               f"{split_dir}/phase3_stats.json")
    write_json({"split": split, "all_passed": ok, "n_scenarios": len(val),
                "n_checks_total": sum(v["n_checks"] for v in val.values()), "per_scenario": val},
               f"{split_dir}/validation_report.json")
    guard()
    log(f"  {len(man)} scenarios, all validations passed = {ok}")
    if not ok:
        raise SystemExit(1)


def cmd_repro(split):
    cfg = json.load(open(CONFIG))
    env = load_env(cfg)
    tmp = tempfile.mkdtemp(prefix="phase3_repro_")
    try:
        build_cache(cfg, split, os.path.join(tmp, "cache"), env)
        cache_ok = all(json.load(open(f"{tmp}/cache/{r}/cache_meta.json"))["sha256"] ==
                       json.load(open(f"{cfg['cache_dir']}/{r}/cache_meta.json"))["sha256"] for r in runs_of(cfg, split))
        man, *_ = simulate(cfg, split, os.path.join(tmp, "cache"), os.path.join(tmp, "out"), env, quiet=True)
        old = pd.read_csv(f"{cfg['output_root']}/{split}/phase3_manifest.csv").set_index("scenario_id")
        new = man.set_index("scenario_id")
        cols = ["sha256_request_seconds", "sha256_candidate_links"]
        mism = [(s, c) for s in old.index for c in cols if old.loc[s, c] != new.loc[s, c]]
    finally:
        shutil.rmtree(tmp)
    res = {"split": split, "cache_hashes_identical": cache_ok, "files_compared": len(old) * 2, "mismatches": mism}
    write_json(res, f"{cfg['output_root']}/{split}/reproducibility_report.json")
    log(json.dumps({k: res[k] for k in ("cache_hashes_identical", "files_compared")}) + f" mismatches={len(mism)}")
    if mism or not cache_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "simulate", "repro-check"])
    ap.add_argument("--split", choices=["val", "test"], required=True)
    a = ap.parse_args()
    {"cache": cmd_cache, "simulate": cmd_simulate, "repro-check": cmd_repro}[a.cmd](a.split)

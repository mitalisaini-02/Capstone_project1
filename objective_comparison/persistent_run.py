"""Run Experiment B with python -m objective_comparison.persistent_run."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from objective_comparison.allocation import Snapshot
from objective_comparison.sinr import snapshot
from src.marketplace.eligibility import ScenarioInputs
from src.network.coverage import load_gnb_sites
from src.network.technology import load_technology
from src.network.tr37885 import TR37885SidelinkModel
from src.reliability.lstm_distance import LSTMDistanceSurvivalModel
from src.reliability.policy import ThresholdPolicy
from src.utils.integrity import sha256_file


from objective_comparison.run import power_rows, prepare, read_json, xy
from objective_comparison.persistent import PersistentScenario


def run_scenario(cfg, phase, run, fresh, pc5, gnb, sites, row, out, limit):
    theta = cfg["reliability_threshold"]
    channel = cfg["channel"]
    path = Path(phase["output_root"]) / cfg["split"] / row.scenario_id
    market = Path(row.phase2_path)
    source = ScenarioInputs(row.scenario_id, str(path), str(market))
    columns = ["time_s", "pue_id", "cue_id"]
    source.cl = source.cl.merge(fresh, on=columns, suffixes=("_cached", ""), validate="many_to_one")
    if not np.allclose(source.cl.p_survive, source.cl.p_survive_cached, atol=2e-6, rtol=0, equal_nan=True):
        raise ValueError(f"{row.scenario_id}: candidate predictions disagree with fresh inference")
    source.cl["feasible_primary"] &= source.cl.p_survive >= theta
    candidates = pd.read_csv(path / "candidate_links.csv.gz",
                             usecols=columns + ["channel_scenario", "realized_state", "request_id"])
    candidates = candidates[candidates.channel_scenario == channel]
    candidates = candidates.merge(source.cl[source.cl.channel_scenario == channel][
        columns + ["request_id", "feasible_primary", "realized_link_capacity_mbps", "p_survive"]],
        on=columns + ["request_id"], validate="one_to_one")
    channel_by_t = {t: g for t, g in candidates.groupby("time_s")}
    available = pd.read_csv(market / "pue_resources.csv.gz", usecols=["time_s", "pue_id", "available"])
    available_by_t = {t: g.pue_id.to_numpy() for t, g in available[available.available].groupby("time_s")}
    sites_xy = np.array([s[1:] for s in sites])
    site_idx = {s[0]: i for i, s in enumerate(sites)}
    raw = []
    sessions = PersistentScenario(cfg)
    for epoch, p in enumerate(source.problems(channel, "primary")):
        if limit is not None and epoch >= limit:
            break
        t0 = perf_counter()
        t = p.time_s
        cue_xy = xy(run, t, p.cue_ids)
        candidates_t = channel_by_t.get(t)
        active = available_by_t.get(t, np.empty(0, dtype=int))
        # Include inactive candidates for diagnostics, on unique non-transmitting channels.
        present = active if candidates_t is None else np.union1d(active, candidates_t.pue_id)
        channels = np.where(np.isin(present, active), present % cfg["pc5_reuse_channels"],
                            cfg["pc5_reuse_channels"] + np.arange(len(present)))
        states = np.full((p.n_requests, len(present)), "LOS", dtype="U5")
        pm = {i: idx for idx, i in enumerate(present)}
        jm = {r: idx for idx, r in enumerate(p.request_ids)}
        if candidates_t is not None:
            ii = candidates_t.pue_id.map(pm).to_numpy()
            jj = candidates_t.request_id.map(jm).to_numpy()
            states[jj, ii] = candidates_t.realized_state
        powers = power_rows(snapshot(pc5, xy(run, t, present), cue_xy, channels, states))
        selected_providers = np.array([pm[i] for i in p.pue_ids[p.link_pue]], dtype=int)
        link_sinr = powers["sinr"][p.link_request, selected_providers]
        cell_radio = power_rows(snapshot(gnb, sites_xy, cue_xy, np.zeros(len(sites)),
                                         np.full((p.n_requests, len(sites)), "LOS")))
        cell_se = np.zeros(p.n_requests)
        identifiers = dict(scenario_id=row.scenario_id, time_s=t, eta=cfg["eta"])
        for j in np.flatnonzero(p.covered):
            k = site_idx[p.cell_ids[p.cell_index[j]]]
            cell_se[j] = np.log2(1 + cell_radio["sinr"][j, k])
        s = Snapshot(p, np.log2(1 + link_sinr), cell_se, pc5.bw_hz / 1e6, gnb.bw_hz / 1e6,
                     cfg["eta"], cfg["epsilon"], cfg["min_selected_mbps"], theta)
        common_s = perf_counter() - t0
        for method, metric in sessions.step(s):
            raw.append(dict(**identifiers, algorithm=method, shared_snapshot_s=common_s,
                            available_pues=len(active), **metric))
    sessions.write(row.scenario_id, out, raw)


def run_group(args):
    import torch

    cfg, phase, run_id, rows, out, limit = args
    torch.set_num_threads(1)
    r = phase["reliability"]
    model = LSTMDistanceSurvivalModel(r["checkpoint"], r["model_config"], r["scaler"], r["metadata"])
    tech = load_technology(phase["technology_file"], phase["technology"])
    if model.range_mismatch(tech.discovery_range_m):
        raise ValueError("Discovery radius does not match LSTM label radius")
    stats = {}
    run, fresh = prepare(cfg, phase, run_id, model, stats)
    pc5 = tech.link_model()
    gnb = TR37885SidelinkModel({**pc5.p, **cfg["gnb_radio_overrides"]})
    sites = load_gnb_sites(phase["gnb_sites_file"])
    for row in rows.itertuples(index=False):
        run_scenario(cfg, phase, run, fresh, pc5, gnb, sites, row, out, limit)
    print(f"{run_id}: completed {len(rows)} scenarios", flush=True)
    return run_id, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="objective_comparison/config.json")
    parser.add_argument("--output")
    parser.add_argument("--split", choices=["val", "test"])
    parser.add_argument("--scenarios", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.split:
        cfg["split"] = args.split
    phase = read_json(cfg["phase3_config"])
    r = phase["reliability"]
    meta = read_json(r["metadata"])
    expected = meta["sha256"]
    actual = {path: sha256_file(path) for path in expected}
    if actual != expected:
        raise ValueError("Checkpoint, architecture or scaler hash mismatch")
    if cfg["reliability_threshold"] != ThresholdPolicy.primary_from_metadata(r["metadata"]).theta:
        raise ValueError("Configured reliability threshold differs from existing primary policy")
    manifest = Path(phase["output_root"]) / cfg["split"] / "phase3_manifest.csv"
    rows = pd.read_csv(manifest).head(args.scenarios) if args.scenarios else pd.read_csv(manifest)
    cfg["output"] = "objective_comparison/persistent_results"
    cfg["experiment"] = "persistent_initial_only"
    out = Path(args.output or cfg["output"])
    out.mkdir(parents=True)
    jobs = [(cfg, phase, run_id, group, out, args.epochs) for run_id, group in rows.groupby("run_id")]
    start = perf_counter()
    if args.workers == 1:
        stats = dict(map(run_group, jobs))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            stats = dict(pool.map(run_group, jobs))
    tech = load_technology(phase["technology_file"], phase["technology"])
    pc5 = tech.link_model()
    metadata = dict(config=cfg, phase3_config=phase, pc5_profile=pc5.p,
                    gnb_profile={**pc5.p, **cfg["gnb_radio_overrides"]}, lstm=stats,
                    checkpoint_hashes=actual, manifest_sha256=sha256_file(manifest),
                    scenario_ids=rows.scenario_id.tolist(), epoch_limit=args.epochs,
                    wall_time_s=perf_counter() - start)
    if {path: sha256_file(path) for path in actual} != actual:
        raise RuntimeError("A trained artifact changed during evaluation")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

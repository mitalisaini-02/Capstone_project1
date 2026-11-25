"""Run with python -m objective_comparison.run from the repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from objective_comparison.allocation import Snapshot, choose
from objective_comparison.sinr import snapshot
from src.marketplace.eligibility import ScenarioInputs
from src.mobility import load_run
from src.network.coverage import load_gnb_sites
from src.network.technology import load_technology
from src.network.tr37885 import TR37885SidelinkModel
from src.reliability.lstm_distance import LSTMDistanceSurvivalModel
from src.reliability.policy import ThresholdPolicy
from src.simulation.reliability_layer import build_reliability_table
from src.utils.integrity import sha256_file


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare(cfg, phase, run_id, model, inputs):
    root = Path(phase["connectivity_dir"]) / run_id
    run = load_run(run_id, phase["raw_mobility_dir"])
    conn = pd.read_csv(root / "pairwise_connectivity.csv.gz")
    fresh = build_reliability_table(conn, run.df, model)
    old = pd.read_csv(Path(phase["cache_dir"]) / run_id / "reliability.csv.gz")
    keys = ["time_s", "pue_id", "cue_id"]
    paired = fresh.merge(old, on=keys, suffixes=("", "_old"), validate="one_to_one")
    probabilities = fresh.p_survive.dropna()
    if (len(fresh) != len(old) or len(fresh) != len(paired) or probabilities.empty
            or not probabilities.between(0., 1.).all()
            or not np.allclose(paired.p_survive, paired.p_survive_old, atol=2e-6, rtol=0, equal_nan=True)
            or not np.array_equal(paired.p_survive >= cfg["reliability_threshold"],
                                  paired.p_survive_old >= cfg["reliability_threshold"])):
        raise ValueError(f"{run_id}: fresh LSTM predictions failed validation")
    inputs.update(n_candidates=len(fresh), n_predictions=len(probabilities),
                  min_probability=float(probabilities.min()), max_probability=float(probabilities.max()),
                  max_cached_delta=float(abs(paired.p_survive - paired.p_survive_old).max()),
                  n_pues=len(run.providers), n_cues=len(run.consumers))
    return run, fresh[keys + ["p_survive"]]


def xy(run, t, ids):
    return np.asarray([run.state(t, i)[:2] for i in ids]).reshape(-1, 2)


def power_rows(radio):
    distance, loss, desired, interference, noise, sinr, db = radio
    return dict(distance_m=distance, path_loss_db=loss, received_mw=desired,
                interference_mw=interference, noise_mw=np.broadcast_to(noise, sinr.shape),
                sinr=sinr, sinr_db=db)


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
    raw, trials, links = [], [], []
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
        if candidates_t is not None:
            for k, rec in enumerate(candidates_t.itertuples(index=False)):
                links.append(dict(**identifiers, radio="PC5", transmitter=rec.pue_id, receiver=rec.cue_id,
                                  request_id=rec.request_id, p_survive=rec.p_survive, eligible=rec.feasible_primary,
                                  capacity_mbps=rec.realized_link_capacity_mbps,
                                  resource_channel=channels[ii[k]],
                                  transmitting=bool(channels[ii[k]] < cfg["pc5_reuse_channels"]),
                                  **{key: vals[jj[k], ii[k]] for key, vals in powers.items()}))
        for j in np.flatnonzero(p.covered):
            k = site_idx[p.cell_ids[p.cell_index[j]]]
            cell_se[j] = np.log2(1 + cell_radio["sinr"][j, k])
            links.append(dict(**identifiers, radio="gNodeB", transmitter=p.cell_ids[p.cell_index[j]],
                              receiver=p.cue_ids[j], request_id=p.request_ids[j], eligible=True,
                              resource_channel=0, transmitting=True,
                              capacity_mbps=p.cell_capacity_mbps[p.cell_index[j]],
                              **{key: vals[j, k] for key, vals in cell_radio.items()}))
        s = Snapshot(p, np.log2(1 + link_sinr), cell_se, pc5.bw_hz / 1e6, gnb.bw_hz / 1e6,
                     cfg["eta"], cfg["epsilon"], cfg["min_selected_mbps"], theta)
        common_s = perf_counter() - t0
        for method in cfg["algorithms"]:
            start = perf_counter()
            notes = {}
            if method == "Random":
                digest = hashlib.sha256(f"{cfg['seed']}:{row.scenario_id}:{t}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                scores = [s.evaluate(s.constructive(method, rng)) for _ in range(cfg["random_trials"])]
                best = choose(scores, s.epsilon)
                metric = scores[best]
                trials.extend(dict(**identifiers, trial=i, selected_trial=i == best, **m)
                              for i, m in enumerate(scores))
            elif method == "Exact":
                plan, notes = s.exact(cfg["solver_time_limit_s"], cfg["solver_relative_gap"])
                metric = s.evaluate(plan)
                if notes["proven_optimal"] and (
                    metric["violations"] or abs(metric["rate"] - notes["stage2_rate"]) > 1e-6
                    or metric["cost"] > notes["stage1_cost"] + cfg["epsilon"] + 1e-9
                ):
                    raise RuntimeError(f"{row.scenario_id} t={t}: solver/evaluator mismatch {notes}, {metric}")
            else:
                metric = s.evaluate(s.constructive(method))
            raw.append(dict(**identifiers, algorithm=method, runtime_s=perf_counter() - start,
                            shared_snapshot_s=common_s, available_pues=len(active), **metric, **notes))
    for name, records in (("epochs", raw), ("trials", trials), ("links", links)):
        pd.DataFrame(records).to_csv(out / f"{row.scenario_id}.{name}.csv.gz", index=False)


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

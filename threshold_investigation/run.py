"""Validation-only threshold sensitivity using existing MEC and Phase 4 solvers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase5_experiments import allocator, ground_truth_check, run_metrics
from scripts.sweep_reliability_threshold import select_threshold
from src.mec.config import Phase5Config
from src.mec.coordinator import MecCoordinator
from src.mec.inputs import Phase5ScenarioInputs
from src.mec.state import StateManager
from src.mobility.trace import load_run
from src.simulation.feasibility import first_failure
from src.utils.integrity import sha256_file, verify_manifest

GRID = (0.50, 0.55, 0.60, 0.62, 0.63, 0.65, 0.67, 0.70, 0.75)
SOURCE = Path("outputs/large_dataset_experiment")
CONFIG = Path("config/mec/phase5.json")
FROZEN = Path("config/marketplace/phase4_frozen.json")
PROTECTED = Path("outputs/PROTECTED_ARTIFACTS.sha256")


def read(path):
    return json.loads(Path(path).read_text())


def classification():
    # Reuse the complete frozen validation confusion counts. No need to reopen sequences.
    frame = pd.read_csv(SOURCE / "reliability_threshold/threshold_sweep_val.csv")
    frame["precision_raw"] = frame.tp / (frame.tp + frame.fp)
    frame["recall_raw"] = frame.tp / (frame.tp + frame.fn)
    frame["specificity"] = frame.tn / (frame.tn + frame.fp)
    frame["balanced_accuracy_raw"] = (frame.recall_raw + frame.specificity) / 2
    frame["f1_score"] = 2 * frame.tp / (2 * frame.tp + frame.fp + frame.fn)
    frame["false_positive_rate"] = 1 - frame.specificity
    frame["false_negative_rate"] = 1 - frame.recall_raw
    frame["predicted_ineligible_count"] = frame.tn + frame.fn
    ci = pd.read_csv(SOURCE / "reliability_threshold_robust/grouped_ci_val.csv")
    return frame.merge(ci.drop(columns=["precision", "recall"]), on="threshold", validate="one_to_one")


def scenario(job):
    sid, cfg, cfg4, method = job
    run_id = sid.split("__")[1]
    inputs = Phase5ScenarioInputs(sid, cfg.phase3_dir("val", sid), cfg.phase2_dir("val", sid), cfg["dt_s"])
    channel = cfg["primary_channel_scenario"]
    # The same existing aware policy is exercised with different in-memory inputs.
    # Phase 5 state construction caches the flags, so create it after each update.
    cl = inputs.cl
    physical = cl.operational.to_numpy(bool)
    window = cl.lstm_window_available.fillna(False).to_numpy(bool)
    frozen = {t: cl[f"feasible_{tag}"].to_numpy(bool, copy=True)
              for t, tag in ((.62, "primary"), (.65, "sensitivity"))}
    wireless = cl.first_failure_unaware.to_numpy()
    run = load_run(run_id, cfg["inputs"]["raw_mobility_dir"])
    episodes = pd.read_csv(Path(cfg["inputs"]["evaluation_only_ground_truth_dir"]) / run_id / "episodes.csv.gz")
    rows = []
    for theta in GRID:
        reliable = cl.p_survive.to_numpy() >= theta
        eligible = physical & window & reliable
        if theta in frozen and not np.array_equal(eligible, frozen[theta]):
            raise ValueError(f"{sid}: eligibility differs from frozen threshold {theta}")
        cl["reliable_primary"] = reliable
        cl["feasible_primary"] = eligible
        cl["first_failure_primary"] = first_failure(
            {"provider_unavailable": cl.provider_ok.to_numpy(bool),
             "insufficient_history": window, "reliability_below_threshold": reliable,
             "wireless_outage": cl.wireless_ok.to_numpy(bool),
             "resource_zero": cl.resource_ok.to_numpy(bool)}, wireless)
        state = StateManager(inputs, run)
        coord = MecCoordinator(inputs, state, channel, "primary", allocator(method, cfg4),
                               cfg.max_interruption_s(), cfg["rate_eps_mbps"], cfg["audit_tolerance_mbps"],
                               cfg["continuity_weight"], proactive=cfg.proactive_params()).run()
        result, report = coord.results(), coord.report()
        sessions = result["sessions"]
        candidates = cl.loc[(cl.channel_scenario == channel) & eligible]
        counts = {key: value for key, value in run_metrics(sessions, report).items()
                  if not key.startswith("sc_")}
        truth = ground_truth_check(result["events"], sessions, state, episodes)
        rows.append(dict(scenario_id=sid, run_id=run_id, threshold=theta, allocator=method,
                         eligible_links=len(candidates), eligible_providers=candidates.pue_id.nunique(),
                         eligible_request_seconds=len(candidates[["time_s", "request_id"]].drop_duplicates()),
                         offered_request_seconds=len(inputs.rs), **counts,
                         **{f"gt_{k}": v for k, v in truth.items()}))
    return rows


def check_integrity(cfg):
    changed, missing = verify_manifest(PROTECTED)
    if changed or missing:
        raise RuntimeError(f"Protected artifacts changed={changed}, missing={missing}")
    cfg.verify_phase4_dependency()
    for path, expected in read(FROZEN)["code_sha256"].items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen Phase 4 source differs: {path}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--workers", type=int, default=6)
    cli.add_argument("--allocator", choices=["greedy", "milp"], default="greedy")
    cli.add_argument("--scenarios", type=int)
    args = cli.parse_args()
    cfg = Phase5Config.load(str(CONFIG))
    check_integrity(cfg)
    cfg4 = read(cfg["inputs"]["phase4_config"])
    manifest = Path(cfg["inputs"]["phase3_root"]) / "val/phase3_manifest.csv"
    ids = pd.read_csv(manifest).scenario_id.tolist()[:args.scenarios]
    classifications = classification()
    selected, rule = select_threshold(classifications.to_dict("records"))
    args.output.mkdir(parents=True)
    classifications.to_csv(args.output / "classification.csv", index=False)
    metadata = dict(split="val", rule=rule, selected=selected, grid=GRID, scenarios=ids,
                    allocator=args.allocator, channel=cfg["primary_channel_scenario"],
                    interruption_tolerance_s=cfg.max_interruption_s(), proactive=vars(cfg.proactive_params()),
                    input_hashes={str(p): sha256_file(p) for p in (CONFIG, FROZEN, PROTECTED, manifest,
                        SOURCE / "reliability_threshold/threshold_sweep_val.csv",
                        SOURCE / "reliability_threshold_robust/grouped_ci_val.csv")})
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    jobs = [(sid, cfg, cfg4, args.allocator) for sid in ids]
    with ProcessPoolExecutor(args.workers) as pool, (args.output / "scenario_results.csv").open("x") as f:
        for n, rows in enumerate(pool.map(scenario, jobs), 1):
            pd.DataFrame(rows).to_csv(f, header=n == 1, index=False)
            f.flush()
            if n % 12 == 0 or n == len(ids):
                print(f"{n}/{len(ids)} validation scenarios", flush=True)
    check_integrity(cfg)


if __name__ == "__main__":
    main()

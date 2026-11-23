"""
generate_sumo_runs.py
---------------------
Generates independent SUMO mobility simulation runs with varied vehicle counts,
densities, and random seeds.

Each run:
  1. Generates random trips using randomTrips.py
  2. Routes trips with duarouter
  3. Parses routed vehicle IDs and assigns roles (~60% PUE, ~40% CUE) reproducibly via run seed
  4. Runs SUMO headlessly via TraCI and extracts mobility CSV
  5. Records requested, routed, simulated, and failed vehicles
  6. Saves: trips.xml, routes.rou.xml, run.sumocfg, mobility.csv, run_config.json

Traffic densities:
  - light:      40–50 vehicles  (target 45, period 22.2s)
  - medium:     60–80 vehicles  (target 70, period 14.3s)
  - heavy:      90–110 vehicles (target 100, period 10.0s)
  - very_heavy: 120–150 vehicles (target 135, period 7.4s)

Do NOT modify Stage 1 (actual_mec_comparison.py) or Stage 2 (dynamic_method_comparison.py).
Do NOT overwrite existing pilot_001, processed datasets, splits, or checkpoints.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Set

# SUMO tools
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))

try:
    import traci
except ImportError as e:
    print(f"ERROR: Cannot import traci: {e}")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NET_FILE = "/home/mitali/Downloads/osm.net.xml"
INFRA_FILE = str(PROJECT_ROOT / "infrastructure.add.xml")
RANDOM_TRIPS_SCRIPT = os.path.join(SUMO_HOME, "tools", "randomTrips.py")

# Traffic density tiers
DENSITY_PROFILES: List[Dict[str, Any]] = [
    {"label": "light",      "target_vehicles": 45,  "period": 22.2},
    {"label": "medium",     "target_vehicles": 70,  "period": 14.3},
    {"label": "heavy",      "target_vehicles": 100, "period": 10.0},
    {"label": "very_heavy", "target_vehicles": 135, "period": 7.4},
]

DEFAULT_DURATION = 1000  # seconds


def _run_cmd(cmd: List[str], cwd: str = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] Command failed: {' '.join(cmd)}")
        print(f"  STDOUT: {result.stdout[-1000:]}")
        print(f"  STDERR: {result.stderr[-1000:]}")
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


def generate_trips(run_dir: Path, period: float, duration: int, seed: int) -> Path:
    trips_file = (run_dir / "trips.xml").resolve()
    cmd = [
        sys.executable,
        RANDOM_TRIPS_SCRIPT,
        "-n", NET_FILE,
        "-o", str(trips_file),
        "--begin", "0",
        "--end", str(duration),
        "--period", str(period),
        "--seed", str(seed),
        "--min-distance", "200.0",
        "--vehicle-class", "passenger"
    ]
    _run_cmd(cmd, cwd=str(run_dir))
    return trips_file


def route_trips(run_dir: Path, trips_file: Path, seed: int) -> Path:
    routes_file = (run_dir / "routes.rou.xml").resolve()
    cmd = [
        "duarouter",
        "-n", NET_FILE,
        "-r", str(trips_file),
        "-o", str(routes_file),
        "--seed", str(seed),
        "--ignore-errors", "true",
        "--no-warnings", "true",
    ]
    _run_cmd(cmd, cwd=str(run_dir))
    return routes_file


def parse_routed_vehicles(routes_file: Path) -> List[str]:
    tree = ET.parse(routes_file)
    return [v.get("id") for v in tree.getroot().findall("vehicle")]


def assign_roles(vids: List[str], seed: int, pue_ratio: float = 0.60) -> Dict[str, str]:
    """
    Reproducibly assign ~60% PUE (provider) and ~40% CUE (consumer) roles.
    Distributes roles uniformly across departure times.
    """
    rng = random.Random(seed)
    shuffled = sorted(vids).copy()
    rng.shuffle(shuffled)
    n_pue = int(round(len(shuffled) * pue_ratio))
    pue_set = set(shuffled[:n_pue])
    return {vid: ("provider" if vid in pue_set else "consumer") for vid in vids}


def write_sumocfg(run_dir: Path, routes_file: Path, duration: int) -> Path:
    cfg_file = (run_dir / "run.sumocfg").resolve()
    content = f"""<configuration>
    <input>
        <net-file value="{NET_FILE}"/>
        <route-files value="{routes_file}"/>
        <additional-files value="{INFRA_FILE}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{duration}"/>
        <step-length value="1"/>
    </time>
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>"""
    cfg_file.write_text(content)
    return cfg_file


def run_sumo_simulation(
    cfg_file: Path,
    mobility_out: Path,
    roles: Dict[str, str],
    duration: int
) -> tuple:
    """
    Executes SUMO simulation and extracts mobility CSV.
    Returns: (records_count, simulated_vids_set)
    """
    sumo_cmd = [
        "sumo",
        "-c", str(cfg_file),
        "--end", str(duration),
        "--step-length", "1",
        "--no-warnings", "true",
    ]

    records = 0
    simulated_vids: Set[str] = set()

    with open(mobility_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "vehicle_id", "role", "x", "y", "speed_mps", "road_id"])

        traci.start(sumo_cmd)
        try:
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0 and step <= duration:
                current_time = traci.simulation.getTime()

                for vid_str in traci.vehicle.getIDList():
                    simulated_vids.add(vid_str)
                    role = roles.get(vid_str, "provider")
                    x, y = traci.vehicle.getPosition(vid_str)
                    speed = traci.vehicle.getSpeed(vid_str)
                    road_id = traci.vehicle.getRoadID(vid_str)

                    writer.writerow([
                        round(current_time, 1),
                        vid_str,
                        role,
                        round(x, 2),
                        round(y, 2),
                        round(speed, 4),
                        road_id,
                    ])
                    records += 1

                traci.simulationStep()
                step += 1
        finally:
            traci.close()

    return records, simulated_vids


def execute_run(
    run_idx: int,
    density: Dict[str, Any],
    seed: int,
    out_base: Path,
    duration: int = DEFAULT_DURATION
) -> Dict[str, Any]:
    run_id = f"run_{run_idx:03d}"
    run_dir = out_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mobility_out = (run_dir / "mobility.csv").resolve()
    cfg_path = run_dir / "run_config.json"

    # Skip if already completed
    if mobility_out.exists() and cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
            if existing.get("status") == "done" and existing.get("records", 0) > 0:
                print(f"  [SKIP] {run_id} already completed ({existing.get('records')} records).")
                return existing
        except Exception:
            pass

    t0 = time.time()
    requested_vehicles = int(round(duration / density["period"]))
    print(f"\n  === {run_id} | {density['label']} (req≈{requested_vehicles}) | seed={seed} ===")

    try:
        trips_file = generate_trips(run_dir, density["period"], duration, seed)
        routes_file = route_trips(run_dir, trips_file, seed)
        routed_vids = parse_routed_vehicles(routes_file)
        roles = assign_roles(routed_vids, seed)

        pue_count = sum(1 for r in roles.values() if r == "provider")
        cue_count = sum(1 for r in roles.values() if r == "consumer")

        cfg_file = write_sumocfg(run_dir, routes_file, duration)
        records, simulated_vids = run_sumo_simulation(cfg_file, mobility_out, roles, duration)

        failed_vids = sorted(list(set(routed_vids) - simulated_vids))
        elapsed = round(time.time() - t0, 1)

        run_cfg = {
            "run_id": run_id,
            "run_idx": run_idx,
            "density_label": density["label"],
            "target_vehicles": density["target_vehicles"],
            "requested_vehicles": requested_vehicles,
            "routed_vehicles": len(routed_vids),
            "simulated_vehicles": len(simulated_vids),
            "pue_count": pue_count,
            "cue_count": cue_count,
            "failed_vehicles_count": len(failed_vids),
            "failed_vehicles": failed_vids,
            "period_s": density["period"],
            "duration_s": duration,
            "seed": seed,
            "records": records,
            "elapsed_s": elapsed,
            "status": "done",
            "mobility_csv": str(mobility_out),
        }
        cfg_path.write_text(json.dumps(run_cfg, indent=2))
        print(f"    ✓ Routed: {len(routed_vids)} | Simulated: {len(simulated_vids)} (PUE: {pue_count}, CUE: {cue_count}) | Records: {records:,} | {elapsed}s")
        return run_cfg

    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        err_cfg = {
            "run_id": run_id,
            "run_idx": run_idx,
            "density_label": density["label"],
            "seed": seed,
            "duration_s": duration,
            "status": "error",
            "error": str(e),
            "elapsed_s": elapsed,
        }
        cfg_path.write_text(json.dumps(err_cfg, indent=2))
        print(f"    ✗ FAILED: {e}")
        return err_cfg


def main():
    parser = argparse.ArgumentParser(description="Generate multi-run SUMO simulations.")
    parser.add_argument("--n_runs", type=int, default=16,
                        help="Number of runs to generate across density tiers (default: 16).")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Simulation duration per run in seconds (default: 1000).")
    parser.add_argument("--seed_base", type=int, default=101,
                        help="Base seed for reproducibility (default: 101).")
    parser.add_argument("--out_dir", type=str, default="data/raw/sumo",
                        help="Output directory.")
    args = parser.parse_args()

    out_base = PROJECT_ROOT / args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" MULTI-RUN SUMO MOBILITY GENERATION")
    print(f" Total runs: {args.n_runs} | Duration: {args.duration}s | Seed base: {args.seed_base}")
    print(f" Output dir: {out_base}")
    print("=" * 70)

    results = []
    n_profiles = len(DENSITY_PROFILES)

    for i in range(args.n_runs):
        density = DENSITY_PROFILES[i % n_profiles]
        seed = args.seed_base + i * 17  # unique seed per run
        res = execute_run(
            run_idx=i,
            density=density,
            seed=seed,
            out_base=out_base,
            duration=args.duration
        )
        results.append(res)

    # Save master manifest
    manifest_path = out_base / "manifest.json"
    done_runs = [r for r in results if r.get("status") == "done"]
    failed_runs = [r for r in results if r.get("status") == "error"]

    manifest = {
        "n_runs_total": len(results),
        "n_runs_done": len(done_runs),
        "n_runs_failed": len(failed_runs),
        "total_records": sum(r.get("records", 0) for r in done_runs),
        "runs": results
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 70)
    print(" GENERATION SUMMARY")
    print("=" * 70)
    print(f" Completed: {len(done_runs)}/{len(results)} runs")
    print(f" Failed:    {len(failed_runs)} runs")
    print(f" Total records extracted: {manifest['total_records']:,}")
    print(f" Manifest saved: {manifest_path}")

    if failed_runs:
        sys.exit(1)


if __name__ == "__main__":
    main()

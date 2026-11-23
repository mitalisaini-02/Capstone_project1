"""
generate_sumo_runs.py
---------------------
Generates N independent SUMO mobility runs with varied vehicle counts,
densities, simulation durations, and random seeds.

Each run:
  1. Generates random trips with randomTrips.py (seed-controlled)
  2. Routes trips with duarouter
  3. Runs SUMO headlessly and extracts mobility CSV via TraCI
  4. Saves: data/raw/sumo/run_NNN/mobility.csv + run_config.json

Usage:
  python scripts/generate_sumo_runs.py [--pilot] [--n_runs 40] [--out_dir data/raw/sumo]

Do NOT modify Stage 1 (actual_mec_comparison.py) or Stage 2 (dynamic_method_comparison.py).
Do NOT overwrite existing processed data, splits, or checkpoints.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Ensure SUMO tools importable
# ---------------------------------------------------------------------------
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))

try:
    import traci
except ImportError as e:
    print(f"ERROR: Cannot import traci — {e}")
    print(f"  SUMO_HOME={SUMO_HOME}; ensure SUMO is installed.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project root (capstone_sim/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NET_FILE = "/home/mitali/Downloads/osm.net.xml"  # fixed OSM network
INFRA_FILE = str(PROJECT_ROOT / "infrastructure.add.xml")
RANDOM_TRIPS_SCRIPT = os.path.join(SUMO_HOME, "tools", "randomTrips.py")

# ---------------------------------------------------------------------------
# Density configurations
# ---------------------------------------------------------------------------
DENSITY_PROFILES: List[Dict[str, Any]] = [
    {"label": "light",      "n_vehicles": 40,  "period": 25.0},
    {"label": "medium",     "n_vehicles": 70,  "period": 14.0},
    {"label": "heavy",      "n_vehicles": 100, "period": 10.0},
    {"label": "very_heavy", "n_vehicles": 130, "period": 7.5},
]

# Default simulation duration per run
DEFAULT_DURATION = 1000  # seconds


def _run_cmd(cmd: List[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, print stderr on failure."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if check and result.returncode != 0:
        print(f"  [ERROR] Command failed: {' '.join(cmd)}")
        print(f"  STDOUT: {result.stdout[-2000:]}")
        print(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


def generate_trips(
    run_dir: Path,
    n_vehicles: int,
    duration: int,
    seed: int,
    period: float
) -> Path:
    """
    Generate random trips XML using SUMO randomTrips.py.
    Returns path to trips file.
    """
    trips_file = run_dir / "trips.xml"
    cmd = [
        sys.executable,
        RANDOM_TRIPS_SCRIPT,
        "-n", NET_FILE,
        "-o", str(trips_file),
        "--begin", "0",
        "--end", str(duration),
        "--period", str(period),
        "--seed", str(seed),
        "--validate",          # discard unroutable trips
        "--fringe-factor", "1.0",
        "--allow-fringe",
        "--random-depart",
    ]
    print(f"    Generating trips (n≈{n_vehicles}, period={period}s, seed={seed})...")
    _run_cmd(cmd, cwd=str(run_dir))
    return trips_file


def route_trips(run_dir: Path, trips_file: Path, seed: int) -> Path:
    """
    Convert trips to routes using duarouter.
    Returns path to routes file.
    """
    routes_file = run_dir / "routes.rou.xml"
    cmd = [
        "duarouter",
        "-n", NET_FILE,
        "-r", str(trips_file),
        "-o", str(routes_file),
        "--seed", str(seed),
        "--ignore-errors", "true",
        "--no-warnings", "true",
    ]
    print(f"    Routing trips → routes...")
    _run_cmd(cmd, cwd=str(run_dir))
    return routes_file


def write_sumocfg(run_dir: Path, routes_file: Path, duration: int) -> Path:
    """Write a SUMO config file for this run."""
    cfg_file = run_dir / "run.sumocfg"
    cfg_content = f"""<configuration>
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
</configuration>
"""
    cfg_file.write_text(cfg_content)
    return cfg_file


def run_sumo_extract(
    cfg_file: Path,
    mobility_out: Path,
    n_vehicles: int,
    duration: int
) -> int:
    """
    Run SUMO via TraCI and extract mobility CSV.
    Provider/consumer split: first 60% of vehicles by sorted ID → providers,
    remaining 40% → consumers.

    Returns the total number of records written.
    """
    # Determine provider threshold dynamically based on vehicle IDs observed
    sumo_cmd = [
        "sumo",
        "-c", str(cfg_file),
        "--end", str(duration),
        "--step-length", "1",
        "--no-warnings", "true",
    ]

    records = 0
    vehicle_roles: Dict[str, str] = {}  # cached role per vehicle_id string

    with open(mobility_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "vehicle_id", "role", "x", "y", "speed_mps", "road_id"])

        traci.start(sumo_cmd)
        try:
            # First pass: collect all vehicle IDs to determine split threshold
            # We'll assign roles dynamically as we see new vehicles
            seen_vehicles: List[str] = []

            step = 0
            while traci.simulation.getMinExpectedNumber() > 0 and step <= duration:
                current_time = traci.simulation.getTime()

                for vid_str in traci.vehicle.getIDList():
                    # Assign role on first encounter
                    if vid_str not in vehicle_roles:
                        seen_vehicles.append(vid_str)
                        # Sort all seen so far; first 60% → provider
                        # Role may be re-assigned as new vehicles appear; we'll
                        # lock assignments once first seen to avoid inconsistency
                        # Simple rule: try to parse numeric ID; if < threshold → provider
                        try:
                            vid_int = int(vid_str)
                        except ValueError:
                            vid_int = abs(hash(vid_str)) % 10000

                        # Provider threshold: 60% of n_vehicles by count
                        threshold = int(n_vehicles * 0.6)
                        vehicle_roles[vid_str] = "provider" if vid_int < threshold else "consumer"

                    role = vehicle_roles[vid_str]
                    x, y = traci.vehicle.getPosition(vid_str)
                    speed = traci.vehicle.getSpeed(vid_str)
                    road_id = traci.vehicle.getRoadID(vid_str)

                    writer.writerow([
                        current_time,
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

    return records


def run_one(
    run_idx: int,
    density: Dict[str, Any],
    seed: int,
    out_base: Path,
    duration: int,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Execute one complete SUMO run: trips → routes → simulate → CSV.
    Returns a summary dict.
    """
    run_id = f"run_{run_idx:03d}"
    run_dir = out_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mobility_out = run_dir / "mobility.csv"
    cfg_path = run_dir / "run_config.json"

    # Skip if already completed successfully
    if mobility_out.exists() and cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
            if existing.get("status") == "done":
                print(f"  [SKIP] {run_id} already completed ({existing.get('records', '?')} records).")
                return existing
        except Exception:
            pass

    run_cfg: Dict[str, Any] = {
        "run_id": run_id,
        "run_idx": run_idx,
        "density_label": density["label"],
        "n_vehicles": density["n_vehicles"],
        "period": density["period"],
        "duration_s": duration,
        "seed": seed,
        "status": "pending",
    }
    cfg_path.write_text(json.dumps(run_cfg, indent=2))

    if dry_run:
        print(f"  [DRY-RUN] Would run {run_id}: {density['label']}, {density['n_vehicles']} vehicles, seed={seed}")
        run_cfg["status"] = "dry_run"
        return run_cfg

    t0 = time.time()
    try:
        print(f"\n  === {run_id} | density={density['label']} | n≈{density['n_vehicles']} | seed={seed} ===")

        trips_file = generate_trips(run_dir, density["n_vehicles"], duration, seed, density["period"])
        routes_file = route_trips(run_dir, trips_file, seed)
        cfg_file = write_sumocfg(run_dir, routes_file, duration)
        records = run_sumo_extract(cfg_file, mobility_out, density["n_vehicles"], duration)

        elapsed = round(time.time() - t0, 1)
        run_cfg.update({
            "status": "done",
            "records": records,
            "mobility_csv": str(mobility_out),
            "elapsed_s": elapsed,
        })
        print(f"    ✓ Done in {elapsed}s — {records:,} records → {mobility_out}")

    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        run_cfg.update({
            "status": "error",
            "error": str(e),
            "elapsed_s": elapsed,
        })
        print(f"    ✗ FAILED: {e}")

    cfg_path.write_text(json.dumps(run_cfg, indent=2))
    return run_cfg


def build_run_schedule(
    n_runs: int,
    seed_base: int = 100
) -> List[Dict[str, Any]]:
    """
    Build a list of run specifications cycling through density profiles.
    Each run gets a unique seed = seed_base + run_idx * 7 (prime step).
    """
    schedule = []
    n_profiles = len(DENSITY_PROFILES)
    for i in range(n_runs):
        density = DENSITY_PROFILES[i % n_profiles]
        seed = seed_base + i * 7  # unique, non-overlapping seeds
        schedule.append({
            "run_idx": i,
            "density": density,
            "seed": seed,
        })
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-run SUMO mobility dataset.")
    parser.add_argument("--pilot", action="store_true",
                        help="Run only 2 pilot simulations (light + medium density).")
    parser.add_argument("--n_runs", type=int, default=40,
                        help="Total number of runs (ignored if --pilot).")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Simulation duration per run in seconds (default: 1000).")
    parser.add_argument("--seed_base", type=int, default=100,
                        help="Base seed; each run uses seed_base + run_idx * 7.")
    parser.add_argument("--out_dir", type=str, default="data/raw/sumo",
                        help="Output base directory for all runs.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be run without executing SUMO.")
    args = parser.parse_args()

    out_base = PROJECT_ROOT / args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        # Two pilot runs: light (40 vehicles, seed=100) + medium (70 vehicles, seed=200)
        schedule = [
            {"run_idx": 0, "density": DENSITY_PROFILES[0], "seed": 100},  # light
            {"run_idx": 1, "density": DENSITY_PROFILES[1], "seed": 200},  # medium
        ]
        print("=== PILOT MODE: 2 runs ===")
    else:
        schedule = build_run_schedule(args.n_runs, seed_base=args.seed_base)
        print(f"=== FULL MODE: {args.n_runs} runs ===")

    print(f"Output dir: {out_base}")
    print(f"Duration per run: {args.duration}s\n")

    results = []
    for spec in schedule:
        result = run_one(
            run_idx=spec["run_idx"],
            density=spec["density"],
            seed=spec["seed"],
            out_base=out_base,
            duration=args.duration,
            dry_run=args.dry_run,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    done = [r for r in results if r.get("status") == "done"]
    errors = [r for r in results if r.get("status") == "error"]
    skipped = [r for r in results if r.get("status") not in ("done", "error", "dry_run")]

    total_records = sum(r.get("records", 0) for r in done)
    print(f"  Completed: {len(done)}/{len(results)}")
    if errors:
        print(f"  Failed:    {len(errors)} runs")
        for r in errors:
            print(f"    - {r['run_id']}: {r.get('error', '?')}")
    print(f"  Total mobility records: {total_records:,}")
    print(f"\n  Run configs saved in: {out_base}/run_NNN/run_config.json")
    print(f"  Mobility CSVs saved in: {out_base}/run_NNN/mobility.csv")

    # Save manifest
    manifest_path = out_base / "manifest.json"
    manifest = {
        "n_runs_attempted": len(results),
        "n_runs_done": len(done),
        "n_runs_failed": len(errors),
        "total_records": total_records,
        "duration_per_run_s": args.duration,
        "runs": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest: {manifest_path}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

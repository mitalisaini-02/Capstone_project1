"""
extract_pilot_mobility.py
--------------------------
Runs the pilot_001 SUMO simulation via TraCI and extracts the mobility CSV.

Pilot parameters:
  - 50 vehicles: IDs 0-29 → providers (PUEs), IDs 30-49 → consumers (CUEs)
  - Duration: 1000 seconds
  - Seed: 101 (already baked into the routes)

Saves: data/raw/sumo/pilot_001/mobility.csv
       data/raw/sumo/pilot_001/run_config.json
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))

import traci

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR    = PROJECT_ROOT / "data" / "raw" / "sumo" / "pilot_001"
CFG_FILE     = PILOT_DIR / "run.sumocfg"
OUTPUT_CSV   = PILOT_DIR / "mobility.csv"
CONFIG_OUT   = PILOT_DIR / "run_config.json"

DURATION     = 1000
N_VEHICLES   = 50
PROVIDER_THRESHOLD = 30   # IDs 0-29 → provider, 30-49 → consumer
RUN_ID       = "pilot_001"
SEED         = 101


def main():
    print(f"[pilot_001] Starting SUMO TraCI extraction...")
    print(f"  Config : {CFG_FILE}")
    print(f"  Output : {OUTPUT_CSV}")

    if not CFG_FILE.exists():
        print(f"ERROR: SUMO config not found: {CFG_FILE}")
        sys.exit(1)

    sumo_cmd = [
        "sumo",
        "-c", str(CFG_FILE),
        "--end", str(DURATION),
        "--step-length", "1",
        "--no-warnings", "true",
    ]

    records = 0
    t0 = time.time()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "vehicle_id", "role", "x", "y", "speed_mps", "road_id"])

        traci.start(sumo_cmd)
        try:
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0 and step <= DURATION:
                current_time = traci.simulation.getTime()

                for vid_str in traci.vehicle.getIDList():
                    try:
                        vid_int = int(vid_str)
                    except ValueError:
                        vid_int = abs(hash(vid_str)) % 10000

                    role = "provider" if vid_int < PROVIDER_THRESHOLD else "consumer"
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

                if step % 100 == 0:
                    print(f"  t={step}s — {records:,} records so far")

        finally:
            traci.close()

    elapsed = round(time.time() - t0, 1)
    print(f"\n  Done in {elapsed}s — {records:,} records written to {OUTPUT_CSV}")

    # Save run config
    cfg = {
        "run_id": RUN_ID,
        "run_idx": 0,
        "density_label": "light_50v",
        "n_vehicles": N_VEHICLES,
        "provider_threshold": PROVIDER_THRESHOLD,
        "duration_s": DURATION,
        "seed": SEED,
        "status": "done",
        "records": records,
        "elapsed_s": elapsed,
        "mobility_csv": str(OUTPUT_CSV),
    }
    CONFIG_OUT.write_text(json.dumps(cfg, indent=2))
    print(f"  Config : {CONFIG_OUT}")


if __name__ == "__main__":
    main()

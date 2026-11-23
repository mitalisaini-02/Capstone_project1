
import os
import sys
import csv
import traci

# Use SUMO's installed Python tools
sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.append(os.path.join(sumo_home, "tools"))

CONFIG = "simulation.sumocfg"
OUTPUT = "mobility.csv"
END_TIME = 300
STEP_LENGTH = 1

with open(OUTPUT, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "time_s",
        "vehicle_id",
        "role",
        "x",
        "y",
        "speed_mps",
        "road_id"
    ])

    sumo_cmd = [
        "sumo",
        "-c", CONFIG,
        "--end", str(END_TIME),
        "--step-length", str(STEP_LENGTH)
    ]

    traci.start(sumo_cmd)

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            current_time = traci.simulation.getTime()

            for vehicle_id in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                road_id = traci.vehicle.getRoadID(vehicle_id)

                # Vehicles 0-29 are providers; 30-49 are consumers
                role = (
                    "provider"
                    if int(vehicle_id) < 30
                    else "consumer"
                )

                writer.writerow([
                    current_time,
                    vehicle_id,
                    role,
                    round(x, 2),
                    round(y, 2),
                    round(speed, 2),
                    road_id
                ])

            traci.simulationStep()

    finally:
        traci.close()

print(f"Mobility dataset saved to {OUTPUT}")

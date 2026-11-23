
import os
import sys
import csv
import random
import math
import traci

# SUMO configuration
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))

CONFIG_FILE = "simulation.sumocfg"
OUTPUT_FILE = "dynamic_marketplace.csv"

# Reproducible simulated parameters (not real network measurements)
SEED = 42
SIMULATION_END = 300
PROVIDER_IDS = {str(i) for i in range(30)}
CONSUMER_IDS = {str(i) for i in range(30, 50)}

PROVIDER_MIN_MBPS = 5.0
PROVIDER_MAX_MBPS = 20.0
PRICE_MIN = 0.01
PRICE_MAX = 0.10
CONSUMER_MIN_MBPS = 3.0
CONSUMER_MAX_MBPS = 10.0

MAX_PROVIDER_DISTANCE_M = 1000.0
GNB_TOTAL_CAPACITY_MBPS = 100.0

random.seed(SEED)

# Fixed profile for each vehicle across time
provider_profiles = {
    pid: {
        "capacity": random.uniform(PROVIDER_MIN_MBPS, PROVIDER_MAX_MBPS),
        "price": random.uniform(PRICE_MIN, PRICE_MAX),
    }
    for pid in sorted(PROVIDER_IDS, key=int)
}

consumer_profiles = {
    cid: {
        "demand": random.uniform(CONSUMER_MIN_MBPS, CONSUMER_MAX_MBPS)
    }
    for cid in sorted(CONSUMER_IDS, key=int)
}

# Output: one row per active consumer per simulation second
fieldnames = [
    "time_s", "consumer_id", "x", "y", "requested_mbps",
    "selected_provider_ids", "provider_rate_mbps",
    "gnb_rate_mbps", "total_rate_mbps", "payment",
    "unmet_demand_mbps"
]

sumo_cmd = [
    "sumo", "-c", CONFIG_FILE,
    "--end", str(SIMULATION_END),
    "--step-length", "1"
]

total_payment = 0.0
total_provider_rate = 0.0
total_gnb_rate = 0.0
total_unmet = 0.0
consumer_seconds = 0

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    traci.start(sumo_cmd)

    try:
        while (
    traci.simulation.getTime() < SIMULATION_END
    and traci.simulation.getMinExpectedNumber() > 0
):
            time_s = traci.simulation.getTime()
            active_ids = set(traci.vehicle.getIDList())

            # Reset per-second provider allocations and gNodeB capacity.
            # This represents capacity available during this time interval.
            provider_used = {pid: 0.0 for pid in PROVIDER_IDS}
            gnb_remaining = GNB_TOTAL_CAPACITY_MBPS

            # Get current positions of active providers
            provider_positions = {}
            for pid in PROVIDER_IDS & active_ids:
                provider_positions[pid] = traci.vehicle.getPosition(pid)

            for cid in sorted(CONSUMER_IDS & active_ids, key=int):
                cx, cy = traci.vehicle.getPosition(cid)
                demand = consumer_profiles[cid]["demand"]
                remaining = demand

                candidates = []
                for pid, (px, py) in provider_positions.items():
                    distance = math.hypot(cx - px, cy - py)

                    if distance > MAX_PROVIDER_DISTANCE_M:
                        continue

                    profile = provider_profiles[pid]
                    available = profile["capacity"] - provider_used[pid]

                    if available > 0:
                        candidates.append((
                            profile["price"],
                            distance,
                            int(pid),
                            pid,
                            available
                        ))

                # Lowest price first; distance breaks price ties
                candidates.sort()

                selected = []
                provider_rate = 0.0
                payment = 0.0

                for price, distance, _, pid, available in candidates:
                    if remaining <= 0:
                        break

                    allocation = min(remaining, available)
                    provider_used[pid] += allocation
                    remaining -= allocation
                    provider_rate += allocation
                    payment += allocation * price
                    selected.append(pid)

                # Aggregate cellular fallback model
                gnb_rate = min(remaining, gnb_remaining)
                gnb_remaining -= gnb_rate
                remaining -= gnb_rate

                total_rate = provider_rate + gnb_rate
                unmet = max(0.0, demand - total_rate)

                writer.writerow({
                    "time_s": time_s,
                    "consumer_id": cid,
                    "x": round(cx, 2),
                    "y": round(cy, 2),
                    "requested_mbps": round(demand, 3),
                    "selected_provider_ids": ";".join(selected),
                    "provider_rate_mbps": round(provider_rate, 3),
                    "gnb_rate_mbps": round(gnb_rate, 3),
                    "total_rate_mbps": round(total_rate, 3),
                    "payment": round(payment, 4),
                    "unmet_demand_mbps": round(unmet, 3)
                })

                total_payment += payment
                total_provider_rate += provider_rate
                total_gnb_rate += gnb_rate
                total_unmet += unmet
                consumer_seconds += 1

            traci.simulationStep()

    finally:
        traci.close()

print("Dynamic marketplace simulation completed.")
print("Output:", OUTPUT_FILE)
print("Consumer-time records:", consumer_seconds)
print("Total provider payment:", round(total_payment, 4))
print("Sum of provider rates across consumer-time records:",
      round(total_provider_rate, 3), "Mbps")
print("Sum of gNodeB rates across consumer-time records:",
      round(total_gnb_rate, 3), "Mbps")
print("Sum of unmet demand across consumer-time records:",
      round(total_unmet, 3), "Mbps")


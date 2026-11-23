
import csv
import random
import math
from collections import defaultdict

# Reproducible synthetic marketplace parameters
random.seed(42)

MOBILITY_FILE = "mobility.csv"
OUTPUT_FILE = "marketplace.csv"

# Initial simulation assumptions (not real 5G measurements)
PROVIDER_COUNT = 30
CONSUMER_DEMAND_MIN = 3.0
CONSUMER_DEMAND_MAX = 10.0
GNB_CAPACITY_MBPS = 100.0

# Provider offers and consumer demands
providers = {}
consumers = {}

for i in range(PROVIDER_COUNT):
    vid = str(i)
    providers[vid] = {
        "capacity_mbps": random.uniform(5, 20),
        "price_per_mbit": random.uniform(0.01, 0.10),
        "used_mbps": 0.0
    }

for i in range(30, 50):
    consumers[str(i)] = {
        "demand_mbps": random.uniform(
            CONSUMER_DEMAND_MIN,
            CONSUMER_DEMAND_MAX
        )
    }

# Read the latest position of each vehicle from SUMO mobility data
latest = {}

with open(MOBILITY_FILE, newline="") as f:
    for row in csv.DictReader(f):
        latest[row["vehicle_id"]] = row

# Find the most recent location of each vehicle
positions = {}

for vid, row in latest.items():
    positions[vid] = (
        float(row["x"]),
        float(row["y"])
    )

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Match consumers to providers using distance and price
results = []
total_payment = 0.0
total_rate = 0.0
total_unmet = 0.0
gnb_remaining = GNB_CAPACITY_MBPS

for consumer_id, info in consumers.items():
    demand = info["demand_mbps"]
    cpos = positions.get(consumer_id)

    candidates = []

    if cpos is not None:
        for provider_id, p in providers.items():
            ppos = positions.get(provider_id)

            if ppos is None:
                continue

            available = p["capacity_mbps"] - p["used_mbps"]

            if available <= 0:
                continue

            dist = distance(cpos, ppos)

            # Simplified proximity-based eligibility assumption
            if dist <= 1000:
                candidates.append((
                    p["price_per_mbit"],
                    dist,
                    provider_id,
                    available
                ))

    # Prefer the cheapest eligible provider
    candidates.sort()

    remaining_demand = demand
    provider_rate = 0.0
    provider_payment = 0.0
    selected_providers = []

    for price, dist, provider_id, available in candidates:
        if remaining_demand <= 0:
            break

        allocation = min(remaining_demand, available)
        if allocation <= 0:
            continue

        providers[provider_id]["used_mbps"] += allocation
        remaining_demand -= allocation
        provider_rate += allocation
        provider_payment += allocation * price
        selected_providers.append(provider_id)

    # Cellular fallback supplies remaining demand if capacity permits
    gnb_rate = min(remaining_demand, gnb_remaining)
    gnb_remaining -= gnb_rate
    remaining_demand -= gnb_rate

    total_rate_for_consumer = provider_rate + gnb_rate
    unmet = max(0.0, demand - total_rate_for_consumer)

    total_payment += provider_payment
    total_rate += total_rate_for_consumer
    total_unmet += unmet

    results.append({
        "consumer_id": consumer_id,
        "requested_mbps": round(demand, 3),
        "selected_provider_ids": ";".join(selected_providers),
        "provider_rate_mbps": round(provider_rate, 3),
        "gnb_rate_mbps": round(gnb_rate, 3),
        "total_rate_mbps": round(total_rate_for_consumer, 3),
        "payment": round(provider_payment, 4),
        "unmet_demand_mbps": round(unmet, 3)
    })

# Save consumer-level results
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Marketplace simulation completed.")
print("Consumers:", len(results))
print("Total consumer payment:", round(total_payment, 4))
print("Total achieved data rate:", round(total_rate, 3), "Mbps")
print("Total unmet demand:", round(total_unmet, 3), "Mbps")
print("Results saved to", OUTPUT_FILE)


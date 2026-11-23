
import csv
import random
import math

random.seed(42)

MOBILITY_FILE = "mobility.csv"
OUTPUT_FILE = "comparison_results.csv"

PROVIDER_COUNT = 30
CONSUMER_COUNT = 20
MAX_DISTANCE_M = 1000.0
GNB_CAPACITY_MBPS = 100.0

# Synthetic parameters: not measured network data
providers = {
    str(i): {
        "capacity": random.uniform(5, 20),
        "price": random.uniform(0.01, 0.10),
    }
    for i in range(PROVIDER_COUNT)
}

consumers = {
    str(i): random.uniform(3, 10)
    for i in range(30, 30 + CONSUMER_COUNT)
}

# Use each vehicle's last recorded position for this initial static baseline
latest = {}
with open(MOBILITY_FILE, newline="") as f:
    for row in csv.DictReader(f):
        latest[row["vehicle_id"]] = row

positions = {
    vid: (float(row["x"]), float(row["y"]))
    for vid, row in latest.items()
}

def run_method(method):
    provider_used = {pid: 0.0 for pid in providers}
    gnb_remaining = GNB_CAPACITY_MBPS

    total_payment = 0.0
    total_rate = 0.0
    total_unmet = 0.0
    selected_count = 0

    # Reset resources for every method
    provider_used = {pid: 0.0 for pid in providers}
    gnb_remaining = GNB_CAPACITY_MBPS

    for cid, demand in consumers.items():
        remaining = demand
        provider_rate = 0.0
        payment = 0.0
        candidates = []

        if method != "gNB-only" and cid in positions:
            cx, cy = positions[cid]

            for pid in providers:
                if pid not in positions:
                    continue

                px, py = positions[pid]
                distance = math.hypot(cx - px, cy - py)
                available = providers[pid]["capacity"] - provider_used[pid]

                if distance <= MAX_DISTANCE_M and available > 0:
                    candidates.append((
                        providers[pid]["price"],
                        distance,
                        pid,
                        available
                    ))

        candidates.sort()

        for price, distance, pid, available in candidates:
            if remaining <= 0:
                break

            allocation = min(remaining, available)
            provider_used[pid] += allocation
            remaining -= allocation
            provider_rate += allocation
            payment += allocation * price

        if provider_rate > 0:
            selected_count += 1

        # gNodeB-only is a modelled reference; same aggregate capacity
        gnb_rate = min(remaining, gnb_remaining)
        gnb_remaining -= gnb_rate
        remaining -= gnb_rate

        achieved = provider_rate + gnb_rate
        unmet = max(0.0, demand - achieved)

        total_payment += payment
        total_rate += achieved
        total_unmet += unmet

    return {
        "method": method,
        "consumers": len(consumers),
        "consumers_with_provider": selected_count,
        "total_payment_model_units": round(total_payment, 4),
        "total_achieved_rate_mbps": round(total_rate, 3),
        "total_unmet_demand_mbps": round(total_unmet, 3),
    }

results = [
    run_method("gNB-only"),
    run_method("cheapest-provider-first"),
    run_method("MEC-objective-placeholder"),
]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Comparison saved to", OUTPUT_FILE)
for result in results:
    print(result)


import csv

INPUT = "dynamic_marketplace.csv"
OUTPUT = "experiment_summary.csv"

with open(INPUT, newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("No records found in dynamic_marketplace.csv")

total_records = len(rows)
provider_records = sum(
    bool(r["selected_provider_ids"].strip()) for r in rows
)
gnb_only_records = total_records - provider_records

total_payment = sum(float(r["payment"]) for r in rows)
provider_rate_sum = sum(float(r["provider_rate_mbps"]) for r in rows)
gnb_rate_sum = sum(float(r["gnb_rate_mbps"]) for r in rows)
unmet_sum = sum(float(r["unmet_demand_mbps"]) for r in rows)

summary = {
    "simulation_records": total_records,
    "provider_selected_records": provider_records,
    "provider_selection_percent": round(
        100 * provider_records / total_records, 2
    ),
    "gnb_only_records": gnb_only_records,
    "gnb_only_percent": round(
        100 * gnb_only_records / total_records, 2
    ),
    "total_payment_model_units": round(total_payment, 4),
    "provider_rate_sum_mbps_records": round(provider_rate_sum, 3),
    "gnb_rate_sum_mbps_records": round(gnb_rate_sum, 3),
    "unmet_demand_sum_mbps_records": round(unmet_sum, 3),
}

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerows(summary.items())

print("Summary saved to", OUTPUT)
for key, value in summary.items():
    print(f"{key}: {value}")


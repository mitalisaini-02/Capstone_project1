#!/usr/bin/env python3
"""
actual_mec_comparison.py
MEC-Assisted Internet Sharing Marketplace
Part 1: Three Allocation Algorithms Comparison

Algorithms implemented:
1. gNodeB-only (Cellular fallback baseline with standard pricing)
2. Cheapest-first greedy (Proximity-constrained greedy baseline)
3. MEC Two-Stage LP using SciPy HiGHS:
   - Stage 1: Minimize total consumer payment subject to QoS / capacity constraints.
   - Stage 2: Maximize achieved data rate within a configured payment tolerance (default: 5%).

All parameters (tolerance, gNodeB price, minimum service fraction, communication range,
capacities, and random seed) are fully configurable via constants and CLI arguments.
"""

import os
import sys
import csv
import math
import random
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.optimize as opt

# =============================================================================
# DEFAULT CONFIGURATION (Configurable via CLI arguments)
# =============================================================================
DEFAULT_SEED = 42
DEFAULT_MOBILITY_FILE = "mobility.csv"
DEFAULT_OUTPUT_SUMMARY = "comparison_results.csv"
DEFAULT_OUTPUT_DETAILED = "comparison_detailed_allocations.csv"

DEFAULT_PROVIDER_COUNT = 30
DEFAULT_CONSUMER_COUNT = 20

DEFAULT_MAX_DISTANCE_M = 1000.0       # Effective communication range (R_com)
DEFAULT_GNB_CAPACITY_MBPS = 100.0     # gNodeB aggregate capacity
DEFAULT_GNB_PRICE = 0.05              # Price per Mbps delivered by gNodeB
DEFAULT_PAYMENT_TOLERANCE = 0.05      # delta = 5% allowed increase in Stage 2
DEFAULT_MIN_SERVICE_FRACTION = 1.00   # Primary baseline QoS constraint (100% demand)

# Provider / Consumer random generation bounds (synthetic simulation inputs)
PROVIDER_CAP_MIN = 5.0
PROVIDER_CAP_MAX = 20.0
PROVIDER_PRICE_MIN = 0.01
PROVIDER_PRICE_MAX = 0.10

CONSUMER_DEMAND_MIN = 3.0
CONSUMER_DEMAND_MAX = 10.0


# =============================================================================
# DATA LOADING & ENVIRONMENT SETUP
# =============================================================================
def generate_profiles(
    provider_count: int = DEFAULT_PROVIDER_COUNT,
    consumer_count: int = DEFAULT_CONSUMER_COUNT,
    seed: int = DEFAULT_SEED
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Generates synthetic provider profiles and consumer demands with a fixed seed.
    Provider IDs: 0 to provider_count - 1
    Consumer IDs: provider_count to provider_count + consumer_count - 1
    """
    rng = random.Random(seed)

    providers = {
        str(i): {
            "capacity": rng.uniform(PROVIDER_CAP_MIN, PROVIDER_CAP_MAX),
            "price": rng.uniform(PROVIDER_PRICE_MIN, PROVIDER_PRICE_MAX)
        }
        for i in range(provider_count)
    }

    consumers = {
        str(i): rng.uniform(CONSUMER_DEMAND_MIN, CONSUMER_DEMAND_MAX)
        for i in range(provider_count, provider_count + consumer_count)
    }

    return providers, consumers


def load_latest_positions(mobility_file: str) -> Dict[str, Tuple[float, float]]:
    """
    Reads vehicle coordinates from mobility.csv, keeping the latest recorded
    position for each vehicle ID.
    """
    if not os.path.exists(mobility_file):
        raise FileNotFoundError(f"Mobility file '{mobility_file}' not found.")

    latest: Dict[str, Dict[str, str]] = {}
    with open(mobility_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            latest[row["vehicle_id"]] = row

    positions = {
        vid: (float(row["x"]), float(row["y"]))
        for vid, row in latest.items()
    }
    return positions


def compute_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# =============================================================================
# ALGORITHM 1: gNodeB-ONLY BASELINE
# =============================================================================
def run_gnb_only(
    consumers: Dict[str, float],
    gnb_capacity: float = DEFAULT_GNB_CAPACITY_MBPS,
    gnb_price: float = DEFAULT_GNB_PRICE
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Routes all consumer demand strictly to the cellular gNodeB.
    Charges gnb_price for all delivered bandwidth.
    """
    gnb_remaining = gnb_capacity
    total_payment = 0.0
    total_rate = 0.0
    total_unmet = 0.0
    consumers_served = 0

    detailed_records = []

    for cid in sorted(consumers.keys(), key=int):
        demand = consumers[cid]
        gnb_allocated = min(demand, gnb_remaining)
        gnb_remaining -= gnb_allocated

        unmet = max(0.0, demand - gnb_allocated)
        payment = gnb_allocated * gnb_price

        total_payment += payment
        total_rate += gnb_allocated
        total_unmet += unmet

        if gnb_allocated > 1e-6:
            consumers_served += 1

        detailed_records.append({
            "method": "gNB-only",
            "consumer_id": cid,
            "requested_mbps": round(demand, 4),
            "provider_rate_mbps": 0.0,
            "gnb_rate_mbps": round(gnb_allocated, 4),
            "total_rate_mbps": round(gnb_allocated, 4),
            "payment_model_units": round(payment, 5),
            "unmet_demand_mbps": round(unmet, 4),
            "selected_providers": ""
        })

    summary = {
        "method": "gNB-only",
        "consumers": len(consumers),
        "consumers_with_provider": 0,
        "consumers_fully_served": sum(1 for r in detailed_records if r["unmet_demand_mbps"] < 1e-4),
        "total_payment_model_units": round(total_payment, 4),
        "total_achieved_rate_mbps": round(total_rate, 3),
        "total_unmet_demand_mbps": round(total_unmet, 3),
        "provider_throughput_mbps": 0.0,
        "gnb_throughput_mbps": round(total_rate, 3),
        "gnb_remaining_capacity_mbps": round(gnb_remaining, 3),
        "cost_per_mbps_delivered": round(total_payment / total_rate, 5) if total_rate > 0 else 0.0,
        "solver_status": "N/A (Deterministic)"
    }

    return summary, detailed_records


# =============================================================================
# ALGORITHM 2: CHEAPEST-FIRST GREEDY BASELINE
# =============================================================================
def run_greedy(
    providers: Dict[str, Dict[str, float]],
    consumers: Dict[str, float],
    positions: Dict[str, Tuple[float, float]],
    max_distance: float = DEFAULT_MAX_DISTANCE_M,
    gnb_capacity: float = DEFAULT_GNB_CAPACITY_MBPS,
    gnb_price: float = DEFAULT_GNB_PRICE
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Greedy heuristic:
    For each consumer, finds eligible providers (dist <= max_distance with remaining capacity).
    Sorts eligible providers by lowest price first (ties broken by distance).
    Allocates available bandwidth up to consumer demand.
    Any residual demand falls back to gNodeB.
    """
    provider_used = {pid: 0.0 for pid in providers}
    gnb_remaining = gnb_capacity

    total_payment = 0.0
    total_rate = 0.0
    total_unmet = 0.0
    total_provider_rate = 0.0
    total_gnb_rate = 0.0
    consumers_with_provider = 0

    detailed_records = []

    for cid in sorted(consumers.keys(), key=int):
        demand = consumers[cid]
        cpos = positions.get(cid)
        remaining = demand
        provider_rate = 0.0
        payment = 0.0
        selected_pids = []

        if cpos is not None:
            candidates = []
            for pid, pinfo in providers.items():
                ppos = positions.get(pid)
                if ppos is None:
                    continue
                dist = compute_euclidean_distance(cpos, ppos)
                avail = pinfo["capacity"] - provider_used[pid]

                if dist <= max_distance and avail > 1e-6:
                    candidates.append((pinfo["price"], dist, pid, avail))

            # Lowest price first, nearest breaks price ties
            candidates.sort(key=lambda item: (item[0], item[1]))

            for price, dist, pid, avail in candidates:
                if remaining <= 1e-6:
                    break
                alloc = min(remaining, avail)
                provider_used[pid] += alloc
                remaining -= alloc
                provider_rate += alloc
                payment += alloc * price
                selected_pids.append(f"{pid}({alloc:.2f}M@{price:.3f})")

        if provider_rate > 1e-6:
            consumers_with_provider += 1

        # Fallback to gNodeB
        gnb_alloc = min(remaining, gnb_remaining)
        gnb_remaining -= gnb_alloc
        remaining -= gnb_alloc
        payment += gnb_alloc * gnb_price

        achieved = provider_rate + gnb_alloc
        unmet = max(0.0, demand - achieved)

        total_payment += payment
        total_rate += achieved
        total_provider_rate += provider_rate
        total_gnb_rate += gnb_alloc
        total_unmet += unmet

        detailed_records.append({
            "method": "cheapest-first-greedy",
            "consumer_id": cid,
            "requested_mbps": round(demand, 4),
            "provider_rate_mbps": round(provider_rate, 4),
            "gnb_rate_mbps": round(gnb_alloc, 4),
            "total_rate_mbps": round(achieved, 4),
            "payment_model_units": round(payment, 5),
            "unmet_demand_mbps": round(unmet, 4),
            "selected_providers": "; ".join(selected_pids)
        })

    summary = {
        "method": "cheapest-first-greedy",
        "consumers": len(consumers),
        "consumers_with_provider": consumers_with_provider,
        "consumers_fully_served": sum(1 for r in detailed_records if r["unmet_demand_mbps"] < 1e-4),
        "total_payment_model_units": round(total_payment, 4),
        "total_achieved_rate_mbps": round(total_rate, 3),
        "total_unmet_demand_mbps": round(total_unmet, 3),
        "provider_throughput_mbps": round(total_provider_rate, 3),
        "gnb_throughput_mbps": round(total_gnb_rate, 3),
        "gnb_remaining_capacity_mbps": round(gnb_remaining, 3),
        "cost_per_mbps_delivered": round(total_payment / total_rate, 5) if total_rate > 0 else 0.0,
        "solver_status": "N/A (Heuristic)"
    }

    return summary, detailed_records


# =============================================================================
# ALGORITHM 3: MEC TWO-STAGE LP USING SCIPY HIGHS
# =============================================================================
def run_mec_two_stage_lp(
    providers: Dict[str, Dict[str, float]],
    consumers: Dict[str, float],
    positions: Dict[str, Tuple[float, float]],
    max_distance: float = DEFAULT_MAX_DISTANCE_M,
    gnb_capacity: float = DEFAULT_GNB_CAPACITY_MBPS,
    gnb_price: float = DEFAULT_GNB_PRICE,
    payment_tolerance: float = DEFAULT_PAYMENT_TOLERANCE,
    min_service_fraction: float = DEFAULT_MIN_SERVICE_FRACTION
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Two-Stage Lexicographic Optimization via SciPy HiGHS:
    Variables:
      r_{i,j}: rate delivered by provider i to consumer j (only for dist(i, j) <= max_distance)
      r_j^g  : rate delivered by gNodeB to consumer j

    Constraints:
      1. Consumer upper bound: sum_i r_{i,j} + r_j^g <= D_j  (for each j)
      2. Provider capacity   : sum_j r_{i,j} <= R_i          (for each i)
      3. gNodeB capacity     : sum_j r_j^g <= R_g_total
      4. QoS / Min Service   : sum_i r_{i,j} + r_j^g >= alpha * D_j (for each j)

    Stage 1:
      Minimize total payment: C = sum_{(i,j)} P_i * r_{i,j} + sum_j P_g * r_j^g
      Yields minimum cost C*.

    Stage 2:
      Maximize total achieved data rate: sum_j (sum_i r_{i,j} + r_j^g)
      Subject to all above constraints PLUS the payment tolerance:
      Total payment <= (1 + delta) * C*
    """
    c_ids = sorted(consumers.keys(), key=int)
    p_ids = sorted(providers.keys(), key=int)

    # 1. Identify eligible provider-consumer communication pairs within range
    eligible_pairs: List[Tuple[str, str]] = []
    for cid in c_ids:
        cpos = positions.get(cid)
        if cpos is None:
            continue
        for pid in p_ids:
            ppos = positions.get(pid)
            if ppos is None:
                continue
            dist = compute_euclidean_distance(cpos, ppos)
            if dist <= max_distance:
                eligible_pairs.append((pid, cid))

    num_pairs = len(eligible_pairs)
    num_c = len(c_ids)
    num_vars = num_pairs + num_c

    # Index mapping:
    # x[0 : num_pairs] -> r_{pid, cid}
    # x[num_pairs + j] -> r_j^g for consumer c_ids[j]
    c_index = {cid: idx for idx, cid in enumerate(c_ids)}
    p_index = {pid: idx for idx, pid in enumerate(p_ids)}

    # Cost vector: Provider prices and gNodeB price
    cost_vector = np.zeros(num_vars)
    for k, (pid, cid) in enumerate(eligible_pairs):
        cost_vector[k] = providers[pid]["price"]
    for j, cid in enumerate(c_ids):
        cost_vector[num_pairs + j] = gnb_price

    # Build Inequality Constraints: A_ub @ x <= b_ub
    A_rows: List[np.ndarray] = []
    b_vals: List[float] = []

    # Constraint 1: Consumer demand upper bound: sum_i r_{i,j} + r_j^g <= D_j
    for j, cid in enumerate(c_ids):
        row = np.zeros(num_vars)
        for k, (p, c) in enumerate(eligible_pairs):
            if c == cid:
                row[k] = 1.0
        row[num_pairs + j] = 1.0
        A_rows.append(row)
        b_vals.append(consumers[cid])

    # Constraint 2: Provider capacity bound: sum_j r_{i,j} <= R_i
    for i, pid in enumerate(p_ids):
        row = np.zeros(num_vars)
        for k, (p, c) in enumerate(eligible_pairs):
            if p == pid:
                row[k] = 1.0
        A_rows.append(row)
        b_vals.append(providers[pid]["capacity"])

    # Constraint 3: gNodeB aggregate capacity bound: sum_j r_j^g <= gnb_capacity
    row_gnb = np.zeros(num_vars)
    for j in range(num_c):
        row_gnb[num_pairs + j] = 1.0
    A_rows.append(row_gnb)
    b_vals.append(gnb_capacity)

    # Constraint 4: QoS minimum service fraction: sum_i r_{i,j} + r_j^g >= alpha * D_j
    # Expressed as: - (sum_i r_{i,j} + r_j^g) <= - alpha * D_j
    for j, cid in enumerate(c_ids):
        row = np.zeros(num_vars)
        for k, (p, c) in enumerate(eligible_pairs):
            if c == cid:
                row[k] = -1.0
        row[num_pairs + j] = -1.0
        A_rows.append(row)
        b_vals.append(-min_service_fraction * consumers[cid])

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0.0, None) for _ in range(num_vars)]

    # ---------------------------------------------------------
    # STAGE 1: MINIMIZE TOTAL CONSUMER PAYMENT
    # ---------------------------------------------------------
    stage1_res = opt.linprog(
        c=cost_vector,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    if not stage1_res.success:
        print(f"[ERROR] Stage 1 LP strictly infeasible with min_service_fraction={min_service_fraction}: {stage1_res.message}")
        solver_status = f"Infeasible (Stage 1 failed: {stage1_res.message})"
        detailed_records = [
            {
                "method": "MEC-lexicographic-LP",
                "consumer_id": cid,
                "requested_mbps": round(consumers[cid], 4),
                "provider_rate_mbps": 0.0,
                "gnb_rate_mbps": 0.0,
                "total_rate_mbps": 0.0,
                "payment_model_units": 0.0,
                "unmet_demand_mbps": round(consumers[cid], 4),
                "selected_providers": ""
            }
            for cid in c_ids
        ]
        total_demand = sum(consumers.values())
        summary = {
            "method": "MEC-lexicographic-LP",
            "consumers": len(consumers),
            "consumers_with_provider": 0,
            "consumers_fully_served": 0,
            "total_payment_model_units": 0.0,
            "total_achieved_rate_mbps": 0.0,
            "total_unmet_demand_mbps": round(total_demand, 3),
            "provider_throughput_mbps": 0.0,
            "gnb_throughput_mbps": 0.0,
            "gnb_remaining_capacity_mbps": round(gnb_capacity, 3),
            "cost_per_mbps_delivered": 0.0,
            "stage1_min_payment": 0.0,
            "payment_budget_cap": 0.0,
            "solver_status": solver_status
        }
        return summary, detailed_records

    C_star = float(stage1_res.fun)
    allowed_payment_budget = (1.0 + payment_tolerance) * C_star

    # ---------------------------------------------------------
    # STAGE 2: MAXIMIZE ACHIEVED DATA RATE WITHIN PAYMENT TOLERANCE
    # ---------------------------------------------------------
    # Objective: maximize sum of all allocated rates = minimize - sum of x
    rate_objective = np.full(num_vars, -1.0)

    # Add payment tolerance budget constraint: cost_vector @ x <= allowed_payment_budget
    A_ub_stage2 = np.vstack([A_ub, cost_vector])
    b_ub_stage2 = np.append(b_ub, allowed_payment_budget)

    stage2_res = opt.linprog(
        c=rate_objective,
        A_ub=A_ub_stage2,
        b_ub=b_ub_stage2,
        bounds=bounds,
        method="highs"
    )

    if not stage2_res.success:
        print(f"[Warning] Stage 2 LP failed: {stage2_res.message}. Falling back to Stage 1 solution.")
        optimal_x = stage1_res.x
        solver_status = f"Stage 1 Optimal, Stage 2 Failed ({stage2_res.message})"
    else:
        R_star = float(-stage2_res.fun)

        # ---------------------------------------------------------
        # STAGE 3: MINIMIZE TOTAL PAYMENT PRESERVING STAGE 2 THROUGHPUT
        # ---------------------------------------------------------
        # Constraint: sum(x) >= R_star - 1e-6 => -sum(x) <= -(R_star - 1e-6)
        row_rate = np.full(num_vars, -1.0)
        A_ub_stage3 = np.vstack([A_ub_stage2, row_rate])
        b_ub_stage3 = np.append(b_ub_stage2, -(R_star - 1e-6))

        stage3_res = opt.linprog(
            c=cost_vector,
            A_ub=A_ub_stage3,
            b_ub=b_ub_stage3,
            bounds=bounds,
            method="highs"
        )

        if not stage3_res.success:
            print(f"[Warning] Stage 3 LP failed: {stage3_res.message}. Using Stage 2 solution.")
            optimal_x = stage2_res.x
            solver_status = f"HiGHS Stage 1 & 2 Optimal, Stage 3 Failed ({stage3_res.message})"
        else:
            optimal_x = stage3_res.x
            solver_status = "HiGHS Three-Stage Lexicographic Optimal"

    # Extract allocations and build detailed results
    pair_allocations = optimal_x[:num_pairs]
    gnb_allocations = optimal_x[num_pairs:]

    total_payment = float(np.dot(cost_vector, optimal_x))
    total_rate = float(np.sum(optimal_x))
    total_provider_rate = float(np.sum(pair_allocations))
    total_gnb_rate = float(np.sum(gnb_allocations))
    total_demand = sum(consumers.values())
    total_unmet = max(0.0, total_demand - total_rate)

    detailed_records = []
    consumers_with_provider = 0

    for j, cid in enumerate(c_ids):
        demand = consumers[cid]
        gnb_rate = float(gnb_allocations[j])

        # Find provider allocations for this consumer
        c_provider_rate = 0.0
        c_payment = gnb_rate * gnb_price
        c_providers_used = []

        for k, (pid, c) in enumerate(eligible_pairs):
            if c == cid:
                alloc = float(pair_allocations[k])
                if alloc > 1e-6:
                    p_price = providers[pid]["price"]
                    c_provider_rate += alloc
                    c_payment += alloc * p_price
                    c_providers_used.append(f"{pid}({alloc:.2f}M@{p_price:.3f})")

        if c_provider_rate > 1e-6:
            consumers_with_provider += 1

        c_total_rate = c_provider_rate + gnb_rate
        c_unmet = max(0.0, demand - c_total_rate)

        detailed_records.append({
            "method": "MEC-lexicographic-LP",
            "consumer_id": cid,
            "requested_mbps": round(demand, 4),
            "provider_rate_mbps": round(c_provider_rate, 4),
            "gnb_rate_mbps": round(gnb_rate, 4),
            "total_rate_mbps": round(c_total_rate, 4),
            "payment_model_units": round(c_payment, 5),
            "unmet_demand_mbps": round(c_unmet, 4),
            "selected_providers": "; ".join(c_providers_used)
        })

    summary = {
        "method": "MEC-lexicographic-LP",
        "consumers": len(consumers),
        "consumers_with_provider": consumers_with_provider,
        "consumers_fully_served": sum(1 for r in detailed_records if r["unmet_demand_mbps"] < 1e-4),
        "total_payment_model_units": round(total_payment, 4),
        "total_achieved_rate_mbps": round(total_rate, 3),
        "total_unmet_demand_mbps": round(total_unmet, 3),
        "provider_throughput_mbps": round(total_provider_rate, 3),
        "gnb_throughput_mbps": round(total_gnb_rate, 3),
        "gnb_remaining_capacity_mbps": round(gnb_capacity - total_gnb_rate, 3),
        "cost_per_mbps_delivered": round(total_payment / total_rate, 5) if total_rate > 0 else 0.0,
        "stage1_min_payment": round(C_star, 4),
        "payment_budget_cap": round(allowed_payment_budget, 4),
        "solver_status": solver_status
    }

    return summary, detailed_records


# =============================================================================
# MASTER RUNNER & COMPARISON WORKFLOW
# =============================================================================
def run_comparison(
    mobility_file: str = DEFAULT_MOBILITY_FILE,
    output_summary: str = DEFAULT_OUTPUT_SUMMARY,
    output_detailed: str = DEFAULT_OUTPUT_DETAILED,
    provider_count: int = DEFAULT_PROVIDER_COUNT,
    consumer_count: int = DEFAULT_CONSUMER_COUNT,
    max_distance: float = DEFAULT_MAX_DISTANCE_M,
    gnb_capacity: float = DEFAULT_GNB_CAPACITY_MBPS,
    gnb_price: float = DEFAULT_GNB_PRICE,
    payment_tolerance: float = DEFAULT_PAYMENT_TOLERANCE,
    min_service_fraction: float = DEFAULT_MIN_SERVICE_FRACTION,
    seed: int = DEFAULT_SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes the three allocation algorithms on identical data and parameters,
    saves the comparison summary and detailed records, and displays formatted output.
    """
    print(f"================================================================================")
    print(f" MEC-Assisted Internet Sharing Marketplace — Algorithmic Comparison")
    print(f"================================================================================")
    print(f" Configuration:")
    print(f"  • Seed                   : {seed}")
    print(f"  • Mobility File          : {mobility_file}")
    print(f"  • Providers              : {provider_count} (IDs 0 to {provider_count - 1})")
    print(f"  • Consumers              : {consumer_count} (IDs {provider_count} to {provider_count + consumer_count - 1})")
    print(f"  • Max Distance (R_com)   : {max_distance} m")
    print(f"  • gNodeB Capacity        : {gnb_capacity} Mbps")
    print(f"  • gNodeB Unit Price      : {gnb_price} model units / Mbps")
    print(f"  • Payment Tolerance (δ)  : {payment_tolerance * 100:.1f}%")
    print(f"  • Min Service Fraction   : {min_service_fraction * 100:.1f}%")
    print(f"================================================================================\n")

    # 1. Load profiles and positions
    providers, consumers = generate_profiles(provider_count, consumer_count, seed)
    positions = load_latest_positions(mobility_file)

    total_demand = sum(consumers.values())
    total_prov_cap = sum(p["capacity"] for p in providers.values())
    print(f" Total Consumer Demand    : {total_demand:.3f} Mbps")
    print(f" Total Provider Capacity  : {total_prov_cap:.3f} Mbps")
    print(f" Total gNodeB Capacity    : {gnb_capacity:.3f} Mbps")
    print(f" Aggregate System Capacity: {total_prov_cap + gnb_capacity:.3f} Mbps\n")

    # 2. Run Algorithm 1: gNodeB-only
    print(" [1/3] Running gNodeB-only baseline...")
    summary_gnb, detailed_gnb = run_gnb_only(
        consumers=consumers,
        gnb_capacity=gnb_capacity,
        gnb_price=gnb_price
    )

    # 3. Run Algorithm 2: Cheapest-first greedy
    print(" [2/3] Running Cheapest-first greedy baseline...")
    summary_greedy, detailed_greedy = run_greedy(
        providers=providers,
        consumers=consumers,
        positions=positions,
        max_distance=max_distance,
        gnb_capacity=gnb_capacity,
        gnb_price=gnb_price
    )

    # 4. Run Algorithm 3: MEC Three-Stage Lexicographic LP
    print(" [3/3] Running MEC Three-Stage Lexicographic LP (SciPy HiGHS)...")
    summary_mec, detailed_mec = run_mec_two_stage_lp(
        providers=providers,
        consumers=consumers,
        positions=positions,
        max_distance=max_distance,
        gnb_capacity=gnb_capacity,
        gnb_price=gnb_price,
        payment_tolerance=payment_tolerance,
        min_service_fraction=min_service_fraction
    )

    summaries = [summary_gnb, summary_greedy, summary_mec]
    all_detailed = detailed_gnb + detailed_greedy + detailed_mec

    # Save summary CSV
    # Unify fieldnames across methods (some keys are MEC-specific)
    all_fieldnames = []
    for s in summaries:
        for k in s.keys():
            if k not in all_fieldnames:
                all_fieldnames.append(k)

    with open(output_summary, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    print(f"\n Summary written to: {output_summary}")

    # Save detailed allocations CSV
    if detailed_gnb:
        with open(output_detailed, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detailed_gnb[0].keys())
            writer.writeheader()
            writer.writerows(all_detailed)
        print(f" Detailed allocations written to: {output_detailed}")

    # Print clean formatted table
    print("\n" + "=" * 105)
    print(f"{'Method':<24} | {'Payment':<10} | {'Achieved':<10} | {'Unmet':<10} | {'PUE Rate':<10} | {'gNB Rate':<10} | {'Cost/Mbps':<10} | {'PUE Users':<10}")
    print("-" * 105)
    for s in summaries:
        print(
            f"{s['method']:<24} | "
            f"{s['total_payment_model_units']:<10.4f} | "
            f"{s['total_achieved_rate_mbps']:<10.3f} | "
            f"{s['total_unmet_demand_mbps']:<10.3f} | "
            f"{s['provider_throughput_mbps']:<10.3f} | "
            f"{s['gnb_throughput_mbps']:<10.3f} | "
            f"{s['cost_per_mbps_delivered']:<10.5f} | "
            f"{s['consumers_with_provider']:<10}"
        )
    print("=" * 105 + "\n")

    # High-level insights
    print(" Key Analytical Takeaways:")
    gnb_p = summary_gnb['total_payment_model_units']
    greedy_p = summary_greedy['total_payment_model_units']
    mec_p = summary_mec['total_payment_model_units']
    greedy_r = summary_greedy['total_achieved_rate_mbps']
    mec_r = summary_mec['total_achieved_rate_mbps']

    print(f"  • gNodeB Baseline : Charges {gnb_p:.4f} units but leaves {summary_gnb['total_unmet_demand_mbps']:.3f} Mbps unmet due to cell capacity cap.")
    print(f"  • Greedy Baseline : Delivers {greedy_r:.3f} Mbps at total payment {greedy_p:.4f} (Cost/Mbps: {summary_greedy['cost_per_mbps_delivered']:.5f}).")
    print(f"  • MEC Two-Stage LP: Delivers {mec_r:.3f} Mbps at total payment {mec_p:.4f} (Cost/Mbps: {summary_mec['cost_per_mbps_delivered']:.5f}).")
    if 'stage1_min_payment' in summary_mec:
        print(f"    - Stage 1 Minimum Payment (C*): {summary_mec['stage1_min_payment']:.4f} units (meets {min_service_fraction*100:.0f}% QoS).")
        print(f"    - Stage 2 Payment Budget Cap   : {summary_mec['payment_budget_cap']:.4f} units (C* + {payment_tolerance*100:.0f}%).")
    print("=" * 105)

    return summaries, all_detailed


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Part 1: MEC-Assisted Internet Sharing Marketplace Comparison")
    parser.add_argument("--mobility-file", type=str, default=DEFAULT_MOBILITY_FILE, help="Path to mobility.csv")
    parser.add_argument("--output-summary", type=str, default=DEFAULT_OUTPUT_SUMMARY, help="Output summary CSV file")
    parser.add_argument("--output-detailed", type=str, default=DEFAULT_OUTPUT_DETAILED, help="Output detailed CSV file")
    parser.add_argument("--providers", type=int, default=DEFAULT_PROVIDER_COUNT, help="Number of Provider UEs (PUEs)")
    parser.add_argument("--consumers", type=int, default=DEFAULT_CONSUMER_COUNT, help="Number of Consumer UEs (CUEs)")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE_M, help="Effective PUE communication range R_com (m)")
    parser.add_argument("--gnb-capacity", type=float, default=DEFAULT_GNB_CAPACITY_MBPS, help="Aggregate gNodeB capacity (Mbps)")
    parser.add_argument("--gnb-price", type=float, default=DEFAULT_GNB_PRICE, help="gNodeB unit price per Mbps")
    parser.add_argument("--payment-tolerance", type=float, default=DEFAULT_PAYMENT_TOLERANCE, help="Stage 2 payment tolerance delta (default: 0.05)")
    parser.add_argument("--min-service-fraction", type=float, default=DEFAULT_MIN_SERVICE_FRACTION, help="Stage 1 min service fraction (default: 0.50)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")

    args = parser.parse_args()

    run_comparison(
        mobility_file=args.mobility_file,
        output_summary=args.output_summary,
        output_detailed=args.output_detailed,
        provider_count=args.providers,
        consumer_count=args.consumers,
        max_distance=args.max_distance,
        gnb_capacity=args.gnb_capacity,
        gnb_price=args.gnb_price,
        payment_tolerance=args.payment_tolerance,
        min_service_fraction=args.min_service_fraction,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dynamic_method_comparison.py
MEC-Assisted Internet Sharing Marketplace
Stage 2: Fair Dynamic Three-Method Comparison

Compares three allocation algorithms on identical 300s SUMO mobility traces:
1. gNodeB-only (Cellular fallback baseline with standard pricing Pg = 0.05)
2. Cheapest-first greedy (Proximity-constrained greedy heuristic with gNodeB fallback)
3. MEC Three-Stage Lexicographic LP (SciPy HiGHS):
   - Stage 1: Minimize total payment subject to 100% QoS (alpha = 1.0) and physical constraints.
   - Stage 2: Maximize achieved data rate within 5% payment tolerance of Stage 1 optimum.
   - Stage 3: Minimize total payment while preserving the maximum throughput achieved in Stage 2.

All methods share identical provider capacities, prices, consumer demands, and positions
at every time step. gNodeB fallback is charged at Pg = 0.05 per Mbps per second in every method.
"""

import os
import sys
import csv
import math
import time
import random
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.optimize as opt

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_SEED = 42
DEFAULT_MOBILITY_FILE = "mobility.csv"
DEFAULT_SUMMARY_OUTPUT = "dynamic_comparison_summary.csv"
DEFAULT_DETAILED_OUTPUT = "dynamic_comparison_detailed.csv"
DEFAULT_TIMESTEP_OUTPUT = "dynamic_comparison_timestep.csv"

DEFAULT_SIMULATION_END_S = 300.0
DEFAULT_STEP_LENGTH_S = 1.0

DEFAULT_PROVIDER_COUNT = 30
DEFAULT_CONSUMER_COUNT = 20

DEFAULT_MAX_DISTANCE_M = 1000.0       # Effective direct communication range (R_com)
DEFAULT_GNB_CAPACITY_MBPS = 100.0     # Aggregate gNodeB cell capacity
DEFAULT_GNB_PRICE = 0.05              # Price per Mbps delivered by gNodeB (model units / Mbps*s)
DEFAULT_PAYMENT_TOLERANCE = 0.05      # delta = 5% tolerance
DEFAULT_MIN_SERVICE_FRACTION = 1.00   # alpha = 1.0 (100% QoS)

PROVIDER_CAP_MIN = 5.0
PROVIDER_CAP_MAX = 20.0
PROVIDER_PRICE_MIN = 0.01
PROVIDER_PRICE_MAX = 0.10

CONSUMER_DEMAND_MIN = 3.0
CONSUMER_DEMAND_MAX = 10.0


# =============================================================================
# DATA STRUCTURES & PROFILE GENERATION
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


def load_mobility_trace(
    mobility_file: str,
    max_time_s: float = DEFAULT_SIMULATION_END_S
) -> Dict[float, Dict[str, Tuple[float, float]]]:
    """
    Loads timestamped vehicle positions from mobility.csv up to max_time_s.
    Returns: {time_s: {vehicle_id: (x, y)}}
    """
    if not os.path.exists(mobility_file):
        raise FileNotFoundError(f"Mobility file '{mobility_file}' not found.")

    trace: Dict[float, Dict[str, Tuple[float, float]]] = {}
    with open(mobility_file, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_s"])
            if t > max_time_s:
                continue
            if t not in trace:
                trace[t] = {}
            trace[t][row["vehicle_id"]] = (float(row["x"]), float(row["y"]))

    return trace


def compute_euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# =============================================================================
# ALGORITHM 1: DYNAMIC gNodeB-ONLY
# =============================================================================
def run_dynamic_gnb_step(
    active_consumers: List[str],
    consumers: Dict[str, float],
    gnb_capacity: float,
    gnb_price: float,
    dt: float = DEFAULT_STEP_LENGTH_S
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Serves active consumers solely through cellular gNodeB up to aggregate capacity.
    Charges gnb_price * allocated_mbps * dt.
    """
    gnb_remaining = gnb_capacity
    step_records = []

    total_pay = 0.0
    total_rate = 0.0
    total_unmet = 0.0

    for cid in active_consumers:
        demand = consumers[cid]
        gnb_alloc = min(demand, gnb_remaining)
        gnb_remaining -= gnb_alloc

        unmet = max(0.0, demand - gnb_alloc)
        payment = gnb_alloc * gnb_price * dt

        total_pay += payment
        total_rate += gnb_alloc
        total_unmet += unmet

        step_records.append({
            "method": "gNB-only",
            "consumer_id": cid,
            "requested_mbps": demand,
            "provider_rate_mbps": 0.0,
            "gnb_rate_mbps": gnb_alloc,
            "total_rate_mbps": gnb_alloc,
            "payment_model_units": payment,
            "unmet_demand_mbps": unmet,
            "selected_providers": ""
        })

    step_summary = {
        "method": "gNB-only",
        "payment": total_pay,
        "rate": total_rate,
        "unmet": total_unmet,
        "provider_rate": 0.0,
        "gnb_rate": total_rate,
        "gnb_remaining": gnb_remaining,
        "infeasible": False,
        "runtime_s": 0.0
    }
    return step_summary, step_records


# =============================================================================
# ALGORITHM 2: DYNAMIC CHEAPEST-FIRST GREEDY
# =============================================================================
def run_dynamic_greedy_step(
    active_consumers: List[str],
    active_providers: List[str],
    consumers: Dict[str, float],
    providers: Dict[str, Dict[str, float]],
    positions: Dict[str, Tuple[float, float]],
    max_distance: float,
    gnb_capacity: float,
    gnb_price: float,
    dt: float = DEFAULT_STEP_LENGTH_S
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Greedy heuristic per time-step:
    For each active consumer, finds eligible providers within max_distance with remaining capacity.
    Sorts by lowest price first (ties broken by distance).
    Residual demand falls back to gNodeB, properly charged at gnb_price * allocated * dt.
    """
    provider_used = {pid: 0.0 for pid in active_providers}
    gnb_remaining = gnb_capacity

    step_records = []
    total_pay = 0.0
    total_rate = 0.0
    total_provider_rate = 0.0
    total_gnb_rate = 0.0
    total_unmet = 0.0

    for cid in active_consumers:
        demand = consumers[cid]
        cpos = positions[cid]
        remaining = demand

        candidates = []
        for pid in active_providers:
            ppos = positions[pid]
            dist = compute_euclidean_distance(cpos, ppos)
            avail = providers[pid]["capacity"] - provider_used[pid]
            if dist <= max_distance and avail > 1e-6:
                candidates.append((providers[pid]["price"], dist, pid, avail))

        # Lowest price first, nearest distance breaks ties
        candidates.sort(key=lambda x: (x[0], x[1]))

        c_provider_rate = 0.0
        c_payment = 0.0
        selected_pids = []

        for price, dist, pid, avail in candidates:
            if remaining <= 1e-6:
                break
            alloc = min(remaining, avail)
            provider_used[pid] += alloc
            remaining -= alloc
            c_provider_rate += alloc
            c_payment += alloc * price * dt
            selected_pids.append(f"{pid}({alloc:.2f}M@{price:.3f})")

        # Fallback to gNodeB
        gnb_alloc = min(remaining, gnb_remaining)
        gnb_remaining -= gnb_alloc
        remaining -= gnb_alloc
        c_payment += gnb_alloc * gnb_price * dt

        achieved = c_provider_rate + gnb_alloc
        unmet = max(0.0, demand - achieved)

        total_pay += c_payment
        total_rate += achieved
        total_provider_rate += c_provider_rate
        total_gnb_rate += gnb_alloc
        total_unmet += unmet

        step_records.append({
            "method": "cheapest-first-greedy",
            "consumer_id": cid,
            "requested_mbps": demand,
            "provider_rate_mbps": c_provider_rate,
            "gnb_rate_mbps": gnb_alloc,
            "total_rate_mbps": achieved,
            "payment_model_units": c_payment,
            "unmet_demand_mbps": unmet,
            "selected_providers": "; ".join(selected_pids)
        })

    step_summary = {
        "method": "cheapest-first-greedy",
        "payment": total_pay,
        "rate": total_rate,
        "unmet": total_unmet,
        "provider_rate": total_provider_rate,
        "gnb_rate": total_gnb_rate,
        "gnb_remaining": gnb_remaining,
        "infeasible": False,
        "runtime_s": 0.0
    }
    return step_summary, step_records


# =============================================================================
# ALGORITHM 3: DYNAMIC MEC THREE-STAGE LEXICOGRAPHIC LP
# =============================================================================
def run_dynamic_mec_step(
    active_consumers: List[str],
    active_providers: List[str],
    consumers: Dict[str, float],
    providers: Dict[str, Dict[str, float]],
    positions: Dict[str, Tuple[float, float]],
    max_distance: float,
    gnb_capacity: float,
    gnb_price: float,
    payment_tolerance: float = DEFAULT_PAYMENT_TOLERANCE,
    min_service_fraction: float = DEFAULT_MIN_SERVICE_FRACTION,
    dt: float = DEFAULT_STEP_LENGTH_S
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    MEC Three-Stage Lexicographic LP per time-step via SciPy HiGHS:
      Stage 1: Minimize total payment subject to 100% QoS (alpha = 1.0) and physical bounds.
      Stage 2: Maximize achieved data rate subject to payment <= (1 + delta) * C*.
      Stage 3: Minimize total payment subject to rate >= R* - 1e-6 and payment <= (1 + delta) * C*.

    Explicitly reports infeasibility if alpha=1.0 cannot be satisfied by available resources.
    """
    t_start = time.perf_counter()

    # 1. Identify eligible communication pairs
    eligible_pairs: List[Tuple[str, str]] = []
    for cid in active_consumers:
        cpos = positions[cid]
        for pid in active_providers:
            ppos = positions[pid]
            dist = compute_euclidean_distance(cpos, ppos)
            if dist <= max_distance:
                eligible_pairs.append((pid, cid))

    num_pairs = len(eligible_pairs)
    num_c = len(active_consumers)
    num_vars = num_pairs + num_c

    # Cost vector (scaled by dt)
    cost_vector = np.zeros(num_vars)
    for k, (pid, cid) in enumerate(eligible_pairs):
        cost_vector[k] = providers[pid]["price"] * dt
    for j, cid in enumerate(active_consumers):
        cost_vector[num_pairs + j] = gnb_price * dt

    # Build Inequality Constraints: A_ub @ x <= b_ub
    A_rows: List[np.ndarray] = []
    b_vals: List[float] = []

    # Constraint 1: Consumer demand upper bound: sum_i r_{i,j} + r_j^g <= D_j
    for j, cid in enumerate(active_consumers):
        row = np.zeros(num_vars)
        for k, (p, c) in enumerate(eligible_pairs):
            if c == cid:
                row[k] = 1.0
        row[num_pairs + j] = 1.0
        A_rows.append(row)
        b_vals.append(consumers[cid])

    # Constraint 2: Provider capacity bound: sum_j r_{i,j} <= R_i
    for i, pid in enumerate(active_providers):
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
    for j, cid in enumerate(active_consumers):
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
    # STAGE 1: MINIMIZE PAYMENT
    # ---------------------------------------------------------
    stage1_res = opt.linprog(c=cost_vector, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not stage1_res.success:
        t_elapsed = time.perf_counter() - t_start
        # Explicitly report infeasibility without silent relaxation
        detailed_records = [
            {
                "method": "MEC-lexicographic-LP",
                "consumer_id": cid,
                "requested_mbps": consumers[cid],
                "provider_rate_mbps": 0.0,
                "gnb_rate_mbps": 0.0,
                "total_rate_mbps": 0.0,
                "payment_model_units": 0.0,
                "unmet_demand_mbps": consumers[cid],
                "selected_providers": ""
            }
            for cid in active_consumers
        ]
        total_demand = sum(consumers[cid] for cid in active_consumers)
        summary = {
            "method": "MEC-lexicographic-LP",
            "payment": 0.0,
            "rate": 0.0,
            "unmet": total_demand,
            "provider_rate": 0.0,
            "gnb_rate": 0.0,
            "gnb_remaining": gnb_capacity,
            "infeasible": True,
            "solver_status": f"Stage 1 Infeasible ({stage1_res.message})",
            "runtime_s": t_elapsed
        }
        return summary, detailed_records

    C_star = float(stage1_res.fun)
    allowed_payment_budget = (1.0 + payment_tolerance) * C_star

    # ---------------------------------------------------------
    # STAGE 2: MAXIMIZE ACHIEVED DATA RATE WITHIN PAYMENT TOLERANCE
    # ---------------------------------------------------------
    rate_objective = np.full(num_vars, -1.0)
    A_ub_stage2 = np.vstack([A_ub, cost_vector])
    b_ub_stage2 = np.append(b_ub, allowed_payment_budget)

    stage2_res = opt.linprog(c=rate_objective, A_ub=A_ub_stage2, b_ub=b_ub_stage2, bounds=bounds, method="highs")

    if not stage2_res.success:
        optimal_x = stage1_res.x
        solver_status = f"Stage 1 Optimal, Stage 2 Failed ({stage2_res.message})"
    else:
        R_star = float(-stage2_res.fun)

        # ---------------------------------------------------------
        # STAGE 3: MINIMIZE PAYMENT PRESERVING MAXIMUM RATE
        # ---------------------------------------------------------
        row_rate = np.full(num_vars, -1.0)
        A_ub_stage3 = np.vstack([A_ub_stage2, row_rate])
        b_ub_stage3 = np.append(b_ub_stage2, -(R_star - 1e-6))

        stage3_res = opt.linprog(c=cost_vector, A_ub=A_ub_stage3, b_ub=b_ub_stage3, bounds=bounds, method="highs")

        if not stage3_res.success:
            optimal_x = stage2_res.x
            solver_status = f"HiGHS Stage 1 & 2 Optimal, Stage 3 Failed ({stage3_res.message})"
        else:
            optimal_x = stage3_res.x
            solver_status = "HiGHS Three-Stage Lexicographic Optimal"

    t_elapsed = time.perf_counter() - t_start

    pair_allocations = optimal_x[:num_pairs]
    gnb_allocations = optimal_x[num_pairs:]

    total_pay = float(np.dot(cost_vector, optimal_x))
    total_rate = float(np.sum(optimal_x))
    total_provider_rate = float(np.sum(pair_allocations))
    total_gnb_rate = float(np.sum(gnb_allocations))
    total_demand = sum(consumers[cid] for cid in active_consumers)
    total_unmet = max(0.0, total_demand - total_rate)

    detailed_records = []
    for j, cid in enumerate(active_consumers):
        demand = consumers[cid]
        gnb_rate = float(gnb_allocations[j])

        c_prov_rate = 0.0
        c_payment = gnb_rate * gnb_price * dt
        c_providers_used = []

        for k, (pid, c) in enumerate(eligible_pairs):
            if c == cid:
                alloc = float(pair_allocations[k])
                if alloc > 1e-6:
                    p_price = providers[pid]["price"]
                    c_prov_rate += alloc
                    c_payment += alloc * p_price * dt
                    c_providers_used.append(f"{pid}({alloc:.2f}M@{p_price:.3f})")

        achieved = c_prov_rate + gnb_rate
        unmet = max(0.0, demand - achieved)

        detailed_records.append({
            "method": "MEC-lexicographic-LP",
            "consumer_id": cid,
            "requested_mbps": demand,
            "provider_rate_mbps": c_prov_rate,
            "gnb_rate_mbps": gnb_rate,
            "total_rate_mbps": achieved,
            "payment_model_units": c_payment,
            "unmet_demand_mbps": unmet,
            "selected_providers": "; ".join(c_providers_used)
        })

    summary = {
        "method": "MEC-lexicographic-LP",
        "payment": total_pay,
        "rate": total_rate,
        "unmet": total_unmet,
        "provider_rate": total_provider_rate,
        "gnb_rate": total_gnb_rate,
        "gnb_remaining": gnb_capacity - total_gnb_rate,
        "infeasible": False,
        "solver_status": solver_status,
        "runtime_s": t_elapsed
    }
    return summary, detailed_records


# =============================================================================
# MASTER DYNAMIC SIMULATION LOOP
# =============================================================================
def run_dynamic_comparison(
    mobility_file: str = DEFAULT_MOBILITY_FILE,
    output_summary: str = DEFAULT_SUMMARY_OUTPUT,
    output_detailed: str = DEFAULT_DETAILED_OUTPUT,
    output_timestep: str = DEFAULT_TIMESTEP_OUTPUT,
    provider_count: int = DEFAULT_PROVIDER_COUNT,
    consumer_count: int = DEFAULT_CONSUMER_COUNT,
    max_distance: float = DEFAULT_MAX_DISTANCE_M,
    gnb_capacity: float = DEFAULT_GNB_CAPACITY_MBPS,
    gnb_price: float = DEFAULT_GNB_PRICE,
    payment_tolerance: float = DEFAULT_PAYMENT_TOLERANCE,
    min_service_fraction: float = DEFAULT_MIN_SERVICE_FRACTION,
    max_time_s: float = DEFAULT_SIMULATION_END_S,
    seed: int = DEFAULT_SEED
) -> Dict[str, Any]:
    """
    Executes all three methods at each second of the 300s SUMO mobility trace
    using identical vehicle states, demands, capacities, and pricing.
    """
    print("=" * 80)
    print(" MEC-Assisted Internet Sharing Marketplace — Dynamic 3-Method Comparison")
    print("=" * 80)
    print(f" Configuration:")
    print(f"  • Seed                     : {seed}")
    print(f"  • Mobility File            : {mobility_file}")
    print(f"  • Simulation Interval      : 0 to {max_time_s} s (dt = 1.0 s)")
    print(f"  • Providers (PUEs)         : {provider_count} (IDs 0 to {provider_count - 1})")
    print(f"  • Consumers (CUEs)         : {consumer_count} (IDs {provider_count} to {provider_count + consumer_count - 1})")
    print(f"  • Communication Range (R)  : {max_distance} m")
    print(f"  • gNodeB Capacity          : {gnb_capacity} Mbps")
    print(f"  • gNodeB Price (Pg)        : {gnb_price} model units / Mbps*s")
    print(f"  • Payment Tolerance (δ)    : {payment_tolerance * 100:.1f}%")
    print(f"  • Target QoS (alpha)       : {min_service_fraction * 100:.1f}%")
    print("=" * 80 + "\n")

    # 1. Load profiles and mobility trace
    providers, consumers = generate_profiles(provider_count, consumer_count, seed)
    trace = load_mobility_trace(mobility_file, max_time_s)

    all_provider_ids = set(providers.keys())
    all_consumer_ids = set(consumers.keys())

    all_detailed_records: List[Dict[str, Any]] = []
    timestep_records: List[Dict[str, Any]] = []

    # Accumulators for aggregate evaluation
    methods = ["gNB-only", "cheapest-first-greedy", "MEC-lexicographic-LP"]
    stats = {
        m: {
            "total_payment": 0.0,
            "total_delivered_mb": 0.0,
            "total_unmet_mb": 0.0,
            "provider_delivered_mb": 0.0,
            "gnb_delivered_mb": 0.0,
            "consumer_records_count": 0,
            "fully_served_records": 0,
            "records_with_provider": 0,
            "infeasible_steps": 0,
            "total_runtime_s": 0.0
        }
        for m in methods
    }

    total_demanded_mb = 0.0
    active_seconds_count = 0
    total_consumer_observations = 0

    time_steps = sorted(trace.keys())
    print(f"Loaded {len(time_steps)} time steps from mobility trace (<= {max_time_s}s).")

    for t in time_steps:
        pos = trace[t]
        active_c = [cid for cid in sorted(all_consumer_ids, key=int) if cid in pos]
        active_p = [pid for pid in sorted(all_provider_ids, key=int) if pid in pos]

        if not active_c:
            continue

        active_seconds_count += 1
        total_consumer_observations += len(active_c)
        step_demand = sum(consumers[cid] for cid in active_c)
        total_demanded_mb += step_demand  # dt = 1.0s -> Mbps * 1s = Mb

        # -------------------------------------------------------------
        # Method 1: gNodeB-only
        # -------------------------------------------------------------
        sum_gnb, det_gnb = run_dynamic_gnb_step(
            active_consumers=active_c,
            consumers=consumers,
            gnb_capacity=gnb_capacity,
            gnb_price=gnb_price
        )

        # -------------------------------------------------------------
        # Method 2: Cheapest-first greedy
        # -------------------------------------------------------------
        sum_gr, det_gr = run_dynamic_greedy_step(
            active_consumers=active_c,
            active_providers=active_p,
            consumers=consumers,
            providers=providers,
            positions=pos,
            max_distance=max_distance,
            gnb_capacity=gnb_capacity,
            gnb_price=gnb_price
        )

        # -------------------------------------------------------------
        # Method 3: MEC Three-Stage Lexicographic LP
        # -------------------------------------------------------------
        sum_mec, det_mec = run_dynamic_mec_step(
            active_consumers=active_c,
            active_providers=active_p,
            consumers=consumers,
            providers=providers,
            positions=pos,
            max_distance=max_distance,
            gnb_capacity=gnb_capacity,
            gnb_price=gnb_price,
            payment_tolerance=payment_tolerance,
            min_service_fraction=min_service_fraction
        )

        step_results = {
            "gNB-only": (sum_gnb, det_gnb),
            "cheapest-first-greedy": (sum_gr, det_gr),
            "MEC-lexicographic-LP": (sum_mec, det_mec)
        }

        # Record detailed per-consumer rows with time_s attached
        for m in methods:
            s_sum, s_det = step_results[m]
            for row in s_det:
                row_copy = dict(row)
                row_copy["time_s"] = t
                all_detailed_records.append(row_copy)

            # Update stats
            st = stats[m]
            st["total_payment"] += s_sum["payment"]
            st["total_delivered_mb"] += s_sum["rate"]
            st["total_unmet_mb"] += s_sum["unmet"]
            st["provider_delivered_mb"] += s_sum["provider_rate"]
            st["gnb_delivered_mb"] += s_sum["gnb_rate"]
            st["consumer_records_count"] += len(active_c)
            st["total_runtime_s"] += s_sum["runtime_s"]
            if s_sum.get("infeasible", False):
                st["infeasible_steps"] += 1

            for r in s_det:
                if r["unmet_demand_mbps"] < 1e-4:
                    st["fully_served_records"] += 1
                if r["provider_rate_mbps"] > 1e-4:
                    st["records_with_provider"] += 1

        # Per-timestep record
        timestep_records.append({
            "time_s": t,
            "active_consumers": len(active_c),
            "active_providers": len(active_p),
            "demanded_rate_mbps": round(step_demand, 3),
            # gNB-only
            "gnb_payment": round(sum_gnb["payment"], 4),
            "gnb_delivered_mbps": round(sum_gnb["rate"], 3),
            "gnb_unmet_mbps": round(sum_gnb["unmet"], 3),
            # Greedy
            "greedy_payment": round(sum_gr["payment"], 4),
            "greedy_delivered_mbps": round(sum_gr["rate"], 3),
            "greedy_unmet_mbps": round(sum_gr["unmet"], 3),
            # MEC LP
            "mec_payment": round(sum_mec["payment"], 4),
            "mec_delivered_mbps": round(sum_mec["rate"], 3),
            "mec_unmet_mbps": round(sum_mec["unmet"], 3),
            "mec_infeasible": sum_mec.get("infeasible", False),
            "mec_runtime_ms": round(sum_mec["runtime_s"] * 1000.0, 2)
        })

    print(f"Dynamic execution finished: {active_seconds_count} active seconds, {total_consumer_observations} consumer-time records.")

    # -------------------------------------------------------------
    # BUILD FINAL COMPARISON SUMMARY TABLE
    # -------------------------------------------------------------
    summary_rows = []
    # Total available capacity over active seconds
    total_gnb_cap_offered_mb = gnb_capacity * active_seconds_count

    for m in methods:
        st = stats[m]
        tot_pay = st["total_payment"]
        tot_deliv = st["total_delivered_mb"]
        tot_unmet = st["total_unmet_mb"]
        n_recs = st["consumer_records_count"]
        fully_served = st["fully_served_records"]
        with_pue = st["records_with_provider"]
        pue_deliv = st["provider_delivered_mb"]
        gnb_deliv = st["gnb_delivered_mb"]
        infeas = st["infeasible_steps"]
        runtime = st["total_runtime_s"]

        avg_rate = tot_deliv / n_recs if n_recs > 0 else 0.0
        pct_fully_served = (fully_served / n_recs * 100.0) if n_recs > 0 else 0.0
        pct_with_pue = (with_pue / n_recs * 100.0) if n_recs > 0 else 0.0
        cost_per_mb = (tot_pay / tot_deliv) if tot_deliv > 0 else 0.0
        gnb_util_pct = (gnb_deliv / total_gnb_cap_offered_mb * 100.0) if total_gnb_cap_offered_mb > 0 else 0.0

        summary_rows.append({
            "method": m,
            "simulation_seconds": max_time_s,
            "active_consumer_seconds": active_seconds_count,
            "total_consumer_records": n_recs,
            "total_demanded_volume_mb": round(total_demanded_mb, 3),
            "total_delivered_volume_mb": round(tot_deliv, 3),
            "total_unmet_demand_mb": round(tot_unmet, 3),
            "average_achieved_rate_mbps": round(avg_rate, 3),
            "fully_served_records": fully_served,
            "fully_served_percent": round(pct_fully_served, 2),
            "records_with_pue": with_pue,
            "records_with_pue_percent": round(pct_with_pue, 2),
            "total_payment_model_units": round(tot_pay, 4),
            "cost_per_mb_delivered": round(cost_per_mb, 5),
            "pue_delivered_volume_mb": round(pue_deliv, 3),
            "gnb_delivered_volume_mb": round(gnb_deliv, 3),
            "gnb_capacity_utilization_pct": round(gnb_util_pct, 2),
            "infeasible_time_steps": infeas,
            "total_runtime_seconds": round(runtime, 4),
            "avg_step_runtime_ms": round(runtime / active_seconds_count * 1000.0, 2) if active_seconds_count > 0 else 0.0
        })

    # Write summary CSV
    with open(output_summary, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary written to: {output_summary}")

    # Write detailed allocations CSV
    if all_detailed_records:
        field_order = [
            "time_s", "method", "consumer_id", "requested_mbps",
            "provider_rate_mbps", "gnb_rate_mbps", "total_rate_mbps",
            "payment_model_units", "unmet_demand_mbps", "selected_providers"
        ]
        with open(output_detailed, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(all_detailed_records)
        print(f"Detailed allocations written to: {output_detailed}")

    # Write per-timestep summary CSV
    if timestep_records:
        with open(output_timestep, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=timestep_records[0].keys())
            writer.writeheader()
            writer.writerows(timestep_records)
        print(f"Time-step records written to: {output_timestep}")

    # Print Formatted Display
    print("\n" + "=" * 120)
    print(f"{'Method':<24} | {'Payment':<10} | {'Delivered Mb':<13} | {'Unmet Mb':<10} | {'Cost/Mb':<10} | {'Full QoS %':<10} | {'PUE Rec %':<10} | {'gNB Util %':<10}")
    print("-" * 120)
    for s in summary_rows:
        print(
            f"{s['method']:<24} | "
            f"{s['total_payment_model_units']:<10.4f} | "
            f"{s['total_delivered_volume_mb']:<13.3f} | "
            f"{s['total_unmet_demand_mb']:<10.3f} | "
            f"{s['cost_per_mb_delivered']:<10.5f} | "
            f"{s['fully_served_percent']:<10.2f} | "
            f"{s['records_with_pue_percent']:<10.2f} | "
            f"{s['gnb_capacity_utilization_pct']:<10.2f}"
        )
    print("=" * 120 + "\n")

    return {"summary": summary_rows}


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Fair Dynamic Three-Method Comparison")
    parser.add_argument("--mobility-file", type=str, default=DEFAULT_MOBILITY_FILE, help="Path to mobility.csv")
    parser.add_argument("--output-summary", type=str, default=DEFAULT_SUMMARY_OUTPUT, help="Summary output CSV")
    parser.add_argument("--output-detailed", type=str, default=DEFAULT_DETAILED_OUTPUT, help="Detailed output CSV")
    parser.add_argument("--output-timestep", type=str, default=DEFAULT_TIMESTEP_OUTPUT, help="Time-step output CSV")
    parser.add_argument("--providers", type=int, default=DEFAULT_PROVIDER_COUNT, help="Provider count")
    parser.add_argument("--consumers", type=int, default=DEFAULT_CONSUMER_COUNT, help="Consumer count")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE_M, help="R_com in meters")
    parser.add_argument("--gnb-capacity", type=float, default=DEFAULT_GNB_CAPACITY_MBPS, help="gNodeB aggregate capacity")
    parser.add_argument("--gnb-price", type=float, default=DEFAULT_GNB_PRICE, help="gNodeB price per Mbps*s")
    parser.add_argument("--payment-tolerance", type=float, default=DEFAULT_PAYMENT_TOLERANCE, help="Stage 2 delta tolerance")
    parser.add_argument("--min-service-fraction", type=float, default=DEFAULT_MIN_SERVICE_FRACTION, help="alpha QoS fraction")
    parser.add_argument("--max-time", type=float, default=DEFAULT_SIMULATION_END_S, help="Simulation horizon in seconds")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")

    args = parser.parse_args()

    run_dynamic_comparison(
        mobility_file=args.mobility_file,
        output_summary=args.output_summary,
        output_detailed=args.output_detailed,
        output_timestep=args.output_timestep,
        provider_count=args.providers,
        consumer_count=args.consumers,
        max_distance=args.max_distance,
        gnb_capacity=args.gnb_capacity,
        gnb_price=args.gnb_price,
        payment_tolerance=args.payment_tolerance,
        min_service_fraction=args.min_service_fraction,
        max_time_s=args.max_time,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

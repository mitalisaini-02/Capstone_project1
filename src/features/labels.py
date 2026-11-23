"""
Label generation module for PUE-CUE link survival over a future prediction horizon.
"""
from typing import Dict, Tuple, Optional
from src.features.kinematics import euclidean_distance


def compute_link_survival_label(
    provider_id: int,
    consumer_id: int,
    current_time_s: float,
    future_horizon_s: float,
    comm_range_m: float,
    lifecycles: Dict[int, Dict[str, any]],
    trajectory_lookup: Dict[Tuple[float, int], Tuple[float, float]],
    step_s: float = 1.0
) -> Optional[int]:
    """
    Computes ground truth link-survival label over [current_time_s + step_s, current_time_s + future_horizon_s].

    Returns:
      1: Pair remains continuously within comm_range_m for the entire future horizon.
      0: Pair exceeds comm_range_m at any point during the future horizon.
      None: Incomplete future observation (vehicle arrived/departed, or simulation ended).
    """
    target_end_t = current_time_s + future_horizon_s

    # 1. Censoring check: both vehicles must remain active through target_end_t
    p_life = lifecycles.get(provider_id)
    c_life = lifecycles.get(consumer_id)

    if p_life is None or c_life is None:
        return None

    if p_life["end_t"] < (target_end_t - 1e-4) or c_life["end_t"] < (target_end_t - 1e-4):
        return None

    # 2. Inspect future trajectory positions
    num_steps = int(round(future_horizon_s / step_s))
    for step_idx in range(1, num_steps + 1):
        future_t = round(current_time_s + step_idx * step_s, 2)

        pos_p = trajectory_lookup.get((future_t, provider_id))
        pos_c = trajectory_lookup.get((future_t, consumer_id))

        if pos_p is None or pos_c is None:
            # Trajectory missing an intermediate second
            return None

        dist = euclidean_distance(pos_p, pos_c)
        if dist > comm_range_m:
            # Link broken at this future second
            return 0

    # Link survived entire future horizon
    return 1

"""
Reusable kinematic functions: distance, velocity estimation, relative speed, and closing rate.
"""
import math
from typing import Tuple, Optional


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculates 2D Euclidean distance in meters between two Cartesian points."""
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])


def estimate_velocity_vector(
    pos_curr: Tuple[float, float],
    pos_prev: Tuple[float, float],
    dt: float,
    speed_mps: Optional[float] = None
) -> Tuple[float, float]:
    """
    Estimates 2D velocity vector (vx, vy) in m/s from consecutive positions.
    Handles irregular or missing timestamps gracefully.
    If speed_mps is provided, normalizes the displacement direction by the logged speed.
    """
    if dt <= 1e-4:
        return (0.0, 0.0)

    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    disp = math.hypot(dx, dy)

    if disp < 1e-5:
        return (0.0, 0.0)

    if speed_mps is not None and speed_mps >= 0.0:
        # Scale unit direction vector by speedometer magnitude
        return (dx / disp * speed_mps, dy / disp * speed_mps)

    return (dx / dt, dy / dt)


def relative_displacement(
    pos_provider: Tuple[float, float],
    pos_consumer: Tuple[float, float]
) -> Tuple[float, float]:
    """Returns displacement vector (dx, dy) pointing from Consumer to Provider in meters."""
    return (pos_provider[0] - pos_consumer[0], pos_provider[1] - pos_consumer[1])


def relative_velocity(
    vel_provider: Tuple[float, float],
    vel_consumer: Tuple[float, float]
) -> Tuple[float, float]:
    """Returns relative velocity vector (dvx, dvy) = v_p - v_c in m/s."""
    return (vel_provider[0] - vel_consumer[0], vel_provider[1] - vel_consumer[1])


def relative_speed(
    vel_provider: Tuple[float, float],
    vel_consumer: Tuple[float, float]
) -> float:
    """Returns magnitude of relative velocity ||v_p - v_c|| in m/s."""
    rel_v = relative_velocity(vel_provider, vel_consumer)
    return math.hypot(rel_v[0], rel_v[1])


def radial_closing_speed(
    pos_provider: Tuple[float, float],
    pos_consumer: Tuple[float, float],
    vel_provider: Tuple[float, float],
    vel_consumer: Tuple[float, float]
) -> float:
    """
    Computes radial closing speed: - d(dist)/dt in m/s.
    Positive value indicates vehicles are moving closer to each other.
    Negative value indicates vehicles are moving farther apart.
    """
    dx = pos_provider[0] - pos_consumer[0]
    dy = pos_provider[1] - pos_consumer[1]
    dist = math.hypot(dx, dy)

    if dist < 1e-5:
        return 0.0

    dvx = vel_provider[0] - vel_consumer[0]
    dvy = vel_provider[1] - vel_consumer[1]

    # dot product: (r_p - r_c) . (v_p - v_c) / dist
    rate_of_separation = (dx * dvx + dy * dvy) / dist
    return -rate_of_separation  # positive when distance is decreasing

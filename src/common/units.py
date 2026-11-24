"""
Single source of truth for units and cost conversion.

Conventions: rates in Mbps, durations in seconds, volumes in decimal GB
(1 GB = 8000 Mb), prices in price units per GB.
"""
MBIT_PER_GB = 8000.0


def volume_gb(rate_mbps, duration_s):
    """Data volume (GB) carried by `rate_mbps` for `duration_s` seconds."""
    return rate_mbps * duration_s / MBIT_PER_GB


def cost(rate_mbps, duration_s, price_per_gb):
    """Payment (price units) for carrying `rate_mbps` for `duration_s` at `price_per_gb`."""
    return price_per_gb * volume_gb(rate_mbps, duration_s)


def rate_mbps_for_volume(volume_gb_, duration_s):
    """Constant rate (Mbps) needed to deliver `volume_gb_` in `duration_s` seconds."""
    return volume_gb_ * MBIT_PER_GB / duration_s

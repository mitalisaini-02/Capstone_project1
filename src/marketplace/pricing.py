"""
Pricing and cost calculation (posted prices).

The consumer pays the provider's posted price per GB for PUE traffic and the gNodeB price per GB
for cellular traffic. Unit conversion comes from src.common.units (1 GB = 8000 Mb).
The optimiser works with the SCALED cost  sum(price_per_gb * rate_mbps)  (identical ordering,
avoids per-second payments ~1e-4 that sit near solver tolerances); payments are converted here.
"""
from dataclasses import dataclass

import numpy as np

from src.common.units import cost as _cost


@dataclass(frozen=True)
class PostedPricePolicy:
    name: str = "posted_price"

    def scaled_cost(self, pue_rate, pue_price, gnb_rate, gnb_price):
        """Objective units: (price per GB) x Mbps."""
        return np.asarray(pue_rate) * np.asarray(pue_price) + np.asarray(gnb_rate) * gnb_price

    def payments(self, pue_rate, pue_price, gnb_rate, gnb_price, dt_s):
        """(pue_revenue, gnb_revenue, consumer_payment) in price units for one epoch."""
        pue = _cost(np.asarray(pue_rate, float), dt_s, np.nan_to_num(np.asarray(pue_price, float)))
        gnb = _cost(np.asarray(gnb_rate, float), dt_s, gnb_price)
        return pue, gnb, pue + gnb

    def budget_scaled(self, demand, max_price_per_gb):
        """C9 right-hand side in scaled units: pi_j^max * d_j."""
        return np.asarray(max_price_per_gb) * np.asarray(demand)

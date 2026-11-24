"""
gNodeB coverage model (separated from data generation).

Pure functions on coordinates: no dependency on mobility, data generation,
the LSTM or the marketplace.
  * load_gnb_sites(xml_path)  -> [(gnb_id, x_m, y_m)] from a SUMO additional file
  * nearest_site(x, y, sites, radius_m) -> serving gNodeB id per position (None if uncovered)
"""
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

import numpy as np

Site = Tuple[str, float, float]


def load_gnb_sites(xml_path: str, poi_type: str = "gNodeB") -> List[Site]:
    """Reads gNodeB POIs (id, x, y) from a SUMO additional file. Order is file order."""
    root = ET.parse(xml_path).getroot()
    sites = [(p.get("id"), float(p.get("x")), float(p.get("y")))
             for p in root.iter("poi") if p.get("type") == poi_type]
    if not sites:
        raise ValueError(f"No {poi_type} POIs in {xml_path}")
    return sites


def nearest_site(x: Sequence[float], y: Sequence[float], sites: List[Site],
                 radius_m: float) -> np.ndarray:
    """
    Serving cell = nearest site within `radius_m` (ties: first site in file order).
    Returns an object array of gNodeB ids, with None where the position is uncovered.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return np.array([], dtype=object)
    sx = np.array([s[1] for s in sites])
    sy = np.array([s[2] for s in sites])
    d = np.hypot(x[:, None] - sx[None, :], y[:, None] - sy[None, :])
    k = d.argmin(axis=1)
    ids = np.array([s[0] for s in sites], dtype=object)[k]
    ids[d[np.arange(len(k)), k] > radius_m] = None
    return ids


def is_covered(x: float, y: float, sites: List[Site], radius_m: float) -> bool:
    return nearest_site([x], [y], sites, radius_m)[0] is not None


def serving_site(x: float, y: float, sites: List[Site], radius_m: float) -> Optional[str]:
    return nearest_site([x], [y], sites, radius_m)[0]

"""
Building footprints for building-aware channel-state classification.

Data: SUMO polygon files produced by `polyconvert` from OpenStreetMap (exact SUMO network
coordinates; see scripts/build_building_geometry.py). OSM buildings are INCOMPLETE and have no
usable heights, so geometry-based line of sight is an optimistic estimate, and every mapped
footprint is assumed to block the 1.6 m antenna path (2-D test).

Geometry rules
  * A direct path is BLOCKED by a building iff the open segment passes through the building's
    INTERIOR. Grazing a vertex or running along a wall does not count as blocking.
  * An endpoint inside or on the boundary of a footprint is reported separately
    (`endpoint_building`); callers must not treat such rows as line of sight.
  * Invalid polygons (unparseable, non-finite, < 3 distinct vertices, not closed, zero area,
    self-intersecting) are rejected and listed with a reason; they never block silently.
  * Missing/malformed files or a layer without any valid building raise errors.

No dependency on the LSTM, reliability, pricing or optimisation modules.
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9
AREA_EPS = 1e-6


# --------------------------------------------------------------------------- primitives
def parse_shape(shape: str) -> np.ndarray:
    pts = [p.split(",") for p in shape.split()]
    return np.array([[float(a), float(b)] for a, b in pts], dtype=float)


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _segments_properly_intersect(p1, p2, q1, q2) -> bool:
    d1 = _cross(q2[0] - q1[0], q2[1] - q1[1], p1[0] - q1[0], p1[1] - q1[1])
    d2 = _cross(q2[0] - q1[0], q2[1] - q1[1], p2[0] - q1[0], p2[1] - q1[1])
    d3 = _cross(p2[0] - p1[0], p2[1] - p1[1], q1[0] - p1[0], q1[1] - p1[1])
    d4 = _cross(p2[0] - p1[0], p2[1] - p1[1], q2[0] - p1[0], q2[1] - p1[1])
    return (d1 * d2 < -EPS) and (d3 * d4 < -EPS)


def is_self_intersecting(poly: np.ndarray) -> bool:
    """True if two non-adjacent edges of a closed ring properly intersect."""
    n = len(poly) - 1
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue   # first and last edges share the closing vertex
            if _segments_properly_intersect(poly[i], poly[i + 1], poly[j], poly[j + 1]):
                return True
    return False


def validate_polygon(poly: Optional[np.ndarray]) -> Optional[str]:
    """Returns None if valid, otherwise a rejection reason."""
    if poly is None or poly.ndim != 2 or poly.shape[1] != 2:
        return "unparseable_shape"
    if not np.all(np.isfinite(poly)):
        return "non_finite_coordinates"
    if len(np.unique(poly, axis=0)) < 3:
        return "too_few_vertices"
    if not np.allclose(poly[0], poly[-1], atol=EPS):
        return "not_closed"
    # Self-intersection is checked before area: a bow-tie's lobes cancel to zero signed area,
    # which would otherwise be reported with the misleading reason "zero_area".
    if is_self_intersecting(poly):
        return "self_intersecting"
    if abs(polygon_area(poly)) < AREA_EPS:
        return "zero_area"
    return None


def on_boundary(poly: np.ndarray, x: float, y: float, tol: float = 1e-7) -> bool:
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), EPS), 0, 1)
    px, py = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    return bool(np.min(np.hypot(px - x, py - y)) <= tol)


def point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    """Ray casting (even-odd). Boundary points may go either way; use on_boundary() for them."""
    xs, ys = poly[:, 0], poly[:, 1]
    x1, y1, x2, y2 = xs[:-1], ys[:-1], xs[1:], ys[1:]
    cond = (y1 > y) != (y2 > y)
    xint = (x2 - x1) * (y - y1) / np.where(cond, y2 - y1, 1.0) + x1
    return bool(np.count_nonzero(cond & (x < xint)) % 2 == 1)


def inside_or_on(poly: np.ndarray, x: float, y: float) -> bool:
    return on_boundary(poly, x, y) or point_in_polygon(poly, x, y)


def strictly_inside(poly: np.ndarray, x: float, y: float) -> bool:
    return (not on_boundary(poly, x, y)) and point_in_polygon(poly, x, y)


def segment_crosses_interior(poly: np.ndarray, px, py, qx, qy) -> bool:
    """True iff the segment P-Q passes through the polygon interior (not just its boundary)."""
    rx, ry = qx - px, qy - py
    a, b = poly[:-1], poly[1:]
    sx, sy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    apx, apy = a[:, 0] - px, a[:, 1] - py
    denom = _cross(rx, ry, sx, sy)
    ts = [0.0, 1.0]
    nz = np.abs(denom) > EPS
    t = np.where(nz, _cross(apx, apy, sx, sy) / np.where(nz, denom, 1.0), np.nan)
    u = np.where(nz, _cross(apx, apy, rx, ry) / np.where(nz, denom, 1.0), np.nan)
    hit = nz & (t >= -EPS) & (t <= 1 + EPS) & (u >= -EPS) & (u <= 1 + EPS)
    ts.extend(np.clip(t[hit], 0.0, 1.0).tolist())
    # collinear overlapping edges: add the projections of their endpoints
    rr = rx * rx + ry * ry
    col = (~nz) & (np.abs(_cross(apx, apy, rx, ry)) <= EPS * max(1.0, np.sqrt(rr)))
    if np.any(col) and rr > 0:
        for ex, ey in ((a[col, 0], a[col, 1]), (b[col, 0], b[col, 1])):
            tt = ((ex - px) * rx + (ey - py) * ry) / rr
            ts.extend(np.clip(tt[(tt >= 0) & (tt <= 1)], 0, 1).tolist())
    ts = np.unique(np.round(np.array(ts), 12))
    for t0, t1 in zip(ts[:-1], ts[1:]):
        if t1 - t0 <= 1e-12:
            continue
        tm = 0.5 * (t0 + t1)
        if strictly_inside(poly, px + tm * rx, py + tm * ry):
            return True
    return False


# --------------------------------------------------------------------------- spatial index
class GridIndex:
    """Uniform grid over building bounding boxes. Deterministic: candidate lists are sorted."""

    def __init__(self, bboxes: np.ndarray, cell_size_m: float):
        if cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        self.cell = float(cell_size_m)
        self.bboxes = bboxes
        self.x0, self.y0 = float(bboxes[:, 0].min()), float(bboxes[:, 1].min())
        cells: Dict[Tuple[int, int], List[int]] = {}
        for k, (xa, ya, xb, yb) in enumerate(bboxes):
            for i in range(self._ix(xa), self._ix(xb) + 1):
                for j in range(self._iy(ya), self._iy(yb) + 1):
                    cells.setdefault((i, j), []).append(k)
        self.cells = {key: sorted(v) for key, v in sorted(cells.items())}

    def _ix(self, x):
        return int(np.floor((x - self.x0) / self.cell))

    def _iy(self, y):
        return int(np.floor((y - self.y0) / self.cell))

    def candidates_bbox(self, xa, ya, xb, yb) -> np.ndarray:
        found = set()
        for i in range(self._ix(xa - EPS), self._ix(xb + EPS) + 1):
            for j in range(self._iy(ya - EPS), self._iy(yb + EPS) + 1):
                found.update(self.cells.get((i, j), ()))
        if not found:
            return np.empty(0, dtype=int)
        c = np.array(sorted(found))
        bb = self.bboxes[c]
        keep = (bb[:, 0] <= xb + EPS) & (bb[:, 2] >= xa - EPS) & (bb[:, 1] <= yb + EPS) & (bb[:, 3] >= ya - EPS)
        return c[keep]


# --------------------------------------------------------------------------- layer
@dataclass
class BuildingLayer:
    polygons: List[np.ndarray]
    ids: List[str]
    coverage: np.ndarray
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    cell_size_m: float = 100.0
    source: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.polygons:
            raise ValueError("Building layer contains no valid building polygons")
        reason = validate_polygon(self.coverage)
        if reason:
            raise ValueError(f"Invalid coverage polygon: {reason}")
        self.bboxes = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                                for p in self.polygons])
        self.index = GridIndex(self.bboxes, self.cell_size_m)

    @property
    def n_buildings(self) -> int:
        return len(self.polygons)

    def in_coverage(self, x: float, y: float) -> bool:
        return inside_or_on(self.coverage, x, y)

    def endpoint_building(self, x: float, y: float) -> int:
        """Index of a building containing (x, y) (inside or on boundary), else -1."""
        for k in self.index.candidates_bbox(x, y, x, y):
            if inside_or_on(self.polygons[k], x, y):
                return int(k)
        return -1

    def blocking_buildings(self, px, py, qx, qy, use_index: bool = True) -> List[int]:
        """Sorted indices of buildings whose interior the segment P-Q passes through."""
        if use_index:
            cand = self.index.candidates_bbox(min(px, qx), min(py, qy), max(px, qx), max(py, qy))
        else:
            cand = range(self.n_buildings)
        return [int(k) for k in cand if segment_crosses_interior(self.polygons[k], px, py, qx, qy)]

    def describe(self) -> Dict:
        return {"n_buildings": self.n_buildings, "n_rejected": len(self.rejected),
                "rejected_reasons": {r: sum(1 for _, x in self.rejected if x == r)
                                     for r in sorted({x for _, x in self.rejected})},
                "cell_size_m": self.cell_size_m, **self.source}


def _read_polys(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Polygon file not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"Malformed polygon file {path}: {e}") from e
    return list(root.iter("poly"))


def load_building_layer(buildings_path: str, coverage_path: str,
                        building_types: Sequence[str] = ("building",),
                        cell_size_m: float = 100.0) -> BuildingLayer:
    """Loads and validates building footprints and the coverage polygon (both SUMO poly XML)."""
    polys, ids, rejected = [], [], []
    for el in _read_polys(buildings_path):
        if el.get("type") not in building_types:
            continue
        pid = el.get("id", "?")
        try:
            poly = parse_shape(el.get("shape", ""))
        except (ValueError, TypeError):
            poly = None
        reason = validate_polygon(poly)
        if reason:
            rejected.append((pid, reason))
        else:
            polys.append(poly)
            ids.append(pid)
    cov_els = _read_polys(coverage_path)
    if len(cov_els) != 1:
        raise ValueError(f"Coverage file must contain exactly one polygon ({coverage_path})")
    coverage = parse_shape(cov_els[0].get("shape", ""))
    return BuildingLayer(polys, ids, coverage, rejected, cell_size_m,
                         {"buildings_path": buildings_path, "coverage_path": coverage_path})

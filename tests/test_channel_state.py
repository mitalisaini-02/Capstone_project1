"""
Tests for building geometry (src/network/buildings.py) and channel-state classification
(src/network/channel_state.py) on a synthetic map with known geometry.

Map (metres): coverage = square (0,0)-(1000,1000); one building B1 = square (400,300)-(600,500);
road A along y = 100; road B along y = 600.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.network.buildings import (BuildingLayer, GridIndex, load_building_layer,
                                   segment_crosses_interior, validate_polygon)
from src.network.channel_state import (OUTPUT_COLUMNS, REASON_ENDPOINT, REASON_OUTSIDE,
                                       AllBuildingBlockedClassifier, AllBuildingClearClassifier,
                                       BuildingGeometryClassifier, RoadTopologyClassifier,
                                       UrbanLosProbability, assign_episode_states, osm_way_of)
from src.network.registry import load_link_model

NR = ("config/network/link_profiles_nr.json", "nr_sidelink_tr37885_v1")
NRH = ("config/network/link_profiles_nr.json", "nr_sidelink_tr37885_v1_highway")
V0 = ("config/marketplace/link_profiles.json", "abstract_sidelink_v0")


def sq(x0, y0, x1, y1):
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1} {x0},{y0}"


def write_poly(path, polys):
    with open(path, "w") as f:
        f.write("<additional>\n")
        for pid, typ, shape in polys:
            f.write(f'  <poly id="{pid}" type="{typ}" shape="{shape}"/>\n')
        f.write("</additional>\n")


class MapFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bpath = os.path.join(cls.tmp.name, "b.poly.xml")
        cls.cpath = os.path.join(cls.tmp.name, "c.poly.xml")
        write_poly(cls.bpath, [("B1", "building", sq(400, 300, 600, 500)),
                               ("park", "leisure", sq(0, 800, 1000, 900))])   # non-building: ignored
        write_poly(cls.cpath, [("cov", "coverage", sq(0, 0, 1000, 1000))])
        cls.layer = load_building_layer(cls.bpath, cls.cpath, cell_size_m=50.0)
        cls.nr = load_link_model(*NR)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pairs(self, rows, edges=None):
        df = pd.DataFrame(rows, columns=["pue_x", "pue_y", "cue_x", "cue_y"], dtype=float)
        df.insert(0, "time_s", np.arange(1.0, len(df) + 1))
        df.insert(1, "pue_id", 1)
        df.insert(2, "cue_id", 2)
        df.insert(3, "distance_m", np.hypot(df.pue_x - df.cue_x, df.pue_y - df.cue_y))
        if edges is not None:
            df["pue_edge"] = [e[0] for e in edges]
            df["cue_edge"] = [e[1] for e in edges]
        return df

    def G(self, **kw):
        return BuildingGeometryClassifier(self.layer, self.nr, **kw)


class TestGeometryClassification(MapFixture):
    def test_same_road_unobstructed(self):
        r = self.G().classify(self.pairs([(100, 100, 900, 100)])).iloc[0]
        self.assertEqual((r.classification_source, bool(r.fallback), bool(r.building_blocked)),
                         ("geometry", False, False))
        pl = float(self.nr.p_los(800.0))
        self.assertAlmostEqual(r.p_los, pl, places=12)
        self.assertAlmostEqual(r.p_nlosv, 1.0 - pl, places=12)
        self.assertEqual(r.p_nlos, 0.0)

    def test_different_roads_building_between(self):
        r = self.G().classify(self.pairs([(500, 100, 500, 600)])).iloc[0]
        self.assertTrue(r.building_blocked)
        self.assertEqual(int(r.n_buildings_crossed), 1)
        self.assertEqual((r.p_los, r.p_nlosv, r.p_nlos), (0.0, 0.0, 1.0))   # urban P_LOS NOT applied

    def test_different_roads_no_building_between(self):
        r = self.G().classify(self.pairs([(100, 100, 100, 600)])).iloc[0]
        self.assertFalse(r.building_blocked)
        self.assertEqual(r.p_nlos, 0.0)
        self.assertAlmostEqual(r.p_los, float(self.nr.p_los(500.0)), places=12)

    def test_intervening_vehicle_is_probabilistic_only(self):
        # The classifier takes no vehicle geometry: vehicle blockage enters only as P(NLOSv).
        r = self.G().classify(self.pairs([(100, 100, 300, 100)])).iloc[0]
        self.assertAlmostEqual(r.p_nlosv, 1.0 - float(self.nr.p_los(200.0)), places=12)
        self.assertGreater(r.p_nlosv, 0.0)

    def test_building_obstruction_takes_precedence(self):
        r = self.G().classify(self.pairs([(500, 100, 500, 600)])).iloc[0]
        self.assertEqual(r.p_nlosv, 0.0)      # no vehicle-blockage state on a building-blocked link
        d = np.array([r.distance_m])
        cap = self.nr.expected_capacity_given_states(d, 0.0, 0.0, 1.0)
        nlos_only = self.nr._expect(d, "NLOS", self.nr.capacity_from_snr_mbps)
        np.testing.assert_array_equal(cap, nlos_only)
        clear = self.nr.expected_capacity_given_states(d, 0.5, 0.5, 0.0)
        self.assertLess(float(cap[0]), float(clear[0]))

    def test_endpoint_inside_building_flagged_not_los(self):
        r = self.G().classify(self.pairs([(500, 400, 500, 900)])).iloc[0]
        self.assertTrue(r.fallback)
        self.assertEqual(r.fallback_reason, REASON_ENDPOINT)
        self.assertFalse(r.primary_eligible)
        self.assertTrue(pd.isna(r.building_blocked))
        self.assertEqual(r.p_los, 0.0)
        self.assertEqual(r.classification_source, "fallback:F2_all_building_blocked")

    def test_endpoint_on_building_boundary_flagged(self):
        r = self.G().classify(self.pairs([(500, 300, 500, 100)])).iloc[0]
        self.assertEqual(r.fallback_reason, REASON_ENDPOINT)

    def test_outside_coverage(self):
        r = self.G().classify(self.pairs([(100, 100, 1500, 100)])).iloc[0]
        self.assertEqual(r.fallback_reason, REASON_OUTSIDE)
        self.assertFalse(r.primary_eligible)
        self.assertEqual(r.p_los, 0.0)

    def test_configurable_fallback_is_labelled(self):
        r = self.G(fallback=AllBuildingClearClassifier(self.nr)).classify(
            self.pairs([(100, 100, 1500, 100)])).iloc[0]
        self.assertTrue(r.fallback)
        self.assertEqual(r.classification_source, "fallback:F3_all_building_clear")
        self.assertFalse(r.primary_eligible)

    def test_grazing_is_not_blocking(self):
        b1 = self.layer.polygons[0]
        self.assertFalse(segment_crosses_interior(b1, 300, 300, 700, 300))   # along bottom wall
        self.assertFalse(segment_crosses_interior(b1, 300, 400, 500, 200))   # touches corner (400,300)
        self.assertTrue(segment_crosses_interior(b1, 300, 400, 700, 400))    # through the middle

    def test_output_schema(self):
        out = self.G().classify(self.pairs([(100, 100, 900, 100), (500, 100, 500, 600)]))
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertTrue((out.scenario_label == "G").all())
        np.testing.assert_allclose(out.p_los + out.p_nlosv + out.p_nlos, 1.0)


class TestScenarioApplicability(MapFixture):
    def test_highway_model_rejected_on_urban_map(self):
        with self.assertRaises(ValueError):
            BuildingGeometryClassifier(self.layer, load_link_model(*NRH))
        with self.assertRaises(ValueError):
            UrbanLosProbability(self.nr, "highway")

    def test_model_without_los_probability_rejected(self):
        with self.assertRaises(TypeError):
            AllBuildingClearClassifier(load_link_model(*V0))

    def test_urban_plos_only_on_building_clear_rows(self):
        out = self.G().classify(self.pairs([(100, 100, 900, 100), (500, 100, 500, 600), (500, 400, 500, 900)]))
        blocked_or_fb = out.building_blocked.fillna(True).astype(bool)
        self.assertTrue((out.loc[blocked_or_fb, "p_los"] == 0).all())
        self.assertTrue((out.loc[blocked_or_fb, "p_nlosv"] == 0).all())

    def test_f1_f2_f3(self):
        p = self.pairs([(100, 100, 900, 100), (100, 100, 100, 600), (100, 100, 300, 100)],
                       edges=[("123#0", "-123#4"), ("123#0", "456#1"), (":j_0", "123#0")])
        f1 = RoadTopologyClassifier(self.nr).classify(p)
        self.assertGreater(f1.p_los.iloc[0], 0.0)                  # same OSM way
        self.assertEqual(f1.p_nlos.iloc[1], 1.0)                   # different way
        self.assertEqual(f1.classification_source.iloc[2], "road_topology_internal_edge")
        self.assertEqual(f1.p_nlos.iloc[2], 1.0)
        self.assertTrue((AllBuildingBlockedClassifier().classify(p).p_nlos == 1.0).all())
        f3 = AllBuildingClearClassifier(self.nr).classify(p)
        self.assertTrue((f3.p_nlos == 0.0).all())
        self.assertEqual(osm_way_of("-129545126#1"), "129545126")
        self.assertIsNone(osm_way_of(":10223231882_2"))


class TestInvalidGeometry(unittest.TestCase):
    def test_polygon_validation_reasons(self):
        self.assertIsNone(validate_polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float)))
        self.assertEqual(validate_polygon(np.array([[0, 0], [np.nan, 0], [1, 1], [0, 0]])), "non_finite_coordinates")
        self.assertEqual(validate_polygon(np.array([[0, 0], [1, 0], [0, 0]], float)), "too_few_vertices")
        self.assertEqual(validate_polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)), "not_closed")
        self.assertEqual(validate_polygon(np.array([[0, 0], [1, 0], [2, 0], [0, 0]], float)), "zero_area")
        self.assertEqual(validate_polygon(np.array([[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]], float)),
                         "self_intersecting")

    def test_files(self):
        with tempfile.TemporaryDirectory() as d:
            cov = os.path.join(d, "c.xml")
            write_poly(cov, [("cov", "coverage", sq(0, 0, 10, 10))])
            with self.assertRaises(FileNotFoundError):
                load_building_layer(os.path.join(d, "missing.xml"), cov)
            bad = os.path.join(d, "bad.xml")
            open(bad, "w").write("<additional><poly id='x'")
            with self.assertRaises(ValueError):
                load_building_layer(bad, cov)
            only_invalid = os.path.join(d, "inv.xml")
            write_poly(only_invalid, [("a", "building", "0,0 1,1 1,0 0,1 0,0"), ("b", "building", "0,0 abc")])
            with self.assertRaises(ValueError):
                load_building_layer(only_invalid, cov)
            mixed = os.path.join(d, "mixed.xml")
            write_poly(mixed, [("ok", "building", sq(1, 1, 2, 2)), ("bow", "building", "0,0 1,1 1,0 0,1 0,0"),
                               ("nan", "building", "0,0 nan,0 1,1 0,0")])
            layer = load_building_layer(mixed, cov)
            self.assertEqual(layer.n_buildings, 1)
            self.assertEqual(sorted(r for _, r in layer.rejected), ["non_finite_coordinates", "self_intersecting"])
            two = os.path.join(d, "two.xml")
            write_poly(two, [("a", "c", sq(0, 0, 1, 1)), ("b", "c", sq(0, 0, 2, 2))])
            with self.assertRaises(ValueError):
                load_building_layer(mixed, two)

    def test_non_finite_pair_positions_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b, c = os.path.join(d, "b.xml"), os.path.join(d, "c.xml")
            write_poly(b, [("B", "building", sq(4, 4, 6, 6))])
            write_poly(c, [("cov", "coverage", sq(0, 0, 10, 10))])
            g = BuildingGeometryClassifier(load_building_layer(b, c), load_link_model(*NR))
            bad = pd.DataFrame({"time_s": [1.0], "pue_id": [1], "cue_id": [2], "distance_m": [1.0],
                                "pue_x": [np.nan], "pue_y": [0.0], "cue_x": [1.0], "cue_y": [0.0]})
            with self.assertRaises(ValueError):
                g.classify(bad)


def _random_layer(seed, n=250):
    rng = np.random.default_rng(seed)
    polys = []
    for _ in range(n):
        cx, cy = rng.uniform(0, 2000, 2)
        k = rng.integers(3, 9)
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        rad = rng.uniform(5, 40, k)
        pts = np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]
        polys.append(np.vstack([pts, pts[:1]]))
    cov = np.array([[-100, -100], [2100, -100], [2100, 2100], [-100, 2100], [-100, -100]], float)
    return [p for p in polys if validate_polygon(p) is None], cov


class TestSpatialIndex(unittest.TestCase):
    def test_index_matches_brute_force_random(self):
        polys, cov = _random_layer(1)
        layer = BuildingLayer(polys, [str(i) for i in range(len(polys))], cov, cell_size_m=75.0)
        rng = np.random.default_rng(2)
        for _ in range(600):
            px, py, qx, qy = rng.uniform(0, 2000, 4)
            self.assertEqual(layer.blocking_buildings(px, py, qx, qy, use_index=True),
                             layer.blocking_buildings(px, py, qx, qy, use_index=False))

    def test_index_reproducible_and_cell_size_independent(self):
        polys, cov = _random_layer(3)
        ids = [str(i) for i in range(len(polys))]
        a = BuildingLayer(polys, ids, cov, cell_size_m=60.0)
        b = BuildingLayer(polys, ids, cov, cell_size_m=60.0)
        c = BuildingLayer(polys, ids, cov, cell_size_m=500.0)
        self.assertEqual(a.index.cells, b.index.cells)
        rng = np.random.default_rng(4)
        for _ in range(200):
            s = rng.uniform(0, 2000, 4)
            self.assertEqual(a.blocking_buildings(*s), c.blocking_buildings(*s))

    def test_grid_rejects_bad_cell(self):
        with self.assertRaises(ValueError):
            GridIndex(np.array([[0, 0, 1, 1]], float), 0.0)

    @unittest.skipUnless(os.path.exists("data/derived/geometry/buildings.poly.xml"), "geometry not built")
    def test_index_matches_brute_force_real_geometry(self):
        layer = load_building_layer("data/derived/geometry/buildings.poly.xml",
                                    "data/derived/geometry/coverage.poly.xml")
        self.assertEqual(layer.n_buildings, 3951)
        self.assertEqual(layer.rejected, [])
        rng = np.random.default_rng(5)
        bb = layer.bboxes
        for _ in range(150):
            x0, y0 = rng.uniform([500, 900], [5000, 4100])
            ang, L = rng.uniform(0, 2 * np.pi), rng.uniform(50, 1000)
            x1, y1 = x0 + L * np.cos(ang), y0 + L * np.sin(ang)
            idx = layer.blocking_buildings(x0, y0, x1, y1)
            # brute force over ALL buildings (bbox overlap is a necessary condition, not the grid)
            ov = np.where((bb[:, 0] <= max(x0, x1)) & (bb[:, 2] >= min(x0, x1)) &
                          (bb[:, 1] <= max(y0, y1)) & (bb[:, 3] >= min(y0, y1)))[0]
            brute = [int(k) for k in ov if segment_crosses_interior(layer.polygons[k], x0, y0, x1, y1)]
            self.assertEqual(idx, brute)


class TestEpisodeStates(MapFixture):
    def test_episode_rule(self):
        rows = [(100, 100, 100 + 20 * t, 100) for t in range(1, 21)]           # moving apart on road A
        p = self.pairs(rows)
        p.loc[10:, "time_s"] += 5                                                # gap -> second episode
        g = self.G().classify(p)
        a = assign_episode_states(g, "run_x", seed=11)
        b = assign_episode_states(g, "run_x", seed=11)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.episode_id.nunique(), 2)
        self.assertEqual(a.groupby("episode_id").episode_u.nunique().max(), 1)  # one U per episode
        self.assertNotEqual(a.episode_u.iloc[0], a.episode_u.iloc[-1])
        # with a fixed U and growing distance, LOS can only switch to NLOSv once (never back)
        for _, ep in a.groupby("episode_id"):
            s = (ep.sort_values("time_s").realized_state == "LOS").astype(int).to_numpy()
            self.assertTrue(np.all(np.diff(s) <= 0))
        f3 = assign_episode_states(AllBuildingClearClassifier(self.nr).classify(p), "run_x", seed=11)
        np.testing.assert_array_equal(a.episode_u.to_numpy(), f3.episode_u.to_numpy())   # shared across scenarios

    def test_blocked_seconds_always_nlos(self):
        g = self.G().classify(self.pairs([(500, 100, 500, 600)] * 3))
        a = assign_episode_states(g, "run_x", seed=1)
        self.assertTrue((a.realized_state == "NLOS").all())


class TestAdditiveCapacityInterface(unittest.TestCase):
    def test_matches_same_street_expectation(self):
        m = load_link_model(*NR)
        d = np.array([50.0, 300.0, 900.0])
        pl = m.p_los(d)
        np.testing.assert_allclose(m.expected_capacity_given_states(d, pl, 1 - pl, 0.0),
                                   m.expected_capacity_mbps(d), rtol=1e-12)
        np.testing.assert_allclose(m.outage_probability_given_states(d, 0.0, pl, 1 - pl, 0.0),
                                   m.outage_probability(d, 0.0), rtol=1e-12)
        with self.assertRaises(ValueError):
            m.expected_capacity_given_states(d, 0.5, 0.2, 0.0)


if __name__ == "__main__":
    unittest.main()

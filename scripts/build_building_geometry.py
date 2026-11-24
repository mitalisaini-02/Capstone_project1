#!/usr/bin/env python3
"""
Build building footprints in the EXACT SUMO network coordinate system with SUMO `polyconvert`.

Inputs (config/network/geometry.json): the SUMO net, the OSM file the net was built from,
and the polyconvert type map. Outputs (new directory, write-once):
  <output_dir>/buildings.poly.xml   polyconvert output (all polygon types; loader keeps buildings)
  <output_dir>/coverage.poly.xml    the OSM download bounding box projected the same way
                                    (the area where building data exist)
  <output_dir>/geometry_manifest.json  input hashes, tool version, counts, map scenario

The map scenario is inferred from road types: 'urban' if no motorway/trunk edges exist.
Building data are OpenStreetMap footprints and are INCOMPLETE (see docs/network_models.md).
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils.integrity import sha256_file  # noqa: E402


def osm_bounds(osm_path):
    for _, el in ET.iterparse(osm_path, events=("start",)):
        if el.tag == "bounds":
            return {k: float(v) for k, v in el.attrib.items() if k in ("minlat", "minlon", "maxlat", "maxlon")}
        if el.tag in ("node", "way"):
            break
    raise ValueError(f"{osm_path} has no <bounds> element; cannot define building coverage")


def net_road_types(net_path):
    counts = collections.Counter()
    for _, el in ET.iterparse(net_path, events=("end",)):
        if el.tag == "edge":
            if el.get("function") != "internal" and el.get("type"):
                counts[el.get("type")] += 1
            el.clear()
    return dict(counts)


def run_polyconvert(net, osm, typemap, out):
    cmd = ["polyconvert", "--net-file", net, "--osm-files", osm, "--type-file", typemap, "-o", out]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"polyconvert failed ({res.returncode}): {res.stderr[-2000:]}")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/network/geometry.json")
    cfg = json.load(open(ap.parse_args().config))
    out_dir = cfg["output_dir"]
    if os.path.exists(out_dir):
        raise SystemExit(f"ABORT: {out_dir} already exists (overwrite protection).")
    for k in ("net_file", "osm_file", "polyconvert_typemap"):
        if not os.path.exists(cfg[k]):
            raise SystemExit(f"ABORT: {k} not found: {cfg[k]}")
    version = subprocess.run(["polyconvert", "--version"], capture_output=True, text=True).stdout.splitlines()[0]

    os.makedirs(out_dir)
    bpath = os.path.join(out_dir, "buildings.poly.xml")
    cmd = run_polyconvert(cfg["net_file"], cfg["osm_file"], cfg["polyconvert_typemap"], bpath)

    b = osm_bounds(cfg["osm_file"])
    with tempfile.TemporaryDirectory() as tmp:
        bbox_osm = os.path.join(tmp, "bbox.osm")
        with open(bbox_osm, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6">\n'
                    f' <node id="-1" lat="{b["minlat"]}" lon="{b["minlon"]}"/>\n'
                    f' <node id="-2" lat="{b["minlat"]}" lon="{b["maxlon"]}"/>\n'
                    f' <node id="-3" lat="{b["maxlat"]}" lon="{b["maxlon"]}"/>\n'
                    f' <node id="-4" lat="{b["maxlat"]}" lon="{b["minlon"]}"/>\n'
                    ' <way id="-10"><nd ref="-1"/><nd ref="-2"/><nd ref="-3"/><nd ref="-4"/><nd ref="-1"/>'
                    '<tag k="building" v="yes"/></way>\n</osm>\n')
        tmp_out = os.path.join(tmp, "coverage.poly.xml")
        run_polyconvert(cfg["net_file"], bbox_osm, cfg["polyconvert_typemap"], tmp_out)
        shape = next(ET.parse(tmp_out).getroot().iter("poly")).get("shape")
    cpath = os.path.join(out_dir, "coverage.poly.xml")
    with open(cpath, "x") as f:
        f.write('<additional>\n    <poly id="building_data_coverage" type="coverage" '
                f'shape="{shape}"/>\n</additional>\n')

    types = collections.Counter(p.get("type") for p in ET.parse(bpath).getroot().iter("poly"))
    roads = net_road_types(cfg["net_file"])
    highway_edges = sum(v for k, v in roads.items() if k in cfg["highway_road_types"])
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "tool": version, "command": cmd,
        "inputs": {k: {"path": cfg[k], "sha256": sha256_file(cfg[k])}
                   for k in ("net_file", "osm_file", "polyconvert_typemap")},
        "osm_bounds_wgs84": b,
        "polygon_type_counts": dict(types),
        "building_polygons_raw": int(sum(types[t] for t in cfg["building_types"])),
        "net_road_type_counts": roads,
        "highway_scenario_edges": highway_edges,
        "map_scenario": "urban" if highway_edges == 0 else "mixed",
        "outputs": {"buildings": bpath, "coverage": cpath,
                    "sha256": {bpath: sha256_file(bpath), cpath: sha256_file(cpath)}},
        "data_caveat": "OpenStreetMap building footprints are incomplete; heights unavailable.",
    }
    with open(os.path.join(out_dir, "geometry_manifest.json"), "x") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: manifest[k] for k in ("tool", "building_polygons_raw", "map_scenario",
                                               "highway_scenario_edges")}, indent=2))


if __name__ == "__main__":
    main()

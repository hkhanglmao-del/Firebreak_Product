"""Regression tests against the English Firebreak R1 source variant.
These tests extract pure functions and never launch Streamlit or access live APIs.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path as FilePath

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, maximum_flow
from scipy.ndimage import binary_dilation, distance_transform_edt, zoom, label
from matplotlib.path import Path as MplPath

SOURCE_FILE = FilePath(__file__).resolve().parents[1] / "firebreak_hardened_r1_english.py"
TREE = ast.parse(SOURCE_FILE.read_text(encoding="utf-8"))
FUNCTION_NAMES = {
    "draw_thick_line",
    "_local_raster_inventory_signature",
    "_solve_cropped_mincut",
    "_load_local_geotiff_grid",
    "load_local_dem_grid",
    "load_local_worldcover_grid",
    "_conservative_access_time_p90",
    "fetch_api_osm_landscape",
    "_osm_api_map_xml_to_elements",
    "_build_single_ignition_mask",
    "compute_tobler_arrival_time",
    "compute_continuous_fire_arrival_field",
    "run_ca_step_vectorized",
    "verify_topological_separation",
}


def make_namespace(shape=(9, 9)):
    ns = {
        "np": np,
        "sp": sp,
        "dijkstra": dijkstra,
        "maximum_flow": maximum_flow,
        "binary_dilation": binary_dilation,
        "distance_transform_edt": distance_transform_edt,
        "zoom": zoom,
        "label": label,
        "Path": MplPath,
        "FilePath": FilePath,
        "GRID_SHAPE": shape,
        "LAT_SPAN_DEG": 0.15,
        "LON_SPAN_DEG": 0.15,
        "NOMINAL_PIXEL_M": 30.0,
        "CA_STEP_TO_HOURS": 0.5,
        "STATE_EMPTY": 0,
        "STATE_TREE": 1,
        "STATE_BURNING": 2,
        "STATE_BURNED": 3,
        "STATE_FIREBREAK": 4,
        "_WEYL_OFFSETS": (0.0, 0.6180339, 0.2360679, 0.8541019,
                           0.4721359, 0.0901699, 0.7082039, 0.3262379),
        "OVERPASS_MIRRORS": ["https://overpass.test/api/interpreter"],
        "LOCAL_DEM_DIR": FilePath("/unused-dem"),
        "LOCAL_WORLDCOVER_DIR": FilePath("/unused-worldcover"),
        "time": __import__("time"),
    }
    selected = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTION_NAMES]
    missing = FUNCTION_NAMES - {n.name for n in selected}
    assert not missing, f"Could not extract functions: {missing}"
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(SOURCE_FILE), "exec"), ns)
    return ns


def env_for(shape, water=None, buildings=None, roads=None):
    fuel = np.ones(shape, dtype=np.float32)
    water = np.zeros(shape, dtype=bool) if water is None else water.copy()
    buildings = np.zeros(shape, dtype=bool) if buildings is None else buildings.copy()
    roads = np.zeros(shape, dtype=bool) if roads is None else roads.copy()
    fuel[water | buildings | roads] = 0.0
    return {
        "elevation": np.zeros(shape, dtype=np.float32),
        "fuel_matrix": fuel,
        "water_mask": water,
        "building_mask": buildings,
        "road_mask": roads,
        "dy_m": 30.0,
        "dx_m": 30.0,
    }


def test_worldcover_raster_is_reprojected_and_water_class_is_masked(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    ns = make_namespace((9, 9))
    ns["LOCAL_WORLDCOVER_DIR"] = tmp_path
    arr = np.full((9, 9), 10, dtype=np.uint8)
    arr[4, 4] = 80
    transform = from_bounds(-0.075, -0.075, 0.075, 0.075, 9, 9)
    tif = tmp_path / "ESA_WorldCover_10m_2021_v200_test_Map.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=9, width=9, count=1,
                       dtype=arr.dtype, crs="EPSG:4326", transform=transform, nodata=0) as ds:
        ds.write(arr, 1)
    # A wrong product in the same folder must not silently override the Map layer.
    quality = np.full((9, 9), 30, dtype=np.uint8)
    wrong_tif = tmp_path / "ESA_WorldCover_10m_2021_v200_test_InputQuality.tif"
    with rasterio.open(wrong_tif, "w", driver="GTiff", height=9, width=9, count=1,
                       dtype=quality.dtype, crs="EPSG:4326", transform=transform, nodata=0) as ds:
        ds.write(quality, 1)

    classes, water, source, coverage = ns["load_local_worldcover_grid"](0.0, 0.0, np.zeros((9, 9), bool))
    assert classes.shape == (9, 9)
    assert classes[4, 4] == 80
    assert water[4, 4]
    assert "ESA WorldCover 2021 v200" in source
    assert coverage == pytest.approx(100.0)


def test_continuous_fire_arrival_does_not_route_through_water_stripe():
    shape = (9, 9)
    ns = make_namespace(shape)
    water = np.zeros(shape, dtype=bool)
    water[:, 4] = True
    env = env_for(shape, water=water)
    fire = np.zeros(shape, dtype=bool)
    fire[4, 1] = True
    arrival = ns["compute_continuous_fire_arrival_field"](
        fire, env, wind_speed=0.0, wind_deg=0.0, humidity=30.0, prcp=0.0,
        temp=20.0, p_base=0.24, allow_spotting=False
    )
    assert np.isfinite(arrival[4, 2])
    assert np.isinf(arrival[:, 4]).all()
    assert np.isinf(arrival[:, 5:]).all()


def test_ca_does_not_cross_full_height_water_barrier():
    shape = (9, 9)
    ns = make_namespace(shape)
    water = np.zeros(shape, dtype=bool)
    water[:, 4] = True
    env = env_for(shape, water=water)
    grid = np.full(shape, ns["STATE_TREE"], dtype=np.uint8)
    grid[water] = ns["STATE_EMPTY"]
    grid[4, 1] = ns["STATE_BURNING"]
    burn_time = np.zeros(shape, dtype=np.int32)
    rng = np.random.default_rng(2026)
    for _ in range(6):
        grid, burn_time = ns["run_ca_step_vectorized"](
            grid, burn_time, env, 0.0, 0.0, 30.0, 0.0, 20.0,
            5.0, 100, rng, active_barrier_mask=None, allow_spotting=False
        )
    assert not np.any((grid[:, 4] == ns["STATE_BURNING"]) | (grid[:, 4] == ns["STATE_BURNED"]))
    assert np.all(grid[:, 5:] == ns["STATE_TREE"])


def test_ca_forbids_diagonal_corner_cut_between_nonburnable_cells():
    shape = (5, 5)
    ns = make_namespace(shape)
    water = np.zeros(shape, dtype=bool)
    water[2, 3] = True
    water[3, 2] = True
    env = env_for(shape, water=water)
    grid = np.full(shape, ns["STATE_TREE"], dtype=np.uint8)
    grid[water] = ns["STATE_EMPTY"]
    grid[2, 2] = ns["STATE_BURNING"]
    burn_time = np.zeros(shape, dtype=np.int32)
    grid2, _ = ns["run_ca_step_vectorized"](
        grid, burn_time, env, 0.0, 0.0, 30.0, 0.0, 20.0,
        5.0, 100, np.random.default_rng(4), active_barrier_mask=None, allow_spotting=False
    )
    assert grid2[3, 3] == ns["STATE_TREE"]


def test_tobler_access_does_not_cross_water_stripe():
    shape = (9, 9)
    ns = make_namespace(shape)
    water = np.zeros(shape, dtype=bool)
    water[:, 4] = True
    roads = np.zeros(shape, dtype=bool)
    roads[:, 1] = True
    elevation = np.zeros(shape, dtype=np.float32)
    slope = np.zeros(shape, dtype=np.float32)
    arrival, _ = ns["compute_tobler_arrival_time"](
        elevation, slope, roads, 30.0, 30.0, water_mask=water,
        building_mask=np.zeros(shape, dtype=bool)
    )
    assert np.isfinite(arrival[:, 2:4]).all()
    assert np.isinf(arrival[:, 4:]).all()


def test_osm_closed_water_polygon_is_returned_as_water_mask(monkeypatch):
    shape = (41, 41)
    ns = make_namespace(shape)
    # OSM way enclosing a small square lake at the center of the 0.15-degree AOI.
    coords = [(-0.02, -0.02), (-0.02, 0.02), (0.02, 0.02), (0.02, -0.02), (-0.02, -0.02)]
    geometry = [{"lat": lat, "lon": lon} for lat, lon in coords]
    payload = {"elements": [{"type": "way", "id": 1, "tags": {"natural": "water"}, "geometry": geometry}]}

    class Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload

    fake_requests = types.SimpleNamespace(post=lambda *args, **kwargs: Response())
    ns["requests"] = fake_requests
    road, building, water, *_ = ns["fetch_api_osm_landscape"](0.0, 0.0)
    assert water.shape == shape
    assert water[20, 20]
    assert water.sum() > 0
    assert not road.any()
    assert not building.any()



@pytest.mark.parametrize("waterway", ["river", "canal", "stream", "riverbank"])
def test_osm_open_waterway_is_returned_as_water_mask(waterway):
    shape = (41, 41)
    ns = make_namespace(shape)
    geometry = [
        {"lat": 0.03, "lon": -0.03},
        {"lat": 0.00, "lon": 0.00},
        {"lat": -0.03, "lon": 0.03},
    ]
    payload = {
        "elements": [{
            "type": "way", "id": 12,
            "tags": {"waterway": waterway},
            "geometry": geometry,
        }]
    }

    class Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload

    ns["requests"] = types.SimpleNamespace(post=lambda *args, **kwargs: Response())
    road, building, water, *_ = ns["fetch_api_osm_landscape"](0.0, 0.0)
    assert water.any(), f"OSM waterway={waterway!r} was not rasterized"
    assert not road.any()
    assert not building.any()


def test_topology_verifier_does_not_cut_diagonal_corner_between_blocked_cells():
    shape = (5, 5)
    ns = make_namespace(shape)
    source = np.zeros(shape, dtype=bool)
    source[1, 1] = True
    target = np.zeros(shape, dtype=bool)
    target[2, 2] = True
    # Only source and target are traversable. The direct diagonal must be rejected
    # because both orthogonal side cells are blocked.
    nonburnable = np.ones(shape, dtype=bool)
    nonburnable[1, 1] = False

    separated, leaked = ns["verify_topological_separation"](
        source, target, np.zeros(shape, bool), np.zeros(shape, bool),
        nonburnable_mask=nonburnable
    )
    assert separated
    assert leaked == 0


def test_local_raster_inventory_signature_changes_when_tiles_change(tmp_path):
    ns = make_namespace((5, 5))
    dem_dir = tmp_path / "dem"
    wc_dir = tmp_path / "worldcover"
    dem_dir.mkdir()
    wc_dir.mkdir()
    ns["LOCAL_DEM_DIR"] = dem_dir
    ns["LOCAL_WORLDCOVER_DIR"] = wc_dir
    signature_empty = ns["_local_raster_inventory_signature"]()

    dem_tile = dem_dir / "sapa.tif"
    dem_tile.write_bytes(b"dem-v1")
    signature_dem = ns["_local_raster_inventory_signature"]()
    assert signature_dem != signature_empty
    assert ns["_local_raster_inventory_signature"]() == signature_dem

    wc_tile = wc_dir / "worldcover.tif"
    wc_tile.write_bytes(b"wc-v1")
    signature_both = ns["_local_raster_inventory_signature"]()
    assert signature_both != signature_dem

    dem_tile.write_bytes(b"dem-v2-content")
    signature_updated = ns["_local_raster_inventory_signature"]()
    assert signature_updated != signature_both


def test_cropped_mincut_returns_a_valid_source_side_partition():
    shape = (9, 9)
    ns = make_namespace(shape)
    costs = np.ones(shape, dtype=np.float32)
    # Make a cheap vertical cut so the max-flow/min-cut primitive has a small,
    # deterministic synthetic case independent of real GIS data.
    costs[:, 4] = 0.25
    source = np.zeros(shape, dtype=bool)
    source[4, 2] = True
    sink = np.zeros(shape, dtype=bool)
    sink[0, :] = True
    sink[-1, :] = True
    sink[:, 0] = True
    sink[:, -1] = True

    partition, sink_reached = ns["_solve_cropped_mincut"](
        costs, source, sink, dx_m=30.0, dy_m=30.0,
        diag_scale=float(np.hypot(30.0, 30.0) / 30.0)
    )

    assert sink_reached is False
    assert partition.shape == shape
    assert partition[source].all()
    assert not partition[sink].any()
    assert partition.any()


def test_topology_verifier_treats_water_as_nonburnable():
    shape = (9, 9)
    ns = make_namespace(shape)
    source = np.zeros(shape, dtype=bool); source[4, 1] = True
    target = np.zeros(shape, dtype=bool); target[4, 7] = True
    water = np.zeros(shape, dtype=bool); water[:, 4] = True
    no_route, leaked = ns["verify_topological_separation"](
        source, target, np.zeros(shape, bool), np.zeros(shape, bool), nonburnable_mask=water
    )
    assert no_route
    assert leaked == 0


def test_access_deadline_uses_p90_and_fails_if_any_work_pixel_unreachable():
    ns = make_namespace((5, 5))
    calc = ns["_conservative_access_time_p90"]
    p90, all_accessible = calc(np.arange(1.0, 11.0))
    assert p90 == pytest.approx(9.1)
    assert all_accessible is True
    p90_bad, all_accessible_bad = calc(np.array([1.0, 2.0, np.inf]))
    assert np.isinf(p90_bad)
    assert all_accessible_bad is False
    p90_empty, all_accessible_empty = calc(np.array([]))
    assert p90_empty == 0.0
    assert all_accessible_empty is True


def test_late_barrier_does_not_erase_existing_fire():
    shape = (7, 7)
    ns = make_namespace(shape)
    env = env_for(shape)
    grid = np.full(shape, ns["STATE_TREE"], dtype=np.uint8)
    grid[3, 3] = ns["STATE_BURNING"]
    burn_time = np.ones(shape, dtype=np.int32)
    barrier = np.zeros(shape, dtype=bool)
    barrier[3, 3] = True
    next_grid, _ = ns["run_ca_step_vectorized"](
        grid, burn_time, env, 0.0, 0.0, 30.0, 0.0, 20.0,
        0.0, 100, np.random.default_rng(7), active_barrier_mask=barrier,
        allow_spotting=False
    )
    assert next_grid[3, 3] in (ns["STATE_BURNING"], ns["STATE_BURNED"])
    assert next_grid[3, 3] != ns["STATE_FIREBREAK"]


def test_ignition_uncertainty_candidates_do_not_create_simultaneous_fires():
    shape = (21, 21)
    ns = make_namespace(shape)
    env = env_for(shape)
    single, single_seed = ns["_build_single_ignition_mask"](
        [(10, 10)], 10, 10, 2, env
    )
    alternatives, chosen_seed = ns["_build_single_ignition_mask"](
        [(10, 10), (6, 6), (14, 14), (10, 14)], 10, 10, 2, env
    )
    assert single_seed == chosen_seed == (10, 10)
    assert np.array_equal(single, alternatives)
    assert int(single.sum()) == 13  # one radius-2 disk, not a union of hypothetical origins


def test_ignition_mask_excludes_water_buildings_roads_and_blocked_cells():
    shape = (21, 21)
    ns = make_namespace(shape)
    env = env_for(shape)
    env["water_mask"][10, 11] = True
    env["building_mask"][11, 10] = True
    env["road_mask"][9, 10] = True
    blocked = np.zeros(shape, dtype=bool)
    blocked[10, 9] = True
    ignition, seed = ns["_build_single_ignition_mask"](
        [(10, 10)], 10, 10, 2, env, blocked_mask=blocked
    )
    assert seed == (10, 10)
    assert ignition[10, 10]
    assert not ignition[10, 11]
    assert not ignition[11, 10]
    assert not ignition[9, 10]
    assert not ignition[10, 9]


def test_osm_map_xml_parser_retains_road_and_building_ways():
    ns = make_namespace((9, 9))
    payload = b'''<osm version="0.6">
      <node id="1" lat="0.01" lon="-0.01" />
      <node id="2" lat="0.01" lon="0.01" />
      <node id="3" lat="-0.01" lon="0.01" />
      <node id="4" lat="-0.01" lon="-0.01" />
      <way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/></way>
      <way id="20"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="building" v="yes"/></way>
      <way id="30"><nd ref="1"/><nd ref="2"/><tag k="landuse" v="forest"/></way>
    </osm>'''
    parsed = ns["_osm_api_map_xml_to_elements"](payload)
    highway = next(el for el in parsed["elements"] if el.get("tags", {}).get("highway") == "primary")
    building = next(el for el in parsed["elements"] if el.get("tags", {}).get("building") == "yes")
    assert len(highway["geometry"]) == 2
    assert len(building["geometry"]) == 5
    assert not any(el.get("tags", {}).get("landuse") == "forest" for el in parsed["elements"])


def test_osm_official_map_endpoint_fallback_rasterizes_roads_without_overpass():
    ns = make_namespace((41, 41))
    payload = b'''<osm version="0.6">
      <node id="1" lat="0.03" lon="-0.03" />
      <node id="2" lat="0.00" lon="0.00" />
      <node id="3" lat="-0.03" lon="0.03" />
      <way id="12"><nd ref="1"/><nd ref="2"/><nd ref="3"/><tag k="highway" v="primary"/></way>
    </osm>'''

    class Response:
        content = payload
        status_code = 200
        def raise_for_status(self):
            pass

    class FakeRequests:
        @staticmethod
        def get(*args, **kwargs):
            return Response()
        @staticmethod
        def post(*args, **kwargs):
            raise AssertionError("Overpass should not be called when the official OSM API succeeds")

    ns["requests"] = FakeRequests
    road, building, water, road_count, *_ = ns["fetch_api_osm_landscape"](0.0, 0.0)
    assert road.any()
    assert road_count == 1
    assert not building.any()
    assert not water.any()

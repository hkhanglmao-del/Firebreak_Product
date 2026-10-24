import gc
import time
from pathlib import Path as FilePath
import requests
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, dijkstra
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt, zoom, label
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.path import Path
import streamlit as st

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION AND SYSTEM CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="Wildfire spread simulation and firebreak planning",
    layout="wide",
    initial_sidebar_state="expanded"
)

GRID_SHAPE = (540, 540)
# Optional local research data. See HUONG_DAN_DU_LIEU_VA_KIEM_THU.md for expected files and provenance.
APP_DIR = FilePath(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
LOCAL_DEM_DIR = DATA_DIR / "dem"
LOCAL_WORLDCOVER_DIR = DATA_DIR / "worldcover"
LAT_SPAN_DEG = 0.15          # Khung không gian vĩ độ: 0.15 độ (~16.6 km)
LON_SPAN_DEG = 0.15          # Khung không gian kinh độ: 0.15 độ
NOMINAL_PIXEL_M = 30.0       # Kích thước pixel danh nghĩa: 30m
CA_STEP_TO_HOURS = 0.5       # 1 bước thời gian mô phỏng (timestep) = 0.5 giờ

# Định mức cơ sở tham chiếu NWCG Fireline Handbook (PMS 410-1):
# Tốc độ phát quang thủ công trên địa hình bằng (0°): ~12m dài / người / giờ
# -> Để phát quang 1 đoạn 30m trên đất bằng cần ~2.5 giờ công (Person-Hours)
BASE_PERSON_HOURS_PER_30M = 2.5

STATE_EMPTY     = 0  # Đường giao thông / Mặt nước / Không có thực bì
STATE_TREE      = 1  # Rừng / Thực bì chưa cháy
STATE_BURNING   = 2  # Đang cháy
STATE_BURNED    = 3  # Đã cháy rụi
STATE_FIREBREAK = 4  # Tuyến đường băng đề xuất trong mô hình

REGION_PRESETS = {
    "Hoang Lien Son – Sa Pa (Lao Cai)": {
        "lat": 22.3750, "lon": 103.8250, "ign_row": 270, "ign_col": 270
    },
    "Da Lat Pine Forest (Lam Dong)": {
        "lat": 11.9404, "lon": 108.4583, "ign_row": 240, "ign_col": 300
    },
    "Ba Vi National Park (Hanoi)": {
        "lat": 21.0600, "lon": 105.3600, "ign_row": 280, "ign_col": 260
    },
    "Hong Linh Mountain (Ha Tinh)": {
        "lat": 18.5200, "lon": 105.7200, "ign_row": 320, "ign_col": 220
    },
    "Son Tra Peninsula (Da Nang)": {
        "lat": 16.1150, "lon": 108.2650, "ign_row": 260, "ign_col": 270
    },
    "Custom coordinates (enter latitude/longitude below)": {
        "lat": 21.3000, "lon": 104.1000, "ign_row": 270, "ign_col": 270
    }
}

OVERPASS_MIRRORS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]


# ==============================================================================
# LAYER 1: GEOGRAPHIC DATA, LOCAL GEOTIFF PRIORITY, AND PERSISTENT CACHE
# ==============================================================================
def compute_projected_pixel_scales(lat_center):
    """Tính kích thước thực (mét) của 1 pixel theo vĩ độ trên hệ quy chiếu WGS84."""
    meters_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * np.radians(lat_center))
    meters_per_deg_lon = 111412.84 * np.cos(np.radians(lat_center))
    dy_m = (LAT_SPAN_DEG * meters_per_deg_lat) / GRID_SHAPE[0]
    dx_m = (LON_SPAN_DEG * meters_per_deg_lon) / GRID_SHAPE[1]
    pixel_area_ha = (dy_m * dx_m) / 10000.0
    return float(dy_m), float(dx_m), float(pixel_area_ha)


def _load_local_geotiff_grid(folder, lat_center, lon_center, categorical=False, required_filename_tokens=None):
    """Reproject local GeoTIFF tiles covering the requested AOI to GRID_SHAPE.

    Returns (grid, valid_mask, files_used). Missing local files are not an error;
    partial coverage is returned explicitly so callers can fail closed where needed.
    The source tiles are never modified.
    """
    folder = FilePath(folder)
    paths = sorted(set(folder.glob("*.tif")) | set(folder.glob("*.tiff"))) if folder.exists() else []
    if required_filename_tokens:
        tokens = tuple(str(token).casefold() for token in required_filename_tokens)
        paths = [path for path in paths if all(token in path.name.casefold() for token in tokens)]
    if not paths:
        return None, None, []

    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject, transform_bounds
    except ImportError as exc:
        raise RuntimeError(
            "GeoTIFF files were found, but rasterio is missing. Install it with: pip install rasterio"
        ) from exc

    south, north = lat_center - LAT_SPAN_DEG / 2.0, lat_center + LAT_SPAN_DEG / 2.0
    west, east = lon_center - LON_SPAN_DEG / 2.0, lon_center + LON_SPAN_DEG / 2.0
    rows, cols = GRID_SHAPE
    dst_transform = from_bounds(west, south, east, north, cols, rows)
    dtype = np.uint16 if categorical else np.float32
    fill_value = 0 if categorical else np.nan
    out = np.full(GRID_SHAPE, fill_value, dtype=dtype)
    valid = np.zeros(GRID_SHAPE, dtype=bool)
    used = []

    for path in paths:
        try:
            with rasterio.open(path) as ds:
                if ds.count < 1 or ds.crs is None:
                    continue
                b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=11)
                if b[2] <= west or b[0] >= east or b[3] <= south or b[1] >= north:
                    continue
                tmp = np.full(GRID_SHAPE, fill_value, dtype=dtype)
                reproject(
                    source=rasterio.band(ds, 1),
                    destination=tmp,
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    src_nodata=ds.nodata,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:4326",
                    dst_nodata=fill_value,
                    resampling=Resampling.nearest if categorical else Resampling.bilinear,
                )
                tmp_valid = (tmp != 0) if categorical else np.isfinite(tmp)
                take = tmp_valid & (~valid)
                out[take] = tmp[take]
                valid |= tmp_valid
                used.append(path.name)
        except Exception as exc:
            raise RuntimeError(f"Could not read GeoTIFF {path.name}: {exc}") from exc

    if not used or not np.any(valid):
        return None, None, []
    return out, valid, used


def load_local_dem_grid(lat_center, lon_center, water_mask):
    """Load local high-resolution DEM tiles; only water may have nodata gaps."""
    dem, valid, used = _load_local_geotiff_grid(
        LOCAL_DEM_DIR, lat_center, lon_center, categorical=False
    )
    if dem is None:
        return None, None
    missing_land = (~valid) & (~np.asarray(water_mask, dtype=bool))
    if np.any(missing_land):
        pct = 100.0 * float(np.mean(valid))
        raise RuntimeError(
            f"The local DEM covers only {pct:.1f}% of the grid and has missing cells outside water. "
            f"Add the missing tiles covering the full map extent to {LOCAL_DEM_DIR}."
        )
    if np.any(~valid):
        # DEM nodata over water is not terrain evidence; fill only for smooth display/gradient
        # calculations. Water itself is non-burnable and excluded from fire-travel graphs.
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        dem[~valid] = dem[tuple(nearest[:, ~valid])]
    return dem.astype(np.float32), "Local GeoTIFF: " + ", ".join(used)


def load_local_worldcover_grid(lat_center, lon_center, water_mask):
    """Load ESA WorldCover categorical codes; reject incomplete land coverage."""
    classes, valid, used = _load_local_geotiff_grid(
        LOCAL_WORLDCOVER_DIR, lat_center, lon_center, categorical=True,
        required_filename_tokens=("2021", "v200", "map")
    )
    if classes is None:
        if LOCAL_WORLDCOVER_DIR.exists() and any(
            path.suffix.casefold() in {".tif", ".tiff"} for path in LOCAL_WORLDCOVER_DIR.iterdir()
        ):
            raise RuntimeError(
                f"No WorldCover 2021 v200 Map tile covering the map area was found in {LOCAL_WORLDCOVER_DIR}. "
                "Place only GeoTIFF files for the correct Map layer and version in this folder; do not use InputQuality."
            )
        return None, None, "", 0.0
    water = np.asarray(water_mask, dtype=bool) | (classes == 80)
    land_domain = ~water
    coverage = float(np.mean(valid[land_domain])) * 100.0 if np.any(land_domain) else 100.0
    if coverage < 99.0:
        raise RuntimeError(
            f"ESA WorldCover covers only {coverage:.1f}% of cells not identified as water. "
            f"Add all relevant 10 m tiles to {LOCAL_WORLDCOVER_DIR}; "
            "NoData cells must not silently be treated as forest."
        )
    return classes.astype(np.uint16), water, "ESA WorldCover 2021 v200: " + ", ".join(used), coverage


def _largest_connected_component(mask, reference_mask=None):
    """Return one 8-connected component, preferring overlap with a reference mask.

    This deliberately refuses to display disconnected, satellite route fragments as
    if they were one coherent firebreak. The caller still must verify topology.
    """
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count <= 1:
        return mask.copy()

    areas = np.bincount(components.ravel(), minlength=count + 1)
    areas[0] = 0
    chosen = 0
    if reference_mask is not None:
        ref = np.asarray(reference_mask, dtype=bool) & mask
        overlaps = np.bincount(components[ref].ravel(), minlength=count + 1) if np.any(ref) else np.zeros(count + 1, dtype=np.int64)
        overlaps[0] = 0
        if overlaps.max(initial=0) > 0:
            candidates = np.flatnonzero(overlaps == overlaps.max())
            chosen = int(candidates[np.argmax(areas[candidates])])
    if chosen == 0:
        chosen = int(np.argmax(areas[1:]) + 1)
    return components == chosen


def _effective_line_workforce(total_manpower, squad_size, work_pixels, squad_factor=1.0):
    """Estimate workers that can be productively assigned to this line at once."""
    total = max(0.0, float(total_manpower))
    front_capacity = max(0.0, float(work_pixels)) * max(1.0, float(squad_size))
    effective = min(total, front_capacity) + 0.65 * max(0.0, total - front_capacity)
    return max(1.0, effective * max(0.1, float(squad_factor)))


def _compute_target_standoff_px(
    total_manpower, squad_size, p_base, wind_speed, temp, humidity, prcp,
    allow_spotting, uncertainty_px, is_smoke_report, sim_hours
):
    """Deterministic personnel/intensity-dependent initial stand-off for route search."""
    fire_intensity = (
        (float(p_base) / 0.24)
        * float(np.exp(
            0.035 * (float(wind_speed) - 5.0)
            + 0.015 * (float(temp) - 20.0)
            - 0.008 * (float(humidity) - 55.0)
            - 0.5 * float(prcp)
        ))
        * (1.12 if allow_spotting else 1.0)
        * (1.0 + 0.025 * float(uncertainty_px)
           + (0.08 if is_smoke_report else 0.0)
           + 0.01 * float(sim_hours))
    )
    crew_power = (
        (max(1.0, float(total_manpower)) / 700.0) ** 0.78
        * (max(1.0, float(squad_size)) / 20.0) ** 0.20
    )
    return int(np.clip(round(3.6 * fire_intensity / max(0.18, crew_power)), 1, 16))


def draw_thick_line(mask, r0, c0, r1, c1, thickness=2):
    rows, cols = mask.shape
    num_pts = int(max(abs(r1 - r0), abs(c1 - c0)) * 2) + 2
    rs = np.clip(np.round(np.linspace(r0, r1, num_pts)).astype(int), 0, rows - 1)
    cs = np.clip(np.round(np.linspace(c0, c1, num_pts)).astype(int), 0, cols - 1)
    for dr in range(-thickness, thickness + 1):
        for dc in range(-thickness, thickness + 1):
            if dr**2 + dc**2 <= thickness**2:
                rr = np.clip(rs + dr, 0, rows - 1)
                cc = np.clip(cs + dc, 0, cols - 1)
                mask[rr, cc] = True


def fetch_api_dem_grid(lat_center, lon_center, grid_n=20):
    """Fetch real SRTM 90 m elevation samples from OpenTopoData, in memory.

    The public OpenTopoData endpoint allows up to 100 locations per request and
    approximately one request per second. A 20x20 grid is therefore sent in
    four batches. Missing elevations are rejected rather than replaced with
    synthetic terrain.
    """
    south, north = lat_center - LAT_SPAN_DEG / 2.0, lat_center + LAT_SPAN_DEG / 2.0
    west, east = lon_center - LON_SPAN_DEG / 2.0, lon_center + LON_SPAN_DEG / 2.0

    lats = np.linspace(north, south, grid_n)
    lons = np.linspace(west, east, grid_n)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()
    elev_results = []

    url = "https://api.opentopodata.org/v1/srtm90m"
    headers = {"User-Agent": "FirebreakOptimizationResearch/2.0 (Hanoi-VN)"}
    batch_size = 100

    with requests.Session() as session:
        for i in range(0, len(flat_lats), batch_size):
            batch_lats = flat_lats[i:i + batch_size]
            batch_lons = flat_lons[i:i + batch_size]
            locations = "|".join(
                f"{lat:.5f},{lon:.5f}"
                for lat, lon in zip(batch_lats, batch_lons)
            )

            try:
                resp = session.post(
                    url,
                    data={"locations": locations, "interpolation": "bilinear"},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"OpenTopoData SRTM 90m API request failed (batch {i // batch_size + 1}): {exc}"
                ) from exc
            except ValueError as exc:
                raise RuntimeError("OpenTopoData API returned invalid JSON.") from exc

            if payload.get("status") != "OK":
                raise RuntimeError(f"OpenTopoData API error: {payload}")

            results = payload.get("results", [])
            if len(results) != len(batch_lats):
                raise RuntimeError(
                    "OpenTopoData returned an unexpected number of elevation samples."
                )
            batch_elev = [item.get("elevation") for item in results]
            if any(value is None for value in batch_elev):
                raise RuntimeError(
                    "OpenTopoData returned missing elevations; refusing to synthesize terrain."
                )
            elev_results.extend(batch_elev)

            if i + batch_size < len(flat_lats):
                time.sleep(1.1)

    coarse_dem = np.asarray(elev_results, dtype=np.float32).reshape((grid_n, grid_n))
    scale_r = GRID_SHAPE[0] / grid_n
    scale_c = GRID_SHAPE[1] / grid_n
    dem_540 = zoom(coarse_dem, (scale_r, scale_c), order=3).astype(np.float32)
    return dem_540[:GRID_SHAPE[0], :GRID_SHAPE[1]]


def fetch_api_osm_landscape(lat_center, lon_center):
    """
    Truy vấn trực tiếp 100% Đường giao thông & Khu dân cư từ OpenStreetMap Overpass API.
    Sử dụng cú pháp 'out geom;' siêu nhẹ trên RAM, không lưu file xuống ổ cứng.
    """
    south, north = lat_center - LAT_SPAN_DEG / 2.0, lat_center + LAT_SPAN_DEG / 2.0
    west, east   = lon_center - LON_SPAN_DEG / 2.0, lon_center + LON_SPAN_DEG / 2.0
    rows, cols   = GRID_SHAPE

    query = f"""
    [out:json][timeout:25];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track"]({south},{west},{north},{east});
      way["landuse"~"residential|commercial|industrial|retail|village|farmyard"]({south},{west},{north},{east});
      way["natural"="water"]({south},{west},{north},{east});
      way["natural"="coastline"]({south},{west},{north},{east});
      way["waterway"~"riverbank|river|canal|stream"]({south},{west},{north},{east});
      way["landuse"="reservoir"]({south},{west},{north},{east});
      way["water"~"lake|reservoir|pond|basin"]({south},{west},{north},{east});
      node["place"~"city|town|village|hamlet|isolated_dwelling|suburb"]({south},{west},{north},{east});
    );
    out geom;
    """

    headers = {
        "User-Agent": "FirebreakOptimizationResearch/2.0 (HighSchool-Academic-Research; Hanoi-VN)",
        "Accept-Encoding": "gzip, deflate"
    }

    data = None
    used_mirror = ""
    last_err = None

    for mirror in OVERPASS_MIRRORS:
        try:
            resp = requests.post(mirror, data={"data": query}, headers=headers, timeout=20)
            if resp.status_code == 429:
                last_err = f"429 Too Many Requests ({mirror})"
                time.sleep(0.4)
                continue
            resp.raise_for_status()
            data = resp.json()
            used_mirror = mirror.split("//")[1].split("/")[0]
            break
        except Exception as e:
            last_err = e
            continue

    if data is None:
        raise RuntimeError(f"All Overpass API servers are currently unavailable: {last_err}")

    road_mask = np.zeros(GRID_SHAPE, dtype=bool)
    building_mask = np.zeros(GRID_SHAPE, dtype=bool)
    water_mask = np.zeros(GRID_SHAPE, dtype=bool)
    coastline_mask = np.zeros(GRID_SHAPE, dtype=bool)
    coastline_water_seeds = np.zeros(GRID_SHAPE, dtype=bool)

    def latlon_to_rc(lat, lon):
        """Map WGS84 coordinates to continuous row/column coordinates; do not clamp."""
        r = (north - float(lat)) / (north - south) * (rows - 1)
        c = (float(lon) - west) / (east - west) * (cols - 1)
        return float(r), float(c)

    def clip_segment_to_grid(p0, p1):
        """Liang-Barsky clip for a segment in continuous (row, col) coordinates."""
        r0, c0 = p0
        r1, c1 = p1
        dr, dc = r1 - r0, c1 - c0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dr, r0), (dr, rows - 1 - r0), (-dc, c0), (dc, cols - 1 - c0)):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
        if t0 > t1:
            return None
        return ((r0 + t0 * dr, c0 + t0 * dc), (r0 + t1 * dr, c0 + t1 * dc))

    def clip_polygon_to_grid(points):
        """Sutherland-Hodgman clip for polygons in continuous (row, col) coordinates."""
        poly = [tuple(map(float, p)) for p in points]
        boundaries = ((0, 0.0, True), (0, float(rows - 1), False),
                      (1, 0.0, True), (1, float(cols - 1), False))
        for axis, bound, keep_greater in boundaries:
            if not poly:
                break
            out = []
            prev = poly[-1]
            prev_inside = (prev[axis] >= bound) if keep_greater else (prev[axis] <= bound)
            for curr in poly:
                curr_inside = (curr[axis] >= bound) if keep_greater else (curr[axis] <= bound)
                if curr_inside != prev_inside:
                    denom = curr[axis] - prev[axis]
                    if abs(denom) > 1e-12:
                        t = (bound - prev[axis]) / denom
                        out.append((prev[0] + t * (curr[0] - prev[0]),
                                    prev[1] + t * (curr[1] - prev[1])))
                if curr_inside:
                    out.append(curr)
                prev, prev_inside = curr, curr_inside
            poly = out
        return np.asarray(poly, dtype=np.float64)

    road_count = 0
    bld_count = 0
    place_count = 0

    for el in data.get("elements", []):
        el_type = el.get("type")
        tags = el.get("tags", {})

        if el_type == "node" and "place" in tags:
            pr_f, pc_f = latlon_to_rc(el["lat"], el["lon"])
            if not (0 <= pr_f < rows and 0 <= pc_f < cols):
                continue
            place_count += 1
            bld_count += 1
            pr, pc = int(round(pr_f)), int(round(pc_f))
            ptype = tags.get("place", "village")
            rad = 15 if ptype in {"city", "town"} else (10 if ptype == "village" else 6)
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr**2 + dc**2 <= rad**2:
                        nr, nc = pr + dr, pc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            building_mask[nr, nc] = True

        elif el_type == "way" and "geometry" in el:
            geom = el["geometry"]
            pts = [latlon_to_rc(pt["lat"], pt["lon"])
                   for pt in geom if "lat" in pt and "lon" in pt]
            if len(pts) < 2:
                continue

            if "highway" in tags:
                road_count += 1
                hw_type = tags.get("highway", "")
                thick = 2 if hw_type in {"motorway", "trunk", "primary", "secondary"} else 1
                for i in range(len(pts) - 1):
                    clipped = clip_segment_to_grid(pts[i], pts[i + 1])
                    if clipped is None:
                        continue
                    (r0, c0), (r1, c1) = clipped
                    draw_thick_line(road_mask, r0, c0, r1, c1, thickness=thick)

            elif tags.get("natural") == "coastline":
                # OSM coastline direction convention: land left, water right.
                for i in range(len(pts) - 1):
                    clipped = clip_segment_to_grid(pts[i], pts[i + 1])
                    if clipped is None:
                        continue
                    (r0, c0), (r1, c1) = clipped
                    draw_thick_line(coastline_mask, r0, c0, r1, c1, thickness=0)
                    dr, dc = r1 - r0, c1 - c0
                    norm = max(float(np.hypot(dr, dc)), 1e-6)
                    # In row/column coordinates, the right-hand normal is (dc, -dr).
                    sr = int(round((r0 + r1) / 2.0 + 4.0 * dc / norm))
                    sc = int(round((c0 + c1) / 2.0 - 4.0 * dr / norm))
                    if 0 <= sr < rows and 0 <= sc < cols:
                        coastline_water_seeds[sr, sc] = True

            elif (tags.get("natural") == "water"
                  or tags.get("waterway") in {"riverbank", "river", "canal", "stream"}
                  or tags.get("landuse") == "reservoir"
                  or tags.get("water") in {"lake", "reservoir", "pond", "basin"}):
                # Closed water polygons are preferred; open mapped water features are rasterized
                # conservatively as a narrow non-burnable corridor.
                if len(pts) >= 3 and (
                    np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 0.5
                ):
                    poly_rc = clip_polygon_to_grid(pts)
                    if len(poly_rc) >= 3:
                        min_r = max(0, int(np.floor(poly_rc[:, 0].min())))
                        max_r = min(rows - 1, int(np.ceil(poly_rc[:, 0].max())))
                        min_c = max(0, int(np.floor(poly_rc[:, 1].min())))
                        max_c = min(cols - 1, int(np.ceil(poly_rc[:, 1].max())))
                        if max_r >= min_r and max_c >= min_c:
                            rr, cc = np.meshgrid(
                                np.arange(min_r, max_r + 1), np.arange(min_c, max_c + 1), indexing="ij"
                            )
                            grid_pts = np.vstack((rr.ravel(), cc.ravel())).T
                            inside = Path(poly_rc).contains_points(grid_pts).reshape(rr.shape)
                            water_mask[min_r:max_r + 1, min_c:max_c + 1] |= inside
                else:
                    for i in range(len(pts) - 1):
                        clipped = clip_segment_to_grid(pts[i], pts[i + 1])
                        if clipped is None:
                            continue
                        (r0, c0), (r1, c1) = clipped
                        draw_thick_line(water_mask, r0, c0, r1, c1, thickness=1)

            elif "landuse" in tags and len(pts) >= 3:
                poly_rc = clip_polygon_to_grid(pts)
                if len(poly_rc) < 3:
                    continue
                bld_count += 1
                min_r = max(0, int(np.floor(poly_rc[:, 0].min())))
                max_r = min(rows - 1, int(np.ceil(poly_rc[:, 0].max())))
                min_c = max(0, int(np.floor(poly_rc[:, 1].min())))
                max_c = min(cols - 1, int(np.ceil(poly_rc[:, 1].max())))
                if max_r > min_r and max_c > min_c:
                    rr, cc = np.meshgrid(
                        np.arange(min_r, max_r + 1), np.arange(min_c, max_c + 1), indexing="ij"
                    )
                    grid_pts = np.vstack((rr.ravel(), cc.ravel())).T
                    path = Path(poly_rc)
                    inside = path.contains_points(grid_pts).reshape(rr.shape)
                    building_mask[min_r:max_r + 1, min_c:max_c + 1] |= inside

    # Use coastline direction to identify the ocean-side connected component(s).
    # The line is slightly thickened because the spread and connectivity checks use 8-neighbours.
    if np.any(coastline_water_seeds) and np.any(coastline_mask):
        coast_barrier = binary_dilation(coastline_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        components, _ = label(~coast_barrier, structure=np.ones((3, 3), dtype=np.uint8))
        seed_labels = components[coastline_water_seeds & (~coast_barrier)]
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size:
            counts = np.bincount(seed_labels, minlength=int(components.max()) + 1)
            counts[0] = 0
            ocean_label = int(np.argmax(counts))
            if ocean_label > 0:
                water_mask |= components == ocean_label
        water_mask |= coast_barrier & binary_dilation(water_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)

    # OSM transport/settlement layers can overlap mapped water; physical water is non-burnable.
    building_mask[road_mask] = False
    building_mask[water_mask] = False
    water_mask[coastline_mask] = water_mask[coastline_mask] | binary_dilation(
        water_mask, structure=np.ones((3, 3), dtype=bool), iterations=1
    )[coastline_mask]
    return road_mask, building_mask, water_mask, road_count, bld_count, place_count, used_mirror


def fetch_realtime_weather_openmeteo(lat, lon):
    """Lấy thời tiết thực tế tại tọa độ qua Open-Meteo Forecast API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"
        f"&wind_speed_unit=ms"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    curr = resp.json().get("current", {})
    return {
        "wind_speed": float(curr.get("wind_speed_10m", 5.0)),
        "wind_deg": float(curr.get("wind_direction_10m", 135.0)),
        "humidity": float(curr.get("relative_humidity_2m", 60.0)),
        "prcp": float(curr.get("precipitation", 0.0)),
        "temp": float(curr.get("temperature_2m", 22.0))
    }


# ==============================================================================
# LAYER 2: SLOPE, ASPECT, AND TOBLER HIKING FUNCTION (1993)
# ==============================================================================
def compute_slope_difficulty_and_manpower(slope_deg, road_mask, building_mask, aspect_deg=None):
    """
    Tính Hệ số Độ khó Phát quang liên tục K(theta), Hệ số Thực bì theo Hướng phơi nắng (Aspect)
    và Chi phí Giờ công (Person-Hours / pixel) dựa trên 4 cấp độ dốc + giãn diện tích mặt dốc 1/cos(theta).
    """
    slope_clipped = np.clip(slope_deg, 0.0, 75.0).astype(np.float32)
    cos_slope = np.cos(np.radians(slope_clipped))
    surface_stretch = 1.0 / np.clip(cos_slope, 0.25, 1.0)

    # Hàm độ khó liên tục qua 4 cấp độ dốc (loại bỏ điểm nhảy gián đoạn ở 15°, 25°, 35°)
    difficulty_factor = np.ones_like(slope_clipped, dtype=np.float32)
    tier1 = (slope_clipped >= 0.0) & (slope_clipped < 15.0)
    tier2 = (slope_clipped >= 15.0) & (slope_clipped < 25.0)
    tier3 = (slope_clipped >= 25.0) & (slope_clipped < 35.0)
    tier4 = (slope_clipped >= 35.0)

    difficulty_factor[tier1] = 1.0 + 0.35 * (slope_clipped[tier1] / 15.0)
    val_15 = 1.35
    difficulty_factor[tier2] = val_15 * np.exp(0.0314 * (slope_clipped[tier2] - 15.0))
    val_25 = val_15 * float(np.exp(0.0314 * 10.0))  # ~1.85
    difficulty_factor[tier3] = val_25 * np.exp(0.0415 * (slope_clipped[tier3] - 25.0))
    val_35 = val_25 * float(np.exp(0.0415 * 10.0))  # ~2.80
    difficulty_factor[tier4] = val_35 * np.exp(0.0450 * (slope_clipped[tier4] - 35.0))

    # Tích hợp hướng phơi sườn dốc (aspect_deg): sườn phơi nắng Tây-Nam (202.5°) có thảm thực bì
    # khô và dày hơn sườn khuất nắng Đông-Bắc (22.5°), làm thay đổi công phát quang và hướng cắt
    if aspect_deg is not None:
        slope_weight = np.sin(np.radians(np.clip(slope_clipped, 0.0, 60.0)))
        aspect_brush_factor = (
            1.0 + 0.14 * slope_weight * np.cos(np.radians(aspect_deg - 202.5))
        ).astype(np.float32)
    else:
        aspect_brush_factor = np.ones_like(slope_clipped, dtype=np.float32)

    total_multiplier = (difficulty_factor * surface_stretch * aspect_brush_factor).astype(np.float32)
    person_hours_matrix = (BASE_PERSON_HOURS_PER_30M * total_multiplier).astype(np.float32)

    # Đường giao thông hỗ trợ giảm ~50% công phát quang khi neo tuyến (dọn thực bì ven đường & canh giữ)
    person_hours_matrix[road_mask] = 1.25
    person_hours_matrix[building_mask] = 1000000.0

    return person_hours_matrix, total_multiplier


def compute_tobler_arrival_time(elevation, slope_deg, road_mask, dy_m, dx_m, water_mask=None, building_mask=None):
    """
    Tính thời gian tiếp cận tích lũy từ road_mask bằng đồ thị 8 hướng có trọng số Tobler.
    Hàm không có lớp dữ liệu riêng cho trạm kiểm lâm hoặc đường mòn tuần tra.
    walk_speed_kmh là tốc độ tham chiếu theo độ dốc cục bộ không dấu; thời gian tích lũy
    sử dụng độ chênh cao có dấu của từng cạnh.
    """
    rows, cols = GRID_SHAPE
    elevation = np.asarray(elevation, dtype=np.float32)
    slope_deg = np.asarray(slope_deg, dtype=np.float32)
    road_mask = np.asarray(road_mask, dtype=bool)
    water_mask = np.zeros(GRID_SHAPE, dtype=bool) if water_mask is None else np.asarray(water_mask, dtype=bool)
    building_mask = np.zeros(GRID_SHAPE, dtype=bool) if building_mask is None else np.asarray(building_mask, dtype=bool)
    if any(m.shape != GRID_SHAPE for m in (elevation, slope_deg, road_mask, water_mask, building_mask)):
        raise ValueError("elevation, slope_deg, road_mask, water_mask, and building_mask must all have the same GRID_SHAPE.")

    walkable = (~water_mask) & (~building_mask)
    N = rows * cols
    idx = np.arange(N, dtype=np.int32).reshape((rows, cols))

    local_grade = np.tan(np.radians(np.clip(slope_deg, 0.0, 70.0)))
    local_speed_kmh = 6.0 * np.exp(-3.5 * np.abs(local_grade + 0.05)) * 0.55
    local_speed_kmh = np.clip(local_speed_kmh, 0.55, 4.0).astype(np.float32)

    def edge_hours(distance_m, dz):
        grade = np.clip(dz / max(float(distance_m), 1e-6), -1.8, 1.8)
        speed = 6.0 * np.exp(-3.5 * np.abs(grade + 0.05)) * 0.55
        speed = np.clip(speed, 0.55, 4.0)
        return ((float(distance_m) / 1000.0) / speed).astype(np.float64)

    u_parts, v_parts, w_parts = [], [], []

    # Build directed edges only between walkable cells. Diagonal movement cannot cut a corner
    # between water/settlement cells, matching the access assumptions used by the planner.
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in directions:
        src_r = slice(max(0, -dr), rows - max(0, dr))
        src_c = slice(max(0, -dc), cols - max(0, dc))
        dst_r = slice(max(0, dr), rows - max(0, -dr))
        dst_c = slice(max(0, dc), cols - max(0, -dc))
        valid = walkable[src_r, src_c] & walkable[dst_r, dst_c]
        if dr != 0 and dc != 0:
            valid &= walkable[src_r, dst_c] & walkable[dst_r, src_c]
        if not np.any(valid):
            continue
        dist_m = float(np.hypot(dr * dy_m, dc * dx_m))
        u = idx[src_r, src_c][valid]
        v = idx[dst_r, dst_c][valid]
        dz = (elevation[dst_r, dst_c] - elevation[src_r, src_c])[valid]
        u_parts.append(u.astype(np.int32, copy=False))
        v_parts.append(v.astype(np.int32, copy=False))
        w_parts.append(edge_hours(dist_m, dz))

    source_road_mask = road_mask & walkable
    if np.any(source_road_mask):
        source_nodes = idx[source_road_mask].ravel()
    else:
        border_mask = np.zeros(GRID_SHAPE, dtype=bool)
        border_mask[0, :] = True; border_mask[-1, :] = True
        border_mask[:, 0] = True; border_mask[:, -1] = True
        source_nodes = idx[border_mask & walkable].ravel()
    if len(source_nodes) == 0:
        raise ValueError("No valid access source: at least one road pixel or map-border pixel outside water and settlements is required.")

    all_u = np.concatenate(u_parts)
    all_v = np.concatenate(v_parts)
    all_w = np.concatenate(w_parts)
    del u_parts, v_parts, w_parts

    # Add explicit zero-weight virtual-source arcs in the same COO construction. Sparse-matrix
    # addition can drop explicit zero entries, which would silently disconnect the super-source.
    super_source = N
    su = np.full(len(source_nodes), super_source, dtype=np.int32)
    sv = source_nodes.astype(np.int32, copy=False)
    sw = np.zeros(len(source_nodes), dtype=np.float64)
    all_u = np.concatenate([all_u, su])
    all_v = np.concatenate([all_v, sv])
    all_w = np.concatenate([all_w, sw])
    del su, sv, sw
    adj_extended = sp.csr_matrix(
        (all_w, (all_u, all_v)), shape=(N + 1, N + 1), dtype=np.float64
    )
    del all_u, all_v, all_w

    dist_hours = dijkstra(csgraph=adj_extended, directed=True, indices=super_source)[:N]
    del adj_extended

    # Preserve accumulated Dijkstra travel time. The displayed local speed remains a reference statistic.
    arrival_time = dist_hours.reshape(GRID_SHAPE).astype(np.float32)
    arrival_time[~walkable] = np.inf
    return arrival_time, local_speed_kmh


def _local_raster_inventory_signature():
    """Stable cache key from local DEM/WorldCover file names, sizes, and modification times."""
    inventory = []
    for label, folder in (("dem", LOCAL_DEM_DIR), ("worldcover", LOCAL_WORLDCOVER_DIR)):
        if not folder.exists():
            continue
        paths = sorted(set(folder.glob("*.tif")) | set(folder.glob("*.tiff")))
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            inventory.append((label, path.name, int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(inventory)


@st.cache_data(show_spinner=False, persist="disk", max_entries=6)
def fetch_and_build_api_environment(lat_center, lon_center, data_signature):
    """Build static terrain/OSM layers; persistent cache permits reuse after restart.

    High-resolution local GeoTIFFs are preferred when present. Otherwise elevation falls
    back to a 20x20 OpenTopoData sample and fuel falls back to an explicitly labelled proxy.
    A cache miss still requires the corresponding API(s); caching is not preloaded offline data.
    """
    dy_m, dx_m, pixel_area_ha = compute_projected_pixel_scales(lat_center)

    road_mask, building_mask, water_mask, road_cnt, bld_cnt, place_cnt, mirror_used = fetch_api_osm_landscape(
        lat_center, lon_center
    )

    local_elevation, dem_source = load_local_dem_grid(lat_center, lon_center, water_mask)
    if local_elevation is None:
        elevation = fetch_api_dem_grid(lat_center, lon_center, grid_n=20)
        dem_source = "OpenTopoData SRTM90m: 20x20 elevation samples interpolated to a 540x540 grid (not a 30 m DEM)"
    else:
        elevation = local_elevation

    dy, dx = np.gradient(elevation, dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    aspect_deg = (np.degrees(np.arctan2(-dx, dy)) % 360.0).astype(np.float32)

    slope_sin = np.sin(np.radians(np.clip(slope_deg, 0.0, 60.0))).astype(np.float32)
    aspect_flammability = (
        1.0 + 0.16 * slope_sin * np.cos(np.radians(aspect_deg - 202.5))
    ).astype(np.float32)

    worldcover_classes, wc_water_mask, landcover_source, landcover_coverage_pct = load_local_worldcover_grid(
        lat_center, lon_center, water_mask
    )
    if wc_water_mask is not None:
        water_mask = wc_water_mask
        # ESA WorldCover 2021 class -> heuristic relative fuel availability (not measured biomass).
        # 10 tree, 20 shrub, 30 grass, 40 cropland, 90 herbaceous wetland, 95 mangroves.
        class_fuel = np.zeros(GRID_SHAPE, dtype=np.float32)
        class_fuel[worldcover_classes == 10] = 1.00
        class_fuel[worldcover_classes == 20] = 0.82
        class_fuel[worldcover_classes == 30] = 0.58
        class_fuel[worldcover_classes == 40] = 0.28
        class_fuel[worldcover_classes == 90] = 0.12
        class_fuel[worldcover_classes == 95] = 0.90
        class_fuel[worldcover_classes == 100] = 0.10
        fuel_matrix = class_fuel * aspect_flammability
    else:
        worldcover_classes = np.zeros(GRID_SHAPE, dtype=np.uint16)
        fuel_matrix = aspect_flammability.copy()
        landcover_source = "Aspect-based proxy — no actual land-cover layer available"
        landcover_coverage_pct = 0.0

    # Non-burnable barriers take precedence over heuristic or classified fuel values.
    fuel_matrix[water_mask] = 0.0
    fuel_matrix[building_mask] = 0.0
    fuel_matrix[road_mask] = 0.0

    manpower_cost, difficulty_mult = compute_slope_difficulty_and_manpower(
        slope_deg, road_mask, building_mask, aspect_deg=aspect_deg
    )
    arrival_time, walk_speed_kmh = compute_tobler_arrival_time(
        elevation, slope_deg, road_mask, dy_m, dx_m,
        water_mask=water_mask, building_mask=building_mask
    )

    ls = LightSource(azdeg=315, altdeg=42)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "vn_forest_terrain",
        ["#1b3b22", "#26532b", "#3a6b33", "#658339", "#94994a", "#b5a36b", "#dcd3b2"]
    )
    hillshade_rgb = ls.shade(
        elevation,
        cmap=terrain_cmap,
        vert_exag=2.0,
        blend_mode='soft'
    )[..., :3].astype(np.float32)

    return {
        "elevation": elevation,
        "slope_deg": slope_deg,
        "aspect_deg": aspect_deg,
        "aspect_flammability": aspect_flammability,
        "road_mask": road_mask,
        "building_mask": building_mask,
        "water_mask": water_mask,
        "worldcover_classes": worldcover_classes,
        "landcover_source": landcover_source,
        "landcover_coverage_pct": landcover_coverage_pct,
        "dem_source": dem_source,
        "fuel_matrix": fuel_matrix,
        "manpower_cost": manpower_cost,
        "difficulty_mult": difficulty_mult,
        "arrival_time": arrival_time,
        "walk_speed_kmh": walk_speed_kmh,
        "hillshade_rgb": hillshade_rgb,
        "dy_m": dy_m,
        "dx_m": dx_m,
        "pixel_area_ha": pixel_area_ha,
        "road_cnt": road_cnt,
        "bld_cnt": bld_cnt,
        "place_cnt": place_cnt,
        "mirror_used": mirror_used
    }


# ==============================================================================
# LAYER 3: 3D FIREBREAK LENGTH AND TERRAIN-DIFFICULTY ANALYSIS
# ==============================================================================
def _skeletonize_binary_mask(mask):
    """Làm mảnh mặt nạ nhị phân bằng Zhang-Suen thinning trên khung bao (Bounding Box) siêu nhanh."""
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool)

    coords = np.argwhere(mask_bool)
    r0 = max(0, int(coords[:, 0].min()) - 2)
    r1 = min(mask_bool.shape[0], int(coords[:, 0].max()) + 3)
    c0 = max(0, int(coords[:, 1].min()) - 2)
    c1 = min(mask_bool.shape[1], int(coords[:, 1].max()) + 3)

    image = mask_bool[r0:r1, c0:c1].copy()
    max_iter = max(image.shape)

    for _ in range(max_iter):
        changed_any = False
        for phase in (0, 1):
            pad = np.pad(image.astype(np.uint8), 1, mode="constant")
            p2 = pad[:-2, 1:-1]
            p3 = pad[:-2, 2:]
            p4 = pad[1:-1, 2:]
            p5 = pad[2:, 2:]
            p6 = pad[2:, 1:-1]
            p7 = pad[2:, :-2]
            p8 = pad[1:-1, :-2]
            p9 = pad[:-2, :-2]
            neigh = (p2, p3, p4, p5, p6, p7, p8, p9)
            B = sum(n.astype(np.uint8) for n in neigh)
            A = sum(((~neigh[i].astype(bool)) & neigh[(i + 1) % 8].astype(bool)).astype(np.uint8)
                    for i in range(8))
            if phase == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = image & (B >= 2) & (B <= 6) & (A == 1) & cond
            if np.any(remove):
                image[remove] = False
                changed_any = True
        if not changed_any:
            break

    out = np.zeros_like(mask_bool)
    out[r0:r1, c0:c1] = image
    return out


def compute_2d_and_3d_firebreak_length(firebreak_mask, elevation, slope_deg, dy_m, dx_m):
    """Ước tính chiều dài 2D/3D từ đồ thị 8-lân-cận của skeleton tuyến."""
    fb_pixels = int(np.sum(firebreak_mask))
    pixel_count_km = (fb_pixels * NOMINAL_PIXEL_M) / 1000.0
    if fb_pixels == 0:
        return 0.0, 0.0, 0.0

    skeleton = _skeletonize_binary_mask(firebreak_mask)
    skel_pixels = int(np.sum(skeleton))
    if skel_pixels == 0:
        return 0.0, 0.0, pixel_count_km

    total_2d_m = 0.0
    total_3d_m = 0.0
    elev = np.asarray(elevation, dtype=np.float32)
    slope = np.asarray(slope_deg, dtype=np.float32)

    m_h = skeleton[:, :-1] & skeleton[:, 1:]
    dz_h = elev[:, 1:][m_h] - elev[:, :-1][m_h]
    cos_h = np.cos(np.radians(np.clip(slope[:, :-1][m_h], 0.0, 75.0)))
    total_2d_m += float(np.sum(m_h)) * dx_m
    total_3d_m += float(np.sum(np.maximum(np.hypot(dx_m, dz_h), dx_m / np.maximum(0.25, cos_h))))

    m_v = skeleton[:-1, :] & skeleton[1:, :]
    dz_v = elev[1:, :][m_v] - elev[:-1, :][m_v]
    cos_v = np.cos(np.radians(np.clip(slope[:-1, :][m_v], 0.0, 75.0)))
    total_2d_m += float(np.sum(m_v)) * dy_m
    total_3d_m += float(np.sum(np.maximum(np.hypot(dy_m, dz_v), dy_m / np.maximum(0.25, cos_v))))

    diag_m = float(np.hypot(dy_m, dx_m))
    m_d1 = skeleton[:-1, :-1] & skeleton[1:, 1:] & (~skeleton[:-1, 1:]) & (~skeleton[1:, :-1])
    dz_d1 = elev[1:, 1:][m_d1] - elev[:-1, :-1][m_d1]
    cos_d1 = np.cos(np.radians(np.clip(slope[:-1, :-1][m_d1], 0.0, 75.0)))
    total_2d_m += float(np.sum(m_d1)) * diag_m
    total_3d_m += float(np.sum(np.maximum(np.hypot(diag_m, dz_d1), diag_m / np.maximum(0.25, cos_d1))))

    m_d2 = skeleton[:-1, 1:] & skeleton[1:, :-1] & (~skeleton[:-1, :-1]) & (~skeleton[1:, 1:])
    dz_d2 = elev[1:, :-1][m_d2] - elev[:-1, 1:][m_d2]
    cos_d2 = np.cos(np.radians(np.clip(slope[:-1, 1:][m_d2], 0.0, 75.0)))
    total_2d_m += float(np.sum(m_d2)) * diag_m
    total_3d_m += float(np.sum(np.maximum(np.hypot(diag_m, dz_d2), diag_m / np.maximum(0.25, cos_d2))))

    min_step = min(float(dx_m), float(dy_m))
    if total_2d_m < max(1, skel_pixels - 1) * min_step * 0.25:
        mean_cos = float(np.mean(np.cos(np.radians(np.clip(slope[skeleton], 0.0, 75.0)))))
        total_2d_m = max(1, skel_pixels - 1) * ((dy_m + dx_m) / 2.0)
        total_3d_m = total_2d_m / max(0.25, mean_cos)

    return total_3d_m / 1000.0, total_2d_m / 1000.0, pixel_count_km


def analyze_firebreak_slope_tiers(
    firebreak_mask, slope_deg, manpower_cost, len_3d_km, effective_manpower, geom_2d_ratio=1.0
):
    fb_total_px = int(np.sum(firebreak_mask))
    if fb_total_px == 0:
        return pd.DataFrame()

    fb_slopes = slope_deg[firebreak_mask]
    fb_costs  = manpower_cost[firebreak_mask] * float(geom_2d_ratio)

    tiers = [
        ("Tier 1: Gentle (0°–15°)", (fb_slopes >= 0.0) & (fb_slopes < 15.0), "1.00×–1.40× (standard)"),
        ("Tier 2: Moderate (15°–25°)", (fb_slopes >= 15.0) & (fb_slopes < 25.0), "1.40×–2.05× (moderate difficulty)"),
        ("Tier 3: Steep (25°–35°)", (fb_slopes >= 25.0) & (fb_slopes < 35.0), "2.05×–3.40× (high difficulty)"),
        ("Tier 4: Very steep (>35°)", (fb_slopes >= 35.0), "3.40×–7.50× (very difficult)")
    ]

    rows_data = []
    eff_workers = max(1.0, float(effective_manpower))

    for label, mask_t, diff_desc in tiers:
        px_cnt = int(np.sum(mask_t))
        pct = (px_cnt / fb_total_px * 100.0) if fb_total_px > 0 else 0.0
        seg_len_3d_km = len_3d_km * (px_cnt / fb_total_px)
        person_hours = float(np.sum(fb_costs[mask_t]))
        est_hours = person_hours / eff_workers
        mean_slp = float(np.mean(fb_slopes[mask_t])) if px_cnt > 0 else 0.0

        rows_data.append({
            "Phân cấp Độ khó Phát quang": label,
            "Số Pixel": px_cnt,
            "Tỷ trọng (%)": round(pct, 1),
            "Độ dốc TB (°)": round(mean_slp, 1),
            "Chiều dài 3D (km)": round(seg_len_3d_km, 2),
            "Hệ số Độ khó & Mặt dốc": diff_desc,
            "Khối lượng (Giờ công)": round(person_hours, 1),
            "TG Thi công Phân bổ (giờ)": round(est_hours, 2)
        })

    return pd.DataFrame(rows_data)


def verify_topological_separation(source_mask, target_mask, barrier_mask, road_mask, nonburnable_mask=None):
    """Kiểm định độc lập tính phân tách tô-pô 8 hướng bằng thuật toán loang BFS."""
    rows, cols = GRID_SHAPE
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    barrier_mask = np.asarray(barrier_mask, dtype=bool)
    road_mask = np.asarray(road_mask, dtype=bool)
    if any(m.shape != GRID_SHAPE for m in (source_mask, target_mask, barrier_mask, road_mask)):
        return False, -1
    if not np.any(target_mask):
        return False, -1

    blocked = barrier_mask | road_mask
    if nonburnable_mask is not None:
        nonburnable_mask = np.asarray(nonburnable_mask, dtype=bool)
        if nonburnable_mask.shape != GRID_SHAPE:
            return False, -1
        # Targets remain traversable for the purpose of detecting whether fire can reach them.
        blocked |= nonburnable_mask & (~target_mask)
    visited = np.zeros(GRID_SHAPE, dtype=bool)
    start_pts = np.argwhere(source_mask & (~blocked))
    # An empty effective source is not evidence that the targets are separated.
    if len(start_pts) == 0:
        return False, -1

    for r, c in start_pts:
        visited[r, c] = True

    queue_r = list(start_pts[:, 0])
    queue_c = list(start_pts[:, 1])
    head = 0

    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while head < len(queue_r):
        cr, cc = queue_r[head], queue_c[head]
        head += 1
        for dr, dc in neighbors_8:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if dr != 0 and dc != 0:
                    # Match CA: a diagonal move is allowed only if both orthogonal side cells are passable.
                    if blocked[cr, nc] or blocked[nr, cc]:
                        continue
                if not visited[nr, nc] and not blocked[nr, nc]:
                    visited[nr, nc] = True
                    queue_r.append(nr)
                    queue_c.append(nc)

    leaked_targets = int(np.sum(visited & target_mask))
    return (leaked_targets == 0), leaked_targets


# ==============================================================================
# LAYER 4: IGNITION BACK-CASTING, CONTINUOUS SPREAD FIELD, AND DYNAMIC MIN-CUT
# ==============================================================================
def compute_backcasted_ignition_seeds(
    obs_row, obs_col, preburn_hours, uncertainty_radius_px,
    wind_speed, wind_deg, is_smoke_plume_report, env, rng
):
    rows, cols = GRID_SHAPE
    elev = env["elevation"]
    dy_m, dx_m = env["dy_m"], env["dx_m"]

    wind_rad = np.radians(wind_deg)
    wind_vr = np.cos(wind_rad)
    wind_vc = -np.sin(wind_rad)

    r_clip = int(np.clip(obs_row, 2, rows - 3))
    c_clip = int(np.clip(obs_col, 2, cols - 3))
    dz_dr = (elev[r_clip + 1, c_clip] - elev[r_clip - 1, c_clip]) / (2.0 * dy_m)
    dz_dc = (elev[r_clip, c_clip + 1] - elev[r_clip, c_clip - 1]) / (2.0 * dx_m)
    norm_slope = np.hypot(dz_dr, dz_dc) + 1e-6
    uphill_vr = dz_dr / norm_slope
    uphill_vc = dz_dc / norm_slope

    if is_smoke_plume_report:
        effective_drift_h = max(0.75, float(preburn_hours))
        drift_rate_px_per_hour = 1.2 + 0.35 * wind_speed
        total_drift_px = drift_rate_px_per_hour * effective_drift_h

        comb_vr = 0.8 * wind_vr + 0.2 * uphill_vr
        comb_vc = 0.8 * wind_vc + 0.2 * uphill_vc
        comb_norm = np.hypot(comb_vr, comb_vc) + 1e-6

        est_origin_r = int(np.clip(round(obs_row - total_drift_px * (comb_vr / comb_norm)), 14, rows - 15))
        est_origin_c = int(np.clip(round(obs_col - total_drift_px * (comb_vc / comb_norm)), 14, cols - 15))
    else:
        est_origin_r, est_origin_c = int(obs_row), int(obs_col)

    struct_8 = np.ones((3, 3), dtype=bool)
    building_guard = binary_dilation(env["building_mask"], structure=struct_8, iterations=3)
    water_mask = np.asarray(env.get("water_mask", np.zeros(GRID_SHAPE, dtype=bool)), dtype=bool)
    burnable_forest = (
        (env["fuel_matrix"] > 0.2) & (~env["road_mask"]) & (~building_guard) & (~water_mask)
    )

    # Limit simultaneously initialized candidate centers to a compact cluster.
    # The full report uncertainty still affects planning/deadline allowances, but
    # must not create many separated ignition spots that force an arbitrary route.
    seed_spread_px = min(6, int(round(max(0.0, float(uncertainty_radius_px)) * 0.45)))
    candidate_offsets = [(0, 0)]
    if seed_spread_px > 0:
        r_outer = max(1, seed_spread_px)
        r_inner = max(1, int(round(seed_spread_px * 0.55)))
        for ang_deg in range(0, 360, 45):
            rad_a = np.radians(ang_deg)
            candidate_offsets.append((int(round(r_outer * np.sin(rad_a))), int(round(r_outer * np.cos(rad_a)))))
            candidate_offsets.append((int(round(r_inner * np.sin(rad_a))), int(round(r_inner * np.cos(rad_a)))))
        for _ in range(4):
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.25, 1.0) * seed_spread_px
            candidate_offsets.append((int(round(dist * np.sin(angle))), int(round(dist * np.cos(angle)))))

    valid_seeds = []
    seen = set()
    for dr, dc in candidate_offsets:
        sr = int(np.clip(est_origin_r + dr, 10, rows - 11))
        sc = int(np.clip(est_origin_c + dc, 10, cols - 11))
        if (sr, sc) not in seen and burnable_forest[sr, sc]:
            valid_seeds.append((sr, sc))
            seen.add((sr, sc))

    if not valid_seeds:
        forest_pts = np.argwhere(burnable_forest)
        if len(forest_pts) == 0:
            raise ValueError(
                "No burnable forest cell was found near the back-cast location; a reliable simulation cannot be initialized."
            )
        dists = (forest_pts[:, 0] - est_origin_r)**2 + (forest_pts[:, 1] - est_origin_c)**2
        # Never silently move the ignition to an arbitrary forest pixel elsewhere on the map.
        # Permit a small local snap, or the explicitly supplied report-uncertainty radius.
        max_relocation_px = max(6, min(30, int(uncertainty_radius_px)))
        nearby = dists <= max_relocation_px ** 2
        if not np.any(nearby):
            raise ValueError(
                f"No burnable forest cell was found within {max_relocation_px} pixels of the back-cast center. "
                "Check the reported coordinates, location uncertainty, or OSM data. The model will not move the fire to an unrelated location."
            )
        local_distances = np.where(nearby, dists, np.inf)
        nr, nc = forest_pts[int(np.argmin(local_distances))]
        valid_seeds.append((int(nr), int(nc)))
        est_origin_r, est_origin_c = int(nr), int(nc)

    return est_origin_r, est_origin_c, valid_seeds


def compute_continuous_fire_arrival_field(
    current_fire_mask, env, wind_speed, wind_deg, humidity, prcp, temp, p_base, allow_spotting
):
    """Compute shortest modeled fire arrival times over burnable cells only.

    Non-burnable cells (roads, settlements, mapped water and all zero-fuel classes) are
    not graph nodes, and diagonal transitions cannot cut across a non-burnable corner.
    This is a deterministic travel-time approximation, not a calibrated fire forecast.
    """
    rows, cols = GRID_SHAPE
    N = rows * cols
    idx = np.arange(N, dtype=np.int32).reshape((rows, cols))
    elev = np.asarray(env["elevation"], dtype=np.float32)
    dy_m, dx_m = float(env["dy_m"]), float(env["dx_m"])
    road_mask = np.asarray(env["road_mask"], dtype=bool)
    building_mask = np.asarray(env["building_mask"], dtype=bool)
    water_mask = np.asarray(env.get("water_mask", np.zeros(GRID_SHAPE, dtype=bool)), dtype=bool)
    fuel = np.asarray(env["fuel_matrix"], dtype=np.float32)
    burnable = (fuel > 0.0) & (~road_mask) & (~building_mask) & (~water_mask)

    w_moisture = float(np.exp(-0.015 * max(0.0, humidity - 30.0)))
    w_rain = float(np.exp(-2.0 * max(0.0, prcp)))
    w_temp = float(np.exp(0.024 * (float(temp) - 20.0)))
    spotting_boost = (1.0 + 0.06 * max(0.0, wind_speed - 4.0)) if allow_spotting else 1.0
    wind_rad = np.radians(wind_deg)
    wind_vr = float(np.cos(wind_rad))
    wind_vc = float(-np.sin(wind_rad))

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    u_list, v_list, w_list = [], [], []
    for dr, dc in directions:
        dist_2d = float(np.hypot(dr * dy_m, dc * dx_m))
        src_r = slice(max(0, -dr), rows - max(0, dr))
        src_c = slice(max(0, -dc), cols - max(0, dc))
        dst_r = slice(max(0, dr), rows - max(0, -dr))
        dst_c = slice(max(0, dc), cols - max(0, -dc))

        src_burnable = burnable[src_r, src_c]
        dst_burnable = burnable[dst_r, dst_c]
        valid = src_burnable & dst_burnable
        # Match the CA's non-corner-cutting rule around roads, buildings, water and zero-fuel cells.
        if dr != 0 and dc != 0:
            side_a = burnable[src_r, dst_c]
            side_b = burnable[dst_r, src_c]
            valid &= side_a & side_b
        if not np.any(valid):
            continue

        u_nodes = idx[src_r, src_c][valid]
        v_nodes = idx[dst_r, dst_c][valid]
        dz = (elev[dst_r, dst_c] - elev[src_r, src_c])[valid]
        dist_3d = np.hypot(dist_2d, dz)
        dist_norm = np.maximum(dist_3d / NOMINAL_PIXEL_M, 1e-6)
        local_slope_deg = np.degrees(np.arctan2(dz, dist_2d))
        w_slope = np.exp(0.035 * np.clip(local_slope_deg, -45.0, 60.0))

        vec_norm = float(np.hypot(dr, dc))
        cos_wind = (dr * wind_vr + dc * wind_vc) / vec_norm
        if cos_wind >= 0:
            w_wind = np.exp(0.085 * wind_speed * cos_wind) * (spotting_boost if cos_wind > 0.3 else 1.0)
        else:
            w_wind = np.exp(0.25 * wind_speed * cos_wind)

        fuel_dst = np.clip(fuel[dst_r, dst_c][valid], 0.08, 1.35)
        p_single = np.clip(
            p_base * fuel_dst * w_slope * w_wind * w_moisture * w_rain * w_temp / np.sqrt(dist_norm),
            0.008, 0.96
        )
        p_front = 1.0 - (1.0 - p_single) ** 2.6
        step_hours = (CA_STEP_TO_HOURS * 0.85) / np.clip(p_front, 0.02, 0.98)
        u_list.append(u_nodes.astype(np.int32, copy=False))
        v_list.append(v_nodes.astype(np.int32, copy=False))
        w_list.append(step_hours.astype(np.float64))

    source_mask = np.asarray(current_fire_mask, dtype=bool) & burnable
    fire_sources = idx[source_mask].ravel()
    if len(fire_sources) == 0:
        return np.full(GRID_SHAPE, np.inf, dtype=np.float32)

    super_u = np.full(len(fire_sources), N, dtype=np.int32)
    super_w = np.zeros(len(fire_sources), dtype=np.float64)
    u_list.append(super_u)
    v_list.append(fire_sources.astype(np.int32, copy=False))
    w_list.append(super_w)
    all_u = np.concatenate(u_list)
    all_v = np.concatenate(v_list)
    all_w = np.concatenate(w_list)
    del u_list, v_list, w_list

    spread_graph = sp.csr_matrix((all_w, (all_u, all_v)), shape=(N + 1, N + 1), dtype=np.float64)
    del all_u, all_v, all_w
    cont_arrival = dijkstra(csgraph=spread_graph, directed=True, indices=N)[:N]
    del spread_graph
    arrival = cont_arrival.reshape(GRID_SHAPE).astype(np.float32)
    arrival[~burnable] = np.inf
    return arrival


_WEYL_OFFSETS = (
    0.0, 0.6180339, 0.2360679, 0.8541019,
    0.4721359, 0.0901699, 0.7082039, 0.3262379
)


def run_ca_step_vectorized(
    grid, burn_time, env, wind_speed, wind_deg, humidity, prcp, temp,
    p_base, burn_duration, rng, active_barrier_mask=None, allow_spotting=True,
    blocked_spread_mask=None
):
    """
    Bước chuyển trạng thái CA vector hóa trên địa hình 3D.
    Dùng số ngẫu nhiên chung theo pixel và timestep để ghép cặp kịch bản đối chứng.
    Dịch pha Weyl tách ngưỡng theo hướng; CRN hỗ trợ so sánh có kiểm soát nhưng không bảo đảm
    tính đơn điệu và không hàm ý rằng kịch bản có đường băng luôn cháy ít hơn.
    """
    next_grid = grid.copy()
    next_burn_time = burn_time.copy()

    road_mask = np.asarray(env["road_mask"], dtype=bool)
    building_mask = np.asarray(env["building_mask"], dtype=bool)
    water_mask = np.asarray(env.get("water_mask", np.zeros(GRID_SHAPE, dtype=bool)), dtype=bool)
    if active_barrier_mask is not None:
        active_barrier_mask = np.asarray(active_barrier_mask, dtype=bool)
        # A road can be used as part of the blocking route, but remains OSM road,
        # not newly constructed firebreak. Never overwrite cells that have already
        # ignited or burned: a late-built line cannot magically extinguish them.
        constructed_pixels = (
            active_barrier_mask & (~road_mask) & (~building_mask) & (~water_mask)
            & (grid != STATE_BURNING) & (grid != STATE_BURNED)
        )
        next_grid[constructed_pixels] = STATE_FIREBREAK

    # If fire reached a planned line before its completion, preserve the burning cell
    # and let it continue to spread. The final dynamic validation will reject that plan.
    burning_mask = (grid == STATE_BURNING)

    # Rút cố định 1 ma trận ngẫu nhiên trên mỗi timestep để đồng bộ tuyệt đối chuỗi RNG
    base_rand_grid = rng.random(GRID_SHAPE, dtype=np.float32)
    spot_jump_seed = int(rng.integers(0, 3))

    if not np.any(burning_mask):
        return next_grid, next_burn_time

    next_burn_time[burning_mask] += 1
    burned_out = burning_mask & (next_burn_time > burn_duration)
    next_grid[burned_out] = STATE_BURNED

    w_moisture = np.exp(-0.015 * max(0.0, humidity - 30.0))
    w_rain = np.exp(-2.0 * max(0.0, prcp))
    w_temp = np.exp(0.024 * (float(temp) - 20.0))

    wind_rad = np.radians(wind_deg)
    wind_vr = np.cos(wind_rad)
    wind_vc = -np.sin(wind_rad)

    dy_m, dx_m = env["dy_m"], env["dx_m"]
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    tree_mask = (
        (next_grid == STATE_TREE) & (env["fuel_matrix"] > 0.0)
        & (~road_mask) & (~building_mask) & (~water_mask)
    )
    spread_passable = (
        (env["fuel_matrix"] > 0.0) & (~road_mask) & (~building_mask) & (~water_mask)
    )
    if active_barrier_mask is not None:
        tree_mask &= (~active_barrier_mask)
        spread_passable &= (~active_barrier_mask)
    if blocked_spread_mask is not None:
        # Protected/buffer areas block spread without being mislabeled as constructed lines.
        blocked_spread_mask = np.asarray(blocked_spread_mask, dtype=bool)
        tree_mask &= (~blocked_spread_mask)
        spread_passable &= (~blocked_spread_mask)

    elev = env["elevation"]

    for dir_idx, (dr, dc) in enumerate(neighbors):
        dist_2d_m = float(np.hypot(dr * dy_m, dc * dx_m))

        shifted_burning = np.zeros_like(burning_mask)
        src_r = slice(max(0, -dr), GRID_SHAPE[0] - max(0, dr))
        src_c = slice(max(0, -dc), GRID_SHAPE[1] - max(0, dc))
        dst_r = slice(max(0, dr), GRID_SHAPE[0] - max(0, -dr))
        dst_c = slice(max(0, dc), GRID_SHAPE[1] - max(0, -dc))

        shifted_burning[dst_r, dst_c] = burning_mask[src_r, src_c]
        candidate_mask = shifted_burning & tree_mask
        if dr != 0 and dc != 0:
            # Do not allow fire to slip diagonally between two non-burnable corner cells.
            side_a = spread_passable[src_r, dst_c]
            side_b = spread_passable[dst_r, src_c]
            corner_ok = np.zeros_like(candidate_mask)
            corner_ok[dst_r, dst_c] = (side_a & side_b)
            candidate_mask &= corner_ok
        if not np.any(candidate_mask):
            continue

        dz_cand = elev[dst_r, dst_c][candidate_mask[dst_r, dst_c]] - elev[src_r, src_c][candidate_mask[dst_r, dst_c]]

        dist_3d_m = np.hypot(dist_2d_m, dz_cand)
        dist_norm = dist_3d_m / NOMINAL_PIXEL_M

        local_slope_deg = np.degrees(np.arctan2(dz_cand, dist_2d_m))
        w_slope = np.exp(0.035 * local_slope_deg)

        vec_norm = float(np.hypot(dr, dc))
        cos_wind = (dr * wind_vr + dc * wind_vc) / vec_norm
        if cos_wind >= 0:
            w_wind = np.exp(0.085 * wind_speed * cos_wind)
        else:
            w_wind = np.exp(0.25 * wind_speed * cos_wind)

        p_spread = (
            p_base
            * env["fuel_matrix"][candidate_mask]
            * w_slope
            * w_wind
            * w_moisture
            * w_rain
            * w_temp
            / (dist_norm ** 0.5)
        )
        p_spread = np.clip(p_spread, 0.0, 0.95)

        rand_vals = (base_rand_grid[candidate_mask] + _WEYL_OFFSETS[dir_idx]) % 1.0
        ignited = rand_vals < p_spread
        if np.any(ignited):
            cand_indices = np.argwhere(candidate_mask)
            ignited_coords = cand_indices[ignited]
            next_grid[ignited_coords[:, 0], ignited_coords[:, 1]] = STATE_BURNING
            next_burn_time[ignited_coords[:, 0], ignited_coords[:, 1]] = 1

    if wind_speed > 8.0 and allow_spotting:
        spot_rate = 0.02 * min(1.8, float(w_temp))
        spot_vals = (base_rand_grid[burning_mask] + 0.5142286) % 1.0
        burning_coords = np.argwhere(burning_mask)
        spotters = burning_coords[spot_vals < spot_rate]
        for r, c in spotters:
            jump_dist = 2 + ((int(r) * 7 + int(c) * 13 + spot_jump_seed) % 3)
            jr = r + int(round(wind_vr * jump_dist))
            jc = c + int(round(wind_vc * jump_dist))
            if 0 <= jr < GRID_SHAPE[0] and 0 <= jc < GRID_SHAPE[1]:
                if tree_mask[jr, jc]:
                    next_grid[jr, jc] = STATE_BURNING
                    next_burn_time[jr, jc] = 1

    return next_grid, next_burn_time


def _solve_cropped_mincut(safe_cost_matrix, source_mask, sink_mask, dx_m, dy_m, diag_scale):
    """
    Giải bài toán Max-Flow / Min-Cut trên khung bao cục bộ bao quanh vùng tìm kiếm.
    Biên của khung bao được xem là phía sink; vì vậy đây là nghiệm của bài toán cắt cục bộ
    trong collar đã định nghĩa, không được mặc định là tương đương đồ thị toàn cục.
    """
    rows, cols = GRID_SHAPE
    active_region = ~sink_mask
    if not np.any(active_region):
        return np.zeros(GRID_SHAPE, dtype=bool), True

    coords = np.argwhere(active_region | source_mask)
    r0 = max(0, int(coords[:, 0].min()) - 2)
    r1 = min(rows, int(coords[:, 0].max()) + 3)
    c0 = max(0, int(coords[:, 1].min()) - 2)
    c1 = min(cols, int(coords[:, 1].max()) + 3)

    sub_r = r1 - r0
    sub_c = c1 - c0
    sub_N = sub_r * sub_c
    SOURCE, SINK = sub_N, sub_N + 1
    sub_idx = np.arange(sub_N, dtype=np.int32).reshape((sub_r, sub_c))

    sub_cost = safe_cost_matrix[r0:r1, c0:c1]
    sub_src_mask = source_mask[r0:r1, c0:c1].copy()
    sub_snk_mask = sink_mask[r0:r1, c0:c1].copy()

    sub_snk_mask[0, :] = True; sub_snk_mask[-1, :] = True
    sub_snk_mask[:, 0] = True; sub_snk_mask[:, -1] = True
    sub_src_mask[sub_snk_mask] = False

    if not np.any(sub_src_mask):
        return np.zeros(GRID_SHAPE, dtype=bool), True

    eh_u, eh_v = sub_idx[:, :-1].ravel(), sub_idx[:, 1:].ravel()
    ch = ((sub_cost[:, :-1].ravel() + sub_cost[:, 1:].ravel()) / 2.0) * (dx_m / NOMINAL_PIXEL_M)

    ev_u, ev_v = sub_idx[:-1, :].ravel(), sub_idx[1:, :].ravel()
    cv = ((sub_cost[:-1, :].ravel() + sub_cost[1:, :].ravel()) / 2.0) * (dy_m / NOMINAL_PIXEL_M)

    ed1_u, ed1_v = sub_idx[:-1, :-1].ravel(), sub_idx[1:, 1:].ravel()
    cd1 = ((sub_cost[:-1, :-1].ravel() + sub_cost[1:, 1:].ravel()) / 2.0) * diag_scale

    ed2_u, ed2_v = sub_idx[:-1, 1:].ravel(), sub_idx[1:, :-1].ravel()
    cd2 = ((sub_cost[:-1, 1:].ravel() + sub_cost[1:, :-1].ravel()) / 2.0) * diag_scale

    u_b = np.concatenate([eh_u, eh_v, ev_u, ev_v, ed1_u, ed1_v, ed2_u, ed2_v])
    v_b = np.concatenate([eh_v, eh_u, ev_v, ev_u, ed1_v, ed1_u, ed2_v, ed2_u])
    c_b = np.concatenate([ch, ch, cv, cv, cd1, cd1, cd2, cd2])
    # Scale capacities so the sum of all finite arcs remains safely within int32.
    # The sum of finite capacities + 1 is then a valid upper bound for an infinity arc.
    raw_capacity_sum = float(np.sum(c_b, dtype=np.float64))
    capacity_scale = min(10.0, 200_000_000.0 / max(raw_capacity_sum, 1.0))
    c_int = np.maximum(1, np.rint(c_b * capacity_scale)).astype(np.int32)
    safe_infinity = int(np.sum(c_int, dtype=np.int64)) + 1
    if safe_infinity >= np.iinfo(np.int32).max:
        raise OverflowError("Graph capacity exceeds the int32 limit; reduce the graph size or resolution.")
    s_idx = sub_idx[sub_src_mask].ravel()
    t_idx = sub_idx[sub_snk_mask].ravel()

    u_s = np.full(len(s_idx), SOURCE, dtype=np.int32)
    v_t = np.full(len(t_idx), SINK, dtype=np.int32)
    c_s = np.full(len(s_idx), safe_infinity, dtype=np.int32)
    c_t = np.full(len(t_idx), safe_infinity, dtype=np.int32)

    all_u = np.concatenate([u_b, u_s, s_idx, t_idx, v_t])
    all_v = np.concatenate([v_b, s_idx, u_s, v_t, t_idx])
    all_c = np.concatenate([c_int, c_s, np.zeros_like(c_s), c_t, np.zeros_like(c_t)])

    graph = sp.csr_matrix((all_c, (all_u, all_v)), shape=(sub_N + 2, sub_N + 2), dtype=np.int32)
    res = maximum_flow(graph, SOURCE, SINK)

    # SciPy returns a skew-symmetric flow matrix: residual capacity is C - F.
    # Adding F.T again double-counts reverse flow and corrupts source reachability.
    residual = (graph - res.flow).tocsr()
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()

    reachable = np.zeros(sub_N + 2, dtype=bool)
    reachable[SOURCE] = True
    queue = [SOURCE]
    indptr, indices = residual.indptr, residual.indices
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for n in indices[indptr[curr]:indptr[curr + 1]]:
            if not reachable[n]:
                reachable[n] = True
                queue.append(n)

    sink_reached = bool(reachable[SINK])
    del graph, res, residual

    full_partition = np.zeros(GRID_SHAPE, dtype=bool)
    if not sink_reached:
        full_partition[r0:r1, c0:c1] = reachable[:sub_N].reshape((sub_r, sub_c))

    return full_partition, sink_reached


def _conservative_access_time_p90(access_times):
    """Return (P90 access time in hours, all_cells_accessible).

    An inaccessible construction pixel makes the candidate infeasible; it is not
    silently dropped from the percentile. Empty work masks require no travel.
    """
    values = np.asarray(access_times, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0, True
    if not np.all(np.isfinite(values)):
        return float("inf"), False
    return float(np.percentile(values, 90)), True


@st.cache_data(show_spinner=False, ttl=1800, max_entries=4)
def run_full_simulation_and_mincut(
    lat_center, lon_center,
    ign_row, ign_col, uncertainty_px, is_smoke_report, initial_radius, preburn_hours,
    wind_speed, wind_deg, humidity, prcp, temp,
    sim_hours, total_manpower, squad_size, p_base, allow_spotting
):
    env = fetch_and_build_api_environment(lat_center, lon_center, _local_raster_inventory_signature())
    dy_m, dx_m, pixel_area_ha = env["dy_m"], env["dx_m"], env["pixel_area_ha"]
    diag_scale = float(np.hypot(dy_m, dx_m) / ((dy_m + dx_m) / 2.0))
    struct_8 = np.ones((3, 3), dtype=bool)

    rng_init = np.random.default_rng(42)
    burn_duration = 3

    est_origin_r, est_origin_c, valid_seeds = compute_backcasted_ignition_seeds(
        ign_row, ign_col, preburn_hours, uncertainty_px,
        wind_speed, wind_deg, is_smoke_report, env, rng_init
    )

    boundary_mask = np.zeros(GRID_SHAPE, dtype=bool)
    boundary_mask[0, :] = True; boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True; boundary_mask[:, -1] = True
    sink_mask = env["building_mask"] | boundary_mask
    sink_guard_band = binary_dilation(sink_mask, structure=struct_8, iterations=2) | sink_mask
    outer_guard_band = binary_dilation(sink_guard_band, structure=struct_8, iterations=1)

    grid_t0 = np.full(GRID_SHAPE, STATE_TREE, dtype=np.uint8)
    burn_time_t0 = np.zeros(GRID_SHAPE, dtype=np.int32)
    grid_t0[env["fuel_matrix"] == 0.0] = STATE_EMPTY
    grid_t0[env["road_mask"]] = STATE_EMPTY

    for sr, sc in valid_seeds:
        for dr in range(-initial_radius, initial_radius + 1):
            for dc in range(-initial_radius, initial_radius + 1):
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < GRID_SHAPE[0] and 0 <= nc < GRID_SHAPE[1]:
                    if dr**2 + dc**2 <= initial_radius**2:
                        if (
                            env["fuel_matrix"][nr, nc] > 0.0
                            and not env["road_mask"][nr, nc]
                            and not outer_guard_band[nr, nc]
                        ):
                            grid_t0[nr, nc] = STATE_BURNING
                            burn_time_t0[nr, nc] = 1

    if not np.any(grid_t0 == STATE_BURNING):
        # Never silently relocate the ignition to an arbitrary burnable cell.
        # The backcast routine already constrained candidate seeds to the declared uncertainty.
        raise ValueError(
            "No valid burning cell could be initialized at the back-cast location within the selected radius. "
            "Check the coordinates, ignition radius, report uncertainty, water/land-cover masks, and settlements. "
            "The model stops rather than creating another fire at an unsupported location."
        )

    preburn_steps = int(round(preburn_hours / CA_STEP_TO_HOURS))
    for _ in range(preburn_steps):
        grid_t0, burn_time_t0 = run_ca_step_vectorized(
            grid_t0, burn_time_t0, env, wind_speed, wind_deg, humidity, prcp, temp,
            p_base, burn_duration, rng_init, active_barrier_mask=None,
            allow_spotting=allow_spotting, blocked_spread_mask=outer_guard_band
        )
    # T=0 is a pre-construction state. Assert that no temporary guard was rendered as a firebreak.
    grid_t0[grid_t0 == STATE_FIREBREAK] = STATE_TREE
    grid_t0[env["road_mask"]] = STATE_EMPTY
    burn_time_t0[grid_t0 != STATE_BURNING] = 0

    # ==========================================================================
    # LƯỢT 1 (PASS 1): KỊCH BẢN ĐỐI CHỨNG KHÔNG CHẶN (UNCONTROLLED BASELINE)
    # ==========================================================================
    total_steps = int(round(sim_hours / CA_STEP_TO_HOURS)) + 1
    history_uncontrolled = np.zeros((total_steps, GRID_SHAPE[0], GRID_SHAPE[1]), dtype=np.uint8)
    history_uncontrolled[0] = grid_t0.copy()

    grid_pass1 = grid_t0.copy()
    bt_pass1 = burn_time_t0.copy()
    rng_pass1 = np.random.default_rng(2026)

    for t in range(1, total_steps):
        grid_pass1, bt_pass1 = run_ca_step_vectorized(
            grid_pass1, bt_pass1, env, wind_speed, wind_deg, humidity, prcp, temp,
            p_base, burn_duration, rng_pass1, active_barrier_mask=None, allow_spotting=allow_spotting
        )
        history_uncontrolled[t] = grid_pass1.copy()

    # ==========================================================================
    # THUẬT TOÁN ĐỒ THỊ MIN-CUT / MAX-FLOW PHỤ THUỘC ĐỘNG VÀO NHÂN LỰC & MỌI INPUT
    # ==========================================================================
    current_fire_mask = (history_uncontrolled[0] == STATE_BURNING) | (history_uncontrolled[0] == STATE_BURNED)

    ca_fire_arrival = np.full(GRID_SHAPE, np.inf, dtype=np.float32)
    for t in range(total_steps):
        mask = (history_uncontrolled[t] == STATE_BURNING) | (history_uncontrolled[t] == STATE_BURNED)
        first_burn = mask & np.isinf(ca_fire_arrival)
        ca_fire_arrival[first_burn] = t * CA_STEP_TO_HOURS

    cont_fire_arrival = compute_continuous_fire_arrival_field(
        current_fire_mask, env, wind_speed, wind_deg, humidity, prcp, temp, p_base, allow_spotting
    )
    fire_arrival_time = np.minimum(ca_fire_arrival, cont_fire_arrival).astype(np.float32)

    heat_fatigue_mult = 1.0 + 0.012 * max(0.0, float(temp) - 20.0) + 0.005 * max(0.0, 14.0 - float(temp))
    rain_march_drag = 1.0 + 0.035 * max(0.0, float(prcp))

    cost_matrix = (env["manpower_cost"] * heat_fatigue_mult).astype(np.float32)
    arrival_time = (env["arrival_time"] * rain_march_drag).astype(np.float32)

    dist_from_fire_px = distance_transform_edt(~current_fire_mask).astype(np.float32)
    init_boundary = binary_dilation(current_fire_mask, structure=struct_8, iterations=1) & (~current_fire_mask)
    base_perim_px = max(24.0, float(np.sum(init_boundary)))
    ring_perimeter_px = base_perim_px + 2.0 * np.pi * dist_from_fire_px

    squad_factor = (float(squad_size) / 20.0) ** 0.22
    frontline_capacity = ring_perimeter_px * float(squad_size)
    effective_workforce = (
        np.minimum(float(total_manpower), frontline_capacity)
        + 0.65 * np.maximum(0.0, float(total_manpower) - frontline_capacity)
    ) * squad_factor

    tactical_line_factor = 1.65 + 0.02 * float(sim_hours) + (0.25 if allow_spotting else 0.0)
    ring_clearance_time = (cost_matrix * ring_perimeter_px * tactical_line_factor) / np.maximum(
        10.0, effective_workforce
    )
    local_squad_time = (cost_matrix * 1.1) / max(1.0, float(squad_size))

    uncertainty_allowance = 0.04 * float(uncertainty_px)
    smoke_allowance = 0.16 if is_smoke_report else 0.0
    preburn_allowance = 0.05 * float(preburn_hours)
    radius_allowance = 0.04 * float(initial_radius)
    horizon_allowance = 0.018 * float(sim_hours)
    spotting_allowance = (0.14 + 0.03 * max(0.0, wind_speed - 6.0)) if allow_spotting else 0.0

    effective_march_delay = 0.28 * arrival_time
    total_operation_time = (
        effective_march_delay
        + ring_clearance_time
        + 0.25 * local_squad_time
        + uncertainty_allowance
        + smoke_allowance
        + preburn_allowance
        + radius_allowance
        + horizon_allowance
        + spotting_allowance
    ).astype(np.float32)

    # Estimate when a basic crew could finish an initial perimeter. Use the actual
    # uncontrolled CA frame at that horizon to seed the planner, so higher manpower
    # plans against an earlier/smaller footprint and lower manpower against a later/larger one.
    initial_front = (
        binary_dilation(current_fire_mask, structure=struct_8, iterations=1)
        & (~current_fire_mask)
        & (~outer_guard_band)
        & (env["fuel_matrix"] > 0.0)
        & (~env["road_mask"])
        & (~env["building_mask"])
        & (~env["water_mask"])
    )
    if np.any(initial_front):
        initial_work_pixels = int(np.sum(initial_front))
        initial_crew = _effective_line_workforce(
            total_manpower, squad_size, initial_work_pixels, squad_factor
        )
        initial_person_hours = float(np.sum(cost_matrix[initial_front], dtype=np.float64))
        initial_travel_hours = float(np.mean(arrival_time[initial_front]))
        planning_horizon_hours = initial_travel_hours + initial_person_hours / initial_crew
    else:
        planning_horizon_hours = CA_STEP_TO_HOURS

    planning_horizon_hours = float(np.clip(
        planning_horizon_hours, CA_STEP_TO_HOURS, max(CA_STEP_TO_HOURS, float(sim_hours))
    ))
    forecast_step = int(np.clip(
        np.ceil(planning_horizon_hours / CA_STEP_TO_HOURS), 1, total_steps - 1
    ))
    forecast_state = history_uncontrolled[forecast_step]
    forecast_fire_raw = (
        (forecast_state == STATE_BURNING) | (forecast_state == STATE_BURNED)
    ) & (env["fuel_matrix"] > 0.0) & (~env["road_mask"]) & (~outer_guard_band)

    # Add a bounded, explicitly wind-aligned lead to the CA footprint. The lead uses the
    # same px/hour heuristic as the smoke back-casting model and is capped to avoid an
    # unbounded tail. It makes the planner protect where the front is moving, not only
    # where it was at T=0. The factor remains a model assumption, not a field-calibrated rate.
    wind_rad = np.radians(float(wind_deg))
    wind_row, wind_col = float(np.cos(wind_rad)), float(-np.sin(wind_rad))
    wind_lead_px = int(np.clip(
        round((1.2 + 0.35 * max(0.0, float(wind_speed))) * planning_horizon_hours),
        0, 18
    ))
    projected_fire = forecast_fire_raw.copy()
    for step_k in range(1, wind_lead_px + 1):
        sh_r = int(round(step_k * wind_row))
        sh_c = int(round(step_k * wind_col))
        if sh_r == 0 and sh_c == 0:
            continue
        shifted = np.zeros_like(forecast_fire_raw)
        src_r = slice(max(0, -sh_r), GRID_SHAPE[0] - max(0, sh_r))
        src_c = slice(max(0, -sh_c), GRID_SHAPE[1] - max(0, sh_c))
        dst_r = slice(max(0, sh_r), GRID_SHAPE[0] - max(0, -sh_r))
        dst_c = slice(max(0, sh_c), GRID_SHAPE[1] - max(0, -sh_c))
        shifted[dst_r, dst_c] = forecast_fire_raw[src_r, src_c]
        projected_fire |= shifted

    # Slightly connect adjacent CA fragments and the wind-aligned projection into one
    # conservative forecast envelope before selecting the main ignition-connected component.
    # Fill enclosed gaps only in the planner's forecast envelope. Do not mutate the CA
    # history: unburned pockets inside a surrounded footprint belong inside the
    # protected perimeter, but must not be relabeled as actually burning in the simulator.
    projected_envelope = binary_fill_holes(projected_fire)
    forecast_fire_raw = binary_dilation(
        projected_envelope, structure=struct_8, iterations=2
    ) & (env["fuel_matrix"] > 0.0) & (~env["road_mask"]) & (~env["building_mask"]) & (~outer_guard_band)
    forecast_fire_mask = _largest_connected_component(
        forecast_fire_raw, reference_mask=current_fire_mask
    )
    if not np.any(forecast_fire_mask):
        forecast_fire_mask = current_fire_mask & (~outer_guard_band)

    risk_horizon_hours = min(float(sim_hours), planning_horizon_hours + 1.0)
    raw_death_zone = (
        (total_operation_time > fire_arrival_time)
        & (fire_arrival_time <= risk_horizon_hours)
        & (env["fuel_matrix"] > 0.0)
        & (~env["road_mask"])
        & (~outer_guard_band)
    )

    rad = np.radians(wind_deg)
    dr_w, dc_w = float(np.cos(rad)), float(-np.sin(rad))
    wind_shift_steps = min(4, int(round(wind_speed * (0.28 if allow_spotting else 0.18))))

    directional_death = raw_death_zone.copy()
    rows, cols = GRID_SHAPE
    for step_k in range(1, wind_shift_steps + 1):
        sh_r = int(round(step_k * dr_w))
        sh_c = int(round(step_k * dc_w))
        if sh_r == 0 and sh_c == 0:
            continue
        shifted = np.zeros_like(raw_death_zone)
        src_r = slice(max(0, -sh_r), rows - max(0, sh_r))
        src_c = slice(max(0, -sh_c), cols - max(0, sh_c))
        dst_r = slice(max(0, sh_r), rows - max(0, -sh_r))
        dst_c = slice(max(0, sh_c), cols - max(0, -sh_c))
        shifted[dst_r, dst_c] = raw_death_zone[src_r, src_c]
        directional_death |= (shifted & (dist_from_fire_px <= 12.0) & (~outer_guard_band))

    death_zone_mask = (
        binary_dilation(directional_death, structure=struct_8, iterations=1)
        & (~outer_guard_band)
    )

    time_pressure_ratio = total_operation_time / np.maximum(0.35, fire_arrival_time)
    pressure_penalty = 1.0 + 0.65 * np.clip(time_pressure_ratio, 0.0, 3.0) ** 2

    manpower_abundance = float(total_manpower) / 840.0
    radial_conservation_weight = 1.0 + (0.12 * manpower_abundance) * dist_from_fire_px

    dynamic_cost_matrix = (
        cost_matrix
        * pressure_penalty
        * env["aspect_flammability"]
        * radial_conservation_weight
    ).astype(np.float32)

    safe_cost_matrix = np.clip(dynamic_cost_matrix, 0.8, 4000.0)
    # Water is not a construction surface; represent it as a high-cost planning node.
    safe_cost_matrix[env["water_mask"]] = 40000.0
    safe_cost_matrix[death_zone_mask] = 4000.0
    safe_cost_matrix[sink_guard_band] = 40000.0

    # More available crews reduce the initial stand-off; wind and modeled fire intensity
    # increase it. This is a transparent heuristic, not a field-calibrated fireline doctrine.
    target_standoff_px = _compute_target_standoff_px(
        total_manpower, squad_size, p_base, wind_speed, temp, humidity, prcp,
        allow_spotting, uncertainty_px, is_smoke_report, sim_hours
    )
    crew_power_index = (
        (max(1.0, float(total_manpower)) / 700.0) ** 0.78
        * (max(1.0, float(squad_size)) / 20.0) ** 0.20
    )

    buffer_pixels = max(1, target_standoff_px)
    max_buffer = 30
    buffer_step = 1

    status = "No plan satisfies the model's time and workforce constraints."
    accepted = False
    best_firebreak = np.zeros(GRID_SHAPE, dtype=bool)          # complete display/blocking route
    best_construction_mask = np.zeros(GRID_SHAPE, dtype=bool)  # new work only; existing roads excluded
    best_partition = np.zeros(GRID_SHAPE, dtype=bool)
    best_eff_manpower = float(total_manpower)
    best_geom_ratio = 1.0
    best_work_len_3d = 0.0
    best_completion_time_hours = float("inf")
    best_metrics = None
    best_candidate_score = None
    metrics_log = []

    # The planner must wrap the main forecast fire footprint, not disconnected high-cost
    # pixels scattered around the map. This avoids route fragments unrelated to the fire.
    base_seed_zone = (forecast_fire_mask | current_fire_mask) & (~outer_guard_band)
    base_seed_zone = _largest_connected_component(base_seed_zone, reference_mask=current_fire_mask)
    if not np.any(base_seed_zone):
        base_seed_zone = current_fire_mask & (~sink_guard_band)

    while buffer_pixels <= max_buffer:
        source_mask = binary_dilation(
            base_seed_zone, structure=struct_8, iterations=buffer_pixels
        ) & (~outer_guard_band)

        # Low workforce gets a wider search collar for alternative lower-effort alignments;
        # high workforce is allowed to seek a closer cut. Bound it to prevent runaway graphs.
        collar_width = int(np.clip(
            round(5.0 + 3.0 / max(0.35, crew_power_index)), 6, 14
        ))
        search_collar = binary_dilation(source_mask, structure=struct_8, iterations=collar_width) & (~sink_guard_band)
        iter_sink_mask = sink_guard_band | (~search_collar)

        partition_mask, sink_reached = _solve_cropped_mincut(
            safe_cost_matrix, source_mask, iter_sink_mask, dx_m, dy_m, diag_scale
        )

        if sink_reached or not np.any(partition_mask):
            buffer_pixels += buffer_step
            continue

        partition_mask[sink_guard_band] = False

        outer_ring = binary_dilation(partition_mask, structure=struct_8, iterations=1) & (~partition_mask)
        # Keep only one connected, fire-proximal cut component. Existing roads can complete
        # the blocking corridor, but are not colored/charged as newly constructed firebreak.
        route_mask = outer_ring & (~sink_guard_band)
        route_mask = _largest_connected_component(route_mask)
        construction_mask = (
            route_mask & (~env["road_mask"]) & (~env["building_mask"])
            & (~env["water_mask"]) & (~sink_guard_band)
        )

        route_pixels = int(np.sum(route_mask))
        work_pixels = int(np.sum(construction_mask))
        if route_pixels == 0:
            buffer_pixels += buffer_step
            continue

        len_3d_km, len_2d_km, route_pixel_km = compute_2d_and_3d_firebreak_length(
            route_mask, env["elevation"], env["slope_deg"], dy_m, dx_m
        )
        work_len_3d_km, work_len_2d_km, work_pixel_km = compute_2d_and_3d_firebreak_length(
            construction_mask, env["elevation"], env["slope_deg"], dy_m, dx_m
        )

        # Verify the same effective barrier width that will be applied in the CA pass;
        # a one-pixel diagonal line alone may leak under 8-neighbour fire spread.
        candidate_barrier_iters = 3 if (allow_spotting and wind_speed > 8.0) else 1
        candidate_barrier = (
            binary_dilation(route_mask, structure=struct_8, iterations=candidate_barrier_iters)
            & (~partition_mask | route_mask)
            & (~sink_guard_band)
        )
        # Verify the whole planned source envelope (forecast footprint + stand-off), not
        # only the tiny fire mask at T=0. Otherwise a line could pass around today's fire
        # but fail to contain the footprint predicted at construction time.
        topo_valid, leaked_targets = verify_topological_separation(
            source_mask, sink_mask, candidate_barrier, env["road_mask"],
            nonburnable_mask=(env["fuel_matrix"] <= 0.0) | env["water_mask"]
        )

        access_time_p90, all_construction_pixels_accessible = _conservative_access_time_p90(
            arrival_time[construction_mask]
        )
        mean_slope_fb = float(np.mean(env["slope_deg"][construction_mask])) if np.any(construction_mask) else 0.0
        max_slope_fb  = float(np.max(env["slope_deg"][construction_mask])) if np.any(construction_mask) else 0.0

        valid_arrival = fire_arrival_time[route_mask]
        finite_arrival = valid_arrival[np.isfinite(valid_arrival)]
        arrival_observed = len(finite_arrival) > 0
        if arrival_observed:
            # Conservative deadline: earliest modeled arrival, minus one CA timestep for margin.
            min_fire_arrival = max(0.0, float(np.min(finite_arrival)) - CA_STEP_TO_HOURS)
        else:
            # Right-censored: no modeled arrival was found within the simulation horizon.
            # The horizon is not represented as an actual predicted arrival time.
            min_fire_arrival = float(sim_hours)

        available_time = max(0.0, min_fire_arrival - access_time_p90) if np.isfinite(access_time_p90) else 0.0
        max_effective_manpower = _effective_line_workforce(
            total_manpower, squad_size, work_pixels, squad_factor
        )

        work_geom_ratio = (
            work_len_2d_km / work_pixel_km
            if work_pixel_km > 0 and work_len_2d_km > 0 else 1.0
        )
        total_person_hours = float(np.sum(cost_matrix[construction_mask])) * work_geom_ratio
        total_cutting_time = total_person_hours / max(1.0, max_effective_manpower)
        completion_time_hours = access_time_p90 + total_cutting_time

        constraints_met = bool(
            topo_valid and all_construction_pixels_accessible
            and np.isfinite(completion_time_hours)
            and completion_time_hours <= min_fire_arrival and route_pixels > 0
            and (arrival_observed or completion_time_hours < float(sim_hours))
        )

        metrics_entry = {
            "Buffer (px)": buffer_pixels,
            "Số Pixel": route_pixels,
            "Pixel thi công mới": work_pixels,
            "Chiều dài 3D Mặt dốc (km)": round(len_3d_km, 2),
            "Chiều dài 2D Phẳng (km)": round(len_2d_km, 2),
            "Độ dài Pixel (N×30m)": round(route_pixel_km, 2),
            "Độ dốc TB / Max (°)": f"{mean_slope_fb:.1f}° / {max_slope_fb:.1f}°",
            "TG tiếp cận P90 (giờ)": round(access_time_p90, 2) if np.isfinite(access_time_p90) else "Inaccessible",
            "TG Lửa chạm (giờ)": round(min_fire_arrival, 2) if arrival_observed else f"> {sim_hours} h (not reached within the simulation horizon)",
            "Quỹ TG Thi công (giờ)": round(available_time, 2),
            "TG Đào theo Độ dốc (giờ)": round(total_cutting_time, 2),
            "Khối lượng (Giờ công)": round(total_person_hours, 1),
            "Kiểm định Tô-pô": (
                "Pass (separated)" if topo_valid
                else ("Undetermined" if leaked_targets < 0 else f"Leak ({leaked_targets} px)")
            ),
            "Trạng thái Ràng buộc": "Feasible" if constraints_met else "Not feasible"
        }
        metrics_log.append(metrics_entry)

        # Preserve the best topology-verified candidate even if no candidate meets deadline.
        # If no verified candidate exists, leave masks empty instead of displaying a random cut.
        if topo_valid and route_pixels > 0:
            candidate_score = (
                max(0.0, completion_time_hours - min_fire_arrival),
                completion_time_hours,
                work_pixels,
                route_pixels,
                buffer_pixels,
            )
            if best_candidate_score is None or candidate_score < best_candidate_score:
                best_candidate_score = candidate_score
                best_firebreak = route_mask.copy()
                best_construction_mask = construction_mask.copy()
                best_partition = partition_mask.copy()
                best_eff_manpower = max_effective_manpower
                best_geom_ratio = work_geom_ratio
                best_work_len_3d = work_len_3d_km
                best_completion_time_hours = completion_time_hours
                best_metrics = dict(metrics_entry)

        if constraints_met:
            accepted = True
            margin = available_time - total_cutting_time
            status = (
                f"The plan satisfies the model constraints for this scenario "
                f"(Buffer: {buffer_pixels} px | Completion time: {completion_time_hours:.2f} h | "
                f"Time margin: +{margin:.2f} h)"
            )
            # The feasible cut just found is the best result; ensure it is selected.
            best_firebreak = route_mask.copy()
            best_construction_mask = construction_mask.copy()
            best_partition = partition_mask.copy()
            best_eff_manpower = max_effective_manpower
            best_geom_ratio = work_geom_ratio
            best_work_len_3d = work_len_3d_km
            best_completion_time_hours = completion_time_hours
            best_metrics = dict(metrics_entry)
            break

        buffer_pixels += buffer_step

    if not accepted:
        if np.any(best_firebreak):
            status = (
                "No route within the search area satisfies both the time and workforce constraints. "
                "The map shows the best continuous route that passed static topological validation; this route is NOT activated."
            )
        else:
            status = (
                "No continuous route passed topological validation. "
                "The unreliable route has been hidden. Change the parameters or study area, or check the OSM data."
            )

    slope_tiers_df = analyze_firebreak_slope_tiers(
        best_construction_mask, env["slope_deg"], cost_matrix,
        best_work_len_3d, best_eff_manpower, best_geom_ratio
    )

    # ==========================================================================
    # LƯỢT 2 (PASS 2): KỊCH BẢN CÓ TUYẾN ĐƯỜNG BĂNG ĐỀ XUẤT
    # ==========================================================================
    barrier_iters = 3 if (allow_spotting and wind_speed > 8.0) else 1
    thick_barrier = (
        binary_dilation(best_firebreak, structure=struct_8, iterations=barrier_iters)
        & (~best_partition | best_firebreak)
        & (~sink_guard_band)
    )

    history_controlled = np.zeros((total_steps, GRID_SHAPE[0], GRID_SHAPE[1]), dtype=np.uint8)
    history_controlled[0] = grid_t0.copy()

    grid_pass2 = grid_t0.copy()
    bt_pass2 = burn_time_t0.copy()
    rng_pass2 = np.random.default_rng(2026)

    for t in range(1, total_steps):
        current_hour = t * CA_STEP_TO_HOURS
        if accepted:
            # The full line becomes effective only after its modeled completion time.
            active_barrier = (
                thick_barrier
                if current_hour + 1e-9 >= best_completion_time_hours
                else np.zeros_like(thick_barrier)
            )
        else:
            # Do not simulate an infeasible candidate as if a partial line were complete.
            active_barrier = np.zeros_like(thick_barrier)

        grid_pass2, bt_pass2 = run_ca_step_vectorized(
            grid_pass2, bt_pass2, env, wind_speed, wind_deg, humidity, prcp, temp,
            p_base, burn_duration, rng_pass2, active_barrier_mask=active_barrier, allow_spotting=allow_spotting
        )
        history_controlled[t] = grid_pass2.copy()

    uncontrolled_burned = (history_uncontrolled[-1] == STATE_BURNING) | (history_uncontrolled[-1] == STATE_BURNED)
    controlled_burned   = (history_controlled[-1] == STATE_BURNING) | (history_controlled[-1] == STATE_BURNED)
    # Gross saved pixels are for spatial overlay; net reduction must subtract newly burned pixels too.
    reduced_burn_mask = uncontrolled_burned & (~controlled_burned)
    additional_burn_mask = controlled_burned & (~uncontrolled_burned)
    gross_saved_area_ha = float(np.sum(reduced_burn_mask)) * pixel_area_ha
    additional_burn_area_ha = float(np.sum(additional_burn_mask)) * pixel_area_ha
    uncontrolled_ha = float(np.sum(uncontrolled_burned)) * pixel_area_ha
    controlled_ha = float(np.sum(controlled_burned)) * pixel_area_ha
    reduced_area_ha = uncontrolled_ha - controlled_ha  # net reduction; may be negative
    reduction_pct = (reduced_area_ha / uncontrolled_ha * 100.0) if uncontrolled_ha > 0 else 0.0

    # Count both a burned crossing into the sink-side partition and any fire that
    # reached the planned barrier itself before it became effective. A barrier must
    # never erase a burning/burned state when activated.
    breach_pixels = (
        int(np.sum(controlled_burned & ((~best_partition & ~thick_barrier) | thick_barrier)))
        if accepted else 0
    )
    if accepted and breach_pixels > 0:
        accepted = False
        status = (
            f"Dynamic validation failed: {breach_pixels} pixels burned beyond the source side or validated route section "
            "in the controlled scenario. Static topological validation alone does not confirm containment."
        )
        if best_metrics is not None:
            best_metrics["Trạng thái Ràng buộc"] = "Dynamic validation failed"
            best_metrics["Kiểm định Tô-pô"] = "Static pass; dynamic failure"

    final_metrics = best_metrics if best_metrics is not None else {
        "Buffer (px)": buffer_pixels,
        "Số Pixel": 0,
        "Chiều dài 3D Mặt dốc (km)": 0.0,
        "Chiều dài 2D Phẳng (km)": 0.0,
        "Độ dài Pixel (N×30m)": 0.0,
        "Độ dốc TB / Max (°)": "0° / 0°",
        "TG tiếp cận P90 (giờ)": 0.0,
        "TG Lửa chạm (giờ)": 0.0,
        "Quỹ TG Thi công (giờ)": 0.0,
        "TG Đào theo Độ dốc (giờ)": 0.0,
        "Khối lượng (Giờ công)": 0.0,
        "Kiểm định Tô-pô": "Undetermined",
        "Trạng thái Ràng buộc": "Not feasible"
    }

    gc.collect()

    return {
        "history_uncontrolled": history_uncontrolled,
        "history_controlled": history_controlled,
        "reduced_burn_mask": reduced_burn_mask,
        "additional_burn_mask": additional_burn_mask,
        "reduced_area_ha": reduced_area_ha,
        "gross_saved_area_ha": gross_saved_area_ha,
        "additional_burn_area_ha": additional_burn_area_ha,
        "uncontrolled_burned_area_ha": uncontrolled_ha,
        "controlled_burned_area_ha": controlled_ha,
        "reduction_pct": reduction_pct,
        "breach_pixels": breach_pixels,
        "firebreak_mask": best_firebreak,
        "construction_mask": best_construction_mask,
        "death_zone_mask": death_zone_mask,
        "est_origin": (est_origin_r, est_origin_c),
        "accepted": accepted,
        "status": status,
        "final_metrics": final_metrics,
        "metrics_log": pd.DataFrame(metrics_log),
        "slope_tiers_df": slope_tiers_df
    }


# ==============================================================================
# LAYER 5: 3D HILLSHADE MAPS AND CONTROLLED SIMULATION COMPARISON
# ==============================================================================
def render_simulation_frame(env, fire_state, unctrl_state=None, show_saved_diff=False):
    img = env["hillshade_rgb"].copy()

    if show_saved_diff and unctrl_state is not None:
        was_burned = (unctrl_state == STATE_BURNING) | (unctrl_state == STATE_BURNED)
        now_safe = (fire_state == STATE_TREE)
        reduced_now = was_burned & now_safe
        img[reduced_now] = img[reduced_now] * 0.40 + np.array([0.15, 0.95, 0.35], dtype=np.float32) * 0.60

    burned_mask = (fire_state == STATE_BURNED)
    img[burned_mask] = img[burned_mask] * 0.25 + np.array([0.12, 0.10, 0.10], dtype=np.float32) * 0.75

    burning_mask = (fire_state == STATE_BURNING)
    img[burning_mask] = [1.0, 0.18, 0.0]

    fb_mask = (fire_state == STATE_FIREBREAK) & (~env["road_mask"]) & (~env["building_mask"]) & (~env["water_mask"])
    if np.any(fb_mask):
        img[fb_mask] = [0.0, 0.98, 0.85]

    # Keep existing OSM features visually distinct from new construction.
    img[env["water_mask"]] = [0.12, 0.42, 0.82]
    img[env["road_mask"] & (~env["water_mask"])] = [0.75, 0.78, 0.80]
    img[env["building_mask"] & (~env["water_mask"])] = [1.0, 0.72, 0.05]
    return np.clip(img, 0.0, 1.0)


def render_tactical_map(env, history_unctrl, history_ctrl, firebreak_mask, death_zone_mask):
    img = env["hillshade_rgb"].copy()

    unctrl_final = (history_unctrl[-1] == STATE_BURNING) | (history_unctrl[-1] == STATE_BURNED)
    ctrl_final   = (history_ctrl[-1] == STATE_BURNING) | (history_ctrl[-1] == STATE_BURNED)
    reduced_zone = unctrl_final & (~ctrl_final)
    img[reduced_zone] = img[reduced_zone] * 0.45 + np.array([0.20, 0.95, 0.40], dtype=np.float32) * 0.55

    img[ctrl_final] = img[ctrl_final] * 0.35 + np.array([0.98, 0.48, 0.0], dtype=np.float32) * 0.65

    dz_only = death_zone_mask & (~ctrl_final)
    img[dz_only] = img[dz_only] * 0.60 + np.array([0.75, 0.15, 0.45], dtype=np.float32) * 0.40

    img[env["water_mask"]] = [0.12, 0.42, 0.82]
    img[env["road_mask"] & (~env["water_mask"])] = [0.78, 0.80, 0.82]
    img[env["building_mask"] & (~env["water_mask"])] = [1.0, 0.75, 0.0]

    init_burned  = (history_unctrl[0] == STATE_BURNED)
    init_burning = (history_unctrl[0] == STATE_BURNING)
    img[init_burned]  = [0.18, 0.18, 0.18]
    img[init_burning] = [0.90, 0.0, 0.0]

    fb_vis = binary_dilation(firebreak_mask, iterations=1)
    # OSM roads are never painted as new firebreak in the tactical map.
    fb_vis &= (~env["road_mask"]) & (~env["building_mask"]) & (~env["water_mask"])
    img[fb_vis] = [0.0, 0.98, 0.85]
    # Draw source road/building data last so reuse of a road is visibly distinct from clearing.
    img[env["water_mask"]] = [0.12, 0.42, 0.82]
    img[env["road_mask"] & (~env["water_mask"])] = [0.78, 0.80, 0.82]
    img[env["building_mask"] & (~env["water_mask"])] = [1.0, 0.75, 0.0]

    return np.clip(img, 0.0, 1.0)


# ==============================================================================
# MAIN STREAMLIT INTERFACE (SECTIONS A–F)
# ==============================================================================
st.title("Wildfire spread simulation and firebreak planning")
st.caption(
    "Local GeoTIFF terrain data prioritized; sparse OpenTopoData fallback; OpenStreetMap Overpass and Open-Meteo weather. "
    "Methods: synchronized stochastic cellular automata (CRN), Dijkstra–Tobler (1993), slope aspect, and workforce-aware dynamic min-cut."
)

# ------------------------------------------------------------------------------
# SECTION A: STUDY AREA INPUTS (SIDEBAR)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("A. Study area and inputs")

    st.subheader("1. Study area coordinates")
    selected_region = st.selectbox(
        "Select a study area or enter custom coordinates:",
        list(REGION_PRESETS.keys())
    )
    preset = REGION_PRESETS[selected_region]

    if st.session_state.get("last_region") != selected_region:
        st.session_state["last_region"] = selected_region
        st.session_state["lat_input"] = preset["lat"]
        st.session_state["lon_input"] = preset["lon"]
        st.session_state["ign_r_in"]  = int(preset["ign_row"])
        st.session_state["ign_c_in"]  = int(preset["ign_col"])
        try:
            w_auto = fetch_realtime_weather_openmeteo(preset["lat"], preset["lon"])
            st.session_state["w_speed"] = float(w_auto["wind_speed"])
            st.session_state["w_deg"]   = int(round(w_auto["wind_deg"]))
            st.session_state["w_hum"]   = int(round(w_auto["humidity"]))
            st.session_state["w_prcp"]  = float(w_auto["prcp"])
            st.session_state["w_temp"]  = float(round(w_auto["temp"] * 2.0) / 2.0)
        except Exception:
            st.session_state.setdefault("w_speed", 6.0)
            st.session_state.setdefault("w_deg", 135)
            st.session_state.setdefault("w_hum", 55)
            st.session_state.setdefault("w_prcp", 0.0)
            st.session_state.setdefault("w_temp", 22.0)
        st.session_state["force_rerun"] = True

    st.session_state.setdefault("w_temp", 22.0)

    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat_center = st.number_input("Latitude", key="lat_input", format="%.4f", step=0.01)
    with col_lon:
        lon_center = st.number_input("Longitude", key="lon_input", format="%.4f", step=0.01)

    st.subheader("2. Reported fire location and uncertainty (T = 0)")
    col_r, col_c = st.columns(2)
    with col_r:
        ign_row = st.slider("Reported row", 20, 520, key="ign_r_in", step=5)
    with col_c:
        ign_col = st.slider("Reported column", 20, 520, key="ign_c_in", step=5)

    uncertainty_m = st.slider(
        "Reported location uncertainty radius (± m)",
        min_value=0, max_value=900, value=240, step=30,
        help=("The full uncertainty radius is retained for conservative planning. Candidate ignition seeds are limited to a cluster of at most 6 pixels "
              "to avoid creating multiple disconnected fires. If no valid forest cell is available, the model snaps only to the nearest forest cell within the input uncertainty "
              "(with a minimum search radius of 6 pixels); otherwise, it stops with an error instead of relocating the fire to an unrelated area.")
    )
    uncertainty_px = int(uncertainty_m // 30)

    is_smoke_report = st.checkbox(
        "Reported location is a smoke plume or fire front (back-cast the origin using wind)",
        value=True
    )

    initial_radius = st.slider("Initial ignition seed radius (pixels)", 2, 10, 3)
    preburn_hours = st.slider(
        "Fire detection delay (hours before T = 0)",
        min_value=0.0, max_value=8.0, value=2.0, step=0.5
    )

    st.subheader("3. Weather conditions")
    if st.button("Refresh weather data from API", width="stretch"):
        try:
            w_live = fetch_realtime_weather_openmeteo(lat_center, lon_center)
            st.session_state["w_speed"] = float(w_live["wind_speed"])
            st.session_state["w_deg"]   = int(round(w_live["wind_deg"]))
            st.session_state["w_hum"]   = int(round(w_live["humidity"]))
            st.session_state["w_prcp"]  = float(w_live["prcp"])
            st.session_state["w_temp"]  = float(round(w_live["temp"] * 2.0) / 2.0)
            st.session_state["force_rerun"] = True
            st.success(
                f"Retrieved from Open-Meteo: temperature {w_live['temp']}°C, "
                f"wind {w_live['wind_speed']} m/s, relative humidity {w_live['humidity']}%"
            )
        except Exception as e:
            st.error(f"Weather API connection failed: {e}")

    temp       = st.slider("Ambient temperature (°C)", 0.0, 45.0, key="w_temp", step=0.5)
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, key="w_speed", step=0.5)
    wind_deg   = st.slider("Wind direction FROM (meteorological degrees: 0° = north, 90° = east)", 0, 360, key="w_deg", step=5)
    humidity   = st.slider("Relative humidity (%)", 10, 100, key="w_hum", step=5)
    prcp       = st.slider("Precipitation (mm/h)", 0.0, 10.0, key="w_prcp", step=0.2)
    allow_spotting = st.checkbox(
        "Simulate fire spotting across the firebreak when wind speed exceeds 8 m/s",
        value=True
    )

    st.subheader("4. Simulation duration and workforce")
    sim_hours = st.slider("Simulation horizon (hours)", 4, 36, 18, 2)
    total_manpower = st.slider("Total available personnel", 100, 2000, 840, 20)
    squad_size = st.number_input("Crew size per work-front pixel", min_value=5, max_value=50, value=20)
    p_base = st.slider("Base ignition probability (p_base)", 0.12, 0.38, 0.24, 0.01)

    st.markdown("---")
    run_btn = st.button("Run simulation (fetch data and compute)", type="primary", width="stretch")

# ------------------------------------------------------------------------------
# SYNCHRONIZE INPUT STATE AND CALL APIS
# ------------------------------------------------------------------------------
current_ui_params = {
    "lat_center": float(lat_center),
    "lon_center": float(lon_center),
    "ign_row": int(ign_row),
    "ign_col": int(ign_col),
    "uncertainty_px": int(uncertainty_px),
    "is_smoke_report": bool(is_smoke_report),
    "initial_radius": int(initial_radius),
    "preburn_hours": float(preburn_hours),
    "wind_speed": float(wind_speed),
    "wind_deg": float(wind_deg),
    "humidity": float(humidity),
    "prcp": float(prcp),
    "temp": float(temp),
    "sim_hours": int(sim_hours),
    "total_manpower": int(total_manpower),
    "squad_size": int(squad_size),
    "p_base": float(p_base),
    "allow_spotting": bool(allow_spotting)
}

if (
    "sim_params" not in st.session_state
    or run_btn
    or st.session_state.get("force_rerun", False)
    or st.session_state["sim_params"] != current_ui_params
):
    st.session_state["force_rerun"] = False
    st.session_state["sim_params"] = current_ui_params
    st.session_state["current_step"] = 0
    st.session_state.pop("timeline_step_slider", None)

params = st.session_state["sim_params"]

try:
    with st.spinner(
        f"Fetching elevation and OpenStreetMap road/settlement data at ({params['lat_center']:.4f}, {params['lon_center']:.4f}) "
        "and solving the 3D min-cut optimization..."
    ):
        env = fetch_and_build_api_environment(params["lat_center"], params["lon_center"], _local_raster_inventory_signature())
        results = run_full_simulation_and_mincut(**params)
except Exception as run_err:
    st.error(
        f"**Simulation failed** at coordinates "
        f"`({params['lat_center']:.4f}, {params['lon_center']:.4f})`. "
        f"Details: `{type(run_err).__name__}: {run_err}`.\n\n"
        "Check API connectivity or data if the failure occurred while loading the map; check input assumptions if it occurred during simulation initialization. "
        "The application does not synthesize geographic data or substitute a fabricated firebreak when a request fails."
    )
    st.stop()

# Status message describing data sources used for the selected coordinates
st.info(
    f"**Live data status at `({params['lat_center']:.4f}°N, {params['lon_center']:.4f}°E)`:** "
        f"**Elevation:** `{env['elevation'].min():.0f}–{env['elevation'].max():.0f} m` "
    f"(mean slope `{env['slope_deg'].mean():.1f}°`, max `{env['slope_deg'].max():.1f}°`, "
    f"mean aspect `{env['aspect_deg'].mean():.0f}°`) | "
    f"**Temperature:** `{params['temp']:.1f}°C` | "
    f"**OpenStreetMap (`{env['mirror_used']}`):** `{env['road_cnt']} road features` (`{int(np.sum(env['road_mask'])):,} px`), "
    f"`{env['bld_cnt']} settlement/land-use features and {env['place_cnt']} populated places` (`{int(np.sum(env['building_mask'])):,} px`), "
    f"`{int(np.sum(env['water_mask'])):,} px` identified as water/coastline | "
    f"**Elevation source:** {env['dem_source']} | **Land-cover/fuel source:** {env['landcover_source']}"
)

if env["landcover_coverage_pct"] == 0.0:
    st.warning(
        "No local ESA WorldCover GeoTIFF was found. The model is using an aspect-based fuel proxy; "
        "do not interpret burned area or firebreak effectiveness as real-world forest outcomes. "
        "See DATA_GUIDE.md to configure suitable land-cover and elevation data."
    )
elif env["landcover_coverage_pct"] < 99.0:
    st.warning(
        f"Land-cover coverage is {env['landcover_coverage_pct']:.1f}%; results should be treated as preliminary."
    )

history_unctrl = results["history_uncontrolled"]
history_ctrl   = results["history_controlled"]
total_steps    = history_ctrl.shape[0]

# ------------------------------------------------------------------------------
# MAIN LAYOUT: SECTION B (LEFT) AND SECTION C (RIGHT)
# ------------------------------------------------------------------------------
col_b, col_c_map = st.columns(2)

# === SECTION B: FIRE-SPREAD SIMULATION ===
with col_b:
    st.subheader("B. Cellular automata fire-spread simulation")

    sim_view_mode = st.radio(
        "Simulation view:",
        [
            "Firebreak scenario (cellular automata containment)",
            "Control scenario (uncontrolled spread without a firebreak)",
            "Control comparison (highlight estimated avoided burning)"
        ],
        horizontal=True
    )

    ctrl_col1, ctrl_col2 = st.columns([1, 3])
    with ctrl_col1:
        play_clicked = st.button("Play animation", width="stretch")
    with ctrl_col2:
        if play_clicked:
            st.session_state.pop("timeline_step_slider", None)
            selected_step = int(np.clip(st.session_state.get("current_step", 0), 0, total_steps - 1))
            st.caption(f"Playing animation… final frame: step {total_steps - 1}")
        else:
            if "timeline_step_slider" not in st.session_state:
                st.session_state["timeline_step_slider"] = int(
                    np.clip(st.session_state.get("current_step", 0), 0, total_steps - 1)
                )
            selected_step = st.slider(
                "Simulation timeline (time step)",
                min_value=0,
                max_value=total_steps - 1,
                format="Step %d",
                key="timeline_step_slider"
            )
            st.session_state["current_step"] = selected_step

    sim_plot_placeholder = st.empty()
    sim_stats_placeholder = st.empty()

    def display_step_frame(step_idx):
        plt.close("all")
        if "uncontrolled spread" in sim_view_mode:
            frame_state = history_unctrl[step_idx]
            rgb_frame = render_simulation_frame(env, frame_state)
        elif "Control comparison" in sim_view_mode:
            frame_state = history_ctrl[step_idx]
            rgb_frame = render_simulation_frame(
                env, frame_state, unctrl_state=history_unctrl[step_idx], show_saved_diff=True
            )
        else:
            frame_state = history_ctrl[step_idx]
            rgb_frame = render_simulation_frame(env, frame_state)

        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=130)
        ax.imshow(rgb_frame)
        cs = ax.contour(env["elevation"], levels=8, colors="white", alpha=0.25, linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d m")

        if params["uncertainty_px"] > 0:
            uncert_circle = plt.Circle(
                (params["ign_col"], params["ign_row"]),
                params["uncertainty_px"],
                color="#00f5d4", fill=False, linestyle="--", linewidth=1.3
            )
            ax.add_patch(uncert_circle)

        orig_r, orig_c = results["est_origin"]
        ax.plot(params["ign_col"], params["ign_row"], marker="x", color="#00f5d4", markersize=8, markeredgewidth=2)
        if (orig_r, orig_c) != (params["ign_row"], params["ign_col"]):
            ax.plot(orig_c, orig_r, marker="*", color="#ffea00", markersize=9)

        elapsed_h = step_idx * CA_STEP_TO_HOURS
        ax.set_title(
            f"State at T = +{elapsed_h:.1f} h (including {params['preburn_hours']} h detection delay)",
            fontsize=9.5, fontweight="bold"
        )
        ax.axis("off")

        legend_patches = [
            mpatches.Patch(color="#00fad9", label="Activated firebreak (after completion)"),
            mpatches.Patch(color="#ff2e00", label="Active fire front"),
            mpatches.Patch(color="#2a2424", label="Burned area"),
            mpatches.Patch(color="#26f259", label="Estimated reduction in burned area (vs. control)"),
            mpatches.Patch(color="#ffb703", label="OSM settlements (protected targets)")
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7.3, framealpha=0.88)
        plt.tight_layout()
        sim_plot_placeholder.pyplot(fig)
        plt.close(fig)

        burning_px = int(np.sum(frame_state == STATE_BURNING))
        burned_px  = int(np.sum(frame_state == STATE_BURNED))
        burned_ha  = (burning_px + burned_px) * env["pixel_area_ha"]

        unctrl_step = history_unctrl[step_idx]
        unctrl_ha = int(np.sum((unctrl_step == STATE_BURNING) | (unctrl_step == STATE_BURNED))) * env["pixel_area_ha"]
        reduced_step_ha = max(0.0, unctrl_ha - burned_ha)

        sim_stats_placeholder.info(
            f"**Step {step_idx}/{total_steps-1} (+{elapsed_h:.1f} h):** "
            f"Burning: **{burning_px:,} px** | "
            f"Burned area: **{burned_ha:.1f} ha** | "
            f"Estimated burned-area reduction (vs. control): **{reduced_step_ha:.1f} ha**"
        )

    if play_clicked:
        for frame_step in range(0, total_steps, 2):
            st.session_state["current_step"] = frame_step
            display_step_frame(frame_step)
            time.sleep(0.08)
        st.session_state["current_step"] = total_steps - 1
        display_step_frame(total_steps - 1)
        st.rerun()
    else:
        display_step_frame(selected_step)

# === SECTION C: ALGORITHM-PROPOSED FIREBREAK MAP ===
with col_c_map:
    st.subheader("C. Algorithm-proposed firebreak map (T = max)")
    st.write("")
    st.write("")

    tactical_rgb = render_tactical_map(
        env, history_unctrl, history_ctrl,
        results["construction_mask"], results["death_zone_mask"]
    )

    fig_tac, ax_tac = plt.subplots(figsize=(6.5, 6.5), dpi=130)
    ax_tac.imshow(tactical_rgb)
    cs_tac = ax_tac.contour(env["elevation"], levels=8, colors="white", alpha=0.25, linewidths=0.6)
    ax_tac.clabel(cs_tac, inline=True, fontsize=6, fmt="%d m")

    if params["uncertainty_px"] > 0:
        uncert_circle_tac = plt.Circle(
            (params["ign_col"], params["ign_row"]),
            params["uncertainty_px"],
            color="white", fill=False, linestyle="--", linewidth=1.3
        )
        ax_tac.add_patch(uncert_circle_tac)

    orig_r, orig_c = results["est_origin"]
    ax_tac.plot(params["ign_col"], params["ign_row"], marker="x", color="white", markersize=8, markeredgewidth=2)
    if (orig_r, orig_c) != (params["ign_row"], params["ign_col"]):
        ax_tac.plot(orig_c, orig_r, marker="*", color="#ffea00", markersize=9)

    rad_w = np.radians(params["wind_deg"])
    w_dy, w_dx = np.cos(rad_w) * 35, -np.sin(rad_w) * 35
    ax_tac.annotate(
        "", xy=(60 + w_dx, 60 + w_dy), xytext=(60, 60),
        arrowprops=dict(arrowstyle="->", color="white", lw=2.2)
    )
    ax_tac.text(
        25, 22, f"Wind: {params['wind_speed']} m/s ({params['wind_deg']}°) | {params['temp']}°C",
        color="white", fontsize=8, fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.6, pad=2)
    )

    ax_tac.set_title("Proposed firebreak: 3D min-cut and simulation comparison", fontsize=9.5, fontweight="bold")
    ax_tac.axis("off")

    tac_patches = [
        mpatches.Patch(color="#00fad9", label="New firebreak (construction required)"),
        mpatches.Patch(color="#fa7a00", label="Burned area with firebreak"),
        mpatches.Patch(color="#33f266", label=f"Burned only in control ({results['gross_saved_area_ha']:.1f} ha)"),
        mpatches.Patch(color="#e60000", label="Fire perimeter at T = 0"),
        mpatches.Patch(color="#bf2673", label="Inaccessible zone"),
        mpatches.Patch(color="#ffbf00", label="OSM settlements (protected targets)"),
        mpatches.Patch(color="#c7ccd1", label="OSM roads")
    ]
    ax_tac.legend(handles=tac_patches, loc="lower right", fontsize=7.1, framealpha=0.88)
    plt.tight_layout()
    st.pyplot(fig_tac)
    plt.close(fig_tac)

    if results["accepted"]:
        if results["breach_pixels"] == 0:
            net_area = results["reduced_area_ha"]
            if net_area >= 0:
                comparison_text = (
                    f"Difference in burned area (control − firebreak): **{net_area:.1f} ha** "
                    f"({results['reduction_pct']:+.1f}%)."
                )
            else:
                comparison_text = (
                    f"Net increase in burned area: **{abs(net_area):.1f} ha** "
                    f"({results['reduction_pct']:+.1f}%) compared with the control."
                )
            st.success(
                f"**Assessment:** {results['status']}.\n\n"
                f"**Control comparison:** {comparison_text} "
                f"Area burned only in the control: {results['gross_saved_area_ha']:.1f} ha; "
                f"area burned only in the firebreak scenario: {results['additional_burn_area_ha']:.1f} ha."
            )
        else:
            st.warning(
                f"**Assessment:** {results['status']}.\n\n"
                f"**Note:** The proposed route separates the static graph, but "
                f"**{results['breach_pixels']} pixels** burned beyond the validated boundary in the dynamic simulation."
            )
    else:
        st.error(f"**Assessment:** {results['status']}")

# ------------------------------------------------------------------------------
# SECTION D: FIREBREAK METRICS AND MODEL CHECKS
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("D. Firebreak metrics and model checks")
st.caption(
    "Time, labor, and validation-status values are model estimates; they do not replace field evaluation or calibration."
)

fm = results["final_metrics"]

st.markdown(
    """
    <style>
    div[data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.25 !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div[data-testid="stMetricDelta"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.25 !important;
        font-size: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

metric_row_1 = st.columns(3, gap="large")
with metric_row_1[0]:
    st.metric(
        "3D firebreak length",
        f"{fm['Chiều dài 3D Mặt dốc (km)']:.2f} km",
        f"2D length: {fm['Chiều dài 2D Phẳng (km)']:.2f} km · Pixel-based: {fm['Độ dài Pixel (N×30m)']:.2f} km",
    )
with metric_row_1[1]:
    st.metric(
        "Mean / maximum slope",
        fm["Độ dốc TB / Max (°)"],
        f"Labor: {fm['Khối lượng (Giờ công)']:,.0f} person-hours",
    )
with metric_row_1[2]:
    st.metric(
        "P90 access time",
        f"{fm['TG tiếp cận P90 (giờ)']:.2f} h" if isinstance(fm["TG tiếp cận P90 (giờ)"], (int, float, np.integer, np.floating)) else fm["TG tiếp cận P90 (giờ)"],
        f"Mean reference walking speed: {env['walk_speed_kmh'].mean():.2f} km/h",
    )

metric_row_2 = st.columns(3, gap="large")
with metric_row_2[0]:
    st.metric(
        "Available time before fire arrival",
        f"{fm['Quỹ TG Thi công (giờ)']:.2f} h",
        f"Modeled deadline: {fm['TG Lửa chạm (giờ)']} h",
    )
with metric_row_2[1]:
    st.metric(
        "Estimated construction time",
        f"{fm['TG Đào theo Độ dốc (giờ)']:.2f} h",
        f"Personnel: {params['total_manpower']:,} (crew size {params['squad_size']} per work-front pixel)",
    )
with metric_row_2[2]:
    st.metric(
        "Separation check",
        fm["Kiểm định Tô-pô"],
        fm["Trạng thái Ràng buộc"],
    )

col_t1, col_t2 = st.columns([1.15, 1.0])
with col_t1:
    st.markdown("##### 1. Slope difficulty and construction time")
    if not results["slope_tiers_df"].empty:
        slope_table = results["slope_tiers_df"].rename(columns={
            "Phân cấp Độ khó Phát quang": "Slope tier",
            "Số Pixel": "Pixels",
            "Tỷ trọng (%)": "Share (%)",
            "Độ dốc TB (°)": "Mean slope (°)",
            "Chiều dài 3D (km)": "3D length (km)",
            "Hệ số Độ khó & Mặt dốc": "Difficulty factor",
            "Khối lượng (Giờ công)": "Person-hours",
            "TG Thi công Phân bổ (giờ)": "Construction time (h)",
        })
        st.dataframe(slope_table, width="stretch", hide_index=True)
with col_t2:
    st.markdown("##### 2. Buffer search log")
    metrics_table = results["metrics_log"].rename(columns={
        "Buffer (px)": "Buffer (px)",
        "Số Pixel": "Route pixels",
        "Pixel thi công mới": "New construction pixels",
        "Chiều dài 3D Mặt dốc (km)": "3D length (km)",
        "Chiều dài 2D Phẳng (km)": "2D length (km)",
        "Độ dài Pixel (N×30m)": "Pixel-based length (km)",
        "Độ dốc TB / Max (°)": "Mean / max slope (°)",
        "TG tiếp cận P90 (giờ)": "P90 access time (h)",
        "TG Lửa chạm (giờ)": "Fire arrival (h)",
        "Quỹ TG Thi công (giờ)": "Available time (h)",
        "TG Đào theo Độ dốc (giờ)": "Construction time (h)",
        "Khối lượng (Giờ công)": "Person-hours",
        "Kiểm định Tô-pô": "Topological check",
        "Trạng thái Ràng buộc": "Constraints",
    })
    st.dataframe(metrics_table, width="stretch", hide_index=True)

# ------------------------------------------------------------------------------
# SECTION E: ALGORITHMS USED IN THE SYSTEM
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("E. Algorithms used in the system")

st.markdown(
    r"""
    1. **3D-aware stochastic cellular automata with common random numbers (CRN):**
       Simulates fire spread on a $540 \times 540$ grid using Moore neighborhoods (8 directions). It accounts for 3D propagation distance, directional slope, solar exposure (`aspect_deg`), ambient temperature (`temp`), wind, humidity, precipitation, and fire spotting.
    2. **Wind- and slope-aware ignition back-casting with spatial uncertainty sampling:**
       Back-projects the combined wind vector and terrain gradient over the reporting delay ($\Delta t_{\text{delay}}$) to estimate the original ignition location at $T = -\Delta t_{\text{delay}}$. Candidate ignition seeds are kept in a compact cluster of up to 6 pixels rather than treating the entire uncertainty radius ($\pm R$) as simultaneous, disconnected fires. The full uncertainty remains available for conservative planning.
    3. **Virtual-source directed Dijkstra with Tobler's hiking function (1993):**
       Builds a directed 8-neighbor grid graph whose edge costs depend on signed elevation change $\Delta z$ (uphill/downhill) using Tobler's speed function. A single Dijkstra run from a virtual source connected to OpenStreetMap roads produces the access-time grid (`arrival_time`).
    4. **Continuous slope/aspect clearing difficulty and multi-echelon workforce allocation:**
       Combines a continuous difficulty function across four slope tiers ($0^\circ$–$15^\circ$, $15^\circ$–$25^\circ$, $25^\circ$–$35^\circ$, and $>35^\circ$), slope-surface expansion $1/\cos\theta$, a southwesterly aspect-related vegetation factor (`aspect_deg`), a heat-related factor (`temp`), and workforce capacity (`total_manpower`, `squad_size`) to estimate the required stand-off distance.
    5. **Bounding-box cropped max-flow/min-cut with radial forest conservation:**
       Solves `maximum_flow` on a cropped tactical search area around the source region. A radial conservation weight encourages closer routes when more personnel are available and allows the search to consider more favorable terrain when workforce is limited.
    6. **Zhang–Suen skeletonization and 3D firebreak arc-length estimation:**
       Thins the firebreak mask to a one-pixel centerline using Zhang–Suen within a bounding box, then estimates its planimetric 2D length ($L_{\text{2D}}$) and terrain-surface 3D length ($L_{\text{3D}}$).
    7. **Independent 8-connected topological separation check using flood-fill BFS:**
       Independently checks the extracted firebreak mask for an 8-neighbor path from the source fire to protected OpenStreetMap settlements or the map boundary.
    8. **Bicubic DEM interpolation, finite-difference gradients, and hillshading:**
       Interpolates elevation samples from the selected data source, estimates spatial gradients ($\partial z/\partial x, \partial z/\partial y$), derives `slope_deg` and `aspect_deg`, and renders shaded terrain for visualization.
    """
)

# ------------------------------------------------------------------------------
# SECTION F: MODEL ASSUMPTIONS AND LIMITATIONS
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("F. Model assumptions and limitations")
st.caption(
    "For high-resolution study data, place DEM GeoTIFF files in data/dem/ and ESA WorldCover 2021 v200 Map GeoTIFF files in data/worldcover/. "
    "The accompanying data guide explains how to download the tiles required for each study area."
)

st.markdown(
    r"""
    The following assumptions and limitations are documented for scientific transparency. Key parameters have not been calibrated directly against field measurements.

    * **1. Logistics and access assumptions:**
      * Travel times from OSM road cells are accumulated on a directed graph using Dijkstra and Tobler's hiking function (1993), with a heuristic factor `eta_forest = 0.55`. The current model does not include separately mapped ranger stations or patrol trails.
      * Rivers and lakes are treated as non-burnable where they are identified by OSM or WorldCover. Crossing rivers, bridges, and other hydrological obstacles is not fully modeled for personnel movement.
    * **2. Clearing productivity and slope/aspect difficulty:**
      * The baseline productivity on flat terrain ($0^\circ$) is referenced from the average manual handline productivity in the **NWCG Fireline Handbook (PMS 410-1, USFS)** for brush and mixed forest fuels (approximately $12$ linear meters per person-hour, or about $2.5$ person-hours for each $30$ m segment).
      * As slope crosses four tiers ($0^\circ$–$15^\circ$, $15^\circ$–$25^\circ$, $25^\circ$–$35^\circ$, and $>35^\circ$), estimated labor increases continuously through an exponential function combined with slope-surface expansion $1/\cos\theta$, aspect (`aspect_deg`), and temperature (`temp`). These rates have **not been calibrated using field time-and-motion data from Vietnamese forest rangers**, and bulldozer-based construction is not modeled separately.
    * **3. Land-cover and microclimate approximations:**
      * When ESA WorldCover 2021 v200 GeoTIFF files are available in `data/worldcover/`, land-cover classes are mapped to heuristic relative fuel factors for tree cover, shrubland, grassland, cropland, and selected wetland/mangrove classes. These factors are not measurements of biomass or on-site fuel moisture. Without this layer, the model uses an aspect-based proxy; burned-area outputs must not be interpreted as actual forest-area estimates.
      * Roads, settlements, and water are assigned zero fuel. The continuous Dijkstra travel field only creates fire-propagation edges between cells classified as burnable, rather than forcing fuel values to a positive lower bound using `np.clip`.
    * **4. API data resolution and containment-validation scope:**
      * The system prioritizes local DEM GeoTIFFs that cover the full map. If no local DEM is available, it queries OpenTopoData SRTM90m at 400 locations (a $20 \times 20$ grid) and interpolates those samples onto the $540 \times 540$ calculation grid. Sample spacing is approximately 874 m across this map extent, so the interpolated output must not be described as a true 30 m elevation dataset and does not preserve microtopography.
      * Burned-area change is a paired comparison between two simulations using common random seeds. CRN does not guarantee monotonic results: the firebreak scenario may produce more burned area in some locations because the simulated state and access timing differ. Dynamic checks are performed after simulation, but do not replace independent field validation or calibration.
    * **5. Route-planning and workforce guardrails:**
      * The min-cut source region is derived from a preliminary CA fire footprint at a forecast time that depends on workforce. This is an approximation, not a construction schedule modeled by individual pixel or crew front. A firebreak is activated in the simulation only when the plan passes topological checks and its estimated completion time meets the modeled deadline.
      * Candidate ignition seeds are restricted to a cluster of up to 6 pixels to avoid interpreting reporting uncertainty as several disconnected fires. If no valid forest cell is available, the model only snaps to the nearest forest cell within the input uncertainty (with a minimum radius of 6 pixels); otherwise, the simulation stops instead of relocating the fire far away. This is an approximation, not a formal probability distribution for ignition location.
      * Only the largest connected route component is displayed. Disconnected spot fires may require separate lines or crews and are not guaranteed to be contained by the main route.
      * Workforce affects the initial buffer, search-collar width, effective personnel, labor estimates, and the footprint used as the min-cut source. Efficiency factors remain heuristics that have not been calibrated in the field. The output is for research simulation, not operational firefighting guidance.
    """
)
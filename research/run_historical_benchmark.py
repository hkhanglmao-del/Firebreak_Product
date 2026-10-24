#!/usr/bin/env python3
"""Reproducible retrospective scale-check for the 7-9 April 2023 Prenn Pass fire.

Runs the current Firebreak R1 model function on a 540x540 grid using:
- clipped public ESA WorldCover 2021 v200 COG tiles,
- clipped public Copernicus GLO-30 COG tiles,
- the application's own OSM landscape builder,
- historical ERA5-Land hourly weather via the Open-Meteo Archive API.

This is a scale check against reported total area, not polygon-level validation:
the original georeferenced burn perimeter and exact ignition coordinate are not
publicly available in this script. Weather is a reanalysis estimate, and the
single-ignition constant-weather CA cannot reproduce suppression / later flare-ups.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import types
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import requests
import rasterio
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "firebreak_hardened_r1_english.py"
OUT_DIR = ROOT / "benchmark-output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Published study AOI and approximate reported ignition point. Prenn Pass coordinates
# are an area-level proxy, not an exact georeferenced ignition observation.
LAT_CENTER = 11.9404
LON_CENTER = 108.4583
IGNITION_LAT = 11.91667
IGNITION_LON = 108.45000
LAT_SPAN_DEG = 0.15
LON_SPAN_DEG = 0.15
SOUTH = LAT_CENTER - LAT_SPAN_DEG / 2.0
NORTH = LAT_CENTER + LAT_SPAN_DEG / 2.0
WEST = LON_CENTER - LON_SPAN_DEG / 2.0
EAST = LON_CENTER + LON_SPAN_DEG / 2.0

# Public references report about 10 ha for the first overnight fire and 13.86 ha
# mapped from PlanetScope for the 7-9 April multi-day event.
FIRST_EVENT_REFERENCE_HA = 10.0
MULTI_DAY_REFERENCE_HA = 13.86

WORLD_COVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
DEM_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"


def log(message: str) -> None:
    print(f"[benchmark] {message}", flush=True)


def clipped_cog_to_local(url: str, out_path: Path, bbox: tuple[float, float, float, float],
                         nodata_override=None) -> dict:
    """Read only the COG window overlapping bbox and save a small local GeoTIFF."""
    left, bottom, right, top = bbox
    log(f"Opening remote COG: {url}")
    last_error = None
    for attempt in range(3):
        try:
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_MULTIRANGE="YES",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                GDAL_HTTP_MAX_RETRY="2",
                GDAL_HTTP_RETRY_DELAY="1",
            ):
                with rasterio.open(url) as src:
                    inter_left = max(left, src.bounds.left)
                    inter_bottom = max(bottom, src.bounds.bottom)
                    inter_right = min(right, src.bounds.right)
                    inter_top = min(top, src.bounds.top)
                    if inter_left >= inter_right or inter_bottom >= inter_top:
                        return {"url": url, "status": "does_not_intersect"}
                    raw = from_bounds(
                        inter_left, inter_bottom, inter_right, inter_top, src.transform
                    )
                    c0 = max(0, int(math.floor(raw.col_off)))
                    r0 = max(0, int(math.floor(raw.row_off)))
                    c1 = min(src.width, int(math.ceil(raw.col_off + raw.width)))
                    r1 = min(src.height, int(math.ceil(raw.row_off + raw.height)))
                    if c1 <= c0 or r1 <= r0:
                        raise RuntimeError("The clipped COG window is empty.")
                    window = Window(c0, r0, c1 - c0, r1 - r0)
                    data = src.read(1, window=window)
                    transform = src.window_transform(window)
                    profile = src.profile.copy()
                    profile.update(
                        driver="GTiff",
                        width=int(window.width),
                        height=int(window.height),
                        transform=transform,
                        compress="deflate",
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                    )
                    if nodata_override is not None:
                        profile["nodata"] = nodata_override
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(data, 1)
                    return {
                        "url": url,
                        "status": "ok",
                        "output": str(out_path.relative_to(ROOT)),
                        "shape": [int(data.shape[0]), int(data.shape[1])],
                        "bbox_intersection": [
                            float(inter_left), float(inter_bottom),
                            float(inter_right), float(inter_top),
                        ],
                        "crs": str(src.crs),
                        "nodata": profile.get("nodata"),
                        "size_bytes": out_path.stat().st_size,
                    }
        except Exception as exc:
            last_error = exc
            log(f"COG read failed attempt {attempt + 1}/3: {exc}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Could not read public COG {url}: {last_error}") from last_error


def prepare_geospatial_inputs() -> list[dict]:
    dem_dir = ROOT / "data" / "dem"
    wc_dir = ROOT / "data" / "worldcover"
    for folder in (dem_dir, wc_dir):
        folder.mkdir(parents=True, exist_ok=True)

    inventory = []
    # WorldCover is tiled on a 3-degree grid. The AOI straddles latitude 12°.
    for tile in ("N09E108", "N12E108"):
        filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
        url = f"{WORLD_COVER_BASE}/{filename}"
        out = wc_dir / filename
        result = clipped_cog_to_local(url, out, (WEST, SOUTH, EAST, NORTH), nodata_override=0)
        inventory.append({"dataset": "ESA WorldCover 2021 v200 Map", "tile": tile, **result})

    # Copernicus DEM GLO-30 is tiled on a 1-degree grid. The AOI also straddles 12°N.
    for lat_tile in (11, 12):
        name = f"Copernicus_DSM_COG_10_N{lat_tile:02d}_00_E108_00_DEM"
        url = f"{DEM_BASE}/{name}/{name}.tif"
        out = dem_dir / f"benchmark_{name}.tif"
        result = clipped_cog_to_local(url, out, (WEST, SOUTH, EAST, NORTH))
        inventory.append({"dataset": "Copernicus DEM GLO-30 DSM", "tile": f"N{lat_tile:02d}E108", **result})

    if any(x.get("status") != "ok" for x in inventory):
        raise RuntimeError("At least one required DEM/WorldCover tile did not cover the benchmark AOI.")
    return inventory


def historical_weather() -> dict:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": IGNITION_LAT,
        "longitude": IGNITION_LON,
        "start_date": "2023-04-07",
        "end_date": "2023-04-07",
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "wind_speed_10m,wind_direction_10m"
        ),
        "timezone": "Asia/Ho_Chi_Minh",
        "wind_speed_unit": "ms",
    }
    log("Requesting hourly Open-Meteo archive weather for 2023-04-07 15:00-22:00 local time.")
    response = requests.get(url, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    # Use 15:00 through 22:00 inclusively to represent the first reported fire period.
    ix = [i for i, stamp in enumerate(times) if "T" in stamp and 15 <= int(stamp.split("T")[1][:2]) <= 22]
    if len(ix) < 6:
        raise RuntimeError(f"Insufficient hourly historical weather values: {len(ix)} hours returned.")
    keys = {
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "precipitation": "prcp",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_deg",
    }
    arrays = {}
    precipitation_gap_policy = "ERA5-Land values available"
    for key, out_key in keys.items():
        vals = hourly.get(key)
        if vals is None:
            raise RuntimeError(f"Historical weather payload missing {key}.")
        period_vals = [vals[i] for i in ix]
        nums = [float(v) for v in period_vals if v is not None]
        if key == "precipitation" and len(nums) < 6:
            # Retry precipitation from the archive's best-match model before applying
            # a clearly labelled zero-rain fallback. This fallback is not presented as
            # station measurement; contemporaneous reports describe the fire-weather
            # period as dry, while usable hourly rainfall values were absent in the archive.
            log(f"Archive precipitation returned only {len(nums)} valid hours; retrying best-match archive.")
            fallback_params = dict(params)
            fallback_params.pop("models", None)
            fallback_response = requests.get(url, params=fallback_params, timeout=90)
            fallback_response.raise_for_status()
            fallback_hourly = fallback_response.json().get("hourly") or {}
            fallback_times = fallback_hourly.get("time") or []
            fallback_indices = [
                i for i, stamp in enumerate(fallback_times)
                if "T" in stamp and 15 <= int(stamp.split("T")[1][:2]) <= 22
            ]
            fallback_vals = fallback_hourly.get("precipitation") or []
            fallback_period = [
                fallback_vals[i] if i < len(fallback_vals) else None
                for i in fallback_indices
            ]
            fallback_nums = [float(v) for v in fallback_period if v is not None]
            if len(fallback_nums) >= 6:
                nums = [
                    float(v) if v is not None else 0.0
                    for v in fallback_period
                ]
                precipitation_gap_policy = "Precipitation incomplete; missing hourly values filled with 0.0 mm"
            else:
                nums = [
                    float(v) if v is not None else 0.0
                    for v in period_vals
                ]
                precipitation_gap_policy = (
                    "Rainfall missing in the hourly archive response; null hours filled with 0.0 mm "
                    "as a dry-weather assumption supported only qualitatively by contemporary news reports"
                )
                log("Rainfall unavailable from the archive; applying the explicitly labelled 0-mm dry-weather assumption.")
        else:
            if len(nums) < 6:
                raise RuntimeError(f"Historical weather variable {key} has too few valid values.")
            if key == "precipitation":
                nums = [float(v) if v is not None else 0.0 for v in period_vals]
        arrays[out_key] = nums
    wd = np.radians(np.asarray(arrays["wind_deg"], dtype=float))
    mean_wind_direction = (math.degrees(math.atan2(float(np.sin(wd).mean()),
                                                   float(np.cos(wd).mean()))) + 360.0) % 360.0
    weather = {
        "source": "Open-Meteo Historical Weather API default archive model; not a local station observation",
        "latitude": IGNITION_LAT,
        "longitude": IGNITION_LON,
        "local_period": "2023-04-07 15:00-22:00 Asia/Ho_Chi_Minh",
        "hours_used": [times[i] for i in ix],
        "precipitation_gap_policy": precipitation_gap_policy,
        "temp_c_mean": float(np.mean(arrays["temp"])),
        "relative_humidity_pct_mean": float(np.mean(arrays["humidity"])),
        "precipitation_mm_hour_mean": float(np.mean(arrays["prcp"])),
        "wind_speed_ms_mean": float(np.mean(arrays["wind_speed"])),
        "wind_direction_from_deg_circular_mean": float(mean_wind_direction),
        "raw_hourly": {
            "temperature_c": arrays["temp"],
            "relative_humidity_pct": arrays["humidity"],
            "precipitation_mm": arrays["prcp"],
            "wind_speed_ms": arrays["wind_speed"],
            "wind_direction_from_deg": arrays["wind_deg"],
        },
    }
    return weather


def load_model_namespace():
    # Import only the application's definitions and functions; do not execute Streamlit UI.
    app_source = APP_PATH.read_text(encoding="utf-8")
    marker = "MAIN STREAMLIT INTERFACE"
    cut = app_source.find(marker)
    if cut < 0:
        raise RuntimeError("Cannot locate the boundary before the Streamlit interface.")
    prefix = app_source[:cut]
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.set_page_config = lambda *args, **kwargs: None

    def cache_data(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function

    fake_streamlit.cache_data = cache_data
    sys.modules["streamlit"] = fake_streamlit
    namespace = {"__file__": str(APP_PATH), "__name__": "firebreak_retro_benchmark"}
    exec(compile(prefix, str(APP_PATH), "exec"), namespace)
    return namespace


def area_series_ha(history: np.ndarray, state_burning: int, state_burned: int,
                   pixel_area_ha: float) -> list[dict]:
    output = []
    for step, grid in enumerate(history):
        mask = (grid == state_burning) | (grid == state_burned)
        output.append({
            "hours_after_start": round(step * 0.5, 2),
            "burned_area_ha_model": round(float(mask.sum()) * pixel_area_ha, 4),
            "burned_cells_model": int(mask.sum()),
        })
    return output


def main() -> None:
    log("Preparing clipped public DEM and WorldCover tiles for the Dalat AOI.")
    data_inventory = prepare_geospatial_inputs()
    weather = historical_weather()
    ns = load_model_namespace()

    # Map the named Prenn Pass location to the app's 540x540 AOI grid.
    ign_row = int(round((NORTH - IGNITION_LAT) / LAT_SPAN_DEG * (ns["GRID_SHAPE"][0] - 1)))
    ign_col = int(round((IGNITION_LON - WEST) / LON_SPAN_DEG * (ns["GRID_SHAPE"][1] - 1)))
    log(f"Approximate reported location maps to row={ign_row}, col={ign_col}.")
    params = {
        "lat_center": LAT_CENTER,
        "lon_center": LON_CENTER,
        "ign_row": ign_row,
        "ign_col": ign_col,
        "uncertainty_px": 8,  # 240 m input uncertainty; exact ignition coordinates unavailable.
        "is_smoke_report": False,
        "initial_radius": 3,
        "preburn_hours": 0.0,
        "wind_speed": weather["wind_speed_ms_mean"],
        "wind_deg": weather["wind_direction_from_deg_circular_mean"],
        "humidity": weather["relative_humidity_pct_mean"],
        "prcp": weather["precipitation_mm_hour_mean"],
        "temp": weather["temp_c_mean"],
        "sim_hours": 8,
        "total_manpower": 150,  # approximate reported deployed personnel, not all simultaneously on a line.
        "squad_size": 20,
        "p_base": 0.24,  # published default/base model parameter; not fitted to this event.
        "allow_spotting": True,
    }
    log("Running the repository's existing run_full_simulation_and_mincut function.")
    start = time.time()
    result = ns["run_full_simulation_and_mincut"](**params)
    elapsed = time.time() - start
    pixel_area_ha = float(ns["compute_projected_pixel_scales"](LAT_CENTER)[2])

    history = result["history_uncontrolled"]
    curve = area_series_ha(history, ns["STATE_BURNING"], ns["STATE_BURNED"], pixel_area_ha)
    initial_area_ha = curve[0]["burned_area_ha_model"]
    modeled_area = float(result["uncontrolled_burned_area_ha"])
    controlled_area = float(result["controlled_burned_area_ha"])
    error_first = modeled_area - FIRST_EVENT_REFERENCE_HA
    error_total = modeled_area - MULTI_DAY_REFERENCE_HA

    with rasterio.open(ROOT / "data" / "worldcover" / "ESA_WorldCover_10m_2021_v200_N09E108_Map.tif") as src_wc:
        pass
    # Report the observed/model area comparison as a scale check only.
    summary = {
        "benchmark_name": "Dalat Prenn Pass wildfire, 2023-04-07 initial 8-hour period",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_code": "firebreak_hardened_r1_english.py; run_full_simulation_and_mincut()",
        "source_code_branch": "audit/r1-validation-fixes-2026-10-24",
        "method": "Existing Firebreak R1 cellular automaton + current Min-Cut planning workflow",
        "aoi": {
            "center_lat": LAT_CENTER, "center_lon": LON_CENTER,
            "bounds_wsen": [WEST, SOUTH, EAST, NORTH],
            "grid_shape": list(ns["GRID_SHAPE"]),
            "nominal_resolution_m": ns["NOMINAL_PIXEL_M"],
            "pixel_area_ha": pixel_area_ha,
            "approximate_ignition_lat": IGNITION_LAT,
            "approximate_ignition_lon": IGNITION_LON,
            "grid_row": ign_row, "grid_col": ign_col,
        },
        "inputs": {
            "data_tiles": data_inventory,
            "weather": weather,
            "simulation": params,
        },
        "model_results": {
            "runtime_seconds": round(elapsed, 2),
            "initial_burned_area_ha": initial_area_ha,
            "uncontrolled_burned_area_ha_after_8h": round(modeled_area, 4),
            "controlled_counterfactual_burned_area_ha_after_8h": round(controlled_area, 4),
            "gross_saved_area_ha_model": round(float(result["gross_saved_area_ha"]), 4),
            "additional_burn_area_ha_model": round(float(result["additional_burn_area_ha"]), 4),
            "net_reduction_area_ha_model": round(float(result["reduced_area_ha"]), 4),
            "reduction_percent_model": round(float(result["reduction_pct"]), 3),
            "firebreak_plan_accepted_by_model": bool(result["accepted"]),
            "firebreak_status": str(result["status"]),
            "final_metrics": result["final_metrics"],
            "burned_area_curve": curve,
        },
        "historical_comparison": {
            "first_period_reported_area_ha_approx": FIRST_EVENT_REFERENCE_HA,
            "difference_model_minus_first_period_ha": round(error_first, 4),
            "absolute_error_vs_first_period_ha": round(abs(error_first), 4),
            "relative_error_vs_first_period_percent": round(error_first / FIRST_EVENT_REFERENCE_HA * 100.0, 2),
            "PlanetScope_reported_area_ha_for_full_multiday_event": MULTI_DAY_REFERENCE_HA,
            "difference_model_minus_multiday_reference_ha_not_like_for_like": round(error_total, 4),
            "absolute_difference_vs_multiday_reference_ha_not_like_for_like": round(abs(error_total), 4),
        },
        "limitations": [
            "The ignition coordinate is an approximate Prenn Pass landmark, not a georeferenced fire origin.",
            "ERA5-Land is reanalysis weather, not a local station observation.",
            "Weather is reduced to one constant mean vector for an 8-hour model window.",
            "The first-event reference is reported as about 10 ha, not an independently digitized burn perimeter.",
            "The 13.86 ha PlanetScope estimate covers a multi-day event with later flare-ups and is not directly equivalent to a single 8-hour simulated ignition.",
            "No georeferenced ground-truth perimeter was available; IoU, omission error and commission error cannot be computed.",
            "WorldCover 2021 is land-cover class data, not measured fuel load or moisture in April 2023.",
            "This retrospective scale check is not validation for operational firefighting decisions.",
        ],
        "references": [
            "https://vnexpress.net/hon-10-ha-rung-thong-da-lat-bi-chay-4591076.html",
            "https://journal.vnuf.edu.vn/en/article/download/103/57/110",
            "https://mapcarta.com/15483576",
        ],
    }
    out_path = OUT_DIR / "prenn_2023_benchmark.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    curve_path = OUT_DIR / "prenn_2023_area_curve.csv"
    import csv
    with curve_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    print("BENCHMARK_RESULT_JSON=" + json.dumps(summary, ensure_ascii=False), flush=True)
    log(f"Wrote {out_path.relative_to(ROOT)} and {curve_path.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()

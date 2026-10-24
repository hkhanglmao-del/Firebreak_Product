#!/usr/bin/env python3
"""Download the minimum public raster tiles needed by the five Firebreak region presets.

Place this file beside firebreak_hardened_r1.py and run it with Python 3.10+.
Outputs:
  data/worldcover/  ESA WorldCover 2021 v200 Map tiles (EPSG:4326, 10 m nominal)
  data/dem/         Copernicus DEM GLO-30 Public tiles; GLO-90 fallback if GLO-30 unavailable

Public sources:
  WorldCover: ESA-WorldCover/esa-worldcover-datasets official AWS download convention
  DEM: Copernicus DEM public AWS buckets
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import requests

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
WC_DIR = DATA_DIR / "worldcover"
DEM_DIR = DATA_DIR / "dem"
SPAN = 0.15

# These are copied from REGION_PRESETS in the uploaded firebreak_hardened_r1.py.
REGIONS = {
    "Hoang Lien Son - Sa Pa": (22.3750, 103.8250),
    "Da Lat pine forest": (11.9404, 108.4583),
    "Ba Vi National Park": (21.0600, 105.3600),
    "Hong Linh mountain": (18.5200, 105.7200),
    "Son Tra peninsula": (16.1150, 108.2650),
}
WC_GRID_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/esa_worldcover_2020_grid.geojson"
WC_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
DEM30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM90_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
USER_AGENT = "Firebreak-Research-Data-Setup/1.0 (public research data download)"
CHUNK_BYTES = 1024 * 1024


def region_bounds(center: tuple[float, float]) -> tuple[float, float, float, float]:
    lat, lon = center
    # xmin, ymin, xmax, ymax; same 0.15 degree bbox used by the dashboard.
    return (lon - SPAN / 2, lat - SPAN / 2, lon + SPAN / 2, lat + SPAN / 2)


def lon_tile_name(x: int) -> str:
    return ("E" if x >= 0 else "W") + f"{abs(x):03d}"


def lat_tile_name(y: int) -> str:
    return ("N" if y >= 0 else "S") + f"{abs(y):02d}"


def tile_bbox_from_geojson_geometry(geometry: dict) -> tuple[float, float, float, float] | None:
    coords = geometry.get("coordinates")
    if not coords:
        return None
    flat: list[tuple[float, float]] = []
    def walk(node):
        if isinstance(node, (tuple, list)):
            if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
                flat.append((float(node[0]), float(node[1])))
            else:
                for child in node:
                    walk(child)
    walk(coords)
    if not flat:
        return None
    xs = [p[0] for p in flat]
    ys = [p[1] for p in flat]
    return min(xs), min(ys), max(xs), max(ys)


def overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    # Bounds are xmin, ymin, xmax, ymax. Touching at a single edge is enough to include the tile.
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def fallback_worldcover_tile_ids(bounds: tuple[float, float, float, float]) -> set[str]:
    xmin, ymin, xmax, ymax = bounds
    # WorldCover tiles are 3x3 degrees, with names for the lower-left corner.
    lat0 = math.floor(ymin / 3) * 3
    lat1 = math.floor(ymax / 3) * 3
    lon0 = math.floor(xmin / 3) * 3
    lon1 = math.floor(xmax / 3) * 3
    ids = set()
    for lat in range(lat0, lat1 + 1, 3):
        for lon in range(lon0, lon1 + 1, 3):
            ids.add(f"{lat_tile_name(lat)}{lon_tile_name(lon)}")
    return ids


def get_worldcover_tile_ids(session: requests.Session, timeout: int) -> tuple[set[str], bool]:
    all_bounds = [region_bounds(c) for c in REGIONS.values()]
    try:
        r = session.get(WC_GRID_URL, timeout=min(timeout, 25))
        r.raise_for_status()
        grid = r.json()
        ids: set[str] = set()
        for feat in grid.get("features", []):
            props = feat.get("properties", {})
            tile_id = props.get("ll_tile") or props.get("tile")
            geom_bbox = tile_bbox_from_geojson_geometry(feat.get("geometry") or {})
            if tile_id and geom_bbox and any(overlaps(geom_bbox, b) for b in all_bounds):
                ids.add(str(tile_id))
        if ids:
            return ids, True
        raise RuntimeError("tile grid loaded but no intersecting tiles were found")
    except Exception as exc:
        # Grid geometry is only an aid to exact tile discovery. Tile names can safely be derived
        # from ESA's documented 3x3 degree grid as a fallback; downloads are still HTTP-validated.
        print(f"WARNING: Could not read ESA tile index ({exc}); deriving 3x3-degree tile names locally.")
        ids = set()
        for b in all_bounds:
            ids |= fallback_worldcover_tile_ids(b)
        return ids, False


def is_valid_tiff(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 100_000:
            return False
        with path.open("rb") as f:
            head = f.read(4)
        return head in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
    except OSError:
        return False


def stream_download(session: requests.Session, url: str, dest: Path, timeout: int, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_tiff(dest):
        print(f"  EXISTS, skip: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    if dest.exists():
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=(20, timeout), allow_redirects=True) as r:
                if r.status_code == 404:
                    print(f"  NOT FOUND (404): {url}")
                    return False
                r.raise_for_status()
                first = b""
                bytes_written = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        if len(first) < 4:
                            first += chunk[:4 - len(first)]
                        f.write(chunk)
                        bytes_written += len(chunk)
                if first not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                    raise RuntimeError(f"Downloaded response is not a TIFF (HTTP {r.status_code}); refusing to save it")
                if bytes_written < 100_000:
                    raise RuntimeError(f"File looks too small ({bytes_written} bytes); refusing to save it")
                os.replace(tmp, dest)
                print(f"  DOWNLOADED: {dest.name} ({bytes_written / 1e6:.1f} MB)")
                return True
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                wait = attempt * 3
                print(f"  Attempt {attempt}/{retries} failed: {exc}. Retry in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED: {url}\n  Reason: {exc}")
    return False


def dem_key(tile_lat: int, tile_lon: int, resolution: int) -> tuple[str, str]:
    northing = lat_tile_name(tile_lat) + "_00"
    easting = lon_tile_name(tile_lon) + "_00"
    stem = f"Copernicus_DSM_COG_{resolution}_{northing}_{easting}_DEM"
    return stem, f"{stem}/{stem}.tif"


def dem_tiles_needed() -> set[tuple[int, int]]:
    # Copernicus DEM is organized by 1x1 degree tiles. Add each tile touched by an AOI bbox.
    tiles: set[tuple[int, int]] = set()
    for center in REGIONS.values():
        xmin, ymin, xmax, ymax = region_bounds(center)
        lat0, lat1 = math.floor(ymin), math.floor(ymax)
        lon0, lon1 = math.floor(xmin), math.floor(xmax)
        for lat in range(lat0, lat1 + 1):
            for lon in range(lon0, lon1 + 1):
                tiles.add((lat, lon))
    return tiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Download only DEM + WorldCover tiles used by Firebreak's five preset regions.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request read timeout seconds (default 180)")
    parser.add_argument("--dry-run", action="store_true", help="List files/URLs without downloading")
    args = parser.parse_args()

    print("Firebreak data setup — five fixed study areas")
    print(f"Project folder: {APP_DIR}")
    print("AOI bounds (lon_min, lat_min, lon_max, lat_max):")
    for name, center in REGIONS.items():
        print(f"  {name}: {region_bounds(center)}")

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        wc_ids, used_official_index = get_worldcover_tile_ids(session, args.timeout)
        print(f"\nWorldCover 2021 v200 Map tiles ({len(wc_ids)}; {'official ESA tile grid' if used_official_index else 'computed fallback'}):")
        wc_urls = []
        for tile in sorted(wc_ids):
            fn = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
            url = f"{WC_BASE}/{fn}"
            wc_urls.append((url, WC_DIR / fn))
            print(f"  {tile}: {url}")

        dem_tiles = sorted(dem_tiles_needed())
        print(f"\nCopernicus DEM tiles ({len(dem_tiles)} unique 1x1-degree tiles; try GLO-30, fall back to GLO-90):")
        dem_items = []
        for lat, lon in dem_tiles:
            stem30, key30 = dem_key(lat, lon, 10)
            stem90, key90 = dem_key(lat, lon, 30)
            u30 = f"{DEM30_BASE}/{key30}"
            u90 = f"{DEM90_BASE}/{key90}"
            # Keep output file names consistent with the actual dataset resolution.
            dem_items.append((lat, lon, u30, DEM_DIR / f"{stem30}.tif", u90, DEM_DIR / f"{stem90}.tif"))
            print(f"  {lat_tile_name(lat)} / {lon_tile_name(lon)}: GLO-30 {u30}")

        print("\nIMPORTANT: Copernicus GLO-30 is a Digital Surface Model (DSM), not a bare-earth terrain model.")
        print("If a GLO-30 tile is not publicly available, this script will try GLO-90 and log the lower-resolution fallback.")
        if args.dry_run:
            print("\nDRY RUN only: no files downloaded.")
            return 0

        WC_DIR.mkdir(parents=True, exist_ok=True)
        DEM_DIR.mkdir(parents=True, exist_ok=True)
        failures = []
        print("\n=== Download WorldCover ===")
        for url, dest in wc_urls:
            if not stream_download(session, url, dest, args.timeout):
                failures.append(("WorldCover", dest.name, url))

        print("\n=== Download DEM ===")
        for lat, lon, u30, p30, u90, p90 in dem_items:
            if is_valid_tiff(p30):
                print(f"  EXISTS, skip: {p30.name}")
                continue
            print(f"  Trying GLO-30: {p30.name}")
            if stream_download(session, u30, p30, args.timeout):
                continue
            print(f"  Trying GLO-90 fallback: {p90.name}")
            if not stream_download(session, u90, p90, args.timeout):
                failures.append(("DEM", p30.name, f"GLO-30: {u30}; GLO-90: {u90}"))

    print("\n=== Summary ===")
    print(f"WorldCover folder: {WC_DIR}")
    print(f"DEM folder:        {DEM_DIR}")
    if failures:
        print(f"\nERROR: {len(failures)} tile(s) could not be downloaded:")
        for kind, name, url in failures:
            print(f"  [{kind}] {name}\n    {url}")
        print("Do not treat missing tiles as successful coverage. Retry later or report these missing URLs.")
        return 2
    print("All requested files downloaded or already present and passed basic TIFF-header validation.")
    print("Next: run `python validate_firebreak_data.py` if that script is in the project folder, then start Streamlit.")
    print("Note: TIFF header validation is not a full raster/CRS/coverage validation; the app performs deeper checks when reading files.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Complete files are preserved; rerun the command to skip valid files.")
        raise SystemExit(130)

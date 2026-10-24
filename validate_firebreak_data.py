"""Validate spatial coverage of local DEM and ESA WorldCover tiles for the presets or one ROI.

Coverage here uses GeoTIFF bounds only; runtime app separately checks valid raster classes.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parent
APP = ROOT / "firebreak_hardened_r1.py"
DATA = ROOT / "data"


def load_presets():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "REGION_PRESETS" for t in node.targets):
                return ast.literal_eval(node.value)
    raise RuntimeError("Không tìm thấy REGION_PRESETS trong app.")


def bounds_for(lat, lon):
    return lon - 0.15 / 2, lat - 0.15 / 2, lon + 0.15 / 2, lat + 0.15 / 2


def overlap_ratio(target, tile):
    west, south, east, north = target
    left, bottom, right, top = tile
    iw = max(0.0, min(east, right) - max(west, left))
    ih = max(0.0, min(north, top) - max(south, bottom))
    area = max((east - west) * (north - south), 1e-12)
    return min(1.0, iw * ih / area)


def inspect_dir(folder, target, prefer_map=False):
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}) if folder.exists() else []
    if prefer_map:
        map_files = [p for p in files if "map" in p.stem.lower()]
        v200_map_files = [p for p in map_files if "2021" in p.stem.lower() and "v200" in p.stem.lower()]
        if v200_map_files:
            files = v200_map_files
        else:
            print(f"  WARN {folder}: cần file ESA WorldCover 2021 v200 Map với tên file gốc; không đọc nhầm InputQuality/khác version.")
            return None
    candidates = []
    for path in files:
        try:
            with rasterio.open(path) as ds:
                if ds.crs is None:
                    continue
                tile_bounds = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
                ratio = overlap_ratio(target, tile_bounds)
                if ratio > 0:
                    candidates.append((ratio, path, ds.crs.to_string(), ds.dtypes[0], ds.res, ds.nodata, ds.count, tile_bounds))
        except Exception as exc:
            print(f"  WARN {path.name}: không đọc được ({exc})")
    if not candidates:
        return None

    # Estimate spatial ROI coverage by the union of all intersecting tile bounds.
    # This does not inspect NoData values; the app's runtime coverage check does.
    west, south, east, north = target
    lon_samples = np.linspace(west, east, 101)
    lat_samples = np.linspace(south, north, 101)
    lon_grid, lat_grid = np.meshgrid(lon_samples, lat_samples)
    covered = np.zeros(lon_grid.shape, dtype=bool)
    for _, _, _, _, _, _, _, (left, bottom, right, top) in candidates:
        covered |= (lon_grid >= left) & (lon_grid <= right) & (lat_grid >= bottom) & (lat_grid <= top)
    union_coverage = float(np.mean(covered))
    return max(candidates, key=lambda item: item[0]), union_coverage, len(candidates)


def check_target(label, lat, lon):
    target = bounds_for(lat, lon)
    print(f"\n{label}: lat={lat:.4f}, lon={lon:.4f}; bbox={target}")
    for name, folder, prefer_map in (
        ("DEM", DATA / "dem", False),
        ("WorldCover Map", DATA / "worldcover", True),
    ):
        row = inspect_dir(folder, target, prefer_map)
        if row is None:
            print(f"  MISSING {name}: không có GeoTIFF giao với bbox trong {folder}")
            continue
        best, union_coverage, ntiles = row
        ratio, path, crs, dtype, res, nodata, count, _ = best
        status = "OK (tile bounds cover ROI)" if union_coverage >= 0.999 else f"PARTIAL ({union_coverage:.1%} estimated union coverage; ROI edges may be missing)"
        print(f"  {name}: {status}; intersecting tiles={ntiles}; primary={path.name}; CRS={crs}; dtype={dtype}; resolution={res}; nodata={nodata}; bands={count}")
        if name == "WorldCover Map":
            # Read only the pixels inside the target AOI. A full 3-degree
            # WorldCover tile is about 36,000 x 36,000 pixels and can consume
            # multiple gigabytes when read into a masked array.
            with rasterio.open(path) as ds:
                west, south, east, north = target
                roi_bounds = transform_bounds("EPSG:4326", ds.crs, west, south, east, north, densify_pts=21)
                window = from_bounds(*roi_bounds, transform=ds.transform)
                window = window.round_offsets().round_lengths()
                window = window.intersection(Window(0, 0, ds.width, ds.height))
                band = ds.read(1, window=window, masked=True)
                values, counts = np.unique(np.asarray(band.compressed()), return_counts=True)
            print("    Class values observed inside AOI window:", dict(zip(values.tolist()[:20], counts.tolist()[:20])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, help="Kiểm tra một vĩ độ tùy chọn")
    parser.add_argument("--lon", type=float, help="Kiểm tra một kinh độ tùy chọn")
    args = parser.parse_args()
    if (args.lat is None) != (args.lon is None):
        parser.error("Cần cung cấp cả --lat và --lon.")
    if args.lat is not None:
        check_target("Custom ROI", args.lat, args.lon)
        return
    for label, preset in load_presets().items():
        if label.startswith("Tọa độ Bất kỳ"):
            continue
        check_target(label, preset["lat"], preset["lon"])
    print("\nLưu ý: coverage ở đây kiểm tra phạm vi không gian. Nó không chứng minh dữ liệu phân loại đúng hoặc mô hình cháy đã được xác thực.")

if __name__ == "__main__":
    main()

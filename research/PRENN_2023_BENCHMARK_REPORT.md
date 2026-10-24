# Preliminary Historical Scale-Check: Dalat Prenn Pass Wildfire

**Run date:** 24 October 2026  
**Code:** `firebreak_hardened_r1_english.py`, existing `run_full_simulation_and_mincut()` function  
**Benchmark execution:** [GitHub Actions run](https://github.com/hkhanglmao-del/Firebreak_Product/actions/runs/40092705695)

> **Superseded baseline — do not use the numbers below as the current result.** This report was written before the ignition-origin correction. The earlier run initialized simultaneous burning disks at multiple hypothetical origin candidates when location uncertainty was nonzero. The application now selects one valid candidate per realization and logs the chosen seed plus the actual initial burning-cell count, with an automated disk-size guard in the runner. The corrected branch also handles unreachable-cell time-pressure ratios without propagating NaN into the min-cut cost matrix, and the Streamlit cache decorator remains attached to the complete simulation function. The final benchmark must run on this exact tested revision; results and CSV/JSON files must be regenerated before quoting model area metrics. The earlier run also omitted OSM road/building masks after Overpass timeouts; the updated runner now attempts the official OSM map API first, filters out unrelated land-use tags such as forest, rasterizes actual building polygons, and records whether the data source succeeded.

## Executive result

The current default configuration overpredicts the reported area by a large margin on this retrospective scale check. It starts with **14.24 ha already burning**, grows to **31.35 ha at 6 hours** and **36.10 ha at 8 hours**. Contemporary reporting put the first Prenn Pass fire at approximately **10 ha**; a PlanetScope study estimated **13.86 ha for the multi-day 7–9 April event**. These reported values are approximate area references, not a georeferenced perimeter for direct pixel-by-pixel validation. [VnExpress, 8 April 2023](https://vnexpress.net/hon-10-ha-rung-thong-da-lat-bi-chay-4591076.html); [PlanetScope study](https://doi.org/10.55250/jo.vnuf.8.2.2023.097-106).

## Data and run setup

- Area: Prenn Pass / Dalat AOI, approximate ignition proxy at 11.91667°N, 108.45000°E. This is not an exact georeferenced ignition point.
- Model grid: 540 × 540, nominal 30 m; 0.09304 ha per cell.
- Land cover: ESA WorldCover 2021 v200 Map, 100.0% valid coverage on the land domain.
- Elevation: clipped Copernicus DEM GLO-30 DSM tiles north of 11°N and 12°N. This is a surface model and is not a field-validated bare-earth terrain model.
- Weather: Open-Meteo historical archive for 7 April 2023, 15:00–22:00 local time; mean temperature 20.35°C, relative humidity 78.38%, wind speed 3.24 m/s, circular mean wind direction FROM 75.2°, precipitation 0.00 mm/h. This is archive/reanalysis output, not a local station record. Using one mean vector smooths the sharp evening change in temperature and humidity.
- **Important integration limitation:** Overpass timed out in the benchmark environment. The benchmark used real DEM and WorldCover data but did not provide road or building masks. The CA and Min-Cut code were run unchanged, but firebreak access and route metrics are provisional and must not be presented as a full OSM-enabled integration result.

## Default-configuration comparison

| Metric | Firebreak R1 | Historical reference | Difference |
|---|---:|---:|---:|
| Initial area (T=0) | 14.24 ha | ~10 ha final reported first event | Initial footprint is already 42.4% larger |
| Area at 6 hours | 31.35 ha | ~10 ha (approximate event total; timing is not exact) | +21.35 ha (213.5%) |
| Area at 8 hours | 36.10 ha | ~10 ha (approximate event total; timing is not exact) | +26.10 ha (261.0%) |
| Area at 8h vs 13.86 ha multi-day mapped area | 36.10 ha | 13.86 ha across 7–9 April | +22.24 ha (+160.5%); **not like-for-like** |

The starting footprint is the first problem: with the default `uncertainty_px=8` and `initial_radius=3`, the code activates multiple candidate ignition centers at once, producing 153 burning cells (14.24 ha). Location uncertainty should not automatically be interpreted as multiple simultaneous burning centers when estimating burned area. A proper probabilistic treatment would run alternative ignition locations as separate realizations or choose a single origin per run.

## Sensitivity check (single ignition center, radius 2 px)

To isolate how much of the mismatch comes from the initial footprint and base ignition probability, the existing algorithm was rerun with `uncertainty_px=0`, `initial_radius=2`, and only `p_base` varied. The same historical weather, DEM, WorldCover, and code path were retained.

| `p_base` | Model area at 6h | Model area at 8h | Error vs ~10 ha at 6h | Error vs ~10 ha at 8h | Planned line completion |
|---:|---:|---:|---:|---:|---:|
| 0.12 | 4.28 | 4.56 | -57.2% | -54.4% | 6.92 |
| 0.16 | 4.75 | 5.21 | -52.5% | -47.9% | 7.37 |
| 0.20 | 6.70 | 9.12 | -33.0% | -8.8% | 7.63 |
| 0.24 | 7.91 | 11.07 | -20.9% | 10.7% | 8.92 |
| 0.28 | 8.47 | 12.10 | -15.3% | 21.0% | 9.40 |
| 0.32 | 10.23 | 15.07 | 2.3% | 50.7% | 9.28 |
| 0.38 | 13.40 | 21.21 | 34.0% | 112.1% | 9.54 |

A candidate with `p_base=0.32` gives 10.23 ha at 6 h (+2.3%), while `p_base=0.20` gives 9.12 ha at 8 h (−8.8%). **These are in-sample scale fits to one approximate reported area, not validated parameters.** The preferred value changes with the assumed comparison time, which is itself uncertain. Do not report either value as independently validated.

## Firebreak counterfactual

Under the default settings, the planner reports a topologically separated candidate, approximately **8.45 km (3D)**, **8.05 km (2D)**, and **1,028.7 person-hours**. The model's own completion estimate is **10.32 hours**, later than the 8-hour test window. As expected, the firebreak is not active during that run and the controlled counterfactual remains **36.10 ha**, a **0% reduction** over the first 8 hours.

In the single-origin sensitivity cases, the same algorithm also shows no burned-area reduction by 8 hours, even when a plan is accepted. This means the run does **not** demonstrate firebreak effectiveness; it only exercises the planning logic. Since road/building masks were unavailable, route/access values in this benchmark are particularly provisional.

## Conclusion and recommendation for the project

1. Keep this as a **preliminary retrospective scale-check**, not “accuracy validation.” The available references give area estimates but not a georeferenced burn polygon, so IoU, omission rate, commission rate, or spread-boundary overlap cannot be calculated from this run.
2. For the next software revision, treat ignition-location uncertainty as alternative starting locations rather than summing all candidates into a single initial burned footprint. Add a test that the starting burned area depends on the ignition radius, not on the number of hypothetical locations.
3. Do not tune `p_base` on this one event and then call this event a validation set. Use it as a development/calibration case and seek a second independent event for validation.
4. Rerun the firebreak planner with functioning OSM roads/buildings or a fixed, sourced offline OSM extract before reporting the route length, travel time, person-hours, or reduction as meaningful project results.

## References

- [VnExpress: More than 10 ha of pine forest affected near Prenn Pass (8 April 2023)](https://vnexpress.net/hon-10-ha-rung-thong-da-lat-bi-chay-4591076.html)
- [Tuổi Trẻ: Additional fire and cumulative 13 ha affected (9 April 2023)](https://tuoitre.vn/rung-phong-ho-deo-prenn-da-lat-van-chay-20230409112832179.htm)
- [Pham Duc Binh (2023), *Mapping small burned areas using high spatial resolution PlanetScope imagery: a case study of the wildfire in Da Lat city*, DOI: 10.55250/jo.vnuf.8.2.2023.097-106](https://doi.org/10.55250/jo.vnuf.8.2.2023.097-106)

# flood_sim.py — 淹水模擬與撤離判定 (Python)

Given **terrain elevation** and **rainfall**, this computes water depth and velocity on every grid cell over time, then decides **which communities must evacuate and when**.

It's the same model as the web version: a 2D local-inertial shallow-water solver (the LISFLOOD-FP method) with Manning friction, Horton infiltration, urban storm drainage, upstream river inflow and sea level.

## Install

```bash
pip install numpy matplotlib pillow requests pandas
pip install global-land-mask   # optional: sea/land mask for elevation_box.py
pip install rasterio           # optional: read/write GeoTIFF with coordinates
```

## Quick start

```bash
# 1) Try a built-in terrain (valley / basin / coast).
#    This also writes sample input files to demo_valley/inputs/ so you can see the formats.
python flood_sim.py --demo valley --out demo_valley

# 2) Your own elevation and rainfall
python flood_sim.py --dem dem.asc --rain rain.csv --towns towns.csv --out results

# 3) Your own elevation with a design storm (no rain file)
python flood_sim.py --dem dem.tif --rain-total 500 --rain-hours 24 --pattern center --towns towns.csv
```

The folder `examples/` contains ready-made inputs. To run them:

```bash
python flood_sim.py --dem examples/dem.asc --rain examples/rain.csv --towns examples/towns.csv \
  --river-mask examples/river_mask.asc --source 0,53 --source 0,54 --catchment-area 150 --edges S --out results
```

## Real places: Open-Meteo workflow

The included `elevation_box.py` (elevation, ~90 m Copernicus DEM) and `rainfall.py` (past or forecast rain) fetch data for any lat/lon box. The simulator reads their CSVs directly.

**One command** (`run_area.py` fetches both, caches them, then runs the flood model):
```bash
# a past typhoon
python run_area.py --south 25.00 --west 121.45 --north 25.10 --east 121.58 \
    --cell 100 --start 2024-10-02 --end 2024-10-04 --towns my_towns.csv \
    -- --source-latlon 25.095,121.46 --catchment-area 300

# the next 3 days (forecast)
python run_area.py --south 25.00 --west 121.45 --north 25.10 --east 121.58 --forecast 3 --towns my_towns.csv
```
Everything after `--` goes to `flood_sim.py`, for example `--soil clay`, `--evac-depth 0.3` or `--source-latlon`. Re-running reuses the downloads.

**Or step by step:**
```bash
python elevation_box.py --south 25.00 --west 121.45 --north 25.10 --east 121.58 --step-m 100 --out elev.csv
python rainfall.py --start 2024-10-02 --end 2024-10-04 --hourly --south 25.00 --west 121.45 --north 25.10 --east 121.58 --step 0.05
python flood_sim.py --dem elev.csv --rain rainfall_hourly.csv --towns my_towns.csv --out results
```

Things to know:
- **Cell size.** Use `--step-m` / `--cell` of **100 m or less**. Coarser grids blur valleys and riverbanks and change the answer. The API cost is points ÷ 100 calls, and the free tier allows about 10,000 calls a day. A 10 × 10 km box at 100 m is about 10,000 points, or 100 calls.
- **Hourly rain.** Use `rainfall.py --hourly`. Daily totals still work, but they are spread over 24 h (`--daily-pattern`) and miss short downpours.
- **Communities and river entry points by lat/lon.** Use a towns CSV with `name,population,lat,lon,radius_m`, and `--source-latlon lat,lon` where a river enters the box from upstream. Add `--catchment-area` (km² upstream of that point) so the river's flood flow is included.
- **River channel.** You don't need a river mask. With river inflow, the model first runs the river at normal flow for 6 h (`--spinup`). Whatever is wet then counts as the normal river, not as flooding.
- **Sea.** Points flagged `is_water` become sea. Set storm surge with `--sea-level`.
- **Clock times.** Reports show real clock times (Asia/Taipei) as well as T+hours, plus `results_latlon.csv` with depth and evacuation time for every cell, ready to plot or load into GIS.

## Input files

### Elevation (`--dem`)
| Format | Notes |
|---|---|
| `.csv` with `lat,lon,elevation_m` columns | Output of `elevation_box.py`. It is resampled to square cells, and the `is_water` column becomes sea. |
| `.asc` ESRI ASCII grid | **Recommended.** Cell size and coordinates are read from the header. Most GIS tools (QGIS: *Raster → Conversion → Translate*) can export it. |
| `.tif` GeoTIFF | Install `rasterio` to keep coordinates; outputs are then also written as GeoTIFF. |
| `.csv` / `.txt` | Plain number matrix, comma or space separated, metres. **Row 1 = north edge.** Needs `--dx`. |
| `.png` grayscale | 8- or 16-bit. Give `--png-min` / `--png-max` (elevation of black/white) and `--dx`. |

- Cells below `--sea-below` (default 0 m) are sea, held at `--sea-level` (use this for storm surge).
- NODATA cells act as walls.
- Large DEM? `--coarsen 2` averages 2×2 cells (4× fewer cells and ~8× faster).

### Rainfall (`--rain`)
**One series** for the whole area, such as hourly station data from 中央氣象署:
```csv
time_h,rain_mm
0,1.4
1,4.1
2,6.8
```
`rain_mm` is the rain that fell between this row's time and the next row's time. If the column is named `mm_h` or `intensity`, it is read as an intensity instead.

**Several stations** (spatially varying rain, interpolated by inverse-distance weighting). Use one column per station, and give the station locations with `--stations`:
```csv
time_h,山區站,平地站,下游站
0,1.9,1.4,1.1
1,5.7,4.1,3.2
```
```csv
name,row,col          ← or: name,x,y  (map coordinates, same CRS as the .asc/.tif)
山區站,10,15
平地站,50,50
下游站,90,70
```

**Open-Meteo format** (`lat,lon,time,rain_mm` from `rainfall.py --hourly`, or `lat,lon,date,rain_mm` daily) is read directly. Each grid point becomes a rain station.

**Optional:** `--rain-grid ratio.asc` multiplies rain cell by cell (for example a radar-derived ratio map). `--oro 40` adds 40% more rain per 1000 m of elevation.

**No rain file:** use a design storm with `--rain-total`, `--rain-hours`, `--pattern uniform|center|front|back`.

Taiwan CWA 24-hour categories for reference: 大雨 80 mm, 豪雨 200, 大豪雨 350, 超大豪雨 500.

### Communities (`--towns`)
```csv
name,population,row,col,radius_m      ← or use x,y instead of row,col
河濱里,3200,35,41,200
中正里,5400,55,36,280
```

### River inflow (optional)
- `--source row,col` marks where the river enters the map (repeat for wide rivers).
- Then either `--inflow hydrograph.csv` (`time_h,Q_m3s`), or `--catchment-area <km²> --runoff-c 0.8 --lag 2`. The second option uses the rational method Q = C·i·A/3.6 driven by the rainfall.
- `--river-mask` (1 = channel) keeps the river channel itself from counting as flooding inside a community.

### Boundaries
`--edges S` lets water leave through the south edge. Use any of `NSEW`. All edges are open by default for your own DEM.

## Evacuation criteria
A community is **EVACUATE** when either of these holds (all thresholds adjustable):
- ≥ `--area-pct` (20%) of its area is ≥ `--evac-depth` (0.5 m) deep, **or**
- anywhere inside, hazard = depth × (velocity + 0.5) ≥ `--hazard-limit` (1.25). This is the common people-safety rating for flowing water.

It is **WARNING** at ≥ `--warn-depth` (0.3 m) over the same area share.

## Outputs (in `--out`)
| File | What |
|---|---|
| `report.md` | Communities to evacuate in order, with warning / evacuate / all-clear times and peak depth |
| `communities.csv` | Same, as a table |
| `timeline.csv` | Status and depth of every community every 5 minutes |
| `evacuation_map.png` | When each place first reaches evacuation depth (0–3 h, 3–6 h …) |
| `max_depth.png`, `hazard_map.png`, `timeline.png` | Maps and the rainfall / depth chart |
| `max_depth.asc`, `evac_time_h.asc`, `max_hazard.asc`, `max_speed.asc` | Grids for QGIS/ArcGIS (`.tif` too if rasterio is installed) |
| `snapshots/` | Depth maps every N hours with `--snapshot-every N` (can be made into an animation) |

## Run time
The time step shrinks for smaller cells and deeper water. Rough figures on a laptop for a 24-hour storm:
- 100×100 grid: about 15 s
- 250×250 grid: about 2–4 min
- 500×500 grid: about 20–40 min. Use `--coarsen 2` to cut this to a few minutes.

## Limitations
This is a screening model with uniform soil and roughness and no levees, culverts, pumps or sewer network detail beyond a uniform drainage rate. Results depend heavily on DEM quality. Use it for teaching, scenario comparison and preliminary planning, **not** for real evacuation decisions; follow 中央氣象署, 經濟部水利署 and local government orders.

Run `python flood_sim.py -h` for every option.

"""
online.py - download elevation and rainfall from Open-Meteo for flood_sim.py.

Used by `flood_sim.py --dem online` / `--rain online`, by the setup questions
you get when pressing Run with no options, and by run_area.py.

  * Elevation: Open-Meteo Elevation API (Copernicus DEM, ~90 m)
  * Rain:      hourly `precipitation` (mm in the hour before each time stamp)
               from the Historical Weather API for old dates and the Forecast
               API for recent / future dates (up to 16 days ahead).
               Docs: https://open-meteo.com/en/docs

Downloads are cached in a folder (default `openmeteo_cache/`), so running the
same area and dates again costs no API calls. Forecasts are re-downloaded
unless the whole period is in the past.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import date, datetime, timedelta

import elevation_box as eb
import rainfall as rf

# A few Open-Meteo weather models (value for the API `models` parameter).
# best_match picks the highest-resolution model for each place.
RAIN_MODELS = ["best_match", "jma_seamless", "ecmwf_ifs025", "gfs_seamless", "icon_seamless",
               "gem_seamless", "ukmo_seamless", "meteofrance_seamless", "cma_grapes_global"]


def parse_box(text):
    """'south,west,north,east' -> tuple of 4 floats (checked)."""
    try:
        s, w, n, e = (float(v) for v in str(text).replace(" ", "").split(","))
    except ValueError:
        raise SystemExit(f"area must be 'south,west,north,east' in degrees, got {text!r}")
    if not (-90 <= s < n <= 90 and -180 <= w < e <= 180):
        raise SystemExit(f"area {text!r} is not a valid box (need south < north and west < east)")
    return s, w, n, e


def parse_dates(start=None, end=None, forecast=None, today=None):
    today = today or date.today()
    if forecast:
        if not 1 <= forecast <= rf.FORECAST_MAX_DAYS:
            raise SystemExit(f"--forecast must be 1–{rf.FORECAST_MAX_DAYS} days")
        return today, today + timedelta(days=forecast - 1)
    if start:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end) if end else s
        if e < s:
            raise SystemExit("--end must be on or after --start")
        return s, e
    return today, today + timedelta(days=2)  # default: next 3 days


def _key(*parts):
    return hashlib.md5(json.dumps(parts).encode()).hexdigest()[:10]


def fetch_elevation(box, cell=100.0, cache="openmeteo_cache", max_calls=3000, refresh=False):
    """Download a square-cell elevation grid for box; returns the CSV path."""
    import pandas as pd  # noqa: F401  (elevation_box needs it)
    s, w, n, e = box
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"elevation_{_key(round(s, 5), round(w, 5), round(n, 5), round(e, 5), cell)}.csv")
    if os.path.exists(path) and not refresh:
        print(f"Elevation: using cached download {path}")
        return path
    step, step_lon = eb.metre_steps(s, n, cell)
    grid = eb.make_grid(s, w, n, e, step, step_lon)
    calls = -(-len(grid) // eb.BATCH)
    print(f"Elevation: downloading {len(grid):,} points at {cell:g} m from Open-Meteo ({calls} API calls)")
    if calls > max_calls:
        raise SystemExit(f"That is more than {max_calls} API calls (free limit is about 10,000 a day). "
                         f"Use a smaller area or a bigger cell size.")
    grid["elevation_m"] = eb.get_elevations(grid["lat"].tolist(), grid["lon"].tolist())
    grid = eb.add_water(grid)
    grid.to_csv(path, index=False)
    print(f"  saved {path}")
    return path


def fetch_rain(box, start, end, step=0.05, model="best_match", cache="openmeteo_cache", refresh=False):
    """Download hourly rain on a lat/lon grid covering box (padded); returns the CSV path
    (columns lat, lon, time, rain_mm, source)."""
    import pandas as pd
    s, w, n, e = box
    today = date.today()
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"rain_{start}_{end}_{model}_{_key(round(s, 4), round(w, 4), round(n, 4), round(e, 4), step)}.csv")
    if os.path.exists(path) and not refresh and end < today - timedelta(days=1):
        print(f"Rain: using cached download {path}")
        return path
    pad = step / 2
    grid = rf.make_grid(s - pad, w - pad, n + pad, e + pad, step)
    pieces = rf.plan_requests(start, end, today)
    print(f"Rain: downloading hourly rain for {len(grid)} points, {start} to {end}"
          f"{'' if model == 'best_match' else f', model {model}'}")
    lats, lons = grid["lat"].tolist(), grid["lon"].tolist()
    parts = []
    for url, ps, pe in pieces:
        if model != "best_match" and url == rf.FORECAST_URL:
            parts.append(_fetch_model(url, lats, lons, ps, pe, model))
        else:
            parts.append(rf.fetch(url, lats, lons, ps, pe, hourly=True))
    hourly = pd.concat(parts, ignore_index=True).sort_values(["lat", "lon", "time"])
    hourly["rain_mm"] = hourly["rain_mm"].fillna(0.0)
    hourly.to_csv(path, index=False)
    tot = hourly.groupby(["lat", "lon"])["rain_mm"].sum()
    print(f"  saved {path}  (totals {tot.min():.0f}–{tot.max():.0f} mm, peak {hourly['rain_mm'].max():.1f} mm/h)")
    return path


def _fetch_model(url, lats, lons, start, end, model):
    """rainfall.fetch with a `models=` parameter (forecast API only)."""
    import requests
    orig = requests.get

    def get(u, params=None, **kw):
        return orig(u, params={**(params or {}), "models": model}, **kw)

    requests.get = get
    try:
        return rf.fetch(url, lats, lons, start, end, hourly=True)
    finally:
        requests.get = orig


def rain_to_series(long_csv, out_csv):
    """Average a lat/lon rain download into one series (time_h,rain_mm) for a DEM without
    coordinates. Returns the datetime of time_h = 0."""
    import pandas as pd
    df = pd.read_csv(long_csv)
    mean = df.groupby("time")["rain_mm"].mean().sort_index()
    t = pd.to_datetime(mean.index) - pd.Timedelta(hours=1)  # value = hour BEFORE the stamp
    t0 = t[0].to_pydatetime()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "rain_mm"])
        for ti, v in zip(t, mean.values):
            w.writerow([round((ti - t[0]).total_seconds() / 3600, 3), round(float(v), 2)])
    return t0


def write_typed_rain(values, out_csv):
    """'5,10,30,60' (mm in each hour) -> time_h,rain_mm CSV."""
    try:
        v = [float(x) for x in str(values).replace(" ", ",").split(",") if x != ""]
    except ValueError:
        raise SystemExit(f"--rain-values must be numbers separated by commas, got {values!r}")
    if not v or min(v) < 0:
        raise SystemExit("--rain-values needs at least one hour and no negative values")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "rain_mm"])
        for h, mm in enumerate(v):
            w.writerow([h, mm])
    return v

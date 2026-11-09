#!/usr/bin/env python3
"""
run_area.py - one command: fetch elevation + rain for a map box from Open-Meteo,
then run the flood simulation and evacuation assessment.

    python run_area.py --south 25.00 --west 121.45 --north 25.10 --east 121.58 \
        --cell 100 --start 2024-10-02 --end 2024-10-04 --towns my_towns.csv

    # forecast: next 3 days
    python run_area.py --south 25.00 --west 121.45 --north 25.10 --east 121.58 --forecast 3

Or just press Run in an IDE: the default box (lower Keelung River, Taipei) and the
next 3 days of forecast rain are used.

Anything after `--` is passed to flood_sim.py, e.g.
    ... -- --source-latlon 25.09,121.46 --catchment-area 300 --soil clay --evac-depth 0.3

Downloads are cached in the output folder, so re-running with different flood
settings does not call the API again.
"""
import argparse
import json
import os
import sys
from datetime import date, timedelta

import pandas as pd

import elevation_box as eb
import rainfall as rf
import flood_sim


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    p = argparse.ArgumentParser(description="Fetch Open-Meteo elevation + rain for a box and run flood_sim.py")
    # default box: lower Keelung River, Taipei (~6.7 × 10 km). Edit to change the area.
    p.add_argument("--south", type=float, default=25.04)
    p.add_argument("--west", type=float, default=121.55)
    p.add_argument("--north", type=float, default=25.10)
    p.add_argument("--east", type=float, default=121.65)
    p.add_argument("--cell", type=float, default=100.0, help="grid cell size in metres (Open-Meteo DEM is ~90 m)")
    p.add_argument("--start", help="YYYY-MM-DD (past or near future)")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--forecast", type=int, help="next N days instead of --start/--end (max 16)")
    p.add_argument("--rain-step", type=float, default=0.05, help="rain sample spacing in degrees (weather model cells ~0.1°)")
    p.add_argument("--towns", help="communities CSV (name,population,lat,lon,radius_m)")
    p.add_argument("--out", default="area_run")
    p.add_argument("--max-calls", type=int, default=3000, help="refuse bigger elevation downloads (free limit ~10,000 calls/day)")
    p.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    # ---------------- elevation ----------------
    elev_csv = os.path.join(a.out, "elevation.csv")
    key = dict(box=[a.south, a.west, a.north, a.east], cell=a.cell)
    keyfile = os.path.join(a.out, "elevation.json")
    cached = (not a.refresh and os.path.exists(elev_csv) and os.path.exists(keyfile)
              and json.load(open(keyfile)) == key)
    if cached:
        print(f"Elevation: using cached {elev_csv}")
    else:
        step, step_lon = eb.metre_steps(a.south, a.north, a.cell)
        grid = eb.make_grid(a.south, a.west, a.north, a.east, step, step_lon)
        calls = -(-len(grid) // eb.BATCH)
        print(f"Elevation: {len(grid):,} points at {a.cell:g} m -> {calls} API calls")
        if calls > a.max_calls:
            raise SystemExit(f"That is more than --max-calls {a.max_calls}. Use a smaller box or a bigger --cell "
                             f"(calls scale with area / cell²).")
        grid["elevation_m"] = eb.get_elevations(grid["lat"].tolist(), grid["lon"].tolist())
        grid = eb.add_water(grid)
        grid.to_csv(elev_csv, index=False)
        json.dump(key, open(keyfile, "w"))
        print(f"Saved {elev_csv}")

    # ---------------- rainfall (hourly) ----------------
    today = date.today()
    if a.forecast:
        start, end = today, today + timedelta(days=a.forecast - 1)
    elif a.start:
        start = date.fromisoformat(a.start)
        end = date.fromisoformat(a.end) if a.end else start
    else:  # default when nothing is given: next 3 days
        start, end = today, today + timedelta(days=2)
        print("No dates given: using the forecast for the next 3 days")
    rain_csv = os.path.join(a.out, f"rain_{start}_{end}_hourly.csv")
    if not a.refresh and os.path.exists(rain_csv) and end < today:
        print(f"Rain: using cached {rain_csv}")
    else:
        # sample a little beyond the box so interpolation near the edges is sensible
        pad = a.rain_step / 2
        grid = rf.make_grid(a.south - pad, a.west - pad, a.north + pad, a.east + pad, a.rain_step)
        pieces = rf.plan_requests(start, end, today)
        print(f"Rain: {len(grid)} points, {start} to {end} (hourly)")
        lats, lons = grid["lat"].tolist(), grid["lon"].tolist()
        hourly = pd.concat([rf.fetch(u, lats, lons, s, e, hourly=True) for u, s, e in pieces], ignore_index=True)
        hourly = hourly.sort_values(["lat", "lon", "time"])
        hourly.to_csv(rain_csv, index=False)
        tot = hourly.groupby(["lat", "lon"])["rain_mm"].sum()
        print(f"Saved {rain_csv}  (totals {tot.min():.0f}–{tot.max():.0f} mm, peak {hourly['rain_mm'].max():.1f} mm/h)")

    # ---------------- flood simulation ----------------
    args = ["--dem", elev_csv, "--rain", rain_csv, "--out", os.path.join(a.out, "flood")]
    if a.towns:
        args += ["--towns", a.towns]
    print("\nRunning flood_sim.py " + " ".join(args + extra) + "\n")
    flood_sim.main(args + extra)


if __name__ == "__main__":
    main()

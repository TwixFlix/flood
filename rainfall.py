"""
Daily rainfall for a grid of points inside a bounding box, from Open-Meteo.

Picks the right API automatically from your dates:
  - older than ~5 days ago  -> Historical Weather API (archive, back to 1940)
  - recent days / future    -> Forecast API (up to ~16 days ahead)
  - a range spanning both   -> split and merged automatically

Install:
    pip install requests numpy pandas matplotlib

Use --hourly for flood simulation (writes <out>_hourly.csv, which
flood_sim.py --rain reads directly). Daily totals hide the short bursts that
cause flooding.

Examples:
    python rainfall.py --start 2024-10-01 --end 2024-10-05          # past (e.g. a typhoon)
    python rainfall.py --start 2024-10-02 --end 2024-10-04 --hourly # hourly, for flood_sim.py
    python rainfall.py --forecast 7                                  # next 7 days
    python rainfall.py --start 2026-11-01 --end 2026-11-15 --plot    # past + forecast mixed
Or just press Run in PyCharm: the defaults below are used.
"""
import argparse
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_DELAY_DAYS = 5   # archive data lags real time by a few days
FORECAST_PAST_DAYS = 90  # forecast API also serves roughly the last 3 months
FORECAST_MAX_DAYS = 16
BATCH = 50               # locations per request
TIMEZONE = "Asia/Taipei"


def make_grid(south, west, north, east, step):
    lats = np.arange(south, north + step / 2, step)
    lons = np.arange(west, east + step / 2, step)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return pd.DataFrame({"lat": lat_g.ravel().round(6), "lon": lon_g.ravel().round(6)})


def plan_requests(start, end, today=None):
    """Split [start, end] into (url, start, end) pieces for the right API."""
    today = today or date.today()
    if end > today + timedelta(days=FORECAST_MAX_DAYS - 1):
        raise ValueError(f"Forecast only goes about {FORECAST_MAX_DAYS} days ahead "
                         f"(until {today + timedelta(days=FORECAST_MAX_DAYS - 1)}).")
    cutoff = today - timedelta(days=ARCHIVE_DELAY_DAYS)  # first day served by forecast API
    pieces = []
    if start < cutoff:
        pieces.append((ARCHIVE_URL, start, min(end, cutoff - timedelta(days=1))))
    if end >= cutoff:
        pieces.append((FORECAST_URL, max(start, cutoff), end))
    return pieces


def fetch(url, lats, lons, start, end, pause=1.0, retries=4, hourly=False):
    """Return a long DataFrame: lat, lon, date (or time if hourly), rain_mm, source.

    Hourly values are Open-Meteo's precipitation sum over the hour BEFORE each time stamp.
    """
    rows = []
    source = "past" if url == ARCHIVE_URL else "forecast/recent"
    for i in range(0, len(lats), BATCH):
        bl, bo = lats[i:i + BATCH], lons[i:i + BATCH]
        params = {
            "latitude": ",".join(f"{x:.4f}" for x in bl),
            "longitude": ",".join(f"{x:.4f}" for x in bo),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            **({"hourly": "precipitation"} if hourly else {"daily": "precipitation_sum"}),
            "timezone": TIMEZONE,
        }
        for attempt in range(retries):
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:  # rate limited
                time.sleep(15 * (attempt + 1))
                continue
            if r.status_code == 400:
                raise RuntimeError(f"Open-Meteo rejected the request: {r.text}")
            r.raise_for_status()
            break
        else:
            raise RuntimeError("Rate limited too many times; use a bigger --step or fewer days.")

        data = r.json()
        if isinstance(data, dict):  # a single location comes back as one object
            data = [data]
        block, var = ("hourly", "precipitation") if hourly else ("daily", "precipitation_sum")
        for lat, lon, loc in zip(bl, bo, data):
            for d, mm in zip(loc[block]["time"], loc[block][var]):
                rows.append((lat, lon, d, mm, source))
        print(f"  [{source}] {min(i + BATCH, len(lats))}/{len(lats)} points")
        time.sleep(pause)
    return pd.DataFrame(rows, columns=["lat", "lon", "time" if hourly else "date", "rain_mm", "source"])


def plot_total(totals, out_png, title):
    import matplotlib.pyplot as plt

    lats, lons = np.sort(totals["lat"].unique()), np.sort(totals["lon"].unique())
    grid = totals.pivot(index="lat", columns="lon", values="total_rain_mm").to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    half_lat = (lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.05
    half_lon = (lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.05
    extent = [lons[0] - half_lon, lons[-1] + half_lon, lats[0] - half_lat, lats[-1] + half_lat]
    im = ax.imshow(grid, origin="lower", extent=extent, cmap="Blues", vmin=0)
    fig.colorbar(im, ax=ax, label="Total rainfall (mm)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")


def main():
    p = argparse.ArgumentParser()
    # Box (defaults = northern Taiwan; edit these to change the area)
    p.add_argument("--south", type=float, default=24.8)
    p.add_argument("--west", type=float, default=121.3)
    p.add_argument("--north", type=float, default=25.3)
    p.add_argument("--east", type=float, default=122.0)
    p.add_argument("--step", type=float, default=0.1, help="degrees; weather cells are ~10 km, so 0.1 is plenty")
    # Dates: either --start/--end, or --forecast N
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--forecast", type=int, help="next N days from today (max 16)")
    p.add_argument("--hourly", action="store_true", help="hourly rain (needed for flood_sim.py)")
    p.add_argument("--out", default="rainfall")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    a = p.parse_args()

    today = date.today()
    if a.forecast:
        start, end = today, today + timedelta(days=a.forecast - 1)
    elif a.start:
        start = date.fromisoformat(a.start)
        end = date.fromisoformat(a.end) if a.end else start
    else:  # default when nothing is given: next 7 days
        start, end = today, today + timedelta(days=6)
    if end < start:
        raise SystemExit("--end must be on or after --start")

    grid = make_grid(a.south, a.west, a.north, a.east, a.step)
    pieces = plan_requests(start, end, today)
    print(f"{len(grid)} points, {start} to {end}")
    for url, s, e in pieces:
        print(f"  {'past (archive)' if url == ARCHIVE_URL else 'forecast'} API: {s} to {e}")

    lats, lons = grid["lat"].tolist(), grid["lon"].tolist()
    if a.hourly:
        hourly = pd.concat([fetch(url, lats, lons, s, e, hourly=True) for url, s, e in pieces], ignore_index=True)
        hourly = hourly.sort_values(["lat", "lon", "time"])
        hourly.to_csv(f"{a.out}_hourly.csv", index=False)
        print(f"Saved {a.out}_hourly.csv  ({len(hourly)} rows, max {hourly['rain_mm'].max():.1f} mm/h)")
        daily = (hourly.assign(date=hourly["time"].str[:10])
                       .groupby(["lat", "lon", "date"], as_index=False)
                       .agg(rain_mm=("rain_mm", "sum"), source=("source", "first")))
    else:
        daily = pd.concat([fetch(url, lats, lons, s, e) for url, s, e in pieces], ignore_index=True)
        daily = daily.sort_values(["lat", "lon", "date"])
    daily.to_csv(f"{a.out}_daily.csv", index=False)
    print(f"Saved {a.out}_daily.csv  ({len(daily)} rows)")

    totals = (daily.groupby(["lat", "lon"], as_index=False)
                   .agg(total_rain_mm=("rain_mm", "sum"), max_day_mm=("rain_mm", "max")))
    totals.to_csv(f"{a.out}_total.csv", index=False)
    print(f"Saved {a.out}_total.csv  (max total {totals['total_rain_mm'].max():.1f} mm)")

    if a.plot:
        plot_total(totals, f"{a.out}_total.png", f"Total rainfall {start} to {end}")


if __name__ == "__main__":
    main()

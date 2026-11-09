"""
Get elevation + water flag for a grid of points inside a bounding box,
using the free Open-Meteo Elevation API and an offline land/water mask.

Install:
    pip install requests numpy pandas global-land-mask matplotlib

Example (northern Taiwan, ~0.02 deg spacing):
    python elevation_box.py --south 24.8 --west 121.3 --north 25.3 --east 122.0 --step 0.02 --plot

For flood simulation use --step-m (spacing in metres, square cells), e.g.
    python elevation_box.py --south 25.00 --west 121.45 --north 25.10 --east 121.58 --step-m 100
The output CSV (lat, lon, elevation_m, is_water) can be passed straight to
flood_sim.py --dem.
"""
import argparse
import time

import math

import numpy as np
import pandas as pd
import requests

try:
    from global_land_mask import globe
except ImportError:  # optional: fall back to "elevation <= 0 is sea"
    globe = None

URL = "https://api.open-meteo.com/v1/elevation"
BATCH = 100  # max points per Open-Meteo request


def metre_steps(south, north, step_m):
    """Degree steps giving roughly square cells of step_m metres."""
    lat_c = math.radians((south + north) / 2)
    return step_m / 110574.0, step_m / (111320.0 * math.cos(lat_c))


def make_grid(south, west, north, east, step, step_lon=None):
    """Return a DataFrame of lat/lon points covering the box."""
    step_lon = step_lon or step
    lats = np.arange(south, north + step / 2, step)
    lons = np.arange(west, east + step_lon / 2, step_lon)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return pd.DataFrame({"lat": lat_g.ravel().round(6), "lon": lon_g.ravel().round(6)})


def get_elevations(lats, lons, pause=1.0, retries=3):
    """Look up elevations (metres) in batches of 100."""
    heights = []
    n = len(lats)
    for i in range(0, n, BATCH):
        params = {
            "latitude": ",".join(f"{x:.5f}" for x in lats[i:i + BATCH]),
            "longitude": ",".join(f"{x:.5f}" for x in lons[i:i + BATCH]),
        }
        for attempt in range(retries):
            r = requests.get(URL, params=params, timeout=20)
            if r.status_code == 429:          # rate limited: wait and retry
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError("Rate limited too many times; try a larger --step.")
        heights.extend(r.json()["elevation"])
        print(f"  {min(i + BATCH, n)}/{n} points done")
        if i + BATCH < n:
            time.sleep(pause)
    return heights


def add_water(df):
    """Flag ocean/sea points (offline ~1 km mask) and clamp their height to 0."""
    if globe is not None:
        df["is_water"] = globe.is_ocean(df["lat"].to_numpy(), df["lon"].to_numpy())
    else:
        print("  note: global-land-mask not installed; treating elevation <= 0 as sea")
        df["is_water"] = df["elevation_m"].fillna(0) <= 0
    df.loc[df["is_water"], "elevation_m"] = 0.0
    return df


def plot(df, out_png):
    import matplotlib.pyplot as plt

    lats, lons = np.sort(df["lat"].unique()), np.sort(df["lon"].unique())
    grid = df.pivot(index="lat", columns="lon", values="elevation_m").to_numpy()
    water = df.pivot(index="lat", columns="lon", values="is_water").to_numpy().astype(bool)

    land = np.ma.masked_where(water, grid)
    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [lons[0], lons[-1], lats[0], lats[-1]]
    ax.imshow(np.where(water, 1, np.nan), origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=2)
    from matplotlib.colors import LinearSegmentedColormap
    land_cmap = LinearSegmentedColormap.from_list(  # terrain colours without the blue part
        "land", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256)))
    im = ax.imshow(land, origin="lower", extent=extent, cmap=land_cmap, vmin=0)
    fig.colorbar(im, ax=ax, label="Elevation (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Elevation (water in blue)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--south", type=float, default=24.8)
    p.add_argument("--west", type=float, default=121.3)
    p.add_argument("--north", type=float, default=25.3)
    p.add_argument("--east", type=float, default=122.0)
    p.add_argument("--step", type=float, default=0.02, help="spacing in degrees")
    p.add_argument("--step-m", type=float, help="spacing in metres (square cells; overrides --step)")
    p.add_argument("--out", default="elevation_grid.csv")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    a = p.parse_args()

    if a.step_m:
        step, step_lon = metre_steps(a.south, a.north, a.step_m)
    else:
        step, step_lon = a.step, a.step
    df = make_grid(a.south, a.west, a.north, a.east, step, step_lon)
    calls = -(-len(df) // BATCH)
    print(f"{len(df)} points -> {calls} API calls (free limit ~10,000 calls/day)")

    df["elevation_m"] = get_elevations(df["lat"].tolist(), df["lon"].tolist())
    df = add_water(df)
    df.to_csv(a.out, index=False)
    print(f"Saved {a.out}  ({df['is_water'].sum()} water points)")

    if a.plot:
        plot(df, a.out.rsplit(".", 1)[0] + ".png")


if __name__ == "__main__":
    main()
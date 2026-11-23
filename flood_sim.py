#!/usr/bin/env python3
"""
flood_sim.py - 2D flood simulation and evacuation assessment
=============================================================

Given terrain elevation and rainfall, this computes water depth and velocity
on every grid cell over time, then decides which communities must evacuate
and when.

Model: local-inertial shallow-water equations (Bates et al. 2010, the method
used by LISFLOOD-FP), with Manning friction, Horton infiltration, urban storm
drainage, upstream river inflow and a fixed sea level.

Quick start
-----------
    python flood_sim.py --demo valley --out demo_valley
    python flood_sim.py --dem dem.asc --rain rain.csv --towns towns.csv --out results

    # download elevation + rain from Open-Meteo instead of using files
    python flood_sim.py --dem online --rain online --area 25.04,121.55,25.10,121.65 --forecast 3
    python flood_sim.py --dem online --rain online --area 25.00,121.45,25.10,121.58 \
        --start 2024-10-02 --end 2024-10-04 --towns my_towns.csv

    # mix: your own elevation file, rain from Open-Meteo (or typed in by hand)
    python flood_sim.py --dem elevation_grid.csv --rain online --start 2024-10-02 --end 2024-10-04
    python flood_sim.py --dem dem.asc --rain-values 5,20,60,80,40,10

Press Run with no options (e.g. in PyCharm) and it asks you step by step
where each input should come from.

Run `python flood_sim.py -h` for every option, and see README.md for the
input file formats.

This is a simplified screening model for teaching, scenario comparison and
preliminary planning. It does NOT replace official warnings or evacuation
orders.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from datetime import datetime

G = 9.81
# Horton infiltration: (f0 initial mm/h, fc final mm/h, tau decay hours)
SOILS = {
    "sand": (60.0, 20.0, 1.5),
    "loam": (25.0, 5.0, 1.5),
    "clay": (10.0, 1.5, 1.0),
    "saturated": (2.0, 1.0, 1.0),
    "urban": (1.0, 0.5, 1.0),
}


# --------------------------------------------------------------------------
# Grid I/O
# --------------------------------------------------------------------------
@dataclass
class GridMeta:
    cellsize: float | None = None
    xll: float = 0.0          # x of lower-left corner
    yll: float = 0.0          # y of lower-left corner
    nodata: float = -9999.0
    crs_transform: object = None  # rasterio profile when read from GeoTIFF
    geo: dict | None = None       # set when the DEM came from a lat/lon grid

    def latlon_to_rc(self, lat, lon):
        g = self.geo
        x = (lon - g["lonc"]) * 111320.0 * math.cos(math.radians(g["latc"]))
        y = (lat - g["latc"]) * 110574.0
        return (g["y_top"] - y) / g["dx"] - 0.5, (x - g["x_left"]) / g["dx"] - 0.5

    def rc_to_latlon(self, r, c):
        g = self.geo
        x = g["x_left"] + (np.asarray(c) + 0.5) * g["dx"]
        y = g["y_top"] - (np.asarray(r) + 0.5) * g["dx"]
        return (g["latc"] + y / 110574.0,
                g["lonc"] + x / (111320.0 * math.cos(math.radians(g["latc"]))))


def _read_text_matrix(text: str) -> np.ndarray:
    text = text.replace(",", " ").replace(";", " ").replace("\t", " ")
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # header / label line
    if len(rows) < 2:
        raise ValueError("no numeric matrix found")
    w = min(len(r) for r in rows)
    return np.array([r[:w] for r in rows], dtype=np.float64)


XY_ORDER = "auto"   # how to read x/y columns in point files: auto, latlon (x=lat) or lonlat (x=lon)
_ELEV_KEYS = ("elevation_m", "elevation", "elev", "z", "height", "height_m", "h", "alt", "altitude")


def _point_files(path):
    """A point-elevation input can be one file, a folder of files, a glob or a comma list."""
    import glob
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")) + glob.glob(os.path.join(path, "*.csv")))
    elif any(ch in path for ch in "*?["):
        files = sorted(glob.glob(os.path.expanduser(path)))
    elif "," in path and not os.path.exists(path):
        files = [p.strip() for p in path.split(",") if p.strip()]
    else:
        files = [path]
    if not files:
        raise SystemExit(f"no elevation point files found in {path!r}")
    return files


def is_point_dem(path):
    """True for elevation given as points with coordinates (lat/lon CSV, x/y/z JSON, a folder of them)."""
    if not path or path.lower() == "online":
        return False
    if os.path.isdir(path) or any(ch in path for ch in "*?[") or ("," in path and not os.path.exists(path)):
        return True
    ext = os.path.splitext(path)[1].lower()
    return ext == ".json" or is_latlon_csv(path)


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _read_point_json(path):
    """JSON list of points: [{"x": "24.67", "y": "121.29", "z": "1170"}, ...]
    (also lat/lon/elevation keys, [x, y, z] lists, or {"points": [...]})."""
    import json
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = next((v for k, v in data.items() if isinstance(v, list)), [data])
    a, b, z, water, named = [], [], [], [], None
    for p in data:
        if isinstance(p, (list, tuple)):
            a.append(_num(p[0])); b.append(_num(p[1])); z.append(_num(p[2]) if len(p) > 2 else np.nan)
            water.append(False); named = named or "xy"
            continue
        d = {str(k).lower(): v for k, v in p.items()}
        if "lat" in d and ("lon" in d or "lng" in d):
            a.append(_num(d["lat"])); b.append(_num(d.get("lon", d.get("lng")))); named = "latlon"
        else:
            a.append(_num(d.get("x"))); b.append(_num(d.get("y"))); named = named or "xy"
        z.append(next((_num(d[k]) for k in _ELEV_KEYS if k in d), np.nan))
        water.append(str(d.get("is_water", "")).lower() in ("true", "1", "yes"))
    return np.array(a), np.array(b), np.array(z), np.array(water, bool), named


def _read_point_csv(path):
    header, rows = _read_table(path)
    low = [h.lower() for h in header]
    if "lat" in low and ("lon" in low or "lng" in low):
        ia, ib, named = low.index("lat"), low.index("lon" if "lon" in low else "lng"), "latlon"
    else:
        ia, ib, named = low.index("x"), low.index("y"), "xy"
    iel = next(i for i, h in enumerate(low) if h in _ELEV_KEYS)
    iw = low.index("is_water") if "is_water" in low else None
    a = np.array([_num(r[ia]) for r in rows]); b = np.array([_num(r[ib]) for r in rows])
    el = np.array([_num(r[iel]) for r in rows])
    wat = np.array([r[iw].strip().lower() in ("true", "1", "yes") for r in rows]) if iw is not None else np.zeros(len(rows), bool)
    return a, b, el, wat, named


def read_latlon_dem(path: str, dx: float | None) -> tuple[np.ndarray, GridMeta]:
    """Elevation points with coordinates: a lat/lon CSV (e.g. from elevation_box.py), or
    x/y/z JSON or CSV files (one file or a folder, e.g. one file per longitude line).

    The points must form a (roughly) regular lat/lon grid; they are resampled onto a
    square grid in metres so the flow model sees equal cell sizes.
    """
    files = _point_files(path)
    parts = []
    for fpath in files:
        reader = _read_point_json if fpath.lower().endswith(".json") else _read_point_csv
        parts.append(reader(fpath))
    A = np.concatenate([p[0] for p in parts]); B = np.concatenate([p[1] for p in parts])
    el = np.concatenate([p[2] for p in parts]); wat = np.concatenate([p[3] for p in parts])
    named = {p[4] for p in parts}
    if "latlon" in named and len(named) == 1:
        lat, lon = A, B
    else:  # x/y columns: which one is latitude?
        order = XY_ORDER
        if order == "auto":
            if np.nanmax(np.abs(B)) > 90 >= np.nanmax(np.abs(A)):
                order = "latlon"
            elif np.nanmax(np.abs(A)) > 90 >= np.nanmax(np.abs(B)):
                order = "lonlat"
            else:
                order = "lonlat"
                print("  note: can't tell which of x/y is latitude; assuming x = longitude, y = latitude "
                      "(use --xy-order latlon if it is the other way round)")
        lat, lon = (A, B) if order == "latlon" else (B, A)
        if np.nanmax(np.abs(lat)) > 90 or np.nanmax(np.abs(lon)) > 180:
            raise SystemExit("x/y values are not degrees (map coordinates in metres?). Point files need "
                             "latitude/longitude; for metre grids use an .asc or .tif file instead.")
        print(f"  reading x/y as {'x = latitude, y = longitude' if order == 'latlon' else 'x = longitude, y = latitude'}")
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon, el, wat = lat[ok], lon[ok], el[ok], wat[ok]
    if not wat.any():
        wat = np.nan_to_num(el, nan=1.0) <= 0      # no water flags given: height 0 or below = sea (as Open-Meteo returns)
    lats, lons = np.unique(np.round(lat, 6)), np.unique(np.round(lon, 6))
    print(f"  {len(files)} file(s), {len(lat):,} points: lat {lats[0]:.4f}–{lats[-1]:.4f}, "
          f"lon {lons[0]:.4f}–{lons[-1]:.4f}, height {np.nanmin(el):.0f}–{np.nanmax(el):.0f} m")
    if len(lats) < 2 or len(lons) < 2:
        one = f"longitude {lons[0]:.4f}" if len(lons) < 2 else f"latitude {lats[0]:.4f}"
        raise SystemExit(f"these points are a single line ({one}), but a flood map needs an area.\n"
                         f"  Give a folder or several files, one per line, e.g. --dem my_points/ or --dem 'data/*.json',\n"
                         f"  or download the area with --dem online --area south,west,north,east.")
    E = np.full((len(lats), len(lons)), np.nan); W = np.zeros_like(E, dtype=bool)
    ia = np.searchsorted(lats, np.round(lat, 6)); io_ = np.searchsorted(lons, np.round(lon, 6))
    E[ia, io_] = el; W[ia, io_] = wat
    holes = np.isnan(E)
    if holes.any():  # fill gaps from neighbouring points
        if holes.mean() > 0.05:
            print(f"  note: {holes.mean():.0%} of grid positions have no point; filled from neighbours")
        for _ in range(max(E.shape)):
            if not np.isnan(E).any():
                break
            P = np.pad(E, 1, constant_values=np.nan)
            nb = np.stack([P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]])
            cnt = np.isfinite(nb).sum(0)
            m = np.isnan(E) & (cnt > 0)
            E[m] = (np.nansum(nb, 0) / np.maximum(cnt, 1))[m]
        E[np.isnan(E)] = np.nanmean(el)
    latc, lonc = (lats[0] + lats[-1]) / 2, (lons[0] + lons[-1]) / 2
    kx = 111320.0 * math.cos(math.radians(latc)); ky = 110574.0
    dlat = float(np.median(np.diff(lats))); dlon = float(np.median(np.diff(lons)))
    auto_dx = not dx
    if not dx:
        dx = round(max(min(dlat * ky, dlon * kx), 10.0), 1)
    if dx > 150 and not auto_dx:
        print(f"  note: {dx:g} m cells are coarse; fine for a quick test, use --dx 100 or less for real results.")
    elif dx > 150:
        print(f"  WARNING: elevation points are {dx:.0f} m apart. Floods follow features smaller than this, "
              f"so results will be rough.\n           Re-download with elevation_box.py --step-m 100 (or smaller).")
    width, height = (lons[-1] - lons[0]) * kx, (lats[-1] - lats[0]) * ky
    nx, ny = max(int(width // dx) + 1, 2), max(int(height // dx) + 1, 2)
    x_left, y_top = (lons[0] - lonc) * kx, (lats[-1] - latc) * ky
    meta = GridMeta(cellsize=dx, geo=dict(latc=latc, lonc=lonc, x_left=x_left, y_top=y_top, dx=dx,
                                          bounds=(float(lats[0]), float(lons[0]), float(lats[-1]), float(lons[-1]))))
    rr, cc = np.mgrid[0:ny, 0:nx]
    glat, glon = meta.rc_to_latlon(rr, cc)
    fi = np.interp(glat, lats, np.arange(len(lats)))   # handles uneven spacing too
    fj = np.interp(glon, lons, np.arange(len(lons)))
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    i1, j1 = np.minimum(i0 + 1, len(lats) - 1), np.minimum(j0 + 1, len(lons) - 1)
    ti, tj = fi - i0, fj - j0
    T = (E[i0, j0] * (1 - ti) * (1 - tj) + E[i0, j1] * (1 - ti) * tj + E[i1, j0] * ti * (1 - tj) + E[i1, j1] * ti * tj)
    sea = W[np.rint(fi).astype(int), np.rint(fj).astype(int)]
    T = np.where(sea, np.minimum(T, -1.0), T)   # water cells become sea (below 0 m)
    print(f"  lat/lon DEM {len(lats)}×{len(lons)} points ({dlat*ky:.0f}×{dlon*kx:.0f} m) -> "
          f"{ny}×{nx} square cells of {dx:g} m; {int(sea.sum())} sea cells")
    if ny * nx > 300_000:
        print(f"  note: {ny*nx:,} cells is a big run (can take hours). Speed up with --dx 100 or --coarsen 2.")
    return T, meta


def is_latlon_csv(path):
    if os.path.splitext(path)[1].lower() not in (".csv", ".txt"):
        return False
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        first = f.readline().lower().replace(" ", "")
    parts = first.strip().split(",")
    return "lat" in parts and "lon" in parts


def read_grid(path: str, png_min=0.0, png_max=500.0, nodata=None, dx=None) -> tuple[np.ndarray, GridMeta]:
    """Read a 2D grid. Row 0 is the NORTH edge. Returns (array, meta)."""
    ext = os.path.splitext(path)[1].lower()
    meta = GridMeta()
    if is_point_dem(path):
        return read_latlon_dem(path, dx)
    if ext == ".asc":
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        hdr, k = {}, 0
        while k < len(lines):
            parts = lines[k].split()
            if len(parts) == 2 and parts[0][0].isalpha():
                hdr[parts[0].lower()] = float(parts[1])
                k += 1
            else:
                break
        arr = _read_text_matrix("\n".join(lines[k:]))
        cs = hdr.get("cellsize")
        meta.cellsize = cs
        meta.nodata = hdr.get("nodata_value", -9999.0)
        if "xllcorner" in hdr:
            meta.xll = hdr["xllcorner"]
        elif "xllcenter" in hdr and cs:
            meta.xll = hdr["xllcenter"] - cs / 2
        if "yllcorner" in hdr:
            meta.yll = hdr["yllcorner"]
        elif "yllcenter" in hdr and cs:
            meta.yll = hdr["yllcenter"] - cs / 2
        arr = arr.astype(np.float64)
        arr[np.isclose(arr, meta.nodata)] = np.nan
    elif ext in (".csv", ".txt", ".xyz", ".dat"):
        with open(path, encoding="utf-8", errors="replace") as f:
            arr = _read_text_matrix(f.read())
    elif ext in (".tif", ".tiff"):
        try:
            import rasterio  # type: ignore
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float64)
                meta.cellsize = abs(src.transform.a)
                meta.xll = src.bounds.left
                meta.yll = src.bounds.bottom
                if src.nodata is not None:
                    meta.nodata = src.nodata
                    arr[arr == src.nodata] = np.nan
                meta.crs_transform = src.profile
        except ImportError:
            from PIL import Image
            arr = np.array(Image.open(path), dtype=np.float64)
            if arr.ndim == 3:
                arr = arr[..., 0]
            print("  note: rasterio not installed - GeoTIFF read without georeferencing; "
                  "pass --dx for the cell size.")
    elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
        from PIL import Image
        im = Image.open(path)
        a = np.array(im, dtype=np.float64)
        if a.ndim == 3:
            a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
            scale = 255.0
        else:
            scale = 65535.0 if a.max() > 255 else 255.0
        arr = png_min + (png_max - png_min) * a / scale
    else:
        raise ValueError(f"unsupported grid format: {ext}")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < -9000] = np.nan
    return arr, meta


def write_asc(path: str, arr: np.ndarray, meta: GridMeta, dx: float, nodata=-9999.0):
    a = np.where(np.isfinite(arr), arr, nodata)
    ny, nx = a.shape
    with open(path, "w") as f:
        f.write(f"ncols {nx}\nnrows {ny}\nxllcorner {meta.xll}\nyllcorner {meta.yll}\n"
                f"cellsize {dx}\nNODATA_value {nodata}\n")
        np.savetxt(f, a, fmt="%.3f")


def write_geotiff(path: str, arr: np.ndarray, meta: GridMeta) -> bool:
    if meta.crs_transform is None:
        return False
    try:
        import rasterio  # type: ignore
    except ImportError:
        return False
    prof = dict(meta.crs_transform)
    prof.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)
    return True


def coarsen(arr: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return arr
    ny, nx = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
    a = arr[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=(1, 3))


# --------------------------------------------------------------------------
# Rainfall
# --------------------------------------------------------------------------
def _read_table(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        sample = f.read()
    dialect = csv.Sniffer().sniff(sample.splitlines()[0]) if "," not in sample.splitlines()[0] else csv.excel
    rows = [r for r in csv.reader(io.StringIO(sample), dialect) if r and any(c.strip() for c in r)]
    header = [h.strip() for h in rows[0]]
    return header, [[c.strip() for c in r] for r in rows[1:]]


class Rainfall:
    """Rain intensity field in mm/h as a function of time.

    Sources (choose one):
      * synthetic design storm  (total mm, duration h, pattern)
      * single series CSV       time_h, rain_mm
      * multi-station CSV       time_h, <station1>, <station2>, ...  + stations CSV
    Optional: a multiplier grid (same size as the DEM) and an orographic factor.
    """

    def __init__(self, shape):
        self.shape = shape
        self.kind = "synthetic"
        self.total = 0.0
        self.hours = 1.0
        self.pattern = "center"
        self.times = None        # interval start times (s)
        self.ends = None         # interval end times (s)
        self.values = None       # (n_intervals, n_stations) mm/h
        self.weights = None      # (n_stations, ny, nx)
        self.multiplier = None   # (ny, nx)
        self.t0 = None           # datetime of t = 0, when known
        self._cache = (None, None)

    # ---- constructors ----
    @classmethod
    def synthetic(cls, shape, total, hours, pattern):
        r = cls(shape)
        r.total, r.hours, r.pattern = total, max(hours, 0.1), pattern
        return r

    @classmethod
    def from_long(cls, shape, header, rows, meta: GridMeta, daily_pattern="center"):
        """Open-Meteo style: lat, lon, time|date, rain_mm (one row per point per hour/day)."""
        import datetime as dtm
        if meta.geo is None:
            raise SystemExit("this rain file uses lat/lon, so the DEM must too "
                             "(use the CSV from elevation_box.py as --dem)")
        low = [h.lower() for h in header]
        ila, ilo = low.index("lat"), low.index("lon")
        it = next(i for i, h in enumerate(low) if h in ("time", "date", "datetime"))
        ir = next(i for i, h in enumerate(low) if h in ("rain_mm", "precipitation", "precipitation_mm", "rain", "precipitation_sum"))
        daily = len(rows[0][it].strip()) <= 10
        recs, stations, stamps = {}, set(), set()
        for r in rows:
            key = (round(float(r[ila]), 5), round(float(r[ilo]), 5))
            t = dtm.datetime.fromisoformat(r[it].strip())
            if not daily:
                t -= dtm.timedelta(hours=1)  # Open-Meteo hourly value = rain in the hour BEFORE the stamp
            v = r[ir].strip()
            recs[(key, t)] = float(v) if v not in ("", "nan", "None", "NaN") else 0.0
            stations.add(key); stamps.add(t)
        stations, stamps = sorted(stations), sorted(stamps)
        vals = np.array([[recs.get((k, t), 0.0) for k in stations] for t in stamps])
        t0 = stamps[0]
        if daily:
            # spread each day's total over 24 hours
            if daily_pattern == "uniform":
                w = np.full(24, 1 / 24)
            else:
                w = np.array([design_storm(h * 3600 + 1800, 1.0, 24, daily_pattern) for h in range(24)]); w /= w.sum()
            vals = (vals[:, None, :] * w[None, :, None]).reshape(-1, len(stations))
            starts = np.array([(d - t0).total_seconds() / 3600 + h for d in stamps for h in range(24)])
            print(f"  note: daily rain spread over 24 h ('{daily_pattern}' shape). Hourly data "
                  f"(rainfall.py --hourly) gives far more realistic flood peaks.")
        else:
            starts = np.array([(t - t0).total_seconds() / 3600 for t in stamps])
        r = cls(shape)
        r.kind = "stations" if len(stations) > 1 else "series"
        r.times = starts * 3600.0
        r.ends = np.append(r.times[1:], r.times[-1] + 3600.0)
        r.values = vals  # mm per hour interval = mm/h
        r.hours = float(r.ends[-1] / 3600.0)
        r.total = float(vals.mean(axis=1).sum())
        r.t0 = t0
        locs = [meta.latlon_to_rc(la, lo) for la, lo in stations]
        r.weights = idw_weights(shape, locs) if len(stations) > 1 else None
        if len(stations) == 1:
            r.values = vals[:, :1]
        tot = vals.sum(axis=0)
        print(f"  rain: {len(stations)} grid points, {len(stamps)} {'days' if daily else 'hours'} from {t0:%Y-%m-%d %H:%M}, "
              f"total {tot.min():.0f}–{tot.max():.0f} mm, peak {vals.max():.1f} mm/h")
        return r

    @classmethod
    def from_csv(cls, shape, path, stations_path=None, grid_rc=None, meta=None, daily_pattern="center"):
        header, rows = _read_table(path)
        lower = [h.lower() for h in header]
        if "lat" in lower and "lon" in lower:
            return cls.from_long(shape, header, rows, meta, daily_pattern)
        tcol = next((i for i, h in enumerate(lower) if h in ("time_h", "time", "hour", "hours", "t", "時間")), 0)
        data_cols = [i for i in range(len(header)) if i != tcol]
        t = np.array([float(r[tcol]) for r in rows])
        vals = np.array([[float(r[i]) if r[i] not in ("", "-", "T") else 0.0 for i in data_cols] for r in rows])
        order = np.argsort(t)
        t, vals = t[order], vals[order]
        if len(t) == 1:
            ends = t + 1.0
        else:
            dts = np.diff(t)
            ends = np.append(t[1:], t[-1] + dts[-1])
        dur = ends - t
        names = [header[i] for i in data_cols]
        is_intensity = any(k in lower[data_cols[0]] for k in ("mm_h", "mm/h", "per_h", "intensity"))
        r = cls(shape)
        r.kind = "series"
        r.times, r.ends = t * 3600.0, ends * 3600.0
        r.values = vals if is_intensity else vals / dur[:, None]   # -> mm/h
        r.hours = float(ends[-1])
        r.total = float((r.values * dur[:, None]).mean(axis=1).sum())
        if len(names) > 1 or stations_path:
            if not stations_path:
                raise SystemExit("rain file has several station columns: pass --stations with their locations")
            locs = read_stations(stations_path, grid_rc)
            missing = [n for n in names if n not in locs]
            if missing:
                raise SystemExit(f"stations file has no location for: {', '.join(missing)}")
            r.kind = "stations"
            r.weights = idw_weights(shape, [locs[n] for n in names])
            r.station_names = names
        return r

    # ---- evaluation ----
    @property
    def end_s(self):
        return self.hours * 3600.0

    def scalar(self, t):
        """Domain-mean intensity (mm/h) at time t, before multipliers."""
        if self.kind == "synthetic":
            return design_storm(t, self.total, self.hours, self.pattern)
        k = np.searchsorted(self.times, t, side="right") - 1
        if k < 0 or t >= self.ends[-1]:
            return 0.0
        return float(self.values[k].mean())

    def field(self, t):
        """Intensity (mm/h): scalar or (ny, nx) array."""
        if self.kind == "stations":
            k = np.searchsorted(self.times, t, side="right") - 1
            if k < 0 or t >= self.ends[-1]:
                f = 0.0
            elif self._cache[0] == k:
                f = self._cache[1]
            else:
                f = np.tensordot(self.values[k], self.weights, axes=1)
                self._cache = (k, f)
        else:
            f = self.scalar(t)
        if self.multiplier is not None:
            f = f * self.multiplier
        return f


def design_storm(t, total, hours, pattern):
    dur = hours * 3600.0
    if t < 0 or t >= dur or dur <= 0:
        return 0.0
    avg = total / hours
    if pattern == "uniform":
        return avg
    p = 0.15 if pattern == "front" else (0.85 if pattern == "back" else 0.45)
    u = t / dur
    return 2 * avg * u / p if u < p else 2 * avg * (1 - u) / (1 - p)


def read_stations(path, grid_rc):
    """CSV: name, row, col  -or-  name, x, y (map coordinates)."""
    header, rows = _read_table(path)
    low = [h.lower() for h in header]
    out = {}
    for r in rows:
        d = dict(zip(low, r))
        name = d.get("name") or d.get("station") or r[0]
        out[name] = grid_rc(d)
    return out


def idw_weights(shape, locs, power=2.0):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    w = np.empty((len(locs), ny, nx), dtype=np.float32)
    for k, (r, c) in enumerate(locs):
        d2 = (yy - r) ** 2 + (xx - c) ** 2
        w[k] = 1.0 / np.maximum(d2, 0.25) ** (power / 2)
    w /= w.sum(axis=0, keepdims=True)
    return w


class Inflow:
    """Upstream river inflow (m3/s). From a CSV hydrograph or the rational method."""

    def __init__(self):
        self.times = None
        self.q = None
        self.area = 0.0
        self.c = 0.8
        self.lag_h = 2.0
        self.base = 0.1  # m3/s per km2

    @classmethod
    def from_csv(cls, path):
        header, rows = _read_table(path)
        o = cls()
        arr = np.array([[float(r[0]), float(r[1])] for r in rows])
        arr = arr[np.argsort(arr[:, 0])]
        o.times, o.q = arr[:, 0] * 3600.0, arr[:, 1]
        return o

    def at(self, t, rain: Rainfall, oro_pct):
        if self.times is not None:
            return float(np.interp(t, self.times, self.q, left=self.q[0], right=self.q[-1]))
        if self.area <= 0:
            return 0.0
        i = rain.scalar(t - self.lag_h * 3600.0) * (1 + oro_pct / 100 * 0.6)
        return self.base * self.area + self.c * i * self.area / 3.6

    def end_s(self):
        return float(self.times[-1]) if self.times is not None else 0.0


# --------------------------------------------------------------------------
# Communities
# --------------------------------------------------------------------------
@dataclass
class Town:
    name: str
    row: int
    col: int
    radius: float           # cells
    population: int = 0
    cells: tuple = field(default=None, repr=False)   # (rows, cols) index arrays


def read_towns(path, grid_rc, dx):
    """CSV: name, population, radius_m, and either row,col or x,y."""
    header, rows = _read_table(path)
    low = [h.lower() for h in header]
    towns = []
    for r in rows:
        d = dict(zip(low, r))
        rr, cc = grid_rc(d)
        rad = float(d.get("radius_m") or d.get("radius") or 5 * dx) / dx
        towns.append(Town(d.get("name") or r[0], int(round(rr)), int(round(cc)),
                          max(rad, 0.8), int(float(d.get("population") or d.get("pop") or 0))))
    return towns


def make_grid_rc(meta: GridMeta, ny, dx):
    def f(d):
        if "row" in d and "col" in d and d["row"] != "" and d["col"] != "":
            return float(d["row"]), float(d["col"])
        if "lat" in d and "lon" in d and d["lat"] != "":
            if meta.geo is None:
                raise SystemExit("lat/lon locations need a lat/lon DEM (CSV from elevation_box.py)")
            return meta.latlon_to_rc(float(d["lat"]), float(d["lon"]))
        if "x" in d and "y" in d:
            x, y = float(d["x"]), float(d["y"])
            col = (x - meta.xll) / dx - 0.5
            row = ny - 1 - ((y - meta.yll) / dx - 0.5)
            return row, col
        raise SystemExit("location needs row,col or lat,lon or x,y columns")
    return f


def assign_town_cells(towns, T, river):
    ny, nx = T.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    for t in towns:
        m = ((yy - t.row) ** 2 + (xx - t.col) ** 2 <= t.radius ** 2) & np.isfinite(T) & (T >= 0)
        if river is not None:
            m &= ~river
        t.cells = np.nonzero(m)
        if t.cells[0].size == 0:
            print(f"  warning: community '{t.name}' has no land cells inside the grid")


# --------------------------------------------------------------------------
# Hydrodynamic model
# --------------------------------------------------------------------------
@dataclass
class Params:
    dx: float = 40.0
    manning: float = 0.035
    soil: str = "loam"
    drain: float = 78.8         # mm/h inside communities
    oro: float = 0.0            # % rain increase per 1000 m
    sea_level: float = 0.0
    sea_below: float = 0.0      # cells with T < this are sea
    edges: str = ""             # open boundaries, any of N S E W
    evac_depth: float = 0.5
    warn_depth: float = 0.3
    area_pct: float = 20.0
    hazard_limit: float = 1.25
    cfl: float = 0.7
    max_dt: float = 30.0


class FloodModel:
    def __init__(self, T, P: Params, rain: Rainfall, inflow: Inflow, sources, towns, river=None):
        self.P = P
        self.rain, self.inflow = rain, inflow
        self.wall = ~np.isfinite(T)
        self.T = np.where(self.wall, np.nanmax(T) + 1000.0, T).astype(np.float64)
        ny, nx = T.shape
        self.ny, self.nx = ny, nx
        self.D = np.zeros((ny, nx))
        self.qx = np.zeros((ny, nx + 1))
        self.qy = np.zeros((ny + 1, nx))
        self.sea = (self.T < P.sea_below) & ~self.wall
        self.urban = np.zeros((ny, nx), bool)
        for t in towns:
            self.urban[t.cells] = True
        self.sources = [s for s in sources if 0 <= s[0] < ny and 0 <= s[1] < nx]
        self.towns = towns
        self.river = river
        self.t = 0.0
        self.dmax = 0.0
        self.cum_rain = 0.0
        self.max_depth = np.zeros((ny, nx))
        self.max_hazard = np.zeros((ny, nx))
        self.max_speed = np.zeros((ny, nx))
        self.first_evac = np.full((ny, nx), np.nan)
        self.orofac = 1.0 + P.oro / 100.0 / 1000.0 * np.clip(self.T, 0, None)
        self.D[self.sea] = np.maximum(0, P.sea_level - self.T[self.sea])
        self.spinning = False

    def spin_up(self, hours):
        """Run the river at its normal (pre-storm) flow with no rain, so it starts the
        event already flowing. Returns the normally-wet cells (river / lakes)."""
        self.spinning = True
        el = 0.0
        while el < hours * 3600:
            dt = self.stable_dt(); self.step(dt); el += dt
        self.spinning = False
        wet = (self.D >= 0.2) & ~self.sea
        self.max_depth[:] = 0; self.max_hazard[:] = 0; self.max_speed[:] = 0
        self.first_evac[:] = np.nan; self.cum_rain = 0.0
        return wet

    def set_towns(self, towns, river):
        self.towns, self.river = towns, river
        self.urban[:] = False
        for t in towns:
            self.urban[t.cells] = True

    # face flux update (vectorised)
    THETA = 0.7   # de Almeida et al. (2012) weighting; damps checkerboard sloshing in deep, slow water

    @staticmethod
    def _smooth(q, axis):
        """theta-weighted face flux: theta*q + (1-theta)/2*(q_prev + q_next) along the flow axis."""
        th = FloodModel.THETA
        out = q.copy()
        if axis == 1 and q.shape[1] > 2:
            out[:, 1:-1] = th * q[:, 1:-1] + (1 - th) / 2 * (q[:, :-2] + q[:, 2:])
        elif axis == 0 and q.shape[0] > 2:
            out[1:-1, :] = th * q[1:-1, :] + (1 - th) / 2 * (q[:-2, :] + q[2:, :])
        return out

    @staticmethod
    def _face(q, Ha, Hb, za, zb, dt, dx, n2, qs=None):
        hf = np.maximum(Ha, Hb) - np.maximum(za, zb)
        wet = hf > 0.002
        hfs = np.where(wet, hf, 1.0)
        S = (Hb - Ha) / dx
        qn = ((q if qs is None else qs) - G * hfs * dt * S) / (1 + G * dt * n2 * np.abs(q) / (hfs ** (7 / 3)))
        lim = hfs * np.sqrt(G * hfs)
        qn = np.clip(qn, -lim, lim)
        return np.where(wet, qn, 0.0)

    def step(self, dt):
        P, T, D = self.P, self.T, self.D
        dx, n2 = P.dx, P.manning ** 2
        H = T + D
        ghost = 0.003 * dx
        # interior faces
        qxs, qys = self._smooth(self.qx, 1), self._smooth(self.qy, 0)
        self.qx[:, 1:-1] = self._face(self.qx[:, 1:-1], H[:, :-1], H[:, 1:], T[:, :-1], T[:, 1:], dt, dx, n2, qxs[:, 1:-1])
        self.qy[1:-1, :] = self._face(self.qy[1:-1, :], H[:-1, :], H[1:, :], T[:-1, :], T[1:, :], dt, dx, n2, qys[1:-1, :])
        # boundaries: open = free outflow into a ghost cell slightly lower and dry
        e = P.edges.upper()
        if "W" in e:
            c = T[:, 0]; self.qx[:, 0] = -self._face(-self.qx[:, 0], H[:, 0], c - ghost, c, c - ghost, dt, dx, n2)
        else:
            self.qx[:, 0] = 0
        if "E" in e:
            c = T[:, -1]; self.qx[:, -1] = self._face(self.qx[:, -1], H[:, -1], c - ghost, c, c - ghost, dt, dx, n2)
        else:
            self.qx[:, -1] = 0
        if "N" in e:
            c = T[0, :]; self.qy[0, :] = -self._face(-self.qy[0, :], H[0, :], c - ghost, c, c - ghost, dt, dx, n2)
        else:
            self.qy[0, :] = 0
        if "S" in e:
            c = T[-1, :]; self.qy[-1, :] = self._face(self.qy[-1, :], H[-1, :], c - ghost, c, c - ghost, dt, dx, n2)
        else:
            self.qy[-1, :] = 0
        # walls (nodata) block flow
        if self.wall.any():
            w = self.wall
            self.qx[:, 1:-1][w[:, :-1] | w[:, 1:]] = 0
            self.qy[1:-1, :][w[:-1, :] | w[1:, :]] = 0
        # limit outflow so depth never goes negative
        qx, qy = self.qx, self.qy
        out = (np.maximum(-qx[:, :-1], 0) + np.maximum(qx[:, 1:], 0) +
               np.maximum(-qy[:-1, :], 0) + np.maximum(qy[1:, :], 0)) * dt / dx
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sc = np.where(out > D, D / np.where(out > 0, out, 1), 1.0)
        qx[:, 1:] = np.where(qx[:, 1:] > 0, qx[:, 1:] * sc, qx[:, 1:])
        qx[:, :-1] = np.where(qx[:, :-1] < 0, qx[:, :-1] * sc, qx[:, :-1])
        qy[1:, :] = np.where(qy[1:, :] > 0, qy[1:, :] * sc, qy[1:, :])
        qy[:-1, :] = np.where(qy[:-1, :] < 0, qy[:-1, :] * sc, qy[:-1, :])
        # continuity
        D += dt / dx * (qx[:, :-1] - qx[:, 1:] + qy[:-1, :] - qy[1:, :])
        # rain, infiltration, drainage (mm/h -> m)
        rate = 0.0 if self.spinning else self.rain.field(self.t)
        f0, fc, tau = SOILS[P.soil]
        infil = fc + (f0 - fc) * math.exp(-(self.t / 3600.0) / tau)
        toM = dt / 3.6e6
        D += rate * self.orofac * toM
        loss = np.where(self.urban, P.drain + 0.15 * infil, infil)
        D -= loss * toM
        np.maximum(D, 0, out=D)
        D[self.wall] = 0
        # river inflow
        Q = self.inflow.at(0.0 if self.spinning else self.t, self.rain, P.oro)
        if Q > 0 and self.sources:
            add = Q * dt / (dx * dx) / len(self.sources)
            for r, c in self.sources:
                D[r, c] += add
        # sea level
        D[self.sea] = np.maximum(0, P.sea_level - T[self.sea])
        if not self.spinning:
            self.cum_rain += float(np.mean(rate)) * dt / 3600.0
            self.t += dt
        land = ~self.sea
        self.dmax = float(D[land].max()) if land.any() else 0.0

    def stable_dt(self):
        return min(self.P.max_dt, self.P.cfl * self.P.dx / math.sqrt(G * max(self.dmax, 0.3)))

    def speed(self):
        d = np.maximum(self.D, 0.1)
        ux = 0.5 * (self.qx[:, :-1] + self.qx[:, 1:]) / d
        uy = 0.5 * (self.qy[:-1, :] + self.qy[1:, :]) / d
        v = np.minimum(np.hypot(ux, uy), 8.0)
        v[self.D < 0.05] = 0
        return v

    def hazard(self, v=None):
        v = self.speed() if v is None else v
        h = self.D * (v + 0.5)
        h[self.D < 0.05] = 0
        return h

    def update_stats(self):
        land = ~self.sea
        v = self.speed()
        h = self.hazard(v)
        np.maximum(self.max_depth, np.where(land, self.D, 0), out=self.max_depth)
        np.maximum(self.max_speed, np.where(land, v, 0), out=self.max_speed)
        np.maximum(self.max_hazard, np.where(land, h, 0), out=self.max_hazard)
        newly = land & np.isnan(self.first_evac) & (self.D >= self.P.evac_depth)
        if self.river is not None:
            newly &= ~self.river
        self.first_evac[newly] = self.t
        return h

    def town_status(self, town: Town, h):
        P = self.P
        if town.cells[0].size == 0:
            return "safe", 0.0, 0.0, 0.0
        d = self.D[town.cells]
        hz = h[town.cells]
        m = d.size
        fe = float((d >= P.evac_depth).sum()) / m
        fw = float((d >= P.warn_depth).sum()) / m
        th = P.area_pct / 100.0
        if fe >= th or hz.max() >= P.hazard_limit:
            s = "EVACUATE"
        elif fw >= th or fe > 0:
            s = "WARNING"
        else:
            s = "safe"
        return s, float(d.max()), float(np.median(d)), float(hz.max())


# --------------------------------------------------------------------------
# Demo terrains (same as the web version)
# --------------------------------------------------------------------------
def value_noise(seed, n):
    rng = np.random.default_rng(seed)
    S = 64
    g = rng.uniform(-1, 1, (S, S))

    def vn(x, y):
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - xi, y - yi
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        a = g[yi % S, xi % S]; b = g[yi % S, (xi + 1) % S]
        c = g[(yi + 1) % S, xi % S]; d = g[(yi + 1) % S, (xi + 1) % S]
        return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy

    def fbm(x, y, oct=4):
        v, amp, f, tot = 0.0, 1.0, 1.0, 0.0
        for _ in range(oct):
            v = v + amp * vn(x * f, y * f); tot += amp; amp *= 0.5; f *= 2
        return v / tot
    return fbm


def carve(T, pts, z0, z1, depth, halfw, mask):
    n = T.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    pts = np.array(pts, float)
    seg = np.hypot(*(pts[1:] - pts[:-1]).T)
    L = seg.sum()
    best = np.full(T.shape, 1e9); bs = np.zeros(T.shape); acc = 0.0
    for k in range(len(seg)):
        (ax, ay), (bx, by) = pts[k], pts[k + 1]
        vx, vy = bx - ax, by - ay
        t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        d = np.hypot(xx - ax - t * vx, yy - ay - t * vy)
        upd = d < best
        best = np.where(upd, d, best); bs = np.where(upd, (acc + t * seg[k]) / L, bs)
        acc += seg[k]
    m = best <= halfw + 0.01
    bed = z0 + (z1 - z0) * bs - depth
    T[m] = np.minimum(T[m], bed[m])
    mask |= m


def demo_terrain(name, n=100):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    u, v = xx / n, yy / n
    R = np.zeros((n, n), bool)
    if name == "valley":
        nz = value_noise(7, n)
        center = lambda v: 0.5 + 0.07 * np.sin(v * np.pi * 2 * 1.2 + 0.6) + 0.03 * np.sin(v * np.pi * 5)
        c = center(v); dist = np.abs(u - c); w = 0.2
        T = 22 - 12 * v + 3.5 * np.minimum(1, dist / w) + 0.8 * nz(u * 12, v * 12, 3)
        mt = 650 * np.power(np.clip((dist - w) / 0.32, 0, None), 1.4) * (0.75 + 0.5 * (nz(u * 4 + 9, v * 4, 4) + 1) / 2)
        T = T + np.where(dist > w, mt, 0)
        tx, ty = (u - (c + 0.12)) / 0.06, (v - 0.62) / 0.09
        q = tx * tx + ty * ty
        T = T + np.where(q < 1, 9 * (1 - q), 0)
        pts = [[center(k / 20) * n, k / 20 * (n - 1)] for k in range(21)]
        carve(T, pts, 22, 10.2, 1.8, 0.75, R)
        cx = lambda vv: int(round(float(center(vv)) * n))
        src = [(0, x) for x in range(n) if abs(x - float(center(0)) * n) <= 1]
        towns = [("河濱里", 3200, 35, cx(0.35) - 6, 5), ("中正里", 5400, 55, cx(0.55) - 9, 7),
                 ("台地社區", 2100, 62, cx(0.62) + 12, 5), ("下游新村", 1800, 85, cx(0.85) + 8, 5)]
        return T, R, "S", src, 150.0, towns
    if name == "basin":
        nz = value_noise(21, n)
        uu, vv = u - 0.5, v - 0.5
        r = np.hypot(uu, vv * 1.1)
        T = 6 + 8 * r + 1.0 * nz(xx / 8, yy / 8, 3)
        T = T + np.where(r > 0.28, 520 * np.power(np.clip((r - 0.28) / 0.25, 0, None), 1.5) *
                         (0.7 + 0.6 * (nz(xx / 20 + 5, yy / 20, 4) + 1) / 2), 0)
        carve(T, [[8, 8], [30, 30]], 70, 9, 2, 1.0, R)
        carve(T, [[30, 30], [48, 50], [66, 52]], 8.5, 6, 2, 1.0, R)
        carve(T, [[50, 99], [50, 75], [48, 52]], 50, 7, 2, 0.7, R)
        carve(T, [[64, 52], [80, 53], [99, 54]], 6, 1, 2, 2.2, R)
        a, b = (xx - 36) / 9, (yy - 62) / 7
        q = a * a + b * b
        T = T - np.where(q < 1, 2.5 * (1 - q), 0)
        towns = [("中心區", 8200, 44, 52, 7), ("低窪里", 4300, 62, 36, 6), ("北郊", 2600, 28, 44, 5), ("東門", 3100, 58, 66, 5)]
        return T, R, "E", [(8, 8)], 80.0, towns
    if name == "coast":
        nz = value_noise(33, n)
        shore = 0.82 + 0.04 * np.sin(v * 7)
        mtn = 16 + 420 * np.power(np.clip((0.3 - u) / 0.3, 0, None), 1.6) * (0.7 + 0.6 * (nz(u * 6, v * 6, 4) + 1) / 2)
        plain = 16 * np.power(np.clip((shore - u) / (shore - 0.3), 0, None), 1.3) + 0.6 + 0.5 * nz(u * 15, v * 15, 3)
        sea = -2 - 30 * (u - shore)
        T = np.where(u < 0.3, mtn, np.where(u < shore, plain, sea))
        dd = (u - (shore - 0.025)) / 0.015
        T = T + np.where(u < shore, 2.2 * np.exp(-dd * dd), 0)
        a, b = (u - 0.72) / 0.06, (v - 0.3) / 0.12
        q = a * a + b * b
        T = T - np.where(q < 1, 1.4 * (1 - q), 0)
        carve(T, [[0, 55], [30, 58], [55, 52], [75, 60], [99, 62]], 40, -2, 2.5, 0.9, R)
        towns = [("漁港社區", 2400, 66, 78, 5), ("潟湖里", 1900, 30, 72, 5), ("鎮中心", 6100, 44, 52, 7), ("山腳村", 1200, 30, 33, 4)]
        return T, R, "E", [(55, 0)], 100.0, towns
    raise SystemExit(f"unknown demo '{name}' (valley, basin, coast)")


def write_demo_inputs(outdir, name, dx, rain_total, rain_hours, pattern):
    """Write the demo terrain as ready-made input files so users can see the formats."""
    T, R, edges, src, area, towns = demo_terrain(name)
    d = os.path.join(outdir, "inputs")
    os.makedirs(d, exist_ok=True)
    meta = GridMeta(cellsize=dx, xll=250000.0, yll=2650000.0)
    write_asc(os.path.join(d, "dem.asc"), T, meta, dx)
    write_asc(os.path.join(d, "river_mask.asc"), R.astype(float), meta, dx)
    with open(os.path.join(d, "towns.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "population", "row", "col", "radius_m"])
        for nm, pop, r, c, rad in towns:
            w.writerow([nm, pop, r, c, rad * dx])
    with open(os.path.join(d, "rain.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_h", "rain_mm"])
        for h in range(int(math.ceil(rain_hours))):
            mm = (design_storm(h * 3600 + 1800, rain_total, rain_hours, pattern))
            w.writerow([h, round(mm, 1)])
    # a 3-station example for spatial rainfall
    with open(os.path.join(d, "rain_stations.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_h", "山區站", "平地站", "下游站"])
        for h in range(int(math.ceil(rain_hours))):
            mm = design_storm(h * 3600 + 1800, rain_total, rain_hours, pattern)
            w.writerow([h, round(mm * 1.4, 1), round(mm, 1), round(mm * 0.8, 1)])
    with open(os.path.join(d, "stations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name", "row", "col"])
        w.writerow(["山區站", 10, 15]); w.writerow(["平地站", 50, 50]); w.writerow(["下游站", 90, 70])
    with open(os.path.join(d, "sources.txt"), "w") as f:
        f.write(" ".join(f"--source {r},{c}" for r, c in src) + f" --catchment-area {area} --edges {edges}\n")
    return T, R, edges, src, area, [Town(nm, r, c, rad, pop) for nm, pop, r, c, rad in towns]


# --------------------------------------------------------------------------
# Output: maps, charts, reports
# --------------------------------------------------------------------------
def setup_fonts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    want = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", "Noto Sans TC",
            "Arial Unicode MS", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in want if f in have]
    plt.rcParams["font.sans-serif"] = fam + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def hillshade(T, dx):
    Tn = np.where(np.isfinite(T), T, np.nanmin(T))
    gy, gx = np.gradient(Tn, dx)
    az, alt = math.radians(315), math.radians(45)
    slope = np.arctan(np.hypot(gx, gy) * 2)
    aspect = np.arctan2(-gx, gy)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


T0 = None  # datetime of t = 0 when the rain file has real dates


def fmt_h(s, clock=True):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "-"
    h = int(s // 3600); m = int(s % 3600 // 60)
    txt = f"T+{h:02d}:{m:02d}"
    if clock and T0 is not None:
        import datetime as dtm
        txt += f" ({(T0 + dtm.timedelta(seconds=s)):%m/%d %H:%M})"
    return txt


def draw_maps(plt, model: FloodModel, T, towns, results, outdir, P):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Circle, Patch
    hs = hillshade(T, P.dx)
    sea = model.sea
    ny, nx = T.shape
    ext = (0, nx, ny, 0)
    figs = []

    def base(ax):
        ax.imshow(hs, cmap="gray", vmin=-0.2, vmax=1.2, extent=ext, interpolation="bilinear")
        ax.imshow(np.ma.masked_where(~sea, np.ones_like(T)), cmap=ListedColormap(["#7fa8d1"]), extent=ext, alpha=0.9)
        if model.river is not None:
            ax.imshow(np.ma.masked_where(~model.river, np.ones_like(T)), cmap=ListedColormap(["#4f8cc9"]), extent=ext, alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])

    def towns_overlay(ax):
        for t in towns:
            r = results[t.name]
            col = {"EVACUATE": "#c62828", "WARNING": "#e07b00", "safe": "#2e7d32"}[r["worst"]]
            ax.add_patch(Circle((t.col + 0.5, t.row + 0.5), t.radius, fill=False, ec="white", lw=3.2))
            ax.add_patch(Circle((t.col + 0.5, t.row + 0.5), t.radius, fill=False, ec=col, lw=1.8,
                                ls="-" if r["worst"] != "safe" else "--"))
            lab = t.name + (f"\n撤離 {fmt_h(r['evac_t'])}" if r["evac_t"] is not None else "")
            ax.text(t.col + 0.5, t.row - t.radius - 0.8, lab, ha="center", va="bottom", fontsize=8.5,
                    color=col, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    km = nx * P.dx / 1000

    # 1) evacuation map: when each cell first reaches the evacuation depth
    fig, ax = plt.subplots(figsize=(8, 8 * ny / nx + 0.6), dpi=130)
    base(ax)
    fe = model.first_evac / 3600.0
    bounds = [0, 3, 6, 12, 24, 1e9]
    cm = ListedColormap(["#5c0a17", "#a3162a", "#d63a2a", "#f08a24", "#f2d64b"])
    ax.imshow(np.ma.masked_invalid(np.where(sea, np.nan, fe)), cmap=cm, norm=BoundaryNorm(bounds, cm.N),
              extent=ext, alpha=0.85, interpolation="nearest")
    towns_overlay(ax)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cm.colors, ["0–3 h", "3–6 h", "6–12 h", "12–24 h", "> 24 h"])],
              title=f"水深達 {P.evac_depth} m 的時間", loc="lower left", fontsize=8, title_fontsize=8, framealpha=0.9)
    ax.set_title(f"撤離地圖 Evacuation map  ·  {km:.1f} km × {ny*P.dx/1000:.1f} km", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "evacuation_map.png")); plt.close(fig)
    figs.append("evacuation_map.png")

    # 2) maximum depth
    fig, ax = plt.subplots(figsize=(8, 8 * ny / nx + 0.6), dpi=130)
    base(ax)
    md = np.where(sea | (model.max_depth < 0.03), np.nan, model.max_depth)
    if model.river is not None:
        md = np.where(model.river, np.nan, md)
    im = ax.imshow(np.ma.masked_invalid(md), cmap="Blues", vmin=0, vmax=max(1.5, np.nanpercentile(md, 99) if np.isfinite(md).any() else 1.5),
                   extent=ext, alpha=0.9, interpolation="nearest")
    towns_overlay(ax)
    fig.colorbar(im, ax=ax, shrink=0.75, label="最大水深 max depth (m)")
    ax.set_title("最大淹水深度 Maximum flood depth", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "max_depth.png")); plt.close(fig)
    figs.append("max_depth.png")

    # 3) hazard rating
    fig, ax = plt.subplots(figsize=(8, 8 * ny / nx + 0.6), dpi=130)
    base(ax)
    hz = np.where(sea | (model.max_hazard < 0.05), np.nan, model.max_hazard)
    if model.river is not None:
        hz = np.where(model.river, np.nan, hz)
    hb = [0, 0.75, 1.25, 2.0, 1e9]
    hcm = ListedColormap(["#f2d64b", "#f08a24", "#d63a2a", "#7a1020"])
    ax.imshow(np.ma.masked_invalid(hz), cmap=hcm, norm=BoundaryNorm(hb, hcm.N), extent=ext, alpha=0.85, interpolation="nearest")
    towns_overlay(ax)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(hcm.colors, ["低 low <0.75", "中 0.75–1.25", "高 1.25–2", "極高 >2"])],
              title="危險度 hazard = d×(v+0.5)", loc="lower left", fontsize=8, title_fontsize=8, framealpha=0.9)
    ax.set_title("最大危險度 Maximum flood hazard", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "hazard_map.png")); plt.close(fig)
    figs.append("hazard_map.png")
    return figs


def draw_timeline(plt, series, towns, outdir, P):
    t = np.array(series["t"]) / 3600
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), dpi=130, sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    a1.bar(t, series["rain"], width=(t[1] - t[0]) if len(t) > 1 else 0.1, color="#4a8fd0", alpha=0.8, label="雨量 rain (mm/h)")
    a1.set_ylabel("mm/h"); a1.invert_yaxis()
    if max(series["inflow"]) > 0:
        a1b = a1.twinx()
        a1b.plot(t, series["inflow"], color="#1c3e78", lw=1.6, label="上游流量 inflow")
        a1b.set_ylabel("m³/s")
    a1.set_title("降雨與各社區最大水深 Rainfall and community water depth", fontsize=11)
    for tw in towns:
        a2.plot(t, series["towns"][tw.name], lw=1.8, label=tw.name)
    a2.axhline(P.evac_depth, color="#c62828", ls="--", lw=1); a2.text(t[-1], P.evac_depth, " 撤離", color="#c62828", va="center", fontsize=8)
    a2.axhline(P.warn_depth, color="#e07b00", ls=":", lw=1); a2.text(t[-1], P.warn_depth, " 警戒", color="#e07b00", va="center", fontsize=8)
    a2.set_ylabel("社區內最大水深 (m)"); a2.set_xlabel("時間 hours")
    a2.legend(fontsize=8, ncol=min(4, len(towns)) or 1, loc="upper right")
    a2.grid(alpha=0.3); a1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "timeline.png")); plt.close(fig)
    return "timeline.png"


def make_animation(plt, frames, T, model, towns, series, outdir, P, fps=6, mp4=True):
    """Animated map of water depth over time + rain chart with a moving 'now' line.
    Writes flood_animation.gif (and .mp4 when ffmpeg is installed)."""
    import datetime as dtm
    from matplotlib import animation
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    if len(frames) < 2:
        return []
    ny, nx = T.shape
    k = max(1, int(math.ceil(max(ny, nx) / 700)))          # keep frames a sensible size
    hs = hillshade(T, P.dx)[::k, ::k]
    sea = model.sea[::k, ::k]
    ext = (0, nx, ny, 0)
    tall = ny > 1.3 * nx
    if tall:      # map on the left, charts on the right
        fig = plt.figure(figsize=(11, 8), dpi=100)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1, 0.55], wspace=0.25, hspace=0.35,
                              left=0.03, right=0.92, top=0.94, bottom=0.07)
        ax_map = fig.add_subplot(gs[:, 0]); ax_r = fig.add_subplot(gs[0, 1]); ax_d = fig.add_subplot(gs[1, 1], sharex=ax_r)
        ax_txt = fig.add_subplot(gs[2, 1])
    else:         # map on top, charts below
        mh = 6.4 * ny / nx
        fig = plt.figure(figsize=(8, mh + 2.6), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[mh, 1.7], width_ratios=[1.5, 1], hspace=0.22, wspace=0.35,
                              left=0.08, right=0.97, top=0.95, bottom=0.08)
        ax_map = fig.add_subplot(gs[0, :]); ax_r = fig.add_subplot(gs[1, 0]); ax_d = ax_r.twinx()
        ax_txt = fig.add_subplot(gs[1, 1])

    ax_map.imshow(hs, cmap="gray", vmin=-0.2, vmax=1.2, extent=ext, interpolation="bilinear")
    ax_map.imshow(np.ma.masked_where(~sea, np.ones_like(hs)), cmap=ListedColormap(["#7fa8d1"]), extent=ext, alpha=0.9)
    vmax = max(1.0, float(np.nanpercentile([np.nanmax(f["d"]) if np.isfinite(f["d"]).any() else 0 for f in frames], 95)))
    vmax = min(vmax, 3.0)
    im = ax_map.imshow(np.full_like(hs, np.nan), cmap="Blues", vmin=0, vmax=vmax, extent=ext,
                       alpha=0.92, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax_map, fraction=0.035 if not tall else 0.05, pad=0.02)
    cb.set_label("水深 water depth (m)")
    rings = []
    for t in towns:
        c = Circle((t.col, t.row), t.radius, fill=False, ec="#2e7d32", lw=2)
        ax_map.add_patch(c)
        ax_map.text(t.col, t.row - t.radius - 1, t.name, ha="center", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        rings.append(c)
    ax_map.set_xticks([]); ax_map.set_yticks([])
    title = ax_map.set_title("", fontsize=11)

    t_h = np.array(series["t"]) / 3600
    rain = np.array(series["rain"])
    step = (t_h[1] - t_h[0]) if len(t_h) > 1 else 0.1
    ax_r.bar(t_h, rain, width=step, color="#4a8fd0", alpha=0.85)
    ax_r.set_ylabel("雨量 rain (mm/h)", color="#2a6fb0")
    cum = np.cumsum(rain * step)
    ax_c = ax_r.twinx() if tall else None
    if ax_c is not None:
        ax_c.plot(t_h, cum, color="#1c3e78", lw=1.4); ax_c.set_ylabel("累積 total (mm)", color="#1c3e78")
    dmax_series = np.array([f["dmax"] for f in frames]); ft = np.array([f["t"] for f in frames]) / 3600
    ax_d.plot(ft, dmax_series, color="#c62828", lw=1.6, label="最大水深 (河道外) max depth off-channel")
    if towns and tall:
        for tw in towns:
            ax_d.plot(t_h, series["towns"][tw.name], lw=1.0, alpha=0.8, label=tw.name)
        ax_d.axhline(P.evac_depth, color="#c62828", ls="--", lw=0.8)
    ax_d.set_ylabel("水深 depth (m)", color="#c62828")
    if tall:
        ax_d.set_xlabel("小時 hours" if T0 is None else f"小時 hours from {T0:%Y-%m-%d %H:%M}")
        ax_d.legend(fontsize=7, loc="upper left", ncol=2)
        ax_r.set_title("降雨與水深 Rain and water depth", fontsize=10)
    else:
        ax_r.set_xlabel("小時 hours" if T0 is None else f"小時 hours from {T0:%Y-%m-%d %H:%M}")
    for ax in (ax_r, ax_d):
        ax.grid(alpha=0.25)
    now_lines = [ax_r.axvline(0, color="k", lw=1.2)] + ([ax_d.axvline(0, color="k", lw=1.2)] if tall else [])
    real_now = None
    if T0 is not None:
        real_now = (dtm.datetime.now() - T0).total_seconds() / 3600
        if t_h[0] < real_now < t_h[-1]:
            for ax in ([ax_r, ax_d] if tall else [ax_r]):
                ax.axvspan(real_now, t_h[-1] + step, color="#ffcc80", alpha=0.18, lw=0)
            ax_r.text(real_now, ax_r.get_ylim()[1], " 預報 forecast →", fontsize=8, va="top", color="#a65c00")
    ax_txt.axis("off")
    info = ax_txt.text(0.0, 1.0, "", va="top", ha="left", fontsize=9.5, family=plt.rcParams["font.sans-serif"],
                       transform=ax_txt.transAxes)
    colors = {"EVACUATE": "#c62828", "WARNING": "#e07b00", "safe": "#2e7d32"}

    def draw(i):
        f = frames[i]
        im.set_data(f["d"])
        th = f["t"] / 3600
        for ln in now_lines:
            ln.set_xdata([th, th])
        for c, st in zip(rings, f["status"]):
            c.set_edgecolor(colors[st])
        when = fmt_h(f["t"])
        tag = ""
        if real_now is not None:
            tag = "  [預報 forecast]" if th > real_now else "  [過去 past]"
        title.set_text(f"{when}{tag}")
        ev = [tw.name for tw, st in zip(towns, f["status"]) if st == "EVACUATE"]
        wn = [tw.name for tw, st in zip(towns, f["status"]) if st == "WARNING"]
        lines = [f"雨量 rain now:  {f['rain']:.1f} mm/h",
                 f"累積雨量 total so far:  {f['cum']:.0f} mm",
                 f"最大水深 deepest (off river):  {f['dmax']:.2f} m",
                 f"淹水 ≥0.3 m:  {f['area']:.2f} km²"]
        if towns:
            lines.append(f"撤離 evacuate: {', '.join(ev) if ev else '-'}")
            if wn:
                lines.append(f"警戒 warning: {', '.join(wn)}")
        info.set_text("\n".join(lines))
        return [im, title, info, *now_lines, *rings]

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    out = []
    gif = os.path.join(outdir, "flood_animation.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=fps))
    out.append("flood_animation.gif")
    if mp4 and animation.writers.is_available("ffmpeg"):
        try:
            anim.save(os.path.join(outdir, "flood_animation.mp4"),
                      writer=animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"]))
            out.append("flood_animation.mp4")
        except Exception as e:  # noqa: BLE001
            print(f"  (mp4 skipped: {e})")
    plt.close(fig)
    return out


def snapshot(plt, model, T, towns, outdir, P, hs_cache={}):
    from matplotlib.patches import Circle
    if "hs" not in hs_cache:
        hs_cache["hs"] = hillshade(T, P.dx)
    ny, nx = T.shape
    fig, ax = plt.subplots(figsize=(6, 6 * ny / nx + 0.4), dpi=110)
    ax.imshow(hs_cache["hs"], cmap="gray", vmin=-0.2, vmax=1.2, interpolation="bilinear")
    d = np.where(model.D < 0.03, np.nan, model.D)
    ax.imshow(np.ma.masked_invalid(d), cmap="Blues", vmin=0, vmax=2.5, alpha=0.9, interpolation="nearest")
    h = model.hazard()
    for t in towns:
        s = model.town_status(t, h)[0]
        col = {"EVACUATE": "#c62828", "WARNING": "#e07b00", "safe": "#2e7d32"}[s]
        ax.add_patch(Circle((t.col, t.row), t.radius, fill=False, ec=col, lw=2))
    ax.set_title(f"{fmt_h(model.t)}  雨量 {np.mean(model.rain.field(model.t)):.1f} mm/h", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fn = os.path.join(outdir, "snapshots", f"depth_{int(model.t // 60):05d}min.png")
    fig.savefig(fn); plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="2D flood simulation + evacuation assessment (淹水模擬與撤離判定)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("terrain 地形")
    g.add_argument("--demo", choices=["valley", "basin", "coast"], help="use a built-in demo terrain (also writes sample input files)")
    g.add_argument("--dem", help="elevation: .asc (ESRI ASCII), .csv/.txt matrix, lat/lon .csv, x/y/z .json points "
                                 "(one file, a folder, or a glob like 'data/*.json'), .tif (GeoTIFF), grayscale .png, "
                                 "or 'online' to download it from Open-Meteo for --area")
    g.add_argument("--xy-order", choices=["auto", "latlon", "lonlat"], default="auto",
                   help="in x/y/z point files: latlon = x is latitude, lonlat = x is longitude (auto guesses)")
    g.add_argument("--dx", type=float, help="cell size in metres (read from .asc/.tif if omitted)")
    g.add_argument("--png-min", type=float, default=0.0, help="elevation of black pixels (image DEM)")
    g.add_argument("--png-max", type=float, default=500.0, help="elevation of white pixels (image DEM)")
    g.add_argument("--nodata", type=float, help="extra nodata value in the DEM (treated as wall)")
    g.add_argument("--coarsen", type=int, default=1, help="average N×N cells into one (speeds up big DEMs)")
    g.add_argument("--edges", default=None, help="open (outflow) boundaries, e.g. 'S' or 'NSEW'; default all for files")
    g.add_argument("--river-mask", help="grid (same size as DEM) where 1 = river channel; excluded from community checks")
    g.add_argument("--sea-level", type=float, default=0.0, help="sea / storm-surge water level (m)")
    g.add_argument("--sea-below", type=float, default=0.0, help="cells with elevation below this are sea")

    g = p.add_argument_group("rainfall 降雨")
    g.add_argument("--rain", help="CSV: time_h,rain_mm (one series), time_h,<station...> (with --stations), "
                                  "Open-Meteo lat,lon,time,rain_mm, or 'online' to download it")
    g.add_argument("--rain-values", help="type the rain in by hand: mm in each hour, e.g. '5,20,60,80,40,10'")
    g.add_argument("--rain-storm", help="build a storm: TYPE[,total_mm[,hours[,from]]], TYPE = typhoon, thunderstorm, "
                                        "plumrain, single, uniform; e.g. 'typhoon,800,48,S'")
    g.add_argument("--stations", help="CSV of rain-station locations: name,row,col or name,x,y (IDW interpolation)")
    g.add_argument("--rain-grid", help="grid multiplier on rainfall (same size as DEM), e.g. a radar-based ratio map")
    g.add_argument("--rain-total", type=float, default=350.0, help="synthetic storm: total rainfall (mm)")
    g.add_argument("--rain-hours", type=float, default=24.0, help="synthetic storm: duration (h)")
    g.add_argument("--pattern", choices=["uniform", "center", "front", "back"], default="center", help="synthetic storm shape")
    g.add_argument("--daily-pattern", choices=["uniform", "center", "front", "back"], default="center",
                   help="how to spread DAILY rain totals over 24 h (hourly data is better)")
    g.add_argument("--oro", type=float, default=None, help="%% extra rain per 1000 m elevation (default 40 synthetic, 0 for files)")

    g = p.add_argument_group("online data 線上資料 (Open-Meteo, used with --dem online / --rain online)")
    g.add_argument("--area", help="box 'south,west,north,east' in degrees (for --rain online it defaults to the lat/lon DEM's area)")
    g.add_argument("--cell", type=float, default=100.0, help="cell size (m) for --dem online; Open-Meteo's DEM is ~90 m")
    g.add_argument("--start", help="first day of rain YYYY-MM-DD (past back to 1940, or up to 16 days ahead)")
    g.add_argument("--end", help="last day of rain YYYY-MM-DD (default = --start)")
    g.add_argument("--forecast", type=int, help="rain for the next N days (1–16) instead of --start/--end; default 3")
    g.add_argument("--rain-model", default="best_match",
                   help="Open-Meteo weather model for forecasts, e.g. best_match, jma_seamless, ecmwf_ifs025, gfs_seamless")
    g.add_argument("--rain-step", type=float, default=0.05, help="spacing of rain sample points (degrees)")
    g.add_argument("--cache", default="openmeteo_cache", help="folder for downloads (re-used on the next run)")
    g.add_argument("--refresh", action="store_true", help="download again even if a cached copy exists")
    g.add_argument("--max-calls", type=int, default=3000, help="refuse elevation downloads needing more API calls")

    g = p.add_argument_group("river inflow 上游河川")
    g.add_argument("--source", action="append", default=[], help="inflow cell 'row,col' (repeatable)")
    g.add_argument("--source-latlon", action="append", default=[], help="inflow point 'lat,lon' (lat/lon DEM only)")
    g.add_argument("--inflow", help="CSV hydrograph: time_h,Q_m3s")
    g.add_argument("--catchment-area", type=float, default=0.0, help="upstream catchment km² (rational method Q=C·i·A/3.6)")
    g.add_argument("--runoff-c", type=float, default=0.8, help="runoff coefficient C")
    g.add_argument("--lag", type=float, default=2.0, help="catchment lag time (h)")
    g.add_argument("--spinup", type=float, default=None,
                   help="hours of normal river flow before the storm; normally-wet cells become the river mask "
                        "(default 6 when there is river inflow, else 0)")

    g = p.add_argument_group("soil, drainage, criteria 土壤/排水/判定")
    g.add_argument("--soil", choices=list(SOILS), default="loam")
    g.add_argument("--drain", type=float, default=78.8, help="storm-drain capacity inside communities (mm/h)")
    g.add_argument("--manning", type=float, default=0.035, help="Manning roughness n")
    g.add_argument("--theta", type=float, default=0.7,
                   help="flux smoothing 0.7–1.0 (lower = more stable in deep pools; 1 = no smoothing, the old behaviour)")
    g.add_argument("--towns", help="CSV of communities: name,population,radius_m and row,col or x,y")
    g.add_argument("--evac-depth", type=float, default=0.5)
    g.add_argument("--warn-depth", type=float, default=0.3)
    g.add_argument("--area-pct", type=float, default=20.0, help="%% of a community's area at evac depth to order evacuation")
    g.add_argument("--hazard-limit", type=float, default=1.25, help="depth×(velocity+0.5) that triggers evacuation anywhere")

    g = p.add_argument_group("run 執行")
    g.add_argument("--hours-after", type=float, default=8.0, help="keep simulating this long after rain/inflow ends")
    g.add_argument("--duration", type=float, help="total simulated hours (overrides --hours-after)")
    g.add_argument("--report-every", type=float, default=5.0, help="community check interval (minutes)")
    g.add_argument("--snapshot-every", type=float, default=0.0, help="save a depth map every N hours (0 = off)")
    g.add_argument("--no-animation", action="store_true", help="skip flood_animation.gif / .mp4")
    g.add_argument("--animate-every", type=float, default=0.0,
                   help="minutes between animation frames (0 = automatic, about 150 frames)")
    g.add_argument("--fps", type=float, default=6.0, help="animation frames per second")
    g.add_argument("--out", default="flood_results", help="output folder")
    g.add_argument("--no-plots", action="store_true")
    return p


DEM_NAMES = ["elevation_grid.csv", "elevation.csv", "elev.csv", "dem.asc", "dem.tif", "dem.tiff", "dem.csv"]
RAIN_NAMES = ["rainfall_hourly.csv", "rain_hourly.csv", "rainfall_daily.csv", "rain.csv"]
TOWN_NAMES = ["towns.csv", "my_towns.csv", "communities_input.csv"]


def _find(names):
    here = os.path.dirname(os.path.abspath(__file__))
    for d in dict.fromkeys([os.getcwd(), here]):
        for n in names:
            pth = os.path.join(d, n)
            if os.path.exists(pth):
                return pth
    return None


DEFAULT_AREA = "25.04,121.55,25.10,121.65"   # lower Keelung River, Taipei


def _ask(prompt, default=None, choices=None):
    sfx = f" [{default}]" if default not in (None, "") else ""
    while True:
        v = input(f"{prompt}{sfx}: ").strip().strip('"').strip("'")
        if not v and default is not None:
            v = str(default)
        if choices and v not in choices:
            print(f"  please type one of: {', '.join(choices)}")
            continue
        if v or default is not None:
            return v


def _ask_file(prompt, default=None):
    while True:
        v = _ask(prompt, default)
        if v and os.path.exists(os.path.expanduser(v)):
            return os.path.expanduser(v)
        print(f"  file not found: {v!r} (drag the file into this window, or type its full path)")


def setup_questions():
    """Ask where each input comes from; returns an argv list for main()."""
    print("=" * 64)
    print(" 淹水模擬 Flood simulator — setup (press Enter to accept [defaults])")
    print("=" * 64)
    args = []
    found_dem, found_rain, found_towns = _find(DEM_NAMES), _find(RAIN_NAMES), _find(TOWN_NAMES)

    print("\nElevation 地形:\n  1) my own file\n  2) download from Open-Meteo (pick an area)\n  3) built-in demo")
    src = _ask("choose", "1" if found_dem else "2", ["1", "2", "3"])
    area = None
    if src == "1":
        args += ["--dem", _ask_file("elevation file (.asc .tif .csv .png .json) or a folder of x/y/z point files",
                                    found_dem)]
        if is_point_dem(args[-1]):
            dx = _ask("cell size in metres (Enter = 100, runs much faster than full detail; 0 = full detail)", "100")
            if dx not in ("", "0"):
                args += ["--dx", dx]
        elif os.path.splitext(args[-1])[1].lower() in (".csv", ".txt", ".png"):
            dx = _ask("cell size in metres", "40")
            args += ["--dx", dx]
    elif src == "2":
        area = _ask("area as south,west,north,east (degrees)", DEFAULT_AREA)
        args += ["--dem", "online", "--area", area, "--cell", _ask("cell size in metres (100 or less is best)", "100")]
    else:
        args += ["--demo", _ask("demo terrain: valley / basin / coast", "valley", ["valley", "basin", "coast"])]

    if src != "3":
        print("\nRain 降雨:\n  1) my own file\n  2) download from Open-Meteo (past or forecast)"
              "\n  3) type it in (mm in each hour)\n  4) build a storm (typhoon, thunderstorm, plum rain ...)")
        r = _ask("choose", "1" if found_rain else "2", ["1", "2", "3", "4"])
        if r == "1":
            args += ["--rain", _ask_file("rain file (.csv)", found_rain)]
            if found_rain is None or args[-1] != found_rain:
                st = _ask("rain-station locations file (Enter if the file has one series)", "")
                if st:
                    args += ["--stations", st]
        elif r == "2":
            args += ["--rain", "online"]
            dem_is_geo = src == "2" or (src == "1" and is_point_dem(args[1]))
            if not dem_is_geo:
                area = _ask("area for the rain, south,west,north,east (degrees)", DEFAULT_AREA)
            if area and "--area" not in args:
                args += ["--area", area]
            when = _ask("dates: 'f N' = forecast next N days, or 'YYYY-MM-DD YYYY-MM-DD' for a past event", "f 3")
            parts = when.replace(",", " ").split()
            if parts[0].lower() in ("f", "forecast"):
                args += ["--forecast", parts[1] if len(parts) > 1 else "3"]
            else:
                args += ["--start", parts[0]] + (["--end", parts[1]] if len(parts) > 1 else [])
        elif r == "3":
            args += ["--rain-values", _ask("mm of rain in each hour, comma-separated", "10,30,60,80,50,20,10")]
        else:
            import storm
            box = None
            if src == "2":
                import online as _ol
                box = _ol.parse_box(area)
            elif is_point_dem(args[1]):
                print("  (reading the elevation file to find its area...)")
                _, m = read_grid(args[1])
                box = m.geo["bounds"]
            path, oro = storm.ask_storm(_ask, box)
            args += ["--rain", path, "--oro", f"{oro:g}"]

        t = _ask("\ncommunities file (name,population,lat,lon,radius_m) — Enter for none", found_towns or "")
        if t:
            args += ["--towns", t]
        if src in ("1", "2"):
            geo = src == "2" or is_point_dem(args[1])
            sl = _ask(f"river entering the area from upstream? give {'lat,lon' if geo else 'row,col'} "
                      f"where it enters (Enter = none)", "")
            if sl:
                args += ["--source-latlon" if geo else "--source", sl,
                         "--catchment-area", _ask("  upstream catchment area (km²)", "100")]
    args += ["--out", _ask("\noutput folder", "flood_results")]
    print("\nSame run next time without questions:\n  python flood_sim.py " +
          " ".join(f'"{x}"' if " " in x else x for x in args) + "\n")
    return args


def resolve_online_dem(a):
    """--dem online -> download and point a.dem at the CSV."""
    if a.dem and a.dem.lower() == "online":
        import online
        if not a.area:
            raise SystemExit("--dem online needs --area south,west,north,east")
        a.dem = online.fetch_elevation(online.parse_box(a.area), a.cell, a.cache, a.max_calls, a.refresh)


def resolve_online_rain(a, meta):
    """--rain online / --rain-values -> a rain CSV path; returns the datetime of t=0 if known."""
    if getattr(a, "rain_storm", None):
        import storm
        parts = [p.strip() for p in a.rain_storm.split(",")]
        kind = parts[0].lower()
        if kind not in storm.TYPES:
            raise SystemExit(f"--rain-storm type must be one of: {', '.join(storm.TYPES)}")
        _, tot0, hrs0, _, oro0 = storm.TYPES[kind]
        total = float(parts[1]) if len(parts) > 1 and parts[1] else tot0
        hours = float(parts[2]) if len(parts) > 2 and parts[2] else hrs0
        frm = parts[3] if len(parts) > 3 else "S"
        os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
        path = os.path.join(a.out, "inputs", f"rain_{kind}_{total:.0f}mm.csv")
        box = meta.geo["bounds"] if (meta is not None and meta.geo) else None
        if box is None and a.area:
            box = None  # a DEM without coordinates can only use one series
        start = datetime.strptime(a.start, "%Y-%m-%d") if a.start else None
        peak, _ = storm.write_storm(path, kind, total, hours, start, box, frm)
        print(f"Storm: {kind}, {total:g} mm over {hours:g} h, peak {peak:.0f} mm/h -> {path}")
        if a.oro is None:
            a.oro = oro0
        a.rain = path
        return None
    if a.rain_values:
        import online
        os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
        path = os.path.join(a.out, "inputs", "rain_typed.csv")
        v = online.write_typed_rain(a.rain_values, path)
        print(f"Rain typed in: {len(v)} h, total {sum(v):g} mm, peak {max(v):g} mm/h -> {path}")
        a.rain = path
        return None
    if not (a.rain and a.rain.lower() == "online"):
        return None
    import online
    if a.area:
        box = online.parse_box(a.area)
    elif meta is not None and meta.geo and meta.geo.get("bounds"):
        box = meta.geo["bounds"]
        print(f"Rain area taken from the elevation file: {', '.join(f'{v:.4f}' for v in box)}")
    else:
        raise SystemExit("--rain online needs --area south,west,north,east "
                         "(your elevation file has no coordinates)")
    start, end = online.parse_dates(a.start, a.end, a.forecast)
    if not (a.start or a.forecast):
        print("No --start/--forecast given: using the forecast for the next 3 days")
    path = online.fetch_rain(box, start, end, a.rain_step, a.rain_model, a.cache, a.refresh)
    if meta is not None and meta.geo:
        a.rain = path
        return None
    # DEM without coordinates: use the area-average rain as one series
    os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
    series = os.path.join(a.out, "inputs", "rain_online_series.csv")
    t0 = online.rain_to_series(path, series)
    print(f"  elevation file has no coordinates, so the area-average rain is used everywhere ({series})")
    a.rain = series
    return t0


def main(argv=None):
    if argv is None and len(sys.argv) == 1:
        try:
            argv = setup_questions()
        except (EOFError, KeyboardInterrupt):
            print("\n(no answers; looking for files next to the script instead)")
            argv = []
    a = build_parser().parse_args(argv)
    if sum(bool(v) for v in (a.rain, a.rain_values, a.rain_storm)) > 1:
        raise SystemExit("give only one of --rain, --rain-values or --rain-storm")
    global XY_ORDER
    XY_ORDER = a.xy_order
    resolve_online_dem(a)
    if not a.dem and not a.demo:
        # No terrain given (e.g. pressed Run in an IDE): use files found next to the script, else a demo.
        a.dem = _find(DEM_NAMES)
        if a.dem:
            print(f"No --dem given: using {a.dem}")
            if not a.rain:
                a.rain = _find(RAIN_NAMES)
                print(f"  rain: {a.rain}" if a.rain else
                      f"  no rain file found -> design storm {a.rain_total:g} mm / {a.rain_hours:g} h")
            if not a.towns:
                a.towns = _find(TOWN_NAMES)
                if a.towns:
                    print(f"  communities: {a.towns}")
        else:
            a.demo = "valley"
            print("No --dem given and no elevation file found next to the script -> running the 'valley' demo.\n"
                  "  To use your own area: run elevation_box.py and rainfall.py --hourly first, "
                  "or pass --dem <file> --rain <file>.")
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    # ---------------- terrain ----------------
    river = None
    if a.demo:
        dx = a.dx or 40.0
        T, R, edges, src, area, towns = write_demo_inputs(a.out, a.demo, dx, a.rain_total, a.rain_hours, a.pattern)
        meta = GridMeta(cellsize=dx)
        river = R
        sources = src
        if a.edges is None:
            a.edges = edges
        if a.catchment_area == 0 and not a.inflow:
            a.catchment_area = area
        if a.demo == "coast" and a.sea_level == 0.0:
            a.sea_level = 0.8
        print(f"Demo '{a.demo}': sample input files written to {os.path.join(a.out, 'inputs')}")
    elif a.dem:
        T, meta = read_grid(a.dem, a.png_min, a.png_max, a.nodata, a.dx)
        dx = meta.cellsize if meta.geo else (a.dx or meta.cellsize)
        if not dx:
            raise SystemExit("cell size unknown: pass --dx <metres>")
        sources, towns = [], []
        if a.edges is None:
            a.edges = "NSEW"
    else:
        raise SystemExit("give --dem <file> or --demo valley|basin|coast")

    if a.coarsen > 1:
        T = coarsen(T, a.coarsen)
        dx *= a.coarsen
        if river is not None:
            river = coarsen(river.astype(float), a.coarsen) > 0.3
    ny, nx = T.shape
    grid_rc_raw = make_grid_rc(meta, ny * a.coarsen, dx / a.coarsen)
    grid_rc = (lambda d: tuple(v / a.coarsen for v in grid_rc_raw(d))) if a.coarsen > 1 else grid_rc_raw

    if a.river_mask:
        rm, _ = read_grid(a.river_mask)
        rm = coarsen(np.nan_to_num(rm), a.coarsen) if a.coarsen > 1 else np.nan_to_num(rm)
        if rm.shape != T.shape:
            raise SystemExit(f"river mask shape {rm.shape} != DEM {T.shape}")
        river = rm > 0.3
    for s in a.source:
        r, c = (float(v) for v in s.split(","))
        sources.append((int(r / a.coarsen), int(c / a.coarsen)))
    for s in a.source_latlon:
        if meta.geo is None:
            raise SystemExit("--source-latlon needs a lat/lon DEM")
        r, c = meta.latlon_to_rc(*(float(v) for v in s.split(",")))
        sources.append((int(round(r / a.coarsen)), int(round(c / a.coarsen))))
    if a.demo and a.coarsen > 1:
        sources = [(int(r / a.coarsen), int(c / a.coarsen)) for r, c in sources]
    if a.towns:
        towns = read_towns(a.towns, grid_rc, dx)
    elif a.demo and a.coarsen > 1:
        for t in towns:
            t.row //= a.coarsen; t.col //= a.coarsen; t.radius /= a.coarsen
    assign_town_cells(towns, T, river)

    # ---------------- rainfall ----------------
    rain_t0 = resolve_online_rain(a, None if a.demo else meta)
    if a.rain:
        geo_meta = meta
        if a.coarsen > 1 and meta.geo:
            geo_meta = GridMeta(cellsize=dx, geo={**meta.geo, "dx": dx})
        rain = Rainfall.from_csv(T.shape, a.rain, a.stations, grid_rc, geo_meta, a.daily_pattern)
        global T0
        T0 = rain.t0 = rain.t0 or rain_t0
        oro = 0.0 if a.oro is None else a.oro
    else:
        rain = Rainfall.synthetic(T.shape, a.rain_total, a.rain_hours, a.pattern)
        oro = 40.0 if a.oro is None else a.oro
    if a.rain_grid:
        mg, _ = read_grid(a.rain_grid)
        mg = coarsen(mg, a.coarsen) if a.coarsen > 1 else mg
        if mg.shape != T.shape:
            raise SystemExit(f"rain grid shape {mg.shape} != DEM {T.shape}")
        rain.multiplier = np.nan_to_num(mg, nan=1.0)

    inflow = Inflow.from_csv(a.inflow) if a.inflow else Inflow()
    inflow.area, inflow.c, inflow.lag_h = a.catchment_area, a.runoff_c, a.lag
    if (inflow.area > 0 or a.inflow) and not sources:
        print("  warning: inflow given but no --source cell; river inflow ignored")

    P = Params(dx=dx, manning=a.manning, soil=a.soil, drain=a.drain, oro=oro, sea_level=a.sea_level,
               sea_below=a.sea_below, edges=a.edges or "", evac_depth=a.evac_depth, warn_depth=a.warn_depth,
               area_pct=a.area_pct, hazard_limit=a.hazard_limit)
    FloodModel.THETA = min(max(a.theta, 0.5), 1.0)
    model = FloodModel(T, P, rain, inflow, sources, towns, river)
    spin = a.spinup if a.spinup is not None else (6.0 if sources and (inflow.area > 0 or a.inflow) else 0.0)
    if spin > 0:
        wet = model.spin_up(spin)
        if a.river_mask is None:
            river = wet if river is None else (river | wet)
            assign_town_cells(towns, T, river)
            model.set_towns(towns, river)
        print(f"Warm-up: river run at normal flow for {spin:g} h; {int(wet.sum())} normally-wet cells "
              f"{'excluded from community checks' if a.river_mask is None else '(river mask given)'}")

    end_s = max(rain.end_s, inflow.end_s() if a.inflow else rain.end_s + a.lag * 3600 * (inflow.area > 0)) + a.hours_after * 3600
    if a.duration:
        end_s = a.duration * 3600
    print(f"Grid {ny}×{nx}, cell {dx:g} m ({nx*dx/1000:.1f}×{ny*dx/1000:.1f} km) · "
          f"rain {rain.kind} {rain.total:.0f} mm over {rain.hours:g} h · simulate {end_s/3600:.1f} h · "
          f"{len(towns)} communities")

    # ---------------- run ----------------
    plt = None if a.no_plots else setup_fonts()
    if a.snapshot_every > 0 and plt:
        os.makedirs(os.path.join(a.out, "snapshots"), exist_ok=True)
    rep = a.report_every * 60
    next_rep, next_snap, next_print = 0.0, 0.0, 3600.0
    frames, next_anim = [], 0.0
    if plt and not a.no_animation:
        anim_dt = a.animate_every * 60 if a.animate_every > 0 else max(rep, round(end_s / 150 / rep) * rep)
        kk = max(1, int(math.ceil(max(ny, nx) / 700)))
        land_anim = ~model.sea
    else:
        anim_dt = None
    results = {t.name: {"evac_t": None, "warn_t": None, "peak": 0.0, "peak_t": None, "worst": "safe",
                        "evac_end": None} for t in towns}
    series = {"t": [], "rain": [], "inflow": [], "towns": {t.name: [] for t in towns}}
    timeline_rows = []
    rank = {"safe": 0, "WARNING": 1, "EVACUATE": 2}
    while model.t < end_s - 1e-6:
        target = min(next_rep, end_s)
        while model.t < target - 1e-6:
            model.step(min(model.stable_dt(), target - model.t))
        if model.t >= next_rep - 1e-6:
            h = model.update_stats()
            series["t"].append(model.t)
            series["rain"].append(rain.scalar(model.t) * (rain.multiplier.mean() if rain.multiplier is not None else 1))
            series["inflow"].append(inflow.at(model.t, rain, oro) if sources else 0.0)
            for tw in towns:
                s, dmx, dmed, hz = model.town_status(tw, h)
                r = results[tw.name]
                series["towns"][tw.name].append(dmx)
                if s == "EVACUATE" and r["evac_t"] is None:
                    r["evac_t"] = model.t
                if s != "safe" and r["warn_t"] is None:
                    r["warn_t"] = model.t
                if r["evac_t"] is not None and s != "EVACUATE" and r["evac_end"] is None:
                    r["evac_end"] = model.t
                if s == "EVACUATE":
                    r["evac_end"] = None
                if dmx > r["peak"]:
                    r["peak"], r["peak_t"] = dmx, model.t
                if rank[s] > rank[r["worst"]]:
                    r["worst"] = s
                timeline_rows.append([round(model.t / 3600, 3), tw.name, s, round(dmx, 3), round(dmed, 3), round(hz, 3)])
            next_rep += rep
        if anim_dt and model.t >= next_anim - 1e-6:
            dd = model.D[::kk, ::kk]
            land_now = land_anim & ~model.river if model.river is not None else land_anim
            wet = (model.D >= 0.3) & land_now
            frames.append({"t": model.t, "d": np.where((dd < 0.03) | model.sea[::kk, ::kk], np.nan, dd).astype(np.float32),
                           "dmax": float(model.D[land_now].max()) if land_now.any() else 0.0,
                           "rain": float(np.mean(rain.field(model.t))), "cum": model.cum_rain,
                           "area": float(wet.sum()) * dx * dx / 1e6,
                           "status": [model.town_status(tw, model.hazard())[0] for tw in towns]})
            next_anim += anim_dt
        if plt and a.snapshot_every > 0 and model.t >= next_snap - 1e-6:
            snapshot(plt, model, T, towns, a.out, P)
            next_snap += a.snapshot_every * 3600
        if model.t >= next_print:
            el = time.time() - t_start
            eta = el / model.t * (end_s - model.t)
            ev = [tw.name for tw in towns if model.town_status(tw, model.hazard())[0] == "EVACUATE"]
            print(f"  {fmt_h(model.t)}  max depth {model.dmax:5.2f} m  rain total {model.cum_rain:6.1f} mm"
                  f"  evacuate: {', '.join(ev) if ev else '-'}   (ETA {eta:,.0f} s)", flush=True)
            next_print += 3600

    # ---------------- outputs ----------------
    land = ~model.sea
    if river is not None:
        land &= ~river
    flooded_km2 = float(((model.max_depth >= 0.3) & land).sum()) * dx * dx / 1e6
    write_asc(os.path.join(a.out, "max_depth.asc"), np.where(model.wall, np.nan, model.max_depth), meta, dx)
    write_asc(os.path.join(a.out, "evac_time_h.asc"), np.where(np.isnan(model.first_evac), -1, model.first_evac / 3600), meta, dx)
    write_asc(os.path.join(a.out, "max_hazard.asc"), model.max_hazard, meta, dx)
    write_asc(os.path.join(a.out, "max_speed.asc"), model.max_speed, meta, dx)
    if a.coarsen == 1 and write_geotiff(os.path.join(a.out, "max_depth.tif"), model.max_depth, meta):
        write_geotiff(os.path.join(a.out, "evac_time_h.tif"), np.where(np.isnan(model.first_evac), -1, model.first_evac / 3600), meta)

    if meta.geo:
        gm = GridMeta(cellsize=dx, geo={**meta.geo, "dx": dx})
        rr, cc = np.mgrid[0:ny, 0:nx]
        la, lo = gm.rc_to_latlon(rr, cc)
        fe = np.where(np.isnan(model.first_evac), -1, model.first_evac / 3600)
        with open(os.path.join(a.out, "results_latlon.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lat", "lon", "elevation_m", "max_depth_m", "max_speed_ms", "max_hazard", "evac_time_h", "is_sea"])
            for i in range(ny):
                for j in range(nx):
                    w.writerow([f"{la[i,j]:.6f}", f"{lo[i,j]:.6f}", f"{T[i,j]:.2f}", f"{model.max_depth[i,j]:.3f}",
                                f"{model.max_speed[i,j]:.2f}", f"{model.max_hazard[i,j]:.3f}", f"{fe[i,j]:.2f}", int(model.sea[i,j])])
    with open(os.path.join(a.out, "communities.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["name", "population", "worst_status", "warning_from_h", "evacuate_from_h", "safe_again_h",
                    "peak_depth_m", "peak_time_h", "evacuate_at"])
        for tw in towns:
            r = results[tw.name]
            h_ = lambda s: "" if s is None else round(s / 3600, 2)
            w.writerow([tw.name, tw.population, r["worst"], h_(r["warn_t"]), h_(r["evac_t"]),
                        h_(r["evac_end"]), round(r["peak"], 2), h_(r["peak_t"]),
                        fmt_h(r["evac_t"]) if r["evac_t"] is not None else ""])
    with open(os.path.join(a.out, "timeline.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "community", "status", "max_depth_m", "median_depth_m", "max_hazard"])
        w.writerows(timeline_rows)

    figs = []
    if plt:
        figs = draw_maps(plt, model, T, towns, results, a.out, P)
        if series["t"]:
            figs.append(draw_timeline(plt, series, towns, a.out, P))
        if frames:
            print(f"Making animation ({len(frames)} frames)...", flush=True)
            figs += make_animation(plt, frames, T, model, towns, series, a.out, P, a.fps)

    # report
    evac = sorted([tw for tw in towns if results[tw.name]["evac_t"] is not None], key=lambda t: results[t.name]["evac_t"])
    lines = ["# 淹水模擬報告 Flood simulation report", "",
             f"- 範圍 Area: {nx*dx/1000:.1f} × {ny*dx/1000:.1f} km, cell {dx:g} m ({ny}×{nx})",
             f"- 降雨 Rain: {rain.kind}, {rain.total:.0f} mm over {rain.hours:g} h; soil {P.soil}; drainage {P.drain} mm/h",
             f"- 模擬時間 Simulated: {end_s/3600:.1f} h · 運算 run time {time.time()-t_start:.0f} s",
             f"- 淹水 ≥ 0.3 m 面積 Area flooded ≥ 0.3 m: {flooded_km2:.2f} km²",
             f"- 撤離標準 Criteria: ≥ {P.area_pct:g}% of area at ≥ {P.evac_depth} m, or hazard ≥ {P.hazard_limit}",
             "", "## 需撤離社區 Communities to evacuate (in order)", ""]
    if evac:
        lines.append("| 社區 | 人口 | 警戒 | 撤離 | 解除 | 最大水深 |")
        lines.append("|---|---:|---|---|---|---:|")
        for tw in evac:
            r = results[tw.name]
            lines.append(f"| {tw.name} | {tw.population:,} | {fmt_h(r['warn_t'])} | **{fmt_h(r['evac_t'])}** | "
                         f"{fmt_h(r['evac_end'])} | {r['peak']:.2f} m |")
        lines.append("")
        lines.append(f"總撤離人口 Total to evacuate: **{sum(t.population for t in evac):,}**")
    else:
        lines.append("無 None — no community reaches the evacuation criteria.")
    others = [tw for tw in towns if tw not in evac]
    if others:
        lines += ["", "## 其他社區 Other communities", ""]
        for tw in others:
            r = results[tw.name]
            lines.append(f"- {tw.name}: {r['worst']}, peak {r['peak']:.2f} m" + (f", warning from {fmt_h(r['warn_t'])}" if r["warn_t"] else ""))
    lines += ["", "> 簡化模型，僅供教學、情境比較與初步規劃；實際撤離請依中央氣象署、水利署與地方政府指示。",
              "> Screening model only. Follow official warnings and evacuation orders."]
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n=== 撤離判定 Evacuation ===")
    if evac:
        for tw in evac:
            r = results[tw.name]
            print(f"  {tw.name:<8} pop {tw.population:>6,}  evacuate at {fmt_h(r['evac_t'])}"
                  f"  (warning {fmt_h(r['warn_t'])}, peak {r['peak']:.2f} m)")
        print(f"  total population to evacuate: {sum(t.population for t in evac):,}")
    else:
        print("  no community reaches the evacuation criteria")
    for tw in others:
        r = results[tw.name]
        print(f"  {tw.name:<8} {r['worst']:<8} peak {r['peak']:.2f} m")
    print(f"\nFlooded ≥0.3 m: {flooded_km2:.2f} km² · outputs in {a.out}/ "
          f"({', '.join(figs + ['report.md', 'communities.csv', 'timeline.csv', 'max_depth.asc', 'evac_time_h.asc'])})"
          f" · {time.time()-t_start:.0f} s")
    return results


if __name__ == "__main__":
    main()

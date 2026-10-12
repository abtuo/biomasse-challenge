from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.geometry import box

REQUIRED_COLUMNS = {"longitude", "latitude", "agbd", "agbd_se"}

def load_gedi_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing GEDI columns: {sorted(missing)}")
    return df

def quality_summary(df: pd.DataFrame) -> dict:
    out = {
        "n_rows": len(df),
        "agbd_mean": float(df["agbd"].mean()),
        "agbd_median": float(df["agbd"].median()),
        "agbd_p95": float(df["agbd"].quantile(0.95)),
        "agbd_p99": float(df["agbd"].quantile(0.99)),
        "agbd_max": float(df["agbd"].max()),
    }
    if "relative_se" in df:
        out["relative_se_median"] = float(df["relative_se"].median())
        out["relative_se_p95"] = float(df["relative_se"].quantile(0.95))
    return out

def coverage_by_resolution(
    df: pd.DataFrame,
    roi_bounds: tuple[float, float, float, float],
    crs_utm: str,
    scales_m: list[int],
) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    ).to_crs(crs_utm)

    roi = gpd.GeoSeries([box(*roi_bounds)], crs="EPSG:4326").to_crs(crs_utm).iloc[0]

    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    minx, miny, maxx, maxy = roi.bounds

    rows = []
    for s in scales_m:
        ox = np.floor(minx / s) * s
        oy = np.floor(miny / s) * s

        ix = np.floor((x - ox) / s).astype(np.int64)
        iy = np.floor((y - oy) / s).astype(np.int64)
        keys = ix * 10_000_000 + iy
        uniq, counts = np.unique(keys, return_counts=True)

        nx = int(np.ceil((maxx - ox) / s))
        ny = int(np.ceil((maxy - oy) / s))

        xs = ox + (np.arange(nx) + 0.5) * s
        ys = oy + (np.arange(ny) + 0.5) * s
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        total_cells = int(contains_xy(roi, X.ravel(), Y.ravel()).sum())

        uix = uniq // 10_000_000
        uiy = uniq % 10_000_000
        ucx = ox + (uix + 0.5) * s
        ucy = oy + (uiy + 0.5) * s
        inside = contains_xy(roi, ucx, ucy)
        counts_inside = counts[inside]

        covered = int(inside.sum())
        rows.append(
            {
                "resolution_m": s,
                "total_cells": total_cells,
                "covered_cells": covered,
                "coverage_percent": 100.0 * covered / total_cells,
                "mean_shots_per_covered_cell": float(counts_inside.mean()),
                "median_shots_per_covered_cell": float(np.median(counts_inside)),
                "p95_shots_per_covered_cell": float(np.percentile(counts_inside, 95)),
            }
        )
    return pd.DataFrame(rows)

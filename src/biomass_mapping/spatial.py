from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def add_utm_coordinates(
    df: pd.DataFrame,
    crs: str = "EPSG:32629",
    lon: str = "longitude",
    lat: str = "latitude",
) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon], df[lat]),
        crs="EPSG:4326",
    ).to_crs(crs)
    out = df.copy()
    out["x_utm"] = gdf.geometry.x.to_numpy()
    out["y_utm"] = gdf.geometry.y.to_numpy()
    return out

def add_spatial_blocks(
    df: pd.DataFrame,
    block_size_m: float,
    crs: str = "EPSG:32629",
) -> pd.DataFrame:
    out = add_utm_coordinates(df, crs=crs)
    bx = np.floor(out["x_utm"] / block_size_m).astype("int64")
    by = np.floor(out["y_utm"] / block_size_m).astype("int64")
    out["spatial_block"] = bx.astype(str) + "_" + by.astype(str)
    return out

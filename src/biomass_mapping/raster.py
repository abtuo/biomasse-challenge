from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window


def raster_feature_names(src) -> list[str]:
    names = list(src.descriptions)
    if not any(names):
        return [f"band_{i + 1}" for i in range(src.count)]
    return [n if n not in (None, "") else f"band_{i + 1}" for i, n in enumerate(names)]

def sample_raster_blockwise(
    df: pd.DataFrame,
    raster_path: str | Path,
    expected_features: list[str] | None = None,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Sample points by internal raster block to avoid random single-pixel I/O."""
    with rasterio.open(raster_path) as src:
        names = raster_feature_names(src)
        if expected_features is not None and names != expected_features:
            raise ValueError(f"Unexpected raster bands in {raster_path}: {names}; expected {expected_features}")
        xs, ys = transform("EPSG:4326", src.crs, df[lon_col].tolist(), df[lat_col].tolist())
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        values = np.full((len(df), src.count), np.nan, dtype="float64")
        valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        block_h, block_w = src.block_shapes[0]
        block_ids = (rows[valid] // block_h) * int(np.ceil(src.width / block_w)) + cols[valid] // block_w
        valid_idx = np.flatnonzero(valid)
        order = np.argsort(block_ids, kind="stable")
        sorted_ids, sorted_idx = block_ids[order], valid_idx[order]
        _, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
        for start, count in zip(starts, counts):
            point_idx = sorted_idx[start:start + count]
            r0 = int((rows[point_idx][0] // block_h) * block_h)
            c0 = int((cols[point_idx][0] // block_w) * block_w)
            window = Window(c0, r0, min(block_w, src.width - c0), min(block_h, src.height - r0))
            arr = src.read(window=window, masked=True)
            rr, cc = rows[point_idx] - r0, cols[point_idx] - c0
            sampled = arr[:, rr, cc].T
            values[point_idx] = np.ma.filled(sampled, np.nan)
        values[~np.isfinite(values)] = np.nan
    return pd.DataFrame(values, columns=names, index=df.index)

def sample_neighborhood_means(
    df: pd.DataFrame,
    raster_path: str | Path,
    diameter_m: float,
    expected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Mean pixels whose centres lie inside a footprint-sized circle around each point."""
    with rasterio.open(raster_path) as src:
        names = raster_feature_names(src)
        if expected_features is not None and names != expected_features:
            raise ValueError(f"Unexpected raster bands: {names}")
        xs, ys = transform("EPSG:4326", src.crs, df.longitude.tolist(), df.latitude.tolist())
        radius = diameter_m / 2.0
        pad = max(1, int(np.ceil(radius / min(abs(src.res[0]), abs(src.res[1])))))
        out = np.full((len(df), src.count), np.nan, dtype="float64")
        for i, (x, y) in enumerate(zip(xs, ys)):
            row, col = src.index(x, y)
            win = Window(col - pad, row - pad, 2 * pad + 1, 2 * pad + 1)
            arr = src.read(window=win, boundless=True, masked=True)
            rr, cc = np.mgrid[row-pad:row+pad+1, col-pad:col+pad+1]
            px, py = rasterio.transform.xy(src.transform, rr, cc, offset="center")
            px = np.asarray(px).reshape(rr.shape)
            py = np.asarray(py).reshape(rr.shape)
            mask = (px-x) ** 2 + (py-y) ** 2 <= radius ** 2
            if not mask.any():
                mask[pad, pad] = True
            vals = np.ma.filled(arr[:, mask], np.nan)
            out[i] = np.nanmean(vals, axis=1)
        out[~np.isfinite(out)] = np.nan
    return pd.DataFrame(out, columns=names, index=df.index)

def sample_raster_at_points(
    df: pd.DataFrame,
    raster_path: str | Path,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    out = df.copy()
    with rasterio.open(raster_path) as src:
        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            out[lon_col].to_numpy().tolist(),
            out[lat_col].to_numpy().tolist(),
        )
        samples = np.vstack(list(src.sample(zip(xs, ys), masked=True)))
        if np.ma.isMaskedArray(samples):
            samples = samples.filled(np.nan)

        names = raster_feature_names(src)

        for i, name in enumerate(names):
            out[name] = samples[:, i]
    return out

def read_raster_stack(path: str | Path):
    src = rasterio.open(path)
    arr = src.read(masked=True).astype("float32")
    return src, arr

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from .raster import raster_feature_names


def extract_local_features(
    points: pd.DataFrame,
    raster_path: str | Path,
    radius_m: float,
    statistic: str = "mean",
    suffix: str | None = None,
    expected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Extract circular pixel-centre statistics with block-grouped window reads.

    Pixels whose centres fall inside ``radius_m`` are used. When no raster-pixel
    centre falls inside a small circle, the nearest centre pixel is used. This is
    the closest practical representation on 20--25 m grids, not an exact GEDI
    footprint integral.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if statistic != "mean":
        raise ValueError("Only mean is supported for the primary comparison")
    required = {"longitude", "latitude"}
    if not required <= set(points):
        raise ValueError(f"Points are missing columns: {sorted(required - set(points))}")
    with rasterio.open(raster_path) as src:
        names = raster_feature_names(src)
        if expected_features is not None and names != expected_features:
            raise ValueError(f"Unexpected raster bands: {names}; expected {expected_features}")
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(points.longitude.to_numpy(), points.latitude.to_numpy())
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols, xs, ys = map(np.asarray, (rows, cols, xs, ys))
        result = np.full((len(points), src.count), np.nan, dtype="float64")
        valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        block_h, block_w = src.block_shapes[0]
        n_block_cols = int(np.ceil(src.width / block_w))
        block_ids = (rows[valid] // block_h) * n_block_cols + cols[valid] // block_w
        point_ids = np.flatnonzero(valid)
        order = np.argsort(block_ids, kind="stable")
        sorted_blocks, sorted_points = block_ids[order], point_ids[order]
        _, starts, counts = np.unique(sorted_blocks, return_index=True, return_counts=True)
        pad = max(1, int(np.ceil(radius_m / min(abs(src.res[0]), abs(src.res[1])))))
        for start, count in zip(starts, counts):
            ids = sorted_points[start:start + count]
            block_row = int((rows[ids[0]] // block_h) * block_h)
            block_col = int((cols[ids[0]] // block_w) * block_w)
            read_row = block_row - pad
            read_col = block_col - pad
            height = min(block_h, src.height - block_row) + 2 * pad
            width = min(block_w, src.width - block_col) + 2 * pad
            window = Window(read_col, read_row, width, height)
            data = src.read(window=window, boundless=True, masked=True)
            for point_id in ids:
                local_row = int(rows[point_id] - read_row)
                local_col = int(cols[point_id] - read_col)
                row_slice = slice(max(0, local_row-pad), min(height, local_row+pad+1))
                col_slice = slice(max(0, local_col-pad), min(width, local_col+pad+1))
                rr = np.arange(read_row + row_slice.start, read_row + row_slice.stop)
                cc = np.arange(read_col + col_slice.start, read_col + col_slice.stop)
                grid_rows, grid_cols = np.meshgrid(rr, cc, indexing="ij")
                centre_x = src.transform.c + (grid_cols + .5) * src.transform.a
                centre_y = src.transform.f + (grid_rows + .5) * src.transform.e
                circle = (centre_x-xs[point_id])**2 + (centre_y-ys[point_id])**2 <= radius_m**2
                if not circle.any():
                    distances = (centre_x-xs[point_id])**2 + (centre_y-ys[point_id])**2
                    circle.flat[int(np.argmin(distances))] = True
                values = data[:, row_slice, col_slice][:, circle]
                means = np.ma.mean(values, axis=1).filled(np.nan)
                result[point_id] = means
    result[~np.isfinite(result)] = np.nan
    label = suffix or f"support{round(radius_m * 2)}"
    return pd.DataFrame(result, columns=[f"{name}_{label}" for name in names], index=points.index)


def support_geometry_description(radius_m: float, resolutions=(20, 25)) -> str:
    details = ", ".join(
        f"{resolution} m raster: pixel centres within {radius_m:g} m; nearest pixel fallback"
        for resolution in resolutions
    )
    return f"Circular pixel-centre kernel (radius {radius_m:g} m; {details})."

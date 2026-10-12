from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def write_single_band(path: str | Path, array, reference, nodata=np.nan, dtype="float32"):
    profile = reference.profile.copy()
    profile.update(count=1, dtype=dtype, compress="deflate", nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(array, dtype=dtype), 1)

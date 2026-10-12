import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def analyze_raster(path: Path) -> dict:
    with rasterio.open(path) as src:
        # Résolution
        res_x, res_y = src.res

        # Taille disque
        size_gb = path.stat().st_size / (1024 ** 3)

        # Taille brute théorique
        bytes_per_value = np.dtype(src.dtypes[0]).itemsize
        raw_size_gb = (
            src.width
            * src.height
            * src.count
            * bytes_per_value
            / (1024 ** 3)
        )

        # Noms de bandes
        band_names = []
        for i, name in enumerate(src.descriptions):
            band_names.append(name or f"band_{i+1}")

        # Estimation du taux de NoData
        # On lit une version réduite pour ne pas charger tout le raster
        max_side = 1500
        factor = max(src.width, src.height) / max_side

        out_height = max(1, int(src.height / factor))
        out_width = max(1, int(src.width / factor))

        sample = src.read(
            1,
            out_shape=(out_height, out_width),
            masked=True
        )

        if np.ma.isMaskedArray(sample):
            nodata_percent = (
                np.mean(sample.mask) * 100
                if np.ndim(sample.mask) > 0
                else 0.0
            )
        else:
            nodata_percent = 0.0

        return {
            "file": path.name,
            "size_GB": round(size_gb, 3),
            "raw_size_GB_est": round(raw_size_gb, 3),
            "width_px": src.width,
            "height_px": src.height,
            "bands": src.count,
            "resolution_x_m": abs(res_x),
            "resolution_y_m": abs(res_y),
            "crs": str(src.crs),
            "dtype": ", ".join(sorted(set(src.dtypes))),
            "nodata": src.nodata,
            "nodata_percent_est": round(nodata_percent, 2),
            "bounds": str(src.bounds),
            "band_names": ", ".join(band_names),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compare raster metadata and estimated NoData."
    )

    parser.add_argument(
        "rasters",
        nargs="+",
        help="Paths to GeoTIFF files"
    )

    parser.add_argument(
        "--output",
        default="outputs/tables/raster_comparison.csv",
        help="CSV output path"
    )

    args = parser.parse_args()

    results = []

    for raster_path in args.rasters:
        path = Path(raster_path)

        if not path.exists():
            print(f"[WARNING] File not found: {path}")
            continue

        print(f"Analyzing: {path}")

        try:
            results.append(analyze_raster(path))
        except (OSError, ValueError, rasterio.errors.RasterioError) as e:
            print(f"[ERROR] {path}: {e}")

    if not results:
        print("No valid raster found.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("RASTER COMPARISON")
    print("=" * 100)

    cols = [
        "file",
        "size_GB",
        "width_px",
        "height_px",
        "bands",
        "resolution_x_m",
        "resolution_y_m",
        "crs",
        "dtype",
        "nodata_percent_est",
    ]

    print(
        df[cols].to_string(index=False)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output, index=False)

    print(f"\nSaved to: {output}")

    print("\nBand names:")
    for _, row in df.iterrows():
        print(f"\n{row['file']}")
        print(row["band_names"])


if __name__ == "__main__":
    main()

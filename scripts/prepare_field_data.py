from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from shapely.geometry import box

# =========================================================
# 1. Config
# =========================================================

DATASET_NAME = "data354/Africa_Biomass_dataset"

ROI = {
    "west": -7.85,
    "south": 4.95,
    "east": -6.35,
    "north": 6.45,
}

CRS_UTM = "EPSG:32629"

OUTPUT_DIR = Path("data/field")
FIGURE_DIR = Path("outputs/figures/field")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Charger le dataset Hugging Face
# =========================================================

print("Loading Africa Biomass Dataset...")

ds = load_dataset(
    DATASET_NAME,
    split="train"
)

df = ds.to_pandas()

print(f"Rows downloaded: {len(df)}")
print("Columns:", df.columns.tolist())


# =========================================================
# 3. Normaliser les colonnes
# =========================================================

df = df.rename(columns={
    "Latitude": "latitude",
    "Longitude": "longitude",
    "biomass_mg_ha": "field_agbd",
    "dates": "measurement_date",
    "iidentifiant": "plot_id",
})

required = [
    "plot_id",
    "measurement_date",
    "latitude",
    "longitude",
    "field_agbd",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["measurement_date"] = pd.to_datetime(
    df["measurement_date"],
    errors="coerce"
)

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df["field_agbd"] = pd.to_numeric(
    df["field_agbd"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "field_agbd",
    ]
).copy()


# =========================================================
# 4. GeoDataFrame
# =========================================================

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"]
    ),
    crs="EPSG:4326"
)


# =========================================================
# 5. ROI
# =========================================================

roi_geom = box(
    ROI["west"],
    ROI["south"],
    ROI["east"],
    ROI["north"]
)

roi_gdf = gpd.GeoDataFrame(
    {"name": ["study_area"]},
    geometry=[roi_geom],
    crs="EPSG:4326"
)


# =========================================================
# 6. Parcelles strictement dans la ROI
# =========================================================

inside_roi = gdf[
    gdf.geometry.within(roi_geom)
].copy()

print(
    f"Plots strictly inside ROI: {len(inside_roi)}"
)


# =========================================================
# 7. Parcelles proches de la ROI
# =========================================================

# Buffer métrique de 50 km autour de la zone.
# Utile pour voir s'il existe des placettes proches
# qui pourraient servir à une validation régionale.

gdf_utm = gdf.to_crs(CRS_UTM)
roi_utm = roi_gdf.to_crs(CRS_UTM)

buffer_50km = roi_utm.geometry.iloc[0].buffer(
    50_000
)

near_roi_utm = gdf_utm[
    gdf_utm.geometry.within(buffer_50km)
].copy()

near_roi = near_roi_utm.to_crs(
    "EPSG:4326"
)

print(
    f"Plots inside 50 km buffer: {len(near_roi)}"
)


# =========================================================
# 8. Ajouter coordonnées UTM
# =========================================================

inside_roi_utm = inside_roi.to_crs(
    CRS_UTM
)

inside_roi_utm["x_utm"] = (
    inside_roi_utm.geometry.x
)

inside_roi_utm["y_utm"] = (
    inside_roi_utm.geometry.y
)


# =========================================================
# 9. Statistiques
# =========================================================

print("\nField biomass summary:")

print(
    inside_roi_utm["field_agbd"]
    .describe(
        percentiles=[
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )
)


# =========================================================
# 10. Sauvegarder
# =========================================================

all_csv = OUTPUT_DIR / "africa_biomass_all.csv"

roi_csv = OUTPUT_DIR / "africa_biomass_roi.csv"

near_csv = (
    OUTPUT_DIR /
    "africa_biomass_roi_buffer50km.csv"
)

roi_gpkg = (
    OUTPUT_DIR /
    "africa_biomass_roi.gpkg"
)


df.to_csv(
    all_csv,
    index=False
)

inside_roi_utm.drop(
    columns="geometry"
).to_csv(
    roi_csv,
    index=False
)

near_roi.drop(
    columns="geometry"
).to_csv(
    near_csv,
    index=False
)

inside_roi_utm.to_file(
    roi_gpkg,
    driver="GPKG"
)


# =========================================================
# 11. Figure spatiale
# =========================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
)

roi_gdf.boundary.plot(
    ax=ax,
    linewidth=2,
    label="Study area"
)

gdf.plot(
    ax=ax,
    markersize=15,
    alpha=0.35,
    label="All Côte d'Ivoire plots"
)

inside_roi.plot(
    ax=ax,
    markersize=45,
    label="Plots inside ROI"
)

ax.set_title(
    "Africa Biomass field plots and study area"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR /
    "field_plots_study_area.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# 12. Histogramme biomasse terrain ROI
# =========================================================

if len(inside_roi) > 0:

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.hist(
        inside_roi["field_agbd"],
        bins=20
    )

    ax.set_xlabel(
        "Field AGBD (Mg/ha)"
    )

    ax.set_ylabel(
        "Number of plots"
    )

    ax.set_title(
        "Field biomass distribution — study area"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR /
        "field_agbd_distribution.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================================================
# 13. Résumé final
# =========================================================

print("\nSaved:")
print(all_csv)
print(roi_csv)
print(near_csv)
print(roi_gpkg)

print(
    "\nFigure:",
    FIGURE_DIR /
    "field_plots_study_area.png"
)

if len(inside_roi) > 0:
    print(
        FIGURE_DIR /
        "field_agbd_distribution.png"
    )
from __future__ import annotations

import numpy as np
import pandas as pd

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2_INDICES = ["NDVI", "NDMI", "NBR", "NDRE"]
S1_FEATURES = ["VV", "VH", "VV_minus_VH"]
PALSAR_FEATURES = [
    "PALSAR_HH_dB",
    "PALSAR_HV_dB",
    "PALSAR_HH_minus_HV_dB",
    "PALSAR_HV_HH_ratio",
    "PALSAR_HH_HV_ratio",
]
SENSOR_GROUPS = {
    "S2": S2_BANDS + S2_INDICES,
    "S1": S1_FEATURES,
    "PALSAR": PALSAR_FEATURES,
}

SENSOR_COMBINATIONS = {
    "S2": ["S2"],
    "S1": ["S1"],
    "PALSAR": ["PALSAR"],
    "S2+S1": ["S2", "S1"],
    "S2+PALSAR": ["S2", "PALSAR"],
    "S1+PALSAR": ["S1", "PALSAR"],
    "S2+S1+PALSAR": ["S2", "S1", "PALSAR"],
}

def validate_sensor_groups(df: pd.DataFrame, require_all: bool = True) -> dict[str, list[str]]:
    groups = {}
    for sensor, expected in SENSOR_GROUPS.items():
        present = [c for c in expected if c in df.columns]
        if require_all and present != expected:
            missing = [c for c in expected if c not in df.columns]
            raise ValueError(f"{sensor} is missing expected features: {missing}")
        bad = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"{sensor} features must be numeric: {bad}")
        groups[sensor] = present
    return groups

def combination_features(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        name: [feature for sensor in sensors for feature in groups[sensor]]
        for name, sensors in SENSOR_COMBINATIONS.items()
    }

def validate_s2_schema(df: pd.DataFrame, target: str = "agbd") -> list[str]:
    required = {target, "agbd_se", "longitude", "latitude"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    features = [c for c in S2_BANDS + S2_INDICES if c in df.columns]
    if not features:
        raise ValueError("No recognized Sentinel-2 predictor columns were found")
    bad = [c for c in features + [target] if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"Columns must be numeric: {bad}")
    return features

def clean_analysis_rows(df: pd.DataFrame, features: list[str], target: str = "agbd"):
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.loc[out[target].notna()].reset_index(drop=True)

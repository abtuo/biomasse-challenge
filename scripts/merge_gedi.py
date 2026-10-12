from pathlib import Path

import pandas as pd

folder = Path("data/raw/gedi_ci")

files = sorted(
    folder.glob("gedi_l4a_ci_*.csv")
)

print("Files:", len(files))

df = pd.concat(
    [
        pd.read_csv(f)
        for f in files
    ],
    ignore_index=True
)

print(
    "Avant déduplication:",
    len(df)
)

# Important :
# les frontières entre tuiles peuvent théoriquement
# produire quelques doublons.

if "shot_number" in df.columns:

    df = (
        df
        .drop_duplicates(
            subset="shot_number"
        )
        .reset_index(drop=True)
    )

print(
    "Après déduplication:",
    len(df)
)

out = (
    "data/raw/"
    "gedi_l4a_cote_divoire_filtered.csv"
)

df.to_csv(
    out,
    index=False
)

print(
    "Saved:",
    out
)
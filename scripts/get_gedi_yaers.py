import pandas as pd

path = "data/raw/gedi_l4a_tai_filtered.csv"

df = pd.read_csv(path)

# GEDI delta_time = secondes depuis 2018-01-01
origin = pd.Timestamp("2018-01-01 00:00:00")

df["gedi_datetime"] = (
    origin
    + pd.to_timedelta(
        df["delta_time"],
        unit="s"
    )
)

df["gedi_year"] = df["gedi_datetime"].dt.year

print("\nGEDI date range:")
print(df["gedi_datetime"].min())
print(df["gedi_datetime"].max())

print("\nGEDI shots by year:")
counts = (
    df["gedi_year"]
    .value_counts()
    .sort_index()
)

for year, n in counts.items():
    print(f"{year}: {n:,}")

print("\nPercentages:")
percent = counts / len(df) * 100

for year, pct in percent.items():
    print(f"{year}: {pct:.2f}%")
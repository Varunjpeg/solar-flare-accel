"""
scripts/01_fetch_sample.py  (v2 - single satellite, single resolution)
"""
from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts
import matplotlib.pyplot as plt

START = "2022-01-01"
END = "2024-12-31"

print(f"Searching for GOES-16 XRS 1-minute data between {START} and {END} ...")
result = Fido.search(
    a.Time(START, END),
    a.Instrument("XRS"),
    a.goes.SatelliteNumber(16),
    a.Resolution("avg1m"),
)
print(result)

print("Downloading ...")
files = Fido.fetch(result, path="data/raw/{file}")
print("Downloaded files:", files)

print("Loading into a time series ...")
combined = ts.TimeSeries(files, concatenate=True)
df = combined.to_dataframe()

print("\n--- First 5 rows ---")
print(df.head())
print("\n--- Shape (rows, columns) ---")
print(df.shape)
print("\n--- Column names ---")
print(list(df.columns))

df.to_csv("data/raw/goes_g16_avg1m_2022_2024.csv")
print("\nSaved -> data/raw/sample_week_feb2024_g16_avg1m.csv")

plt.figure(figsize=(12, 4))
plt.plot(df.index, df["xrsb"])
plt.yscale("log")
plt.ylabel("XRS-B flux (W/m^2)")
plt.xlabel("Time")
plt.title("GOES-16 XRS-B soft X-ray flux (1-min avg), Feb 20-25 2024")
plt.tight_layout()
plt.savefig("reports/goes_g16_avg1m_2022_2024_plot.png")
print("Saved plot -> reports/sample_week_plot_g16_avg1m.png")
plt.show()
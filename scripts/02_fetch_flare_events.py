"""
scripts/02_fetch_flare_events.py  (v7 - correct CSV subdirectory + exact filename pattern)

Downloads the official NOAA GOES Flare Report for 2022, 2023, 2024.
Source: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/
        goes/multi/l2/data/xrsf-l2-flrpt_science/csv/
File naming: sci_xrsf-l2-flrpt_geo_y{YYYY}_v1-0-1.csv (one file per year).

Why this source instead of HEK:
  - Pre-deduplicated: one row per flare, already reconciled across satellites.
  - Static files: no live-search API, no timeout/hang risk.
  - NOAA authoritative: same labels that define the A/B/C/M/X scale.
  - Contains start/peak/end times + GOES class (the exact fields we need).
"""
import requests
import pandas as pd
from io import StringIO

BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
    "goes/multi/l2/data/xrsf-l2-flrpt_science/csv/"
)
YEARS = [2022, 2023, 2024]
FILE_PATTERN = "sci_xrsf-l2-flrpt_geo_y{year}_v1-0-1.csv"
OUT_PATH = "data/raw/goes_flare_report_2022_2024.csv"

all_dfs = []
for year in YEARS:
    fname = FILE_PATTERN.format(year=year)
    url = BASE_URL + fname
    print(f"Downloading {url} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        print(f"got {len(df)} rows")
        all_dfs.append(df)
    except Exception as ex:
        print(f"FAILED ({ex})")

if not all_dfs:
    print("\nERROR: No data downloaded. Check network / URL.")
    raise SystemExit(1)

combined = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows across all years: {len(combined)}")
print(f"\n--- Column names ---\n{list(combined.columns)}")
print(f"\n--- First 5 rows ---")
print(combined.head().to_string())
print(f"\n--- Data types ---")
print(combined.dtypes)

combined.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
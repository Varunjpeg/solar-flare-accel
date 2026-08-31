"""
scripts/03_filter_flare_events.py
Filter the raw HEK event list down to ONE canonical, authoritative
source (NOAA SWPC), deduplicate, and produce a clean labeled event list.
"""
import pandas as pd

df = pd.read_csv("data/raw/flare_events_raw_feb2024.csv", parse_dates=[
    "event_starttime", "event_peaktime", "event_endtime"
])

print(f"Total raw records: {len(df)}")

# Keep only the official NOAA SWPC source
swpc = df[df["frm_name"] == "SWPC"].copy()
print(f"SWPC records: {len(swpc)}")

# Drop any rows with a missing class (shouldn't be any for SWPC, but be safe)
swpc = swpc.dropna(subset=["fl_goescls"])
swpc = swpc.sort_values("event_starttime").reset_index(drop=True)

# Extract just the letter (A/B/C/M/X) as a separate column - this will
# become our integer class label later
swpc["goes_letter"] = swpc["fl_goescls"].str[0]

print("\n--- Class distribution (this is your class-imbalance preview) ---")
print(swpc["goes_letter"].value_counts())

print("\n--- Full clean event list ---")
print(swpc[["event_starttime", "event_peaktime", "event_endtime", "fl_goescls"]].to_string(index=False))

swpc.to_csv("data/processed/flare_events_clean_feb2024.csv", index=False)
print("\nSaved -> data/processed/flare_events_clean_feb2024.csv")
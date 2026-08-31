"""
scripts/04_check_data_quality.py
Investigate the quality flags on the full 3-year dataset, confirm the
dips correspond to bad-quality periods, and produce a quality-filtered
version of the flux data.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/goes_g16_avg1m_2022_2024.csv", index_col=0, parse_dates=True)

print(f"Total rows: {len(df)}")
print("\n--- xrsb_quality value counts ---")
print(df["xrsb_quality"].value_counts())

bad_pct = (df["xrsb_quality"] != 0).mean() * 100
print(f"\n{bad_pct:.2f}% of rows are flagged non-zero (bad) quality")

# Confirm: do the flux dips line up with bad-quality flags?
suspicious = df[df["xrsb"] < 1e-8]
print(f"\nRows with xrsb < 1e-8 (the anomalous dips): {len(suspicious)}")
print("Quality flag breakdown for those suspicious rows:")
print(suspicious["xrsb_quality"].value_counts())

# Filter to good-quality data only
clean = df[df["xrsb_quality"] == 0].copy()
print(f"\nRows remaining after quality filter: {len(clean)} (dropped {len(df) - len(clean)})")

clean.to_csv("data/processed/goes_g16_avg1m_2022_2024_qualityfiltered.csv")
print("Saved -> data/processed/goes_g16_avg1m_2022_2024_qualityfiltered.csv")

# Re-plot to confirm the dips are gone
plt.figure(figsize=(14, 4))
plt.plot(clean.index, clean["xrsb"])
plt.yscale("log")
plt.ylabel("XRS-B flux (W/m^2)")
plt.xlabel("Time")
plt.title("GOES-16 XRS-B, 2022-2024, quality-filtered")
plt.tight_layout()
plt.savefig("reports/goes_g16_2022_2024_qualityfiltered_plot.png")
print("Saved plot -> reports/goes_g16_2022_2024_qualityfiltered_plot.png")
plt.show()
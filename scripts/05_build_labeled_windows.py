"""
scripts/05_build_labeled_windows.py

PURPOSE
-------
Convert the raw time-series flux data + flare event catalog into a clean
fixed-size (X, y) dataset ready for ML training.

CONCEPT: WHAT IS A "WINDOW"?
-------------------------------
A 1D-CNN takes a *fixed-length* input vector — not an arbitrary-length
time series. So we "window" the data:

    For each flare event:
      - Find its peak time (the moment of maximum X-ray flux).
      - Extract 60 minutes of data centered on the peak:
            30 minutes BEFORE the peak  +  30 minutes AFTER the peak
      - This gives 60 rows (1 per minute) × 2 channels (xrsa, xrsb).
      - Label = GOES class: A=0, B=1, C=2, M=3, X=4

    For background (no-flare) windows:
      - Randomly sample times that are NOT within ±2h of any flare.
      - Label = 0 (no-flare / background).
      - We sample the SAME NUMBER as total flare windows to avoid extreme
        class imbalance between "flare" and "no-flare".

WHY LOG10?
----------
The flux spans many orders of magnitude (e.g. 1e-9 to 1e-3 W/m²).
Neural networks perform best when inputs are roughly on the same scale.
log10 compresses that range so the input features are comparable.

TEMPORAL TRAIN/VAL/TEST SPLIT:
-------------------------------
CRITICAL: We split by DATE, not randomly. If we split randomly, the model
could see data from NEAR a flare event in training but the flare itself in
test — that's data leakage, and would give artificially inflated accuracy.

    2022-01-01 → 2023-06-30  = TRAIN   (~60%)
    2023-07-01 → 2023-12-31  = VAL     (~20%)
    2024-01-01 → 2024-12-31  = TEST    (~20%)

OUTPUT
------
  data/processed/labeled_windows.npz  with keys:
      X_train  shape (N_train, 60, 2)  float32, log10-normalized
      y_train  shape (N_train,)        int8
      X_val    shape (N_val, 60, 2)    float32
      y_val    shape (N_val,)          int8
      X_test   shape (N_test, 60, 2)  float32
      y_test   shape (N_test,)        int8
      class_names  = ['background', 'A', 'B', 'C', 'M', 'X']

  data/processed/windowing_report.txt  — class distributions + split sizes
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
FLUX_CSV        = "data/processed/goes_g16_avg1m_2022_2024_qualityfiltered.csv"
FLARE_CSV       = "data/raw/goes_flare_report_2022_2024.csv"
OUT_NPZ         = "data/processed/labeled_windows.npz"
OUT_REPORT      = "data/processed/windowing_report.txt"

WINDOW_MINUTES  = 60        # total window size in minutes
PRE_PEAK_MIN    = 30        # minutes before peak to include
POST_PEAK_MIN   = 30        # minutes after peak
FLUX_FLOOR      = 1e-9      # clip values below this (physically impossible → set by floor)
FLUX_LOG_MIN    = -9.0      # log10(FLUX_FLOOR)
FLUX_LOG_MAX    = -3.0      # log10(1e-3 W/m², extreme X-class saturation)

TRAIN_END  = pd.Timestamp("2023-06-30 23:59:00")
VAL_END    = pd.Timestamp("2023-12-31 23:59:00")
# test = everything after VAL_END

BG_EXCLUSION_MINUTES = 120  # no-flare windows must be ≥2h from any flare peak
RANDOM_SEED = 42

# Map GOES flare class letter → integer label
# We add 'background' as class 0, then A=1, B=2, C=3, M=4, X=5
LETTER_TO_LABEL = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
CLASS_NAMES = ["background", "A", "B", "C", "M", "X"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("STEP 1 — Loading quality-filtered flux data")
print("=" * 60)
flux = pd.read_csv(FLUX_CSV, index_col=0, parse_dates=True)
flux = flux.sort_index()
print(f"  Flux rows: {len(flux):,}   ({flux.index[0]} to {flux.index[-1]})")
print(f"  Columns:   {list(flux.columns)}")

# Pre-compute log10-clipped arrays for fast numpy slicing later
xrsa_log = np.clip(np.log10(flux["xrsa"].values.astype(np.float32)), FLUX_LOG_MIN, FLUX_LOG_MAX)
xrsb_log = np.clip(np.log10(flux["xrsb"].values.astype(np.float32)), FLUX_LOG_MIN, FLUX_LOG_MAX)
flux_log  = np.stack([xrsa_log, xrsb_log], axis=1)  # shape (N, 2)
# index array for fast lookup: timestamp → integer position
ts_index = flux.index

print(f"\n  flux_log shape: {flux_log.shape}")
print(f"  xrsa_log range: {xrsa_log.min():.2f} … {xrsa_log.max():.2f}")
print(f"  xrsb_log range: {xrsb_log.min():.2f} … {xrsb_log.max():.2f}")


print("\n" + "=" * 60)
print("STEP 2 — Loading flare catalog")
print("=" * 60)
events = pd.read_csv(FLARE_CSV, parse_dates=["time", "start_time", "end_time"])
print(f"  Total events: {len(events):,}")
print(f"  Class distribution (raw):")
events["letter"] = events["flare_class"].str[0].str.upper()
print(events["letter"].value_counts().to_string())

# Filter to only the classes we care about: A, B, C, M, X
# (there can occasionally be null/invalid entries)
events = events[events["letter"].isin(LETTER_TO_LABEL)].copy()
events["label"] = events["letter"].map(LETTER_TO_LABEL).astype(np.int8)
print(f"\n  After filtering to A/B/C/M/X: {len(events):,} events")


def extract_window(peak_ts):
    """
    Given a peak timestamp, return a (60, 2) float32 array of
    log10-clipped [xrsa, xrsb] flux centered on the peak.
    Returns None if data is insufficient (gaps / out of range).
    """
    start_ts = peak_ts - pd.Timedelta(minutes=PRE_PEAK_MIN)
    end_ts   = peak_ts + pd.Timedelta(minutes=POST_PEAK_MIN - 1)

    # Use searchsorted for fast positional lookup
    i_start = ts_index.searchsorted(start_ts)
    i_end   = ts_index.searchsorted(end_ts, side="right")

    segment = flux_log[i_start:i_end]

    if len(segment) < WINDOW_MINUTES:
        return None   # not enough data (gap / edge of dataset)

    # Take exactly WINDOW_MINUTES rows
    return segment[:WINDOW_MINUTES].astype(np.float32)


print("\n" + "=" * 60)
print("STEP 3 — Extracting flare windows")
print("=" * 60)

flare_X, flare_y, flare_ts = [], [], []
skipped = 0

for _, row in events.iterrows():
    peak_ts = row["time"]  # "time" column = peak time in this schema
    w = extract_window(peak_ts)
    if w is None:
        skipped += 1
        continue
    flare_X.append(w)
    flare_y.append(row["label"])
    flare_ts.append(peak_ts)

print(f"  Extracted: {len(flare_X):,} flare windows")
print(f"  Skipped (insufficient data around peak): {skipped}")

# Class distribution of extracted windows
unique, counts = np.unique(flare_y, return_counts=True)
print("\n  Extracted flare class distribution:")
for u, c in zip(unique, counts):
    print(f"    {CLASS_NAMES[u]:12s} (label={u}): {c:5d} windows")


print("\n" + "=" * 60)
print("STEP 4 — Sampling background (no-flare) windows")
print("=" * 60)
"""
CONCEPT: Why do we need background windows?
Our model needs to know what "no flare happening" looks like, not just
classify between flare classes. Without background examples, the model
would always predict a flare class even on quiet-sun data.

We sample random 60-minute windows from times that are far from any flare.
"""

# Build a set of "forbidden" timestamps: ±BG_EXCLUSION_MINUTES from each flare peak
print("  Building exclusion set around all flare peak times...")
peak_timestamps = set(pd.DatetimeIndex(flare_ts))
exclusion_set   = set()
for pt in flare_ts:
    for offset_min in range(-BG_EXCLUSION_MINUTES, BG_EXCLUSION_MINUTES + 1):
        exclusion_set.add(pt + pd.Timedelta(minutes=offset_min))

# Candidate background peak times: any minute NOT in exclusion_set,
# also not too close to the start/end of the dataset (need ±30 min headroom)
headroom = pd.Timedelta(minutes=PRE_PEAK_MIN + 5)
cand_mask = [
    (ts not in exclusion_set)
    and (ts >= ts_index[0] + headroom)
    and (ts <= ts_index[-1] - headroom)
    for ts in ts_index
]
candidates = ts_index[cand_mask]
print(f"  Background candidate timestamps: {len(candidates):,}")

# Sample as many background windows as total flare windows
n_bg_needed = len(flare_X)
sampled_ts  = pd.DatetimeIndex(random.sample(list(candidates), min(n_bg_needed, len(candidates))))

bg_X, bg_y, bg_ts = [], [], []
for ts in sampled_ts:
    w = extract_window(ts)
    if w is None:
        continue
    bg_X.append(w)
    bg_y.append(np.int8(0))  # label 0 = background
    bg_ts.append(ts)          # track timestamp ONLY if window was successfully extracted

print(f"  Extracted {len(bg_X):,} background windows (label=0)")


print("\n" + "=" * 60)
print("STEP 5 — Combining + temporal train/val/test split")
print("=" * 60)
"""
CONCEPT: Temporal split (NOT random shuffle)
We assign each window to a split based on the PEAK TIMESTAMP of the event
that generated it. This guarantees the model never "sees the future" during
training — a fundamental requirement for any real-time forecasting system.

Analogy: if you train a stock predictor, you can't include Jan 15 in training
if it's predicting Jan 14 — same principle here with solar flares.
"""

# Combine all windows with their peak timestamps
all_X  = np.array(flare_X + bg_X, dtype=np.float32)  # shape (N_total, 60, 2)
all_y  = np.array(flare_y + bg_y, dtype=np.int8)       # shape (N_total,)
all_ts = list(flare_ts) + list(bg_ts)                   # timestamps synced with all_X rows

print(f"  Total windows: {len(all_X):,}  (flare: {len(flare_X):,}, background: {len(bg_X):,})")
print(f"  Array shape: X={all_X.shape}, y={all_y.shape}")

# Temporal split
train_mask = np.array([ts <= TRAIN_END           for ts in all_ts])
val_mask   = np.array([TRAIN_END < ts <= VAL_END for ts in all_ts])
test_mask  = np.array([ts > VAL_END              for ts in all_ts])

X_train, y_train = all_X[train_mask], all_y[train_mask]
X_val,   y_val   = all_X[val_mask],   all_y[val_mask]
X_test,  y_test  = all_X[test_mask],  all_y[test_mask]

print(f"\n  TRAIN: {len(X_train):,} windows  (up to 2023-06-30)")
print(f"  VAL:   {len(X_val):,} windows  (2023-07-01 to 2023-12-31)")
print(f"  TEST:  {len(X_test):,} windows  (2024-01-01 to 2024-12-31)")


print("\n" + "=" * 60)
print("STEP 6 - Saving to NPZ")
print("=" * 60)
Path("data/processed").mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT_NPZ,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    class_names=np.array(CLASS_NAMES),
)
print(f"  Saved -> {OUT_NPZ}")
print(f"  File contents:")
with np.load(OUT_NPZ, allow_pickle=True) as f:
    for k, v in f.items():
        print(f"    {k:12s}: shape={v.shape}  dtype={v.dtype}")


# ─── Write report ────────────────────────────────────────────────────────────
report_lines = []
report_lines.append("=" * 60)
report_lines.append("WINDOWING REPORT — goes_g16_avg1m_2022_2024")
report_lines.append("=" * 60)
report_lines.append(f"Window size:   {WINDOW_MINUTES} minutes")
report_lines.append(f"Pre-peak:      {PRE_PEAK_MIN} min")
report_lines.append(f"Post-peak:     {POST_PEAK_MIN} min")
report_lines.append(f"Log10 clip:    [{FLUX_LOG_MIN}, {FLUX_LOG_MAX}]")
report_lines.append(f"")
report_lines.append(f"Total flare events in catalog: {len(events):,}")
report_lines.append(f"Flare windows extracted:       {len(flare_X):,}")
report_lines.append(f"Background windows:            {len(bg_X):,}")
report_lines.append(f"Total windows:                 {len(all_X):,}")
report_lines.append(f"")
for split_name, y_split in [("TRAIN", y_train), ("VAL", y_val), ("TEST", y_test)]:
    report_lines.append(f"--- {split_name} split ({len(y_split)} windows) ---")
    u, c = np.unique(y_split, return_counts=True)
    for label, cnt in zip(u, c):
        report_lines.append(f"  {CLASS_NAMES[label]:12s} (label={label}): {cnt:5d}  ({100*cnt/len(y_split):.1f}%)")
    report_lines.append("")

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(OUT_REPORT, "w") as f:
    f.write(report_text)
print(f"\nReport saved -> {OUT_REPORT}")
print("\nDone! Phase 1 complete. Ready for ML training.")

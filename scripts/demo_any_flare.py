"""
scripts/demo_any_flare.py

PURPOSE:
Interactive demo for faculty. Allows you to specify ANY date and time
from the command line to fetch NOAA data and run the AI prediction.

USAGE:
python scripts/demo_any_flare.py --date 2024-05-14 --time 16:51
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings("ignore")
from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts

MODEL_PATH = "models/flare_1dcnn_float32.pt"
NUM_CLASSES = 6
CLASS_NAMES = ['background', 'A', 'B', 'C', 'M', 'X']

import torch.nn as nn
class FlareCNN1D(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(2, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes))
    def forward(self, x):
        return self.classifier(self.gap(self.conv3(self.conv2(self.conv1(x)))))

def main():
    parser = argparse.ArgumentParser(description="Test AI on any solar flare.")
    parser.add_argument("--date", required=True, help="Date of the flare (YYYY-MM-DD)")
    parser.add_argument("--time", required=True, help="Peak time of the flare (HH:MM)")
    args = parser.parse_args()
    
    target_date = args.date
    peak_time_str = f"{target_date} {args.time}:00"
    
    print("\n" + "="*60)
    print(f"FETCHING LIVE DATA FOR: {peak_time_str}")
    print("="*60)
    
    try:
        peak_ts = pd.Timestamp(peak_time_str)
    except ValueError:
        print("Invalid date/time format. Use YYYY-MM-DD and HH:MM")
        return

    # Fetch data for that specific day
    result = Fido.search(
        a.Time(f"{target_date} 00:00", f"{target_date} 23:59"),
        a.Instrument("XRS"),
        a.Resolution("avg1m")
    )
    
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    files = Fido.fetch(result, path="data/raw/{file}", progress=False)
    
    if not files:
        print("Error: Could not find NOAA GOES-16 data for that date.")
        return
        
    df = ts.TimeSeries(files, concatenate=True).to_dataframe()
    
    # 60 min window: 30 mins before peak to 30 mins after
    start_ts = peak_ts - pd.Timedelta(minutes=30)
    end_ts = peak_ts + pd.Timedelta(minutes=29)
    
    window_df = df.loc[str(start_ts):str(end_ts)]
    
    if len(window_df) < 60:
        print(f"Error: Missing data around {peak_time_str}. Found only {len(window_df)} minutes.")
        return
        
    window_df = window_df.iloc[:60]
    
    xrsa_log = np.clip(np.log10(window_df["xrsa"].values.astype(np.float32)), -9.0, -3.0)
    xrsb_log = np.clip(np.log10(window_df["xrsb"].values.astype(np.float32)), -9.0, -3.0)
    sample_x = np.stack([xrsa_log, xrsb_log], axis=1)
    input_tensor = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

    model = FlareCNN1D(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_label = logits.argmax(1).item()
        
    print("\n" + "="*60)
    print("                      RESULTS")
    print("="*60)
    print(f"AI PREDICTION        : {CLASS_NAMES[pred_label]}-Class Flare")
    print(f"AI CONFIDENCE        : {probabilities[pred_label]*100:.2f}%")
    print("="*60)
    
    # Plot
    time_axis = np.arange(-30, 30)
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, window_df["xrsb"], label="XRS-B", color="red")
    plt.plot(time_axis, window_df["xrsa"], label="XRS-A", color="blue")
    plt.axvline(0, color="gray", linestyle="--", label="Target Time")
    plt.yscale("log")
    plt.title(f"Custom Demo ({target_date}): AI predicted {CLASS_NAMES[pred_label]}-Class ({probabilities[pred_label]*100:.1f}%)")
    plt.xlabel("Minutes relative to target time")
    plt.ylabel("X-Ray Flux (W/m²)")
    plt.grid(True)
    plt.legend()
    
    plot_file = f"reports/custom_flare_{target_date}_{args.time.replace(':','')}.png"
    plt.savefig(plot_file)
    print(f"\nPlot saved to: {plot_file}\n")

if __name__ == "__main__":
    main()

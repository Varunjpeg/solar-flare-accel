"""
scripts/demo_live_internet.py

PURPOSE:
A live demo script for your faculty presentation. 
It connects to the internet, downloads the official NOAA data for the 
famous May 14, 2024 X8.7 solar flare, feeds it into your trained AI, 
and displays the prediction.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Suppress warnings for cleaner demo output
warnings.filterwarnings("ignore")

# Sunpy for downloading space data
from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts

# --- Configuration ---
MODEL_PATH = "models/flare_1dcnn_float32.pt"
OUT_PLOT = "reports/live_internet_demo.png"
NUM_CLASSES = 6
CLASS_NAMES = ['background', 'A', 'B', 'C', 'M', 'X']

# The AI Model Architecture (Must match the trained model)
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

def run_live_demo():
    print("\n" + "="*60)
    print("SOLAR FLARE AI - LIVE INTERNET DEMO")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. FETCH THE DATA FROM NOAA (INTERNET)
    # ---------------------------------------------------------
    # The massive X8.7 flare peaked at 16:51 UTC on May 14, 2024.
    # We will grab the data for that specific day.
    print("\n[1/4] Connecting to NOAA servers to download May 14, 2024 data...")
    result = Fido.search(
        a.Time("2024-05-14 00:00", "2024-05-14 23:59"),
        a.Instrument("XRS"),
        a.goes.SatelliteNumber(16),
        a.Resolution("avg1m")
    )
    
    # Download the file to a temporary raw folder
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    downloaded_files = Fido.fetch(result, path="data/raw/{file}", progress=False)
    
    print("      Data downloaded successfully!")
    
    # Load it into a Pandas Dataframe
    combined = ts.TimeSeries(downloaded_files, concatenate=True)
    df = combined.to_dataframe()
    
    # ---------------------------------------------------------
    # 2. EXTRACT EXACTLY THE 60-MINUTE WINDOW
    # ---------------------------------------------------------
    print("\n[2/4] Slicing exactly 60 minutes of data around the flare peak (16:21 to 17:20 UTC)...")
    
    # The peak was ~16:51. We take 30 mins before, and 30 mins after.
    start_time = "2024-05-14 16:21:00"
    end_time   = "2024-05-14 17:20:00"
    
    # Slice the dataframe
    window_df = df.loc[start_time:end_time]
    
    if len(window_df) < 60:
        print("Error: Could not extract a full 60 minutes. Check data.")
        return
        
    window_df = window_df.iloc[:60] # Force exactly 60 rows just in case
    
    # Preprocess: Log10 scaling exactly how the AI was trained
    xrsa_log = np.clip(np.log10(window_df["xrsa"].values.astype(np.float32)), -9.0, -3.0)
    xrsb_log = np.clip(np.log10(window_df["xrsb"].values.astype(np.float32)), -9.0, -3.0)
    
    # Combine into shape (60, 2)
    sample_x = np.stack([xrsa_log, xrsb_log], axis=1)
    
    # Convert to PyTorch Tensor shape (1 batch, 2 channels, 60 time-steps)
    input_tensor = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

    # ---------------------------------------------------------
    # 3. LOAD THE TRAINED AI
    # ---------------------------------------------------------
    print("\n[3/4] Loading the AI Hardware Prototype...")
    model = FlareCNN1D(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # ---------------------------------------------------------
    # 4. RUN INFERENCE (NOWCASTING)
    # ---------------------------------------------------------
    print("\n[4/4] Feeding the raw satellite data into the AI...")
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
        predicted_label = logits.argmax(1).item()
        
    print("\n" + "="*60)
    print("                      RESULTS")
    print("="*60)
    print(f"OFFICIAL NOAA RECORD : X-Class Flare (X8.7)")
    print(f"AI PREDICTION        : {CLASS_NAMES[predicted_label]}-Class Flare")
    print(f"AI CONFIDENCE        : {probabilities[predicted_label]*100:.2f}%")
    print("="*60)
    
    if CLASS_NAMES[predicted_label] == 'X':
        print("\nSUCCESS! The AI perfectly identified the extreme solar flare.")
    
    # ---------------------------------------------------------
    # 5. GENERATE THE PLOT
    # ---------------------------------------------------------
    time_axis = np.arange(-30, 30)
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, window_df["xrsb"], label="XRS-B (Primary)", color="red", linewidth=2)
    plt.plot(time_axis, window_df["xrsa"], label="XRS-A (Secondary)", color="blue", linewidth=2)
    
    plt.axvline(0, color="gray", linestyle="--", label="Flare Peak (16:51 UTC)")
    
    plt.yscale("log")
    plt.title(f"Live Demo (May 14, 2024): AI predicted {CLASS_NAMES[predicted_label]}-Class Flare ({probabilities[predicted_label]*100:.1f}%)")
    plt.xlabel("Minutes relative to Peak")
    plt.ylabel("X-Ray Flux (W/m²)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT)
    print(f"\nVisual plot saved to: {OUT_PLOT}")
    print("Ready for faculty presentation!\n")

if __name__ == "__main__":
    run_live_demo()

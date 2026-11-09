import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
MODEL_PATH = "models/flare_1dcnn_float32.pt"
DATA_PATH = "data/processed/labeled_windows.npz"
OUT_PLOT = "reports/mentor_demo_plot.png"
NUM_CLASSES = 6
CLASS_NAMES = ['background', 'A', 'B', 'C', 'M', 'X']

# Re-define the model architecture to load weights
import torch.nn as nn
class FlareCNN1D(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return self.classifier(x)

def run_demo():
    print("Loading model...")
    model = FlareCNN1D(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    print("Loading test data...")
    d = np.load(DATA_PATH)
    X_test = d["X_test"]
    y_test = d["y_test"]
    
    # Find a cool flare to show (M or X class)
    # M is class 4, X is class 5
    target_classes = [4, 5] 
    idx_list = [i for i, label in enumerate(y_test) if label in target_classes]
    
    if not idx_list:
        print("No M or X class flares found in test set, picking random.")
        idx = np.random.randint(0, len(y_test))
    else:
        # Pick a random M or X class flare
        idx = np.random.choice(idx_list)
        
    sample_x = X_test[idx] # Shape: (60, 2)
    actual_label = y_test[idx]
    
    # Preprocess for model: (batch, channels, time) -> (1, 2, 60)
    input_tensor = torch.tensor(sample_x, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
    
    print("Running AI inference...")
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
        predicted_label = logits.argmax(1).item()
        
    print(f"\n--- RESULTS ---")
    print(f"Actual Class:    {CLASS_NAMES[actual_label]}")
    print(f"Predicted Class: {CLASS_NAMES[predicted_label]}")
    print(f"AI Confidence:   {probabilities[predicted_label]*100:.2f}%")
    
    # Reverse log10 for plotting so it looks like a real flux graph
    xrsa_real = 10 ** sample_x[:, 0]
    xrsb_real = 10 ** sample_x[:, 1]
    time_axis = np.arange(-30, 30) # 30 min before peak, 30 min after

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, xrsb_real, label="XRS-B (Primary)", color="red")
    plt.plot(time_axis, xrsa_real, label="XRS-A (Secondary)", color="blue")
    plt.axvline(0, color="gray", linestyle="--", label="Flare Peak")
    
    plt.yscale("log")
    plt.title(f"GOES-16 Demo: AI predicted {CLASS_NAMES[predicted_label]}-Class Flare ({probabilities[predicted_label]*100:.1f}%)")
    plt.xlabel("Minutes relative to Peak")
    plt.ylabel("X-Ray Flux (W/m²)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT)
    print(f"\nDemo plot saved to {OUT_PLOT}")

if __name__ == "__main__":
    run_demo()

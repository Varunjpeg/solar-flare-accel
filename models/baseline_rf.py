"""
models/baseline_rf.py

PURPOSE
-------
Train a classical Random Forest classifier on hand-crafted statistical
features of each flux window. This gives us an ACCURACY FLOOR —  a number
we must beat with the 1D-CNN to justify the neural-network (and eventually
hardware) complexity.

CONCEPT: Why a baseline model?
-------------------------------
Before building complex systems (CNN → INT8 → RTL), we should know: can a
simple model already solve the problem well?

  - If a Random Forest hits 95% accuracy → the CNN probably won't add much.
  - If a Random Forest hits 70% → the CNN has room to improve, and the
    extra complexity is justified.

FEATURES (hand-crafted from each 60-min window):
  For each channel (xrsa_log, xrsb_log):
    - mean, std, min, max
    - max minus first value (proxy for "rise from background")
    - argmax (where in the window does the peak occur?)
    - slope (linear fit gradient, proxy for rise/decay rate)
  Also: ratio features between xrsa and xrsb.

  Total: ~15 features per window. Very interpretable.

HARDWARE RELEVANCE:
  A Random Forest is NOT hardware-friendly (many decision trees → large
  memory, irregular access patterns). The 1D-CNN, by contrast, maps to a
  regular MAC array. So even if RF accuracy ≈ CNN accuracy, the CNN wins
  for the hardware accelerator because it's synthesizable into a clean
  datapath. This distinction should appear in your paper's Table 1.

OUTPUT
------
  models/rf_baseline_results.txt  — accuracy, confusion matrix, per-class recall
"""

import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_NPZ  = "data/processed/labeled_windows.npz"
OUT_REPORT = "models/rf_baseline_results.txt"

Path("models").mkdir(exist_ok=True)

print("Loading labeled_windows.npz ...")
data = np.load(DATA_NPZ, allow_pickle=True)
X_train = data["X_train"]  # (N, 60, 2)
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]
class_names = list(data["class_names"])

print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"  Classes: {class_names}")


def extract_features(X):
    """
    X: (N, 60, 2) float32
    Returns: (N, num_features) float32 feature matrix
    """
    feats = []
    for i in range(X.shape[0]):
        w = X[i]          # shape (60, 2)
        row = []
        for ch in range(2):   # ch=0: xrsa_log, ch=1: xrsb_log
            s = w[:, ch]      # shape (60,)
            row.append(s.mean())
            row.append(s.std())
            row.append(s.min())
            row.append(s.max())
            row.append(s.max() - s[0])           # rise magnitude
            row.append(float(np.argmax(s)))      # position of peak in window
            # linear slope via least squares
            t = np.arange(60, dtype=np.float32)
            slope = np.polyfit(t, s, 1)[0]
            row.append(slope)
        # Ratio: xrsb_max / xrsa_max (spectral hardness proxy)
        xrsa_max = w[:, 0].max()
        xrsb_max = w[:, 1].max()
        row.append(xrsb_max - xrsa_max)   # log difference ≡ log ratio
        feats.append(row)
    return np.array(feats, dtype=np.float32)


print("\nExtracting features ...")
F_train = extract_features(X_train)
F_val   = extract_features(X_val)
F_test  = extract_features(X_test)
print(f"  Feature matrix shape: {F_train.shape}")

# ── Train ──────────────────────────────────────────────────────────────────
print("\nTraining Random Forest (n_estimators=200, class_weight='balanced') ...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",   # compensates for class imbalance automatically
    random_state=42,
    n_jobs=-1,
)
rf.fit(F_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 60)
lines.append("RANDOM FOREST BASELINE RESULTS")
lines.append("=" * 60)

for split_name, F, y_true in [
    ("VALIDATION", F_val, y_val),
    ("TEST",       F_test, y_test),
]:
    y_pred = rf.predict(F)
    acc = accuracy_score(y_true, y_pred)
    all_labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred,
                                   labels=all_labels,
                                   target_names=class_names,
                                   zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    lines.append(f"\n--- {split_name} SET ---")
    lines.append(f"Overall accuracy: {acc*100:.2f}%")
    lines.append(f"\nPer-class report:\n{report}")
    lines.append(f"Confusion matrix (rows=true, cols=pred):")
    lines.append(f"  Classes: {class_names}")
    lines.append(str(cm))

report_text = "\n".join(lines)
print(report_text)

with open(OUT_REPORT, "w") as f:
    f.write(report_text)
print(f"\nSaved -> {OUT_REPORT}")
print("\nBaseline done. The 1D-CNN should beat these numbers.")

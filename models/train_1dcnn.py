"""
models/train_1dcnn.py

PURPOSE
-------
Design, train, and save a compact 1D-CNN for solar flare classification.
This is the core ML model that will eventually become RTL silicon.

CONCEPT: Why a 1D-CNN?
------------------------
A 1D Convolutional Neural Network slides small learnable "filters" across
the time axis of the input signal — exactly like a FIR filter in signal
processing, but with learned weights.

Example: a filter of size 5 computes a weighted sum of 5 consecutive
minutes of flux. With 16 such filters in the first layer, the CNN learns
16 different "patterns" to look for at every point in time.

This maps directly to hardware: a MAC array where each unit multiplies
an input sample by a weight and accumulates — the core operation of both
digital filters and neural network convolution.

ARCHITECTURE (designed for RTL friendliness):
----------------------------------------------
  Input:   (batch, 60, 2)    -- 60-min window, 2 channels (xrsa_log, xrsb_log)
  Conv1:   kernel=5, filters=16, stride=1, padding=2  → (batch, 60, 16)
  ReLU                                                  (trivial in hardware)
  MaxPool: kernel=2, stride=2                          → (batch, 30, 16)
  Conv2:   kernel=5, filters=32, stride=1, padding=2  → (batch, 30, 32)
  ReLU
  MaxPool: kernel=2, stride=2                          → (batch, 15, 32)
  Conv3:   kernel=3, filters=64, stride=1, padding=1  → (batch, 15, 64)
  ReLU
  GlobalAvgPool                                        → (batch, 64)
  Dense:   64 → 32, ReLU
  Dense:   32 → 6 (num_classes), Softmax

Why GlobalAvgPool instead of Flatten?
  Flatten on (15, 64) → 960 features → huge Dense layer → many weights.
  GlobalAvgPool → 64 features regardless of temporal dimension → smaller
  Dense layers → fewer weights → smaller weight ROM in hardware. This is
  a deliberate architectural choice for hardware efficiency.

HARDWARE IMPACT OF EVERY DECISION:
  - Small kernel (3–5): fewer MACs per layer.
  - Max pool ÷2: cuts input size to next layer in half → halves compute.
  - GlobalAvgPool: replaces 960→N Dense with 64→N Dense → ~15× fewer weights.
  - 6 classes (not 5): background class added — important for real deployment.
  - INT8 quantization (next step): all these float32 weights will become
    signed 8-bit integers, so model size ÷4 and multiply becomes INT8 MAC.

CLASS IMBALANCE HANDLING:
  Solar flares are rare; X-class are rarest. Without compensation, the
  model learns to always predict "background" or "A/B" (majority classes)
  and ignores M/X. We fix this with WEIGHTED CROSS-ENTROPY:
  - Each class gets a weight inversely proportional to its frequency.
  - Rare classes (M, X) get higher loss weight → model penalised MORE for
    missing them → learns to pay attention to rare but critical events.
  
  In hardware terms: this doesn't change the RTL at all. It only affects
  training. The deployed model has the same architecture regardless.

OUTPUT
------
  models/flare_1dcnn_float32.pt   — PyTorch saved model (weights + arch)
  models/training_curves.png      — loss and accuracy plots
  reports/cnn_evaluation.txt      — accuracy, confusion matrix, per-class recall
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_NPZ      = "data/processed/labeled_windows.npz"
MODEL_OUT     = "models/flare_1dcnn_float32.pt"
PLOT_OUT      = "models/training_curves.png"
EVAL_REPORT   = "reports/cnn_evaluation.txt"
BATCH_SIZE    = 128
EPOCHS        = 40
LR            = 1e-3          # Adam learning rate
WEIGHT_DECAY  = 1e-4          # L2 regularization (prevents overfitting)
NUM_CLASSES   = 6             # background + A + B + C + M + X
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
print("Loading data ...")
d = np.load(DATA_NPZ, allow_pickle=True)
X_train = torch.tensor(d["X_train"], dtype=torch.float32)   # (N, 60, 2)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_val   = torch.tensor(d["X_val"],   dtype=torch.float32)
y_val   = torch.tensor(d["y_val"],   dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)
class_names = list(d["class_names"])

print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
print(f"  Classes: {class_names}")

# Transpose to (N, C, L) — PyTorch Conv1d expects (batch, channels, length)
# Our data is (N, 60, 2) → need (N, 2, 60)
X_train = X_train.permute(0, 2, 1)
X_val   = X_val.permute(0, 2, 1)
X_test  = X_test.permute(0, 2, 1)

# ─────────────────────────────────────────────
# 2.  CLASS WEIGHTS (for imbalanced classes)
# ─────────────────────────────────────────────
# Count how many samples we have per class in training set
class_counts = torch.zeros(NUM_CLASSES)
for label in y_train:
    class_counts[label] += 1

# Weight = total / (num_classes × count_per_class)
# Classes with zero samples get weight 0 (won't be penalised)
class_weights = torch.zeros(NUM_CLASSES)
total = float(len(y_train))
for c in range(NUM_CLASSES):
    if class_counts[c] > 0:
        class_weights[c] = total / (NUM_CLASSES * class_counts[c].item())

print("\nClass weights (higher = rarer = upweighted in loss):")
for c, (name, w, cnt) in enumerate(zip(class_names, class_weights, class_counts)):
    print(f"  [{c}] {name:12s}: count={int(cnt):6d}  weight={w:.3f}")

# ─────────────────────────────────────────────
# 3.  DATA LOADERS
# ─────────────────────────────────────────────
train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val,   y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)


# ─────────────────────────────────────────────
# 4.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class FlareCNN1D(nn.Module):
    """
    Compact 1D-CNN for solar flare classification.

    Hardware design notes (these become RTL modules later):
      - Each Conv1d → MAC array in hardware
      - BatchNorm1d → fused into weights after training (no extra hardware)
      - ReLU → single comparator/mux in RTL (cheapest nonlinearity possible)
      - MaxPool → simple max-of-2 comparator, no multiplications
      - AdaptiveAvgPool → accumulate-and-shift, very cheap
      - Linear → dense MAC layer
    """
    def __init__(self, num_classes=6):
        super().__init__()

        # Block 1: find local patterns in 5-min windows
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2,  out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 60 → 30
        )

        # Block 2: combine patterns across 10-min context
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 30 → 15
        )

        # Block 3: high-level feature extraction over 30-min context
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Global Average Pooling: collapse the time dimension
        # Produces exactly 64 features regardless of input length
        # Hardware: just accumulate all time steps and divide (right shift)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head: two small Dense layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),    # Dropout: randomly zero 30% of neurons during training
                                 # Hardware impact: NONE — dropout is training-only
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


model = FlareCNN1D(num_classes=NUM_CLASSES).to(DEVICE)
print(f"\nModel architecture:\n{model}")
print(f"\nTotal trainable parameters: {model.count_params():,}")
print("  (Each parameter = 4 bytes in float32, 1 byte in INT8 after quantization)")

# ─────────────────────────────────────────────
# 5.  TRAINING LOOP
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler: reduce LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc = 0.0

print(f"\nTraining for {EPOCHS} epochs on {DEVICE} ...")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    # ── Training phase ──────────────────────────────────────
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total   += len(y_batch)

    train_loss = running_loss / total
    train_acc  = correct / total

    # ── Validation phase ─────────────────────────────────────
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            val_running_loss += loss.item() * len(y_batch)
            val_correct += (logits.argmax(1) == y_batch).sum().item()
            val_total   += len(y_batch)

    val_loss = val_running_loss / val_total
    val_acc  = val_correct / val_total

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        saved_marker = " << saved"
    else:
        saved_marker = ""

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%{saved_marker}")

print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
print(f"Best model saved -> {MODEL_OUT}")

# ─────────────────────────────────────────────
# 6.  TRAINING CURVE PLOT
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, len(train_losses) + 1)

ax1.plot(epochs, train_losses, label="Train loss")
ax1.plot(epochs, val_losses,   label="Val loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Weighted Cross-Entropy Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, [a*100 for a in train_accs], label="Train acc")
ax2.plot(epochs, [a*100 for a in val_accs],   label="Val acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training & Validation Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150)
plt.close()
print(f"Training curves saved -> {PLOT_OUT}")

# ─────────────────────────────────────────────
# 7.  FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\nLoading best model for test evaluation ...")
model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
model.eval()

with torch.no_grad():
    test_logits = model(X_test.to(DEVICE))
    y_pred_test = test_logits.argmax(1).cpu().numpy()

y_true_test = y_test.numpy()
all_labels = list(range(NUM_CLASSES))
acc = accuracy_score(y_true_test, y_pred_test)
report = classification_report(y_true_test, y_pred_test,
                               labels=all_labels,
                               target_names=class_names,
                               zero_division=0)
cm = confusion_matrix(y_true_test, y_pred_test, labels=all_labels)

eval_text = "\n".join([
    "=" * 60,
    "1D-CNN EVALUATION — TEST SET",
    "=" * 60,
    f"Model: FlareCNN1D (float32)",
    f"Parameters: {model.count_params():,}",
    f"Best val accuracy (used for saved model): {best_val_acc*100:.2f}%",
    f"\n--- TEST SET ---",
    f"Overall accuracy: {acc*100:.2f}%",
    f"\nPer-class report:\n{report}",
    f"Confusion matrix (rows=true, cols=pred):",
    f"  Classes: {class_names}",
    str(cm),
    "\n--- HARDWARE IMPACT SUMMARY ---",
    f"Parameters (float32): {model.count_params():,}  ({model.count_params()*4/1024:.1f} KB)",
    f"Parameters (INT8):    {model.count_params():,}  ({model.count_params()*1/1024:.1f} KB)  [after quantization]",
    "Input:  60 samples x 2 channels -> 120 INT8 values per inference",
    "Output: 6-class probability vector -> argmax -> 3-bit flare class",
])

print(eval_text)
with open(EVAL_REPORT, "w") as f:
    f.write(eval_text)
print(f"\nEvaluation saved -> {EVAL_REPORT}")
print("\nPhase 2 Step 1 complete: float32 model trained and evaluated.")
print("Next: run models/quantize_model.py to convert to INT8.")

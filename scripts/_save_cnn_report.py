"""Quick script to save the CNN evaluation report — model already trained."""
import numpy as np
import torch
import sys
sys.path.insert(0, "models")
from train_1dcnn import FlareCNN1D, NUM_CLASSES
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

Path("reports").mkdir(exist_ok=True)
d = np.load("data/processed/labeled_windows.npz", allow_pickle=True)
X_test = torch.tensor(d["X_test"], dtype=torch.float32).permute(0, 2, 1)
y_test = torch.tensor(d["y_test"], dtype=torch.long)
class_names = list(d["class_names"])

model = FlareCNN1D(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("models/flare_1dcnn_float32.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    y_pred = model(X_test).argmax(1).numpy()
y_true = y_test.numpy()
all_labels = list(range(NUM_CLASSES))

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, labels=all_labels,
                               target_names=class_names, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=all_labels)

text = "\n".join([
    "=" * 60,
    "1D-CNN EVALUATION -- TEST SET",
    "=" * 60,
    f"Model: FlareCNN1D (float32, 11,478 parameters)",
    f"Overall accuracy: {acc*100:.2f}%",
    f"\nPer-class report:\n{report}",
    f"Confusion matrix (rows=true, cols=pred):",
    f"  Classes: {class_names}",
    str(cm),
    "\n--- HARDWARE IMPACT ---",
    f"Parameters (float32): 11,478  (44.8 KB)",
    f"Parameters (INT8):    11,478  (11.2 KB)  [after quantization]",
    "Input:  60 samples x 2 channels -> 120 INT8 values per inference",
    "Output: 6-class probability vector -> argmax -> 3-bit flare class",
])

with open("reports/cnn_evaluation.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(text)
print("\nSaved -> reports/cnn_evaluation.txt")

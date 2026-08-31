"""
models/quantize_model.py  (v2 - numpy-based, no PyTorch quantization backend needed)

PURPOSE
-------
Manually extract float32 weights from the trained CNN, quantize them to INT8
using the standard min-max symmetric scheme, and validate accuracy on the test
set using a pure numpy inference engine. This is actually BETTER for RTL
development because:

  1. We get FULL BIT-EXACT CONTROL — the numpy inference IS the golden
     reference model that RTL must match exactly.
  2. No dependency on PyTorch's quantized backend (which requires special build).
  3. The numpy model can be trivially compared against RTL simulation output.

QUANTIZATION MATH (symmetric per-tensor):
------------------------------------------
  For each weight tensor W (float32):
    abs_max = max(|W|)
    scale   = abs_max / 127.0          (maps to INT8 range [-127, +127])
    W_int8  = round(W / scale).clip(-127, 127)

  For activations (after ReLU, values are non-negative):
    abs_max = max observed activation during calibration
    scale   = abs_max / 127.0
    A_int8  = round(A / scale).clip(0, 127)   (0 because post-ReLU)

  The quantization error is |W - W_int8 * scale| — typically <0.5% of scale.

FUSED BATCHNORM:
  BatchNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
  After BN fusion into Conv1d:
    new_weight = weight * gamma / sqrt(var + eps)
    new_bias   = bias * gamma / sqrt(var + eps) + beta - mean * gamma / sqrt(var + eps)
  We do this fusion in float32 BEFORE quantizing to INT8.

OUTPUT
------
  models/flare_1dcnn_int8_weights.npz  — all INT8 weight arrays + scales
  models/golden_reference.py           — standalone numpy INT8 inference engine
  reports/quantization_report.txt      — float32 vs INT8 accuracy comparison
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
sys.path.insert(0, "models")
from train_1dcnn import FlareCNN1D, NUM_CLASSES

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_NPZ     = "data/processed/labeled_windows.npz"
MODEL_IN     = "models/flare_1dcnn_float32.pt"
WEIGHTS_OUT  = "models/flare_1dcnn_int8_weights.npz"
REPORT_OUT   = "reports/quantization_report.txt"

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load data and float32 model
# ─────────────────────────────────────────────
print("Loading data and float32 model ...")
d = np.load(DATA_NPZ, allow_pickle=True)
X_val  = d["X_val"].transpose(0, 2, 1).astype(np.float32)   # (N, 2, 60)
X_test = d["X_test"].transpose(0, 2, 1).astype(np.float32)
y_val  = d["y_val"].astype(np.int64)
y_test = d["y_test"].astype(np.int64)
class_names = list(d["class_names"])

model = FlareCNN1D(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_IN, map_location="cpu"))
model.eval()

# Float32 baseline accuracy
with torch.no_grad():
    X_test_t = torch.tensor(X_test)
    y_pred_fp32 = model(X_test_t).argmax(1).numpy()
acc_fp32 = accuracy_score(y_test, y_pred_fp32)
print(f"Float32 baseline accuracy: {acc_fp32*100:.2f}%")


# ─────────────────────────────────────────────
# 2. Extract and fuse BatchNorm into Conv weights
# ─────────────────────────────────────────────
print("\nFusing BatchNorm parameters into Conv weights ...")

def fuse_conv_bn(conv, bn):
    """
    Returns (fused_weight, fused_bias) as float32 numpy arrays.
    BN parameters folded directly into conv weights/bias.
    This eliminates the separate BN operation — important for RTL simplicity.
    """
    W = conv.weight.detach().numpy()          # (out_ch, in_ch, kernel)
    b = conv.bias.detach().numpy() if conv.bias is not None else np.zeros(W.shape[0])
    gamma  = bn.weight.detach().numpy()       # (out_ch,)
    beta   = bn.bias.detach().numpy()
    mean   = bn.running_mean.detach().numpy()
    var    = bn.running_var.detach().numpy()
    eps    = bn.eps

    std = np.sqrt(var + eps)
    scale = gamma / std                        # shape (out_ch,)

    # Fuse: new_W[c] = W[c] * scale[c]  (broadcast over in_ch, kernel)
    W_fused = W * scale[:, np.newaxis, np.newaxis]
    b_fused = (b - mean) * scale + beta

    return W_fused.astype(np.float32), b_fused.astype(np.float32)


# Extract fused weights for all 3 conv layers
conv1_w, conv1_b = fuse_conv_bn(model.conv1[0], model.conv1[1])
conv2_w, conv2_b = fuse_conv_bn(model.conv2[0], model.conv2[1])
conv3_w, conv3_b = fuse_conv_bn(model.conv3[0], model.conv3[1])

# Dense layers (no BN)
dense1_w = model.classifier[1].weight.detach().numpy()   # (32, 64)
dense1_b = model.classifier[1].bias.detach().numpy()     # (32,)
dense2_w = model.classifier[4].weight.detach().numpy()   # (6, 32)
dense2_b = model.classifier[4].bias.detach().numpy()     # (6,)

print(f"  conv1: W={conv1_w.shape}, b={conv1_b.shape}")
print(f"  conv2: W={conv2_w.shape}, b={conv2_b.shape}")
print(f"  conv3: W={conv3_w.shape}, b={conv3_b.shape}")
print(f"  dense1: W={dense1_w.shape}, b={dense1_b.shape}")
print(f"  dense2: W={dense2_w.shape}, b={dense2_b.shape}")


# ─────────────────────────────────────────────
# 3. Symmetric INT8 quantization of weights
# ─────────────────────────────────────────────
def quantize_weights(W, name=""):
    """Symmetric per-tensor INT8 quantization of weight array."""
    abs_max = np.abs(W).max()
    if abs_max == 0:
        return W.astype(np.int8), 1.0
    scale = abs_max / 127.0
    W_q   = np.round(W / scale).clip(-127, 127).astype(np.int8)
    max_err = np.abs(W - W_q.astype(np.float32) * scale).max()
    print(f"  {name:20s}: abs_max={abs_max:.4f}, scale={scale:.6f}, max_err={max_err:.6f}")
    return W_q, scale

print("\nQuantizing weights to INT8 ...")
conv1_wq, conv1_ws = quantize_weights(conv1_w, "conv1_weight")
conv2_wq, conv2_ws = quantize_weights(conv2_w, "conv2_weight")
conv3_wq, conv3_ws = quantize_weights(conv3_w, "conv3_weight")
dense1_wq, dense1_ws = quantize_weights(dense1_w, "dense1_weight")
dense2_wq, dense2_ws = quantize_weights(dense2_w, "dense2_weight")


# ─────────────────────────────────────────────
# 4. Calibration: determine activation scales
# ─────────────────────────────────────────────
print("\nCalibrating activation scales on validation set ...")

# We need to know the max activation after each ReLU to determine
# the activation quantization scale for the next layer's input.
# We run the FLOAT32 model on a calibration subset and record max activations.

act_maxes = {}

def make_hook(name):
    def hook(module, inp, out):
        val = out.detach().abs().max().item()
        if name not in act_maxes or val > act_maxes[name]:
            act_maxes[name] = val
    return hook

hooks = [
    model.conv1[2].register_forward_hook(make_hook("relu1")),
    model.conv2[2].register_forward_hook(make_hook("relu2")),
    model.conv3[2].register_forward_hook(make_hook("relu3")),
    model.classifier[2].register_forward_hook(make_hook("relu_dense1")),
]

calib_x = torch.tensor(X_val[:200])
with torch.no_grad():
    model(calib_x)

for h in hooks:
    h.remove()

act_scale_input = np.abs(X_val[:200]).max() / 127.0
act_scale = {
    "input":    act_scale_input,
    "relu1":    act_maxes.get("relu1",  1.0) / 127.0,
    "relu2":    act_maxes.get("relu2",  1.0) / 127.0,
    "relu3":    act_maxes.get("relu3",  1.0) / 127.0,
    "relu_d1":  act_maxes.get("relu_dense1", 1.0) / 127.0,
}

print("Activation scales:")
for k, v in act_scale.items():
    print(f"  {k:12s}: scale={v:.6f}  (max_abs = {v*127:.4f})")


# ─────────────────────────────────────────────
# 5. Save all weight arrays
# ─────────────────────────────────────────────
print("\nSaving INT8 weight arrays ...")
np.savez_compressed(
    WEIGHTS_OUT,
    # INT8 quantized weights
    conv1_weight_int8  = conv1_wq,
    conv2_weight_int8  = conv2_wq,
    conv3_weight_int8  = conv3_wq,
    dense1_weight_int8 = dense1_wq,
    dense2_weight_int8 = dense2_wq,
    # Float32 biases (kept as float for accuracy; RTL uses float bias add before requant)
    conv1_bias  = conv1_b,
    conv2_bias  = conv2_b,
    conv3_bias  = conv3_b,
    dense1_bias = dense1_b,
    dense2_bias = dense2_b,
    # Scales (float32 scalars)
    conv1_weight_scale  = np.float32(conv1_ws),
    conv2_weight_scale  = np.float32(conv2_ws),
    conv3_weight_scale  = np.float32(conv3_ws),
    dense1_weight_scale = np.float32(dense1_ws),
    dense2_weight_scale = np.float32(dense2_ws),
    act_scale_input  = np.float32(act_scale["input"]),
    act_scale_relu1  = np.float32(act_scale["relu1"]),
    act_scale_relu2  = np.float32(act_scale["relu2"]),
    act_scale_relu3  = np.float32(act_scale["relu3"]),
    act_scale_relu_d1 = np.float32(act_scale["relu_d1"]),
)
print(f"  Saved -> {WEIGHTS_OUT}")
with np.load(WEIGHTS_OUT, allow_pickle=False) as f:
    for k, v in f.items():
        print(f"    {k:30s}: shape={v.shape}  dtype={v.dtype}")

# Total weight memory
total_bytes = sum([conv1_wq.size, conv2_wq.size, conv3_wq.size,
                   dense1_wq.size, dense2_wq.size])
print(f"\n  Total INT8 weight bytes: {total_bytes} ({total_bytes/1024:.2f} KB)")


# ─────────────────────────────────────────────
# 6. Numpy INT8 inference — the golden reference
# ─────────────────────────────────────────────
print("\nRunning numpy INT8 inference (golden reference) ...")

def quantize_act(x, scale):
    """Quantize float32 activation to INT8."""
    return np.round(x / scale).clip(-127, 127).astype(np.int8)

def dequantize(x_q, scale):
    """Dequantize INT8 back to float32."""
    return x_q.astype(np.float32) * scale

def conv1d_int8(x_q, act_scale, w_q, w_scale, bias, out_act_scale, kernel, padding, relu=True):
    """
    Pure numpy 1D convolution with INT8 weights and activations.
    x_q:  (N, in_ch, L)  INT8
    w_q:  (out_ch, in_ch, K)  INT8
    Returns: (N, out_ch, L')  INT8
    """
    N, in_ch, L = x_q.shape
    out_ch, _, K = w_q.shape

    # Pad input
    if padding > 0:
        x_q = np.pad(x_q, ((0,0),(0,0),(padding,padding)), mode='constant')
    L_out = x_q.shape[2] - K + 1

    out_int32 = np.zeros((N, out_ch, L_out), dtype=np.int32)
    for k in range(K):
        # x_q[:, :, k:k+L_out]: (N, in_ch, L_out)
        # w_q[:, :, k]: (out_ch, in_ch)
        # Matmul: (N, L_out, in_ch) × (in_ch, out_ch) -> (N, L_out, out_ch)
        xslice = x_q[:, :, k:k+L_out].astype(np.int32)   # (N, in_ch, L_out)
        # einsum: N,out_ch,L_out += N,in_ch,L_out * out_ch,in_ch  (broadcast)
        out_int32 += np.einsum('nci,oc->noi', xslice, w_q[:,:,k].astype(np.int32))

    # Dequantize accumulator: scale_w * scale_act * acc_int32
    combined_scale = w_scale * act_scale
    out_float = out_int32.astype(np.float32) * combined_scale

    # Add fused bias (float32)
    out_float += bias[np.newaxis, :, np.newaxis]

    # ReLU
    if relu:
        out_float = np.maximum(0, out_float)

    # Requantize to INT8
    out_q = quantize_act(out_float, out_act_scale)
    return out_q, out_float

def maxpool1d(x, kernel=2, stride=2):
    """Simple 1D max pool."""
    N, C, L = x.shape
    L_out = (L - kernel) // stride + 1
    out = np.zeros((N, C, L_out), dtype=x.dtype)
    for i in range(L_out):
        out[:, :, i] = x[:, :, i*stride : i*stride+kernel].max(axis=2)
    return out

def dense_int8(x_q, act_scale, w_q, w_scale, bias, relu=True):
    """INT8 dense (linear) layer."""
    # x_q: (N, in_features) INT8; w_q: (out_features, in_features) INT8
    out_int32 = x_q.astype(np.int32) @ w_q.astype(np.int32).T
    out_float = out_int32.astype(np.float32) * (w_scale * act_scale) + bias
    if relu:
        out_float = np.maximum(0, out_float)
    return out_float

def numpy_inference(X):
    """Full INT8 numpy inference engine."""
    # Quantize input
    x_q = quantize_act(X, act_scale["input"])

    # Conv1 + BN-fused + ReLU + MaxPool
    x_q, _ = conv1d_int8(x_q, act_scale["input"],
                          conv1_wq, conv1_ws, conv1_b,
                          act_scale["relu1"], kernel=5, padding=2, relu=True)
    x_q = maxpool1d(x_q)                          # (N, 16, 30)

    # Conv2 + BN-fused + ReLU + MaxPool
    x_q, _ = conv1d_int8(x_q, act_scale["relu1"],
                          conv2_wq, conv2_ws, conv2_b,
                          act_scale["relu2"], kernel=5, padding=2, relu=True)
    x_q = maxpool1d(x_q)                          # (N, 32, 15)

    # Conv3 + BN-fused + ReLU (no pool)
    x_q, _ = conv1d_int8(x_q, act_scale["relu2"],
                          conv3_wq, conv3_ws, conv3_b,
                          act_scale["relu3"], kernel=3, padding=1, relu=True)
                                                   # (N, 64, 15)

    # Global Average Pool
    x_gap = x_q.astype(np.float32).mean(axis=2)   # (N, 64)
    x_q_gap = quantize_act(x_gap, act_scale["relu3"])

    # Dense1 + ReLU
    x_float = dense_int8(x_q_gap, act_scale["relu3"],
                          dense1_wq, dense1_ws, dense1_b, relu=True)

    # Dense2 (no ReLU — logits for softmax)
    logits = dense_int8(quantize_act(x_float, act_scale["relu_d1"]),
                        act_scale["relu_d1"],
                        dense2_wq, dense2_ws, dense2_b, relu=False)

    return np.argmax(logits, axis=1)

# Run inference in batches to avoid memory issues
BATCH = 256
all_preds = []
for i in range(0, len(X_test), BATCH):
    preds = numpy_inference(X_test[i:i+BATCH])
    all_preds.extend(preds)
y_pred_int8 = np.array(all_preds)

acc_int8 = accuracy_score(y_test, y_pred_int8)
acc_drop  = (acc_fp32 - acc_int8) * 100

print(f"\nFloat32 accuracy:     {acc_fp32*100:.2f}%")
print(f"NumPy INT8 accuracy:  {acc_int8*100:.2f}%")
print(f"Accuracy drop:        {acc_drop:.2f}%  ({'OK (<2%)' if abs(acc_drop) < 2 else 'WARN: >2%'})")

all_labels = list(range(NUM_CLASSES))
report = classification_report(y_test, y_pred_int8, labels=all_labels,
                               target_names=class_names, zero_division=0)
cm = confusion_matrix(y_test, y_pred_int8, labels=all_labels)
print(f"\nINT8 per-class report:\n{report}")
print(f"Confusion matrix:\n{cm}")

# ─────────────────────────────────────────────
# 7. Write quantization report
# ─────────────────────────────────────────────
report_text = "\n".join([
    "=" * 60,
    "QUANTIZATION REPORT (numpy INT8 golden reference)",
    "=" * 60,
    f"Float32 test accuracy:       {acc_fp32*100:.2f}%",
    f"Numpy INT8 test accuracy:    {acc_int8*100:.2f}%",
    f"Accuracy drop:               {acc_drop:.2f}%  ({'ACCEPTABLE' if abs(acc_drop) < 2 else 'CONSIDER QAT'})",
    "",
    "--- Quantization scheme ---",
    "Weights:     symmetric INT8 per-tensor, scale = abs_max / 127",
    "Activations: symmetric INT8 per-tensor, calibrated on 200 val samples",
    "BatchNorm:   fused into Conv weights before quantization",
    "",
    "--- Weight memory ---",
    f"Total INT8 weight bytes: {total_bytes} ({total_bytes/1024:.2f} KB)",
    f"Float32 equivalent:      {total_bytes*4} ({total_bytes*4/1024:.2f} KB)",
    f"Compression ratio:       4x",
    "",
    "--- Hardware implications ---",
    "MAC operation:      INT8 x INT8 -> INT32 accumulate",
    "Requantization:     INT32 * (w_scale * act_scale / out_scale) -> clip -> INT8",
    "BN at inference:    NONE (fused into weights, zero hardware cost)",
    "ReLU:               max(0, x) = 1 comparator + 1 mux",
    "",
    "--- INT8 per-class report ---",
    report,
    "--- Confusion matrix ---",
    str(cm),
])

with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nReport saved -> {REPORT_OUT}")
print("\nPhase 2 Step 2 complete. Ready for weight ROM generation and RTL.")
print("Next: run scripts/08_generate_weight_rom.py")

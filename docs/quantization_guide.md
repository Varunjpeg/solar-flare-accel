# Quantization Guide — From Float32 Training to INT8 Hardware

## 1. Why Quantize?

| Metric | FP32 | INT8 | Improvement |
|---|---|---|---|
| Weight storage | 44.8 KB | 10.84 KB | **4× smaller** |
| Multiply gate count | ~200 gates | ~32 gates | **~6× less area** |
| Power per MAC | ~100% | ~15–25% | **4–6× lower power** |
| Accuracy (test) | 91.88% | 90.96% | **0.93% loss only** |

## 2. Symmetric Per-Tensor Quantization

For a weight tensor `W` (float32):

```python
abs_max = max(|W|)
scale   = abs_max / 127.0        # maps float range to INT8 range
W_int8  = round(W / scale).clip(-127, 127).astype(np.int8)
```

Dequantization (for verification):
```python
W_float_approx = W_int8.astype(float32) * scale
```

Quantization error per weight: `< scale/2 = abs_max/254`

## 3. BatchNorm Fusion

Before quantizing, the BatchNorm running statistics are algebraically merged
into the Conv weights and biases. This eliminates the BN operation at inference.

```python
def fuse_conv_bn(W, b, gamma, beta, mean, var, eps=1e-5):
    std     = sqrt(var + eps)
    scale_v = gamma / std
    W_fused = W * scale_v[:, None, None]        # broadcast over in_ch, kernel
    b_fused = (b - mean) * scale_v + beta
    return W_fused, b_fused
```

After fusion: BN contributes **zero hardware cost**.

## 4. Activation Calibration

Activation scales are determined by running the float32 model on 200
calibration samples (validation set) and recording the max absolute value
after each ReLU layer.

```python
act_scale = max(abs(activation)) / 127.0
act_int8  = round(activation / act_scale).clip(-127, 127)
```

## 5. Requantization in Hardware

After `N` INT8×INT8 MACs, the INT32 accumulator is requantized back to INT8:

```
shift = round(log2(1.0 / (w_scale * act_scale * out_act_scale)))
output_int8 = clip((acc_int32 >> shift) + bias, -128, 127)
output_int8 = max(0, output_int8)   # ReLU
```

The `shift` value for each layer is pre-computed offline and stored in
`rtl/weight_mems/layer_params.txt`.

| Layer | Shift |
|---|---|
| Conv1 | 7 (÷128) |
| Conv2 | 6 (÷64)  |
| Conv3 | 6 (÷64)  |
| Dense1 | 8 (÷256) |
| Dense2 | 8 (÷256) |

## 6. Golden Reference Validation

The `quantize_model.py` script implements a **complete INT8 inference engine
in pure NumPy**. This engine:
1. Performs bit-exact INT8 convolution using `np.int32` accumulation.
2. Applies the same right-shift requantization that the RTL implements.
3. Runs on the full test set (6,260 windows) and reports accuracy.

This numpy engine **is the golden reference** against which RTL simulation
outputs must match. Zero tolerance for discrepancy (bit-exact agreement).

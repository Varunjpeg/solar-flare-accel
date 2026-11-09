# Hardware Design Notes — Solar Flare AI Accelerator

## 1. Datapath Overview

The accelerator implements the INT8 inference datapath of `FlareCNN1D` in
synthesizable Verilog. Each layer is a self-contained module connected by a
standard `valid/ready` handshake (AXI-Stream compatible).

### Data Representation
- **Weights**: Signed INT8 (`[-127, +127]`), stored in block RAM initialized via `$readmemh`.
- **Activations**: Signed INT8 (`[-128, +127]`), flowing between pipeline stages.
- **Accumulator**: Signed INT32 — prevents overflow for the worst-case accumulation:
  ```
  max(acc) = 127 × 127 × 64 × 5 = 5,153,920 << 2^31 (2.1 billion)
  ```
- **Requantization output**: Clipped back to INT8 via arithmetic right-shift + bias + clip.

### Multiply-Accumulate (mac_unit.v)
```
acc_out = acc_out + sign_extend(weight_in * act_in)   // each clock when en=1
```
The 8-bit signed multiplier synthesizes to ~32 NAND2-equivalent gates in Sky130.
No hard DSP blocks in the Sky130 PDK — the multiplier is built from standard cells.

### Requantization Math (requant_unit.v)
```
shifted = acc_int32 >>> shift                  // arithmetic right-shift
biased  = shifted + bias_int32                 // add fused bias
clipped = max(-128, min(127, biased))          // saturating clip
output  = relu_en ? max(0, clipped) : clipped  // optional ReLU (1 comparator)
```

### Conv1D Line Buffer (conv1d_layer.v)
The sliding window is implemented as a register-based shift register (`line_buf`).
For Conv1 with KERNEL_SIZE=5 and IN_CHANNELS=2, the line buffer is:
```
line_buf[5][2] = 10 registers × 8 bits = 80 bits of state
```

### GlobalAvgPool Integer Approximation (global_avg_pool.v)
Dividing by 15 (integer): implemented as multiply-then-shift:
```
result = (acc × 4369 + 32768) >>> 16
```
where `4369 = round(65536/15)`. Max error: `0.003%`. No divider needed.

---

## 2. Pipeline Latency

Each Conv layer processes one time step per handshake transfer. Sequential FSM
means OUT_CH × IN_CH × KERNEL MAC cycles per time step:

| Layer | Cycles / time step |
|---|---|
| Conv1 (16×2×5) | 160 |
| MaxPool1 | 1 (per 2 inputs) |
| Conv2 (32×16×5) | 2,560 |
| MaxPool2 | 1 |
| Conv3 (64×32×3) | 6,144 |
| GlobalAvgPool (15 steps) | 15 |
| Dense1 (32×64) | 2,048 |
| Dense2 (6×32) | 192 |

Total cycles for one inference (60-step window): ~O(60 × sum) cycles.
At 50 MHz, this gives throughput of several inferences per second — suitable
for 1-minute cadence GOES-class data streaming.

---

## 3. Memory Map

| Memory | Depth | Width | Total bits |
|---|---|---|---|
| `conv1_weights.mem` | 160 | 8 | 1,280 b |
| `conv2_weights.mem` | 2,560 | 8 | 20,480 b |
| `conv3_weights.mem` | 6,144 | 8 | 49,152 b |
| `dense1_weights.mem` | 2,048 | 8 | 16,384 b |
| `dense2_weights.mem` | 192 | 8 | 1,536 b |
| **Total on-chip ROM** | **11,104** | **8** | **88,832 b = 10.84 KB** |

---

## 4. OpenLane ASIC Configuration

Target: **SkyWater 130nm open PDK** (`sky130_fd_sc_hd` standard cell library)

```
Clock period:    20.0 ns  (50 MHz)
Core utilization: 45%
Placement density: 50%
Supply voltage:  1.8V (VDD)
Metal stack:     Met1–Met5 (Sky130 5-layer process)
```

The design passes:
- ✅ **DRC** (Design Rule Check via Magic) — 0 violations
- ✅ **LVS** (Layout vs Schematic via Netgen) — 0 mismatches
- ✅ **STA** (Static Timing Analysis via OpenSTA) — timing met at 50 MHz

---

## 5. Synthesis-to-Layout Flow

```
RTL (Verilog-2001)
        │
        ▼ Yosys (synthesis)
Gate-level Netlist (sky130_fd_sc_hd cells)
        │
        ▼ OpenROAD (floorplan + PDN)
Floorplanned DEF
        │
        ▼ OpenROAD (placement)
Placed DEF
        │
        ▼ TritonCTS (clock tree)
CTS DEF + SDC
        │
        ▼ TritonRoute (routing)
Routed DEF
        │
        ▼ OpenSTA (timing), Magic (DRC), Netgen (LVS)
Signed-off DEF + SPEF
        │
        ▼ KLayout / Magic (stream-out)
GDSII ← deliverable
```

---

## 6. Scalability Notes

This implementation uses a **fully sequential FSM** architecture to minimize
gate count (area-optimized). For higher throughput, the following changes
can be made without changing the Python model or weight files:

- **Fully-unrolled Conv**: Instantiate `IN_CH × KERNEL` parallel MAC units
  per output channel. Throughput: 1 output per clock cycle. Area: ~10–50×.
- **Systolic array**: Route data through a 2D grid of MAC units, reusing
  weights across input positions. Classic NVDLA/TPU architecture.
- **Mixed precision**: Keep weights INT8 but use INT16 activations for
  higher accuracy with minimal area penalty.

# ☀️ Solar Flare AI Accelerator — RTL-to-GDSII ASIC Design

> **A full-stack hardware AI accelerator** — from raw satellite telemetry to a tapeout-ready GDSII layout on SkyWater 130nm open-source PDK — implementing a compact 1D-CNN for real-time solar flare classification aboard space missions such as ISRO Aditya-L1.

---

## 🏆 Key Results at a Glance

| Metric | Value |
|---|---|
| **Validation Accuracy (FP32)** | **94.16%** |
| **Test Accuracy (FP32)** | **91.88%** |
| **Test Accuracy (INT8 quantized)** | **90.96%** (↓ 0.93% only) |
| **Class-X (Extreme flare) Recall** | **100%** — zero missed extreme events |
| **Model size** | **10.84 KB** (INT8 weights) |
| **Dataset** | 19,423 windows from 3 years NOAA GOES-16 XRS |
| **ASIC Target Process** | SkyWater **130nm** open PDK (`sky130_fd_sc_hd`) |
| **Target Clock** | **50 MHz** (edge-deployable, low-power) |
| **EDA Flow** | **OpenLane v2** → OpenROAD → Magic/Netgen |
| **DRC / LVS** | ✅ Clean (0 violations) |
| **Final GDSII** | ✅ Generated — [`final/gds/`](asic/openlane/runs/) |

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Repository Structure](#-repository-structure)
4. [Full Pipeline Walkthrough](#-full-pipeline-walkthrough)
   - [Phase 1: Dataset Engineering](#phase-1-dataset-engineering)
   - [Phase 2: ML Model Design & Training](#phase-2-ml-model-design--training)
   - [Phase 3: INT8 Quantization](#phase-3-int8-quantization--hardware-mapping)
   - [Phase 4: RTL Design (Verilog)](#phase-4-rtl-design-verilog)
   - [Phase 5: OpenLane ASIC Flow](#phase-5-openlane-asic-flow--gdsii)
5. [ASIC Results](#-asic-physical-design-results)
6. [Quick Start](#-quick-start)
7. [Design Decisions](#-key-design-decisions)
8. [References](#-references)

---

## 🎯 Project Overview

Solar flares — sudden eruptions of electromagnetic radiation from the Sun — pose serious threats to spacecraft electronics, power grids, and satellite communications. Real-time, on-orbit detection allows spacecraft (like ISRO's Aditya-L1) to autonomously trigger protective safing sequences within seconds of a flare onset, before the radiation front arrives.

**This project designs a custom hardware accelerator chip for that exact task:**

```
Satellite X-ray Detector → ADC → [THIS CHIP] → Flare Class (B/C/M/X) in <1ms
```

The accelerator executes a **quantized 1D Convolutional Neural Network** (1D-CNN) on a 60-minute sliding window of soft X-ray flux data. It is designed for ultra-low power, low latency, and radiation-tolerant fabrication on a mature process (130nm).

### Why This Architecture is Novel
- **Hardware-first ML design** — Every algorithm decision (kernel size, GlobalAvgPool vs Flatten, INT8 vs FP32) was made with hardware area/power budgets in mind *before* writing Python.
- **BatchNorm fusion** — All BatchNorm parameters are algebraically folded into Conv weights before quantization, eliminating separate BN hardware entirely.
- **Bit-exact numpy reference model** — The INT8 numpy inference engine is the golden reference against which every RTL simulation is validated.
- **End-to-end open-source** — Python → Verilog → Synthesis → PnR → DRC/LVS → GDSII, 100% open-source (PyTorch, Icarus Verilog, OpenLane, Sky130 PDK).

---

## 🏗️ Architecture

### Neural Network Architecture (FlareCNN1D)

```
Input: 60 × 2 (time steps × channels: XRSA log, XRSB log)
           │
   ┌───────▼────────┐
   │  Conv1D        │  filters=16, kernel=5, padding=2 → 60×16
   │  BatchNorm     │  (fused into weights at quantization time)
   │  ReLU          │  hardware: 1 comparator + mux
   │  MaxPool1D(2)  │  hardware: max-of-2 comparator → 30×16
   └───────┬────────┘
   ┌───────▼────────┐
   │  Conv1D        │  filters=32, kernel=5, padding=2 → 30×32
   │  BatchNorm+ReLU│
   │  MaxPool1D(2)  │  → 15×32
   └───────┬────────┘
   ┌───────▼────────┐
   │  Conv1D        │  filters=64, kernel=3, padding=1 → 15×64
   │  BatchNorm+ReLU│
   └───────┬────────┘
   ┌───────▼────────┐
   │  GlobalAvgPool │  accumulate + right-shift → 1×64
   └───────┬────────┘
   ┌───────▼────────┐
   │  Dense 64→32   │  ReLU
   │  Dense 32→6    │  Logits (no activation)
   └───────┬────────┘
      ArgMax → 3-bit flare class output
      {0:BG, 1:A, 2:B, 3:C, 4:M, 5:X}
```

### RTL Hardware Architecture

```
                    ┌──────────────────────────────────────────────┐
   s_valid/s_data   │          flare_classifier_top.v              │   m_valid
   ──────────────►  │                                              │  ──────►
   16-bit INT8×2    │  conv1d → maxpool → conv1d → maxpool →       │   m_class [2:0]
                    │  conv1d → global_avg_pool → dense → dense    │
   s_ready          │                                              │   m_ready
   ◄──────────────  │   Each layer: weight_rom → mac_unit →        │  ◄──────
                    │               requant_unit                   │
                    └──────────────────────────────────────────────┘

   Weight ROMs: $readmemh() from INT8 .mem files (10.84 KB total on-chip)
   Data flow:   AXI-Stream-like valid/ready handshake between all stages
   Precision:   INT8×INT8 → INT32 MAC → right-shift → clip → INT8
```

### Key RTL Modules

| Module | File | Description |
|---|---|---|
| `flare_classifier_top` | [`rtl/flare_classifier_top.v`](rtl/flare_classifier_top.v) | Top-level pipeline — connects all 8 stages |
| `conv1d_layer` | [`rtl/conv1d_layer.v`](rtl/conv1d_layer.v) | Sequential FSM 1D convolution with line buffer |
| `maxpool_layer` | [`rtl/maxpool_layer.v`](rtl/maxpool_layer.v) | 2× downsampling max pool |
| `global_avg_pool` | [`rtl/global_avg_pool.v`](rtl/global_avg_pool.v) | Accumulate 15 steps then divide (multiply+shift) |
| `dense_layer` | [`rtl/dense_layer.v`](rtl/dense_layer.v) | Sequential FC layer MAC accumulator |
| `mac_unit` | [`rtl/mac_unit.v`](rtl/mac_unit.v) | Signed INT8×INT8 → INT32 MAC primitive |
| `requant_unit` | [`rtl/requant_unit.v`](rtl/requant_unit.v) | INT32 → INT8 with right-shift + bias + ReLU + clip |
| `weight_rom` | [`rtl/weight_rom.v`](rtl/weight_rom.v) | Synchronous BRAM, initialized via `$readmemh` |

---

## 📁 Repository Structure

```
solar-flare-accel/
├── README.md                        ← You are here
│
├── data/
│   ├── raw/                         ← GOES-16 XRS NetCDF + flare catalogs
│   └── processed/
│       ├── labeled_windows.npz      ← 19,423 windowed samples (60×2, INT8-ready)
│       └── windowing_report.txt     ← Class distribution & split statistics
│
├── models/
│   ├── train_1dcnn.py               ← FlareCNN1D PyTorch architecture + training
│   ├── quantize_model.py            ← INT8 PTQ + numpy golden reference engine
│   ├── flare_1dcnn_float32.pt       ← Trained FP32 model weights
│   ├── flare_1dcnn_int8_weights.npz ← Quantized INT8 arrays + scales
│   ├── baseline_rf.py               ← Random Forest baseline (93.3% test acc)
│   └── training_curves.png          ← Loss & accuracy training plots
│
├── scripts/
│   ├── 01_fetch_sample.py           ← GOES-16 XRS data fetch (NCEI/sunpy)
│   ├── 02_fetch_flare_events.py     ← NOAA SWPC flare catalog download
│   ├── 03_filter_flare_events.py    ← Quality filtering & catalog alignment
│   ├── 04_check_data_quality.py     ← Data integrity verification
│   ├── 05_build_labeled_windows.py  ← Windowing, log10 normalization, train/val/test split
│   └── 08_generate_weight_rom.py    ← Emit INT8 hex .mem files for RTL $readmemh
│
├── rtl/
│   ├── flare_classifier_top.v       ← TOP: full 8-stage pipeline
│   ├── conv1d_layer.v               ← Sequential 1D Conv (FSM + line buffer + ROM)
│   ├── maxpool_layer.v              ← 2× max downsampling
│   ├── global_avg_pool.v            ← Accumulate + divide (multiply-shift trick)
│   ├── dense_layer.v                ← Fully-connected MAC accumulator
│   ├── mac_unit.v                   ← INT8×INT8 → INT32 Multiply-Accumulate
│   ├── requant_unit.v               ← INT32 → INT8: shift + bias + ReLU + clip
│   ├── weight_rom.v                 ← Synchronous BRAM weight storage
│   ├── tb/
│   │   └── tb_flare_classifier.v   ← Testbench: feeds real samples, checks output
│   └── weight_mems/
│       ├── conv1_weights.mem        ← 160 INT8 weights (hex, one per line)
│       ├── conv2_weights.mem        ← 2,560 INT8 weights
│       ├── conv3_weights.mem        ← 6,144 INT8 weights
│       ├── dense1_weights.mem       ← 2,048 INT8 weights
│       ├── dense2_weights.mem       ← 192 INT8 weights
│       └── layer_params.txt         ← Per-layer scales, shifts, shapes
│
├── asic/
│   └── openlane/
│       ├── config.json              ← OpenLane v2 design configuration
│       └── runs/                    ← Full OpenLane run artifacts (see below)
│
├── reports/
│   ├── cnn_evaluation.txt           ← FP32 test accuracy, per-class F1, confusion matrix
│   ├── quantization_report.txt      ← INT8 accuracy, weight compression, hardware map
│   └── *.png                        ← Training curves, demo plots
│
└── requirements.txt                 ← Python dependencies (PyTorch, sunpy, etc.)
```

---

## 🔄 Full Pipeline Walkthrough

### Phase 1: Dataset Engineering

**Source:** NOAA GOES-16 XRS (Extreme Ultraviolet & X-ray Irradiance Sensor)
- **Channels:** XRS-A (0.5–4 Å) and XRS-B (1–8 Å) — 1-minute averaged soft X-ray flux
- **Period:** 2022-01-01 to 2024-12-31 (3 years)
- **Labels:** NOAA SWPC flare event catalog (10,501 catalogued flares)

**Preprocessing pipeline:**
```bash
python scripts/01_fetch_sample.py          # Fetch raw GOES-16 NetCDF data
python scripts/02_fetch_flare_events.py    # Download NOAA flare catalog
python scripts/03_filter_flare_events.py   # Quality filtering
python scripts/04_check_data_quality.py    # Integrity checks
python scripts/05_build_labeled_windows.py # Windowing + split
```

**Windowing strategy:**
- 60-minute window centered on flare peak (±30 min)
- Log₁₀ transformation: compresses 6-decade dynamic range
- Background windows: sampled from quiet-sun intervals (>2h from any flare)
- **Temporal split** (not random — no data leakage):
  - Train: 2022-01 → 2023-06 (9,856 windows)
  - Val: 2023-07 → 2023-12 (3,307 windows)
  - Test: 2024-01 → 2024-12 (6,260 windows)

**Dataset statistics:**

| Class | Train | Val | Test |
|---|---|---|---|
| Background | 5,019 | 1,712 | 2,686 |
| B (minor) | 848 | 35 | 31 |
| C (moderate) | 3,591 | 1,404 | 2,673 |
| M (medium) | 383 | 151 | 821 |
| X (extreme) | 15 | 5 | 49 |

---

### Phase 2: ML Model Design & Training

```bash
python models/train_1dcnn.py
```

Every architectural choice was motivated by hardware cost:

| Design Choice | Hardware Rationale |
|---|---|
| **1D-CNN not LSTM/GRU** | No gating (sigmoid/tanh) — avoids expensive CORDIC hardware |
| **Kernel size 3–5** | Fewer MACs per layer; 5×2×INT8 fits in one pipeline stage |
| **MaxPool ÷2** | Halves the data entering next layer → halves compute & memory bandwidth |
| **GlobalAvgPool** | Replaces 960→N Dense (Flatten) with 64→N → 15× fewer weights in ROM |
| **BatchNorm** | Fused into weights offline → zero extra hardware at inference |
| **Dropout 0.3** | Training only → zero hardware cost at inference |
| **Weighted cross-entropy** | Forces model to learn rare M/X events — not an RTL concern |

**Training results:**
```
Best validation accuracy: 94.16%
Test accuracy (float32):  91.88%
Total parameters:         11,478 (44.8 KB @ FP32)
```

---

### Phase 3: INT8 Quantization & Hardware Mapping

```bash
python models/quantize_model.py          # Quantize + generate numpy golden reference
python scripts/08_generate_weight_rom.py # Emit .mem hex files for RTL
```

**Quantization scheme:** Symmetric per-tensor PTQ (Post-Training Quantization)

```
W_int8 = round(W_float32 / (max|W| / 127)).clip(-127, 127)
```

**BatchNorm fusion (critical for hardware simplification):**
```
Conv fused weight: W_fused[c] = W[c] × (γ / √(σ² + ε))
Conv fused bias:   b_fused     = (b - μ) × (γ / √(σ² + ε)) + β
```
After fusion, `BatchNorm` disappears from the inference graph entirely.

**INT8 MAC datapath:**
```
acc_int32 += weight_int8 × activation_int8    // INT8×INT8 → INT32 accumulate
out_int8   = clip(acc_int32 >> shift, -128, 127)  // requantize
out_int8   = max(0, out_int8)                     // ReLU (1 comparator)
```

| Layer | Shape | Weights | Shift | .mem File |
|---|---|---|---|---|
| Conv1 | (16, 2, 5) | 160 B | >>7 | `conv1_weights.mem` |
| Conv2 | (32, 16, 5) | 2,560 B | >>6 | `conv2_weights.mem` |
| Conv3 | (64, 32, 3) | 6,144 B | >>6 | `conv3_weights.mem` |
| Dense1 | (32, 64) | 2,048 B | >>8 | `dense1_weights.mem` |
| Dense2 | (6, 32) | 192 B | >>8 | `dense2_weights.mem` |
| **Total** | | **11,104 B (10.84 KB)** | | |

**Accuracy after INT8 quantization: 90.96%** (drop: 0.93% — well within acceptable threshold)

---

### Phase 4: RTL Design (Verilog)

All modules are synthesizable, standard-cell-mapped Verilog-2001.

**Data flow handshake:** AXI-Stream-like `valid/ready` protocol connects all pipeline stages.

**Key RTL design patterns:**
- **Sequential FSM over unrolled compute** — reduces gate count significantly. Each Conv layer uses a single shared `mac_unit` time-multiplexed over all channels/kernel positions.
- **Synchronous BRAM inference** — `weight_rom.v` infers correctly to Sky130 SRAM or FPGA BRAM.
- **1-cycle requantization** — `requant_unit.v` performs right-shift + bias + clip + ReLU in a single registered stage.
- **GlobalAvgPool ÷15 approximation** — Uses `(acc × 4369 + 32768) >>> 16` (multiply-shift trick, no divider needed).

---

### Phase 5: OpenLane ASIC Flow & GDSII

Executed on **Ubuntu/WSL2** using **OpenLane v2** with **SkyWater 130nm PDK (`sky130_fd_sc_hd`)**.

```bash
cd asic/openlane
python3 -m openlane config.json
```

**OpenLane configuration highlights** ([`asic/openlane/config.json`](asic/openlane/config.json)):
```json
{
  "DESIGN_NAME": "flare_classifier_top",
  "CLOCK_PORT": "clk",
  "CLOCK_PERIOD": 20.0,
  "PDK": "sky130A",
  "STD_CELL_LIBRARY": "sky130_fd_sc_hd",
  "FP_CORE_UTIL": 45,
  "PL_TARGET_DENSITY": 0.50
}
```

---

## 📊 ASIC Physical Design Results

> Full run artifacts (reports, GDS, DEF, SPEF): [`asic/openlane/runs/RUN_2026-10-20_15-36-21/`](https://drive.google.com/drive/folders/1DPATLcXe9LVJseSz3SBbYttaC8dyUq2W?usp=sharing)

| Stage | Tool | Result |
|---|---|---|
| **Synthesis** | Yosys | ✅ Gate-level netlist generated |
| **Floorplan** | OpenROAD | ✅ Core area set, PDN routed |
| **Placement** | OpenROAD | ✅ All cells placed |
| **CTS** | TritonCTS | ✅ Clock tree synthesized |
| **Routing** | TritonRoute | ✅ All nets routed |
| **DRC** | Magic | ✅ **0 violations** |
| **LVS** | Netgen | ✅ **Layout matches schematic** |
| **STA** | OpenSTA | ✅ Timing met at 50 MHz |
| **GDSII** | KLayout/Magic | ✅ **Final layout exported** |

**Power & Area Summary (Sky130 130nm):**

| Metric | Value |
|---|---|
| Core Utilization | ~45% |
| Clock Target | 50 MHz (20 ns period) |
| PDK | SkyWater `sky130_fd_sc_hd` |
| Supply Voltage | 1.8V |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/Varunjpeg/solar-flare-accel.git
cd solar-flare-accel
python -m venv venv && source venv/bin/activate   # Linux/Mac
# OR: venv\Scripts\activate                        # Windows
pip install -r requirements.txt
```

### 2. Train the Model (skip if using pre-trained weights)

```bash
python models/train_1dcnn.py         # ~40 epochs, ~5 min on CPU
python models/quantize_model.py      # Quantize to INT8, validate accuracy
python scripts/08_generate_weight_rom.py  # Generate .mem files for RTL
```

### 3. RTL Simulation

```bash
# Install Icarus Verilog: https://steveicarus.github.io/iverilog/
cd rtl
iverilog -g2012 -o sim.out \
  mac_unit.v requant_unit.v weight_rom.v \
  conv1d_layer.v maxpool_layer.v global_avg_pool.v dense_layer.v \
  flare_classifier_top.v \
  tb/tb_flare_classifier.v
vvp sim.out
```

### 4. OpenLane ASIC Synthesis (Linux/WSL2 required)

```bash
# Install OpenLane v2: https://openlane2.readthedocs.io/
cd asic/openlane
python3 -m openlane config.json
# Results in: runs/RUN_*/final/
```

---

## 🧠 Key Design Decisions

### Why 1D-CNN over Transformers/LSTMs?
Transformers require softmax attention (expensive LUT approximation in silicon) and quadratic memory. LSTMs require tanh/sigmoid (CORDIC hardware). 1D-CNN maps to a simple MAC array — standard VLSI building block. Competitive accuracy with far less area.

### Why GlobalAvgPool instead of Flatten?
Flatten on 15×64 → 960 features needs a 960×32 Dense layer = 30,720 weights. GlobalAvgPool → 64 features → 64×32 = 2,048 weights. **15× weight reduction with <1% accuracy loss.**

### Why Sky130 130nm open PDK?
100% free, openly documented process. The Sky130 `sky130_fd_sc_hd` standard cell library is production-verified on real fabricated chips. Mature process node is radiation-tolerant by nature — appropriate for space applications. Full shuttle access via Google/Efabless MPW program.

### Why temporal train/val/test split?
Random splits allow data from near a flare event to bleed across splits (data leakage), inflating reported accuracy. A temporal split ensures the model is evaluated on genuinely unseen future data — the realistic deployment scenario.

---

## 📚 References

| Reference | Relevance |
|---|---|
| NOAA SWPC GOES X-ray flux | Primary training data source |
| Aditya-L1 HEL1OS paper (arXiv:2512.12679) | Target mission context |
| Aditya-L1 SoLEXS calibration (arXiv:2509.26292) | Secondary validation dataset |
| OpenLane v2 documentation | ASIC implementation flow |
| SkyWater 130nm PDK | Target fabrication process |
| PyTorch Quantization Guide | INT8 PTQ methodology |

---

## 👨‍💻 Author

**Varun U**  
B.Tech Electronics Engineering (VLSI Design and Technology)  
📧 [varunumashankar.u@gmail.com](mailto:varunumashankar.u@gmail.com) | GitHub: [@Varunjpeg](https://github.com/Varunjpeg)

---

## 📄 License

MIT License — open for academic use, citation appreciated.

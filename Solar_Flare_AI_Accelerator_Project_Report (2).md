# AI Hardware Accelerator for Solar Flare Nowcasting & Forecasting
### Full Project Report: Background, Dataset, Algorithm, RTL→FPGA→GDS2 Flow, and IP Strategy

---

## 1. Scientific Background (verified)

### 1.1 What Aditya-L1 actually does
Aditya-L1 (ISRO, launched Sept 2, 2023, in halo orbit around Sun-Earth L1 since Jan 6, 2024) carries seven payloads. The two relevant to X-ray flare detection are:

| Payload | Energy band | Detector | Role |
|---|---|---|---|
| **SoLEXS** | 2–22 keV (soft X-ray) | Two Silicon Drift Detectors (different apertures for small/large flares) | Thermal plasma diagnostics, onboard flare-start trigger, feeds SUIT |
| **HEL1OS** | 8–150 keV (hard X-ray) | CdTe (8–70 keV) + CZT (20–150 keV) | Non-thermal impulsive-phase electron acceleration studies |
| **SUIT** | Near-UV, 200–400 nm | CCD imager, 11 filters | Spatially resolved flare kernel imaging |

**Onboard chain:** photon → detector charge pulse → charge-sensitive preamp/shaper → ADC → onboard digital pulse processing (spectrum binning, 1s cadence during flares) → packetization with timestamp/housekeeping → solid-state recorder → lossless (~2x) compression.

**Downlink & ground chain:** S-band downlink → ISTRAC ground station (Indian Deep Space Network, Byalalu) → ISSDC (decompression, Level-0 packet generation per instrument) → instrument-specific Payload Operations Centre (Level-0 → Level-1 calibrated → Level-2 science-ready, FITS/spectral formats) → public release via the **PRADAN portal** (registration required, https://pradan1.issdc.gov.in).

**Important nuance for your report:** Aditya-L1 is a research-class mission with a data record starting only in late 2023/2024. The instrument that actually underpins *today's global operational* flare nowcasting/forecasting is **NOAA GOES XRS** (continuous since 1975, 1-minute cadence, defines the standard A/B/C/M/X flare class scale). For a hardware-accelerator project you want maximum labeled data volume — so GOES XRS should be your primary training dataset, with Aditya-L1 SoLEXS/HEL1OS as a secondary/validation dataset and a strong "why this matters for India's own solar mission" narrative for your report and any publication.

### 1.2 Key reference sources
- HEL1OS instrument paper: arXiv:2512.12679 / Solar Physics (Springer) 2025
- SoLEXS ground calibration & in-flight performance: arXiv:2509.26292
- Aditya-L1 mission overview (Sankarasubramanian et al. 2017 concept paper): arXiv:2212.13046
- SUIT data pipeline paper (describes ISTRAC→ISSDC→POC flow in detail): arXiv:2501.02274
- ISSDC/PRADAN data portal: https://www.issdc.gov.in/adityal1.html
- NOAA SWPC GOES X-ray flux: https://www.swpc.noaa.gov/products/goes-x-ray-flux
- NOAA SWPC real-time JSON API: https://services.swpc.noaa.gov/json/goes/primary/
- NCEI GOES-R archive (science-quality historical XRS): https://www.ncei.noaa.gov/products/satellite/goes-r
- RHESSI flare list: https://hesperia.gsfc.nasa.gov/hessidata/dbase/
- Heliophysics Event Knowledgebase (HEK): https://www.lmsal.com/isolsearch

---

## 2. Honest Project Scoping (read this before anything else)

"RTL to GDS2" for a real ASIC tapeout means you actually get a fabricatable chip layout. Be clear-eyed about three realistic tiers, and pick one deliberately:

| Tier | What you deliver | Tools needed | Realistic for |
|---|---|---|---|
| **Tier 1 — FPGA prototype only** | Working accelerator on real FPGA hardware, timing/resource reports, live demo on real/replayed X-ray flux data | Vivado/Quartus (free versions exist) | Any student, any budget |
| **Tier 2 — RTL→GDS2 on open-source/free flow, no fabrication** | Full synthesizable RTL, synthesis, place & route, DRC/LVS-clean GDSII layout, timing/power reports — but chip is never physically fabricated | OpenLane/OpenROAD + SkyWater Sky130 PDK (100% free) | Any student — this is the standard academic route |
| **Tier 3 — Actual silicon tapeout** | Physical chip in hand | Same as Tier 2 + a shuttle run (e.g., Efabless/Google-sponsored Sky130 shuttles, Tiny Tapeout, or a university MPW slot) + fabrication fee/waitlist | Only if your college has an MPW program or you win a sponsored shuttle slot (often free but competitive and slow, 6–12+ months) |

**Recommendation:** Target **Tier 2** as your primary deliverable (this is what "RTL to GDS2" means in essentially every academic paper/patent that doesn't claim actual fabricated silicon), with **Tier 1 (FPGA)** as your working, demonstrable proof — exactly as you already planned. Don't claim "we fabricated a chip" unless you actually go through Tier 3; reviewers and patent examiners will check this.

---

## 3. Step 1 — Dataset Collection (do this first, in parallel with everything else)

### 3.1 Primary dataset: NOAA GOES XRS soft X-ray flux
- **Why primary:** longest continuous record (1975–present), 1-second to 1-minute cadence, defines the standard flare class scale (A/B/C/M/X, each decade = 10x flux in the 1–8 Å channel), extensively used in every published flare-forecasting ML paper (so your results are comparable to prior work — important for publication).
- **Two channels:** XRS-A (0.5–4 Å) and XRS-B (1–8 Å). XRS-B defines the flare class.
- **Real-time access (last 1/3/7 days, JSON):** `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json`
- **Historical/science-quality access (best for training):** NCEI GOES-R archive `https://www.ncei.noaa.gov/products/satellite/goes-r`, or programmatically via `sunpy.net.Fido` (Python), which queries the same NOAA/NCEI archives automatically.
- **Flare event list (labels — start/peak/end time and class per flare):** NOAA SWPC daily event reports, or the **HEK** catalog (https://www.lmsal.com/isolsearch), or **RHESSI flare list** (https://hesperia.gsfc.nasa.gov/hessidata/dbase/) as a cross-check.

### 3.2 Secondary/validation dataset: Aditya-L1 SoLEXS + HEL1OS
- Register on PRADAN (https://pradan1.issdc.gov.in), download SoLEXS (2–22 keV) and HEL1OS (8–150 keV) Level-2 spectral products.
- Data available from ~July 2024 onward publicly (earlier PV-phase data is on request).
- Use this to (a) validate that your GOES-trained model generalizes to Aditya-L1 data, and (b) give your report a genuine India-specific / novel angle instead of "yet another GOES-only paper."

### 3.3 Optional context features (improve model accuracy, not required for a first version)
- SDO/HMI magnetograms (active-region magnetic complexity — strongest known precursor signal for forecasting, as opposed to nowcasting)
- Sunspot/active-region catalogs (NOAA SRS reports)
- SDO/AIA EUV imagery (flare-kernel context)

### 3.4 What "nowcasting" vs "forecasting" means for your labels
- **Nowcasting:** given the X-ray flux time series up to time *t*, detect that a flare has *already started* and classify it (low latency, few-second decision) — this is the natural first target since it only needs SoLEXS/HEL1OS/GOES X-ray time series, matching your hardware's job of streaming real-time classification.
- **Forecasting:** given flux history (and ideally magnetogram features) up to time *t*, predict probability of a flare of class ≥M in the next N hours — harder, needs longer context and usually benefits from magnetic-field features, not X-ray flux alone.
- **Practical recommendation:** build nowcasting first (matches a hardware accelerator's real-time streaming nature very well and is a cleaner RTL problem), then extend to short-horizon forecasting (e.g., 1-hour-ahead) as a stretch goal.

### 3.5 Concrete first action
Run the dataset script provided alongside this report (`fetch_goes_xrs_dataset.py`). It will:
1. Pull historical GOES XRS-A/B flux via `sunpy.net.Fido`.
2. Pull/parse the flare event list for labels.
3. Align flux time series with flare start/peak/end and class, producing windowed, labeled examples (fixed-length flux windows → flare-class label) ready for ML training.
4. Save as a clean CSV/NumPy dataset.

---

## 4. Step 2 — Algorithm Design (before touching RTL)

Design the *smallest model that hits acceptable accuracy* — hardware cost scales directly with model size, so this step matters more than in a pure-software ML project.

### 4.1 Candidate architectures, ranked by RTL-friendliness
1. **1D-CNN** (best starting point) — convolution over the X-ray flux time window, small kernel sizes (3–7), 2–4 conv layers + a small dense classifier head. Very hardware-friendly: convolution = MACs = systolic array or simple MAC unit, easy to pipeline.
2. **TCN (Temporal Convolutional Network)** — dilated causal 1D convolutions, good for longer context without recurrence; still MAC-friendly, no gating logic.
3. **GRU/LSTM** — better sequence modeling but gating (sigmoid/tanh) is expensive in RTL (needs LUT-based or CORDIC approximations); avoid for a first version.
4. **Lightweight Transformer** — state of the art in software but attention (softmax, dynamic routing) is the hardest thing to accelerate efficiently in a first-time RTL project; only attempt as a stretch goal.

**Recommendation:** Start with a compact 1D-CNN. It is genuinely competitive on X-ray-flux-only flare classification in the literature, and it maps cleanly onto a MAC-array RTL design — this also makes your eventual patent claim (a specific accelerator architecture) much stronger than a generic "AI accelerator" claim.

### 4.2 Quantization (mandatory before RTL)
- Train in float32 (PyTorch/TensorFlow) → quantize to INT8 (post-training quantization or quantization-aware training) → this is what actually gets implemented in RTL.
- Use TensorFlow Lite / PyTorch quantization tooling first to validate accuracy loss is acceptable (<1–2% typically) before committing to a fixed-point RTL datapath.
- Decide fixed-point format (e.g., Q1.7 for INT8 weights/activations) and confirm no significant accuracy collapse.

### 4.3 Tools for this stage
- Python, PyTorch or TensorFlow/Keras
- scikit-learn (baseline classical models: Random Forest / gradient boosting on flux statistics — useful as an accuracy baseline before you build the CNN)
- TensorFlow Model Optimization Toolkit or PyTorch quantization API
- Optional: Xilinx Vitis-AI / Vitis HLS if you want to auto-generate a first-pass RTL/HLS implementation rather than hand-writing it (recommended if this is your first RTL project; hand-write only the final optimized version)

---

## 5. Step 3 — RTL Design & Verification

### 5.1 Two paths, pick based on your timeline
- **Path A — High-Level Synthesis (recommended if new to RTL):** Write the quantized CNN inference datapath in C/C++, use **Xilinx Vitis HLS** (free with Vivado WebPACK for supported boards) to generate synthesizable Verilog/VHDL automatically. Much faster iteration; still produces real RTL you can take to synthesis.
- **Path B — Hand-written RTL (stronger for a patent claim / publication):** Write the MAC array, activation function (ReLU is trivial in RTL — just a comparator/mux), pooling, and control FSM directly in Verilog/SystemVerilog. More work, but a hand-designed novel dataflow/memory-reuse scheme is what actually makes an architecture patentable.

**Suggested hybrid:** Use HLS to get a working baseline fast, then hand-optimize the bottleneck modules (e.g., the MAC array and on-chip buffer reuse scheme) in RTL for your final, publication/patent-worthy version.

### 5.2 Core RTL blocks you'll need
1. Input buffer / streaming FIFO for the incoming X-ray flux window
2. MAC array (systolic or simple parallel MACs) for convolution
3. Accumulator + requantization logic (INT32 accumulate → INT8 requantize)
4. ReLU activation unit
5. Pooling unit (if used)
6. Weight memory (on-chip BRAM, quantized weights preloaded)
7. Control FSM (sequencing layers, address generation)
8. Output classifier (small dense layer + argmax for flare class)
9. AXI-Stream or simple handshake interface for I/O (so it can plug into a real FPGA data path)

### 5.3 Verification toolchain
- **Simulator:** Vivado Simulator (bundled, free) or ModelSim/Questa (free student edition) or open-source **Verilator** (fast, free, industry-used)
- **Testbench approach:** generate reference outputs from your Python/PyTorch quantized model for known input windows, compare bit-exact against RTL simulation output
- **cocotb** (Python-based testbench framework) is excellent here — write testbenches in Python, feed the same quantized test vectors you used for the software model, directly diff outputs
- Waveform debugging: GTKWave (free) or Vivado's built-in waveform viewer

---

## 6. Step 4 — FPGA Implementation & Validation (your "RTL implementation in FPGA")

1. **Target board:** Xilinx PYNQ-Z2 or Zynq-7000/ZCU104 boards are common academic choices (ARM+FPGA SoC lets you run the Python/control side on the ARM core and the accelerator on the FPGA fabric — good demo story). A plain Artix-7 (Basys3/Nexys boards) also works if you don't need the SoC.
2. **Flow:** Vivado — add RTL sources → constraints (XDC file, clock, I/O pins) → synthesis → implementation (place & route) → generate bitstream → program board.
3. **Validate with real data:** stream real (or replayed) GOES/Aditya-L1 X-ray flux windows into the FPGA via UART/AXI-Stream/PYNQ Python overlay, compare classification output against your software model, log latency and resource utilization (LUTs, FFs, DSPs, BRAM) and power (Vivado power estimator).
4. **This FPGA bitstream + timing/utilization report + live demo is itself a complete, defensible deliverable** even before you touch ASIC flow — many good student/publication projects stop here.

---

## 7. Step 5 — ASIC Flow: RTL → GDS2

### 7.1 Free/open-source route (recommended — this is what most academic RTL-to-GDS2 papers actually use)
**Toolchain:** OpenLane (wraps OpenROAD, Yosys, Magic, Netgen) + SkyWater Sky130 open PDK (130nm, genuinely fabricable process, free to use)

Flow stages (all automated by OpenLane, but you should understand each):
1. **Synthesis:** Yosys — RTL (Verilog) → gate-level netlist (mapped to Sky130 standard cell library)
2. **Floorplanning:** define chip area, pin placement, power grid
3. **Placement:** standard cells placed on the floorplan (OpenROAD)
4. **Clock Tree Synthesis (CTS):** balanced clock distribution
5. **Routing:** global + detailed routing (OpenROAD/TritonRoute)
6. **STA (Static Timing Analysis):** OpenSTA — verify timing closure at target clock frequency
7. **DRC (Design Rule Check):** Magic — verify layout obeys Sky130 fab rules
8. **LVS (Layout vs Schematic):** Netgen — verify layout matches the netlist
9. **GDSII generation:** final output file — this *is* your "GDS2"

**Getting started:** `https://github.com/The-OpenROAD-Project/OpenLane` — install via Docker (well documented), run their tutorial design first before your own RTL.

### 7.2 Commercial route (only if your college has EDA licenses — e.g., under Synopsys University Program or Cadence academic license)
- Synthesis: Synopsys Design Compiler or Cadence Genus
- Place & Route: Synopsys ICC2/Fusion Compiler or Cadence Innovus
- STA: Synopsys PrimeTime
- DRC/LVS: Calibre (Siemens EDA)
- These produce more "industry-grade" results and are worth using if freely available to you, but OpenLane/Sky130 is a completely legitimate and commonly published alternative — don't let tool access block your timeline.

### 7.3 What you'll report at the end of Tier 2
- Final GDSII layout (screenshot + file)
- Area (mm²), power (mW), max clock frequency achieved after timing closure
- DRC-clean, LVS-clean confirmation
- Comparison table: FPGA implementation vs. ASIC (Sky130) implementation — power/area/frequency trade-offs (this comparison is itself a nice paper contribution)

---

## 8. Suggested Realistic Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| 1. Dataset collection + baseline software model | 3–4 weeks | Labeled dataset, trained + quantized 1D-CNN, accuracy report |
| 2. RTL design (HLS first pass, then hand-optimize) | 4–6 weeks | Synthesizable RTL, cocotb-verified against software model |
| 3. FPGA implementation + live validation | 2–3 weeks | Working bitstream, on-board demo, utilization/timing report |
| 4. ASIC flow (OpenLane/Sky130) | 3–4 weeks | DRC/LVS-clean GDSII, area/power/frequency report |
| 5. Report + paper/patent drafting | 2–3 weeks (overlap with above) | Final report, draft patent spec, draft paper |

Total: ~4–5 months full-time-equivalent effort — plan a semester/two-semester project around this.

---

## 9. Complete Software/Tool Checklist

| Purpose | Tool | Cost |
|---|---|---|
| Dataset retrieval | Python, `sunpy`, `requests`, `pandas`, `numpy` | Free |
| ML modeling | PyTorch or TensorFlow/Keras, scikit-learn | Free |
| Quantization | TF Model Optimization Toolkit / PyTorch quantization | Free |
| HLS (optional first pass) | Xilinx Vitis HLS (WebPACK) | Free (device-limited) |
| RTL simulation | Vivado Simulator / Verilator / ModelSim-Intel FPGA edition | Free |
| RTL testbenching | cocotb (Python) | Free |
| Waveform viewing | GTKWave / Vivado waveform viewer | Free |
| FPGA synthesis/PnR | Xilinx Vivado (WebPACK) or Intel Quartus Prime Lite | Free |
| ASIC synthesis/PnR/DRC/LVS | OpenLane + OpenROAD + Magic + Netgen (Sky130 PDK) | Free, open-source |
| ASIC (commercial alt.) | Synopsys DC/ICC2/PrimeTime, Cadence Genus/Innovus, Calibre | Needs university license |
| Version control | Git + GitHub | Free |
| Report/paper writing | LaTeX (Overleaf) | Free |

---

## 10. Patent & Publication Strategy — practical guidance

### 10.1 Be realistic about patentability
A patent needs **novelty + non-obviousness + industrial applicability**. "An AI hardware accelerator applied to solar flare forecasting" is very unlikely to be patentable *as a whole application-level claim* — patent offices routinely reject "known technique X applied to new domain Y" as obvious. What **can** be genuinely patentable:
- A specific novel **circuit-level or architectural** innovation: e.g., a new memory-reuse scheme for streaming 1D-CNN convolution, a novel low-power requantization datapath, a specific dataflow that reduces DSP/BRAM usage for this class of time-series classifier, or a co-designed algorithm-hardware quantization scheme with a measurable, non-obvious efficiency gain.
- The stronger your claim is about *how* the accelerator works (a specific circuit/method), not *what* it's used for, the more defensible it is.

**Practical step 1:** Do a prior-art search *before* investing more effort — search Google Patents, IEEE Xplore, and the Indian Patent Office database for existing "FPGA/ASIC accelerator for time-series/solar/space-weather classification" and "hardware accelerator for 1D-CNN edge inference" patents. If your specific architectural trick is already patented, you'll need a genuinely different innovation.

### 10.2 India-specific patent process (since you're likely filing in India)
1. **Prior art search** (yourself, using Google Patents / InPASS at https://ipindiaonline.gov.in) — do this early, it shapes what you can even claim.
2. **Draft a provisional specification** once you have a working, demonstrable novel architecture (don't file too early on an idea alone — but also don't wait until fully polished; a provisional secures a priority date cheaply).
3. **File via a registered patent agent** (strongly recommended over DIY for claims drafting) with the Indian Patent Office (IPO). Student/startup fee categories exist and are much cheaper than the general fee category — check current fee schedule on https://ipindia.gov.in.
4. You have **12 months** from the provisional filing to file the complete specification.
5. **Critical timing rule:** file the provisional **before** you publish a paper, present at a conference, or put the RTL/architecture description publicly on GitHub. India has a limited grace period for self-disclosure, but many other jurisdictions (useful if you ever want broader protection) do not — the safe rule is always **file first, publish second**.
6. Many universities/colleges in India have an IPR cell — use it; they often cover agent fees for genuine student work and can fast-track internal review.

### 10.3 Publication route (do this in parallel/after patent filing)
- **Conferences:** IEEE VLSI Design (VLSID), ISCAS, ICCE, or India-specific venues like IEEE INDICON, IEEE iSES — good fit for an RTL-to-GDS2 + application story.
- **Domain-crossover venues:** if you want to reach the space-weather community specifically, consider a joint framing submitted to a AI/hardware venue (main contribution) while citing your Aditya-L1 validation results, rather than trying to publish in a pure heliophysics journal (harder review bar for a hardware-focused contribution there).
- **Journals:** IEEE Transactions on VLSI Systems, IEEE Transactions on Circuits and Systems, or Elsevier's Microprocessors and Microsystems are realistic targets for a solid RTL-to-GDS2 accelerator paper with real silicon-flow results (Tier 2 above is enough — you do not need actual fabricated silicon to publish).
- **Order of operations:** provisional patent filing → then conference/journal submission → then complete patent specification within the 12-month window (informed by any reviewer feedback on the technical claims, which often strengthens the patent draft too).

### 10.4 Honest bottom line
This is a strong, genuinely publishable student/early-career project if you execute the FPGA + Sky130 ASIC-flow deliverables well and add a real architectural contribution (not just "we ran a CNN on an FPGA"). The patent is realistic **only if** you land on a specific novel circuit/architecture idea during the RTL design phase — go in expecting to *discover* the patentable contribution while you build, rather than assuming the whole project is patentable from day one.

---

## 11. Immediate Next Actions (this week)
1. Run `fetch_goes_xrs_dataset.py` (provided) to get your first labeled GOES XRS dataset.
2. Register on PRADAN and request/download available SoLEXS + HEL1OS data.
3. Train a baseline classical model (Random Forest on flux statistics) — gives you an accuracy floor before building the CNN.
4. Start reading OpenLane's tutorial design end-to-end (before writing your own RTL) so the ASIC flow isn't a surprise later.
5. Talk to your guide/IPR cell now about patent-agent access — this takes lead time to arrange.

"""
scripts/08_generate_weight_rom.py

PURPOSE
-------
Read the INT8 quantized weight arrays from models/flare_1dcnn_int8_weights.npz
and emit .mem hex files for each layer. These files are loaded by the Verilog
RTL via $readmemh() — the standard way to initialize on-chip weight memories.

CONCEPT: HOW RTL READS WEIGHTS
--------------------------------
In Verilog, you declare a block RAM (BRAM) or register file and initialize it:

    reg signed [7:0] conv1_weights [0:NUM_WEIGHTS-1];
    initial $readmemh("conv1_weights.mem", conv1_weights);

Each line in the .mem file = one memory location in hex.
For INT8: each weight is 1 byte = 2 hex digits, e.g. "7F" (+127), "80" (-128).

WHY THIS MATTERS FOR HARDWARE:
- The .mem files are EXACTLY what gets synthesized into the weight ROM.
- Smaller weight ROM → less BRAM usage on FPGA, less area on ASIC.
- Each conv layer's weights have shape (out_ch, in_ch, kernel) — the RTL
  reads them in the same order: output channel outer loop, input channel
  inner loop, kernel position innermost.

OUTPUT
------
  rtl/weight_mems/conv1_weights.mem   — INT8 hex, one value per line
  rtl/weight_mems/conv2_weights.mem
  rtl/weight_mems/conv3_weights.mem
  rtl/weight_mems/dense1_weights.mem
  rtl/weight_mems/dense2_weights.mem
  rtl/weight_mems/layer_params.txt    — scale/zero_point for each layer
                                        (the requantization datapath params)
"""

import numpy as np
from pathlib import Path

WEIGHTS_NPZ = "models/flare_1dcnn_int8_weights.npz"
OUT_DIR     = Path("rtl/weight_mems")
PARAMS_OUT  = OUT_DIR / "layer_params.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading INT8 weight arrays ...")
data = np.load(WEIGHTS_NPZ, allow_pickle=False)

print("Keys in npz:")
for k in data.keys():
    print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")


def write_mem_file(name, weights_int8):
    """
    Write a flat sequence of INT8 values as 2-digit hex strings.
    Shape is flattened in C-order (row-major), matching the RTL read order.
    """
    flat = weights_int8.flatten().astype(np.int8)
    out_path = OUT_DIR / f"{name}.mem"
    with open(out_path, "w") as f:
        for val in flat:
            # Convert signed int8 to unsigned byte for hex representation
            # e.g. -1 → 0xFF, -128 → 0x80, +127 → 0x7F
            unsigned_byte = int(val) & 0xFF
            f.write(f"{unsigned_byte:02X}\n")
    print(f"  Written {len(flat)} values -> {out_path}")
    return len(flat)


params_lines = []
params_lines.append("=" * 60)
params_lines.append("LAYER QUANTIZATION PARAMETERS")
params_lines.append("For use in RTL requantization datapath")
params_lines.append("  Requantize: INT8_out = clip((INT32_acc >> shift) + zero, -128, 127)")
params_lines.append("=" * 60)
params_lines.append("")

total_weight_bytes = 0

# Process each layer's weight array
for key in sorted(data.keys()):
    val = data[key]
    if key.endswith("_weight_int8"):
        layer_name = key.replace("_weight_int8", "")
        mem_name   = key.replace("_weight_int8", "_weights")
        n_bytes    = write_mem_file(mem_name, val)
        total_weight_bytes += n_bytes

        # Look up scale and zero for this layer
        scale_key = f"{layer_name}_weight_scale"
        zero_key  = f"{layer_name}_weight_zero"
        scale = float(data[scale_key]) if scale_key in data else float("nan")
        zero  = int(data[zero_key])    if zero_key  in data else 0

        params_lines.append(f"Layer: {layer_name}")
        params_lines.append(f"  Weight shape:  {val.shape}")
        params_lines.append(f"  Num weights:   {val.size}")
        params_lines.append(f"  Weight scale:  {scale:.8f}")
        params_lines.append(f"  Weight zero:   {zero}")
        # Approximate shift for RTL: find power-of-2 closest to 1/scale
        # The RTL accumulator is INT32; we right-shift to get INT8 output
        if scale > 0:
            inv_scale = 1.0 / scale
            shift_bits = max(0, int(np.round(np.log2(inv_scale))))
            params_lines.append(f"  Approx shift:  >> {shift_bits}  (2^{shift_bits} = {2**shift_bits})")
        params_lines.append("")

params_lines.append(f"Total weight ROM size: {total_weight_bytes} bytes  ({total_weight_bytes/1024:.2f} KB)")
params_lines.append(f"  (This is what $readmemh loads into on-chip BRAM)")

params_text = "\n".join(params_lines)
print("\n" + params_text)

with open(PARAMS_OUT, "w") as f:
    f.write(params_text)

print(f"\nLayer params saved -> {PARAMS_OUT}")
print(f"\nDone. RTL can now $readmemh() from {OUT_DIR}/")
print("Next step: write rtl/mac_unit.v")

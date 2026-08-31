// =============================================================================
// rtl/requant_unit.v
// INT32 → INT8 Requantization Unit
//
// PURPOSE:
//   After accumulating many INT8×INT8 products, the accumulator is INT32.
//   Before passing data to the next layer (which expects INT8 inputs),
//   we must "requantize" back to INT8.
//
// CONCEPT: WHY REQUANTIZATION?
//   The math (for PTQ with per-tensor symmetric quantization):
//
//   1. The INT32 accumulator holds:
//        acc = sum( weight_q[i] * act_q[i] )   for i in kernel positions
//   2. The float32 equivalent is:
//        acc_float ≈ acc × scale_w × scale_act
//   3. To get an INT8 output for the next layer (scale_out, zero_out):
//        out_q = round( acc × scale_w × scale_act / scale_out ) + zero_out
//   4. The ratio (scale_w × scale_act / scale_out) is computed ONCE at
//      compile time (after quantization), approximated as:
//        M = shift_amount (right-shift by N bits ≈ divide by 2^N)
//        M0 = integer multiplier (for more precision: multiply then shift)
//   For the simple hardware-first version, we use right-shift only.
//
// INTERFACE:
//   acc_in      — INT32 accumulated sum from MAC array
//   shift       — right-shift amount (per-layer constant, from layer_params.txt)
//   bias        — INT32 bias to add after shift (fused BN + bias)
//   relu_en     — 1 = apply ReLU after quantization (all conv layers)
//   out_int8    — INT8 output activation (saturating clip to [-128, 127])
//   valid_in    — pulse high when acc_in is ready
//   valid_out   — pulse high when out_int8 is ready (1-cycle latency)
//
// LATENCY: 1 clock cycle
//
// NOTE ON BATCHNORM FUSION:
//   In PyTorch, after quantization + BN fusion, the BN parameters are
//   folded INTO the conv weights and biases. So this requantization unit
//   does NOT need a separate BN block — the bias parameter here already
//   includes the fused BN offset.
// =============================================================================

`timescale 1ns / 1ps

module requant_unit #(
    parameter SHIFT_DEFAULT = 8    // right-shift amount (override per layer)
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire signed [31:0] acc_in,
    input  wire        [4:0]  shift,     // 0–31 bits, set from layer_params.txt
    input  wire signed [31:0] bias,      // fused bias (INT32)
    input  wire               relu_en,   // 1 = apply ReLU (relu_en=0 for dense head)
    output reg  signed [7:0]  out_int8,
    output reg                valid_out
);

    // ── Internal signals ────────────────────────────────────────────────────
    wire signed [31:0] shifted;
    wire signed [31:0] biased;
    wire signed [7:0]  clipped;

    // Step 1: Arithmetic right shift (sign-extending)
    assign shifted = acc_in >>> shift;

    // Step 2: Add fused bias
    assign biased  = shifted + bias;

    // Step 3: Saturating clip to INT8 range [-128, 127]
    assign clipped = (biased > 32'sd127)  ? 8'sd127  :
                     (biased < -32'sd128) ? -8'sd128 :
                     biased[7:0];

    // Step 4: ReLU — max(0, clipped)
    // In hardware: a single comparator and mux. Cheapest nonlinearity possible.
    wire signed [7:0] relu_out;
    assign relu_out = (relu_en && (clipped < 8'sd0)) ? 8'sd0 : clipped;

    // ── Output register ─────────────────────────────────────────────────────
    always @(posedge clk) begin
        if (!rst_n) begin
            out_int8  <= 8'sd0;
            valid_out <= 1'b0;
        end else begin
            out_int8  <= relu_out;
            valid_out <= valid_in;
        end
    end

endmodule

// =============================================================================
// rtl/mac_unit.v
// Signed 8-bit × 8-bit Multiply-Accumulate Unit
//
// PURPOSE:
//   This is the fundamental compute primitive of the 1D-CNN accelerator.
//   Every convolution, every dense layer, reduces to:
//       accumulator += weight × input_activation
//   That is exactly what this module does — one multiply and one accumulate
//   per clock cycle, fully pipelined.
//
// WHY INT8?
//   The trained model uses INT8 weights and activations after quantization.
//   - INT8 × INT8 multiplier: ~4x less area than FP32 multiplier in ASIC
//   - INT8 MAC: ~4-8x less power than FP32 MAC
//   - The product of two INT8 values fits in 16 bits, but we accumulate
//     many products so we use a 32-bit accumulator to prevent overflow.
//     (max overflow: 127×127×kernel_size×in_channels = 127²×64×5 ≈ 5M,
//      which fits comfortably in INT32 = max 2.1 billion)
//
// INTERFACE:
//   clk         — system clock
//   rst_n       — active-low synchronous reset
//   en          — enable: 1 = perform MAC, 0 = hold accumulator
//   clr         — clear: synchronous reset of accumulator to 0
//                 (assert before starting each output neuron's accumulation)
//   weight_in   — signed INT8 weight from weight ROM
//   act_in      — signed INT8 input activation from input buffer
//   acc_out     — signed INT32 accumulated result
//   acc_valid   — 1 when acc_out contains a valid accumulated sum
//
// LATENCY: 1 clock cycle (registered output, no combinational loops)
//
// SYNTHESIS TARGET: Sky130 / any standard-cell library
//   The multiply maps to DSP slices on FPGA (Xilinx DSP48E2),
//   or to a standard-cell multiplier on ASIC (Sky130 has no hard DSPs).
// =============================================================================

`timescale 1ns / 1ps

module mac_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        clr,
    input  wire signed [7:0]  weight_in,   // INT8 signed weight
    input  wire signed [7:0]  act_in,      // INT8 signed activation
    output reg  signed [31:0] acc_out,     // INT32 signed accumulator
    output reg                acc_valid
);

    // ── Intermediate product: INT8 × INT8 → INT16 ──────────────────────────
    // Note: SystemVerilog signed × signed = signed product automatically.
    // We widen to 32 bits before adding to the accumulator to prevent overflow.
    wire signed [15:0] product;
    assign product = weight_in * act_in;   // 8-bit × 8-bit = 16-bit product

    // ── Accumulator register ───────────────────────────────────────────────
    always @(posedge clk) begin
        if (!rst_n) begin
            acc_out   <= 32'sd0;
            acc_valid <= 1'b0;
        end else if (clr) begin
            // Clear before starting a new output neuron
            acc_out   <= 32'sd0;
            acc_valid <= 1'b0;
        end else if (en) begin
            // MAC: add sign-extended product to accumulator
            acc_out   <= acc_out + {{16{product[15]}}, product};  // sign extend to 32b
            acc_valid <= 1'b1;
        end
        // If neither clr nor en: hold current value (acc_valid stays)
    end

endmodule

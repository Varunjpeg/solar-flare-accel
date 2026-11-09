// =============================================================================
// rtl/flare_classifier_top.v
// Solar Flare AI Accelerator - Top Level
//
// PURPOSE:
//   Connects the entire 1D-CNN pipeline.
//   Input:  Stream of 2-channel 8-bit solar flux data (1 time step per valid).
//   Output: 3-bit flare class prediction, emitted after the 60th time step.
//
// ARCHITECTURE:
//   Input (2) -> Conv1 (16) -> MaxPool1 -> Conv2 (32) -> MaxPool2 -> 
//   Conv3 (64) -> GlobalAvgPool -> Dense1 (32) -> Dense2 (6) -> ArgMax -> Class
// =============================================================================

`timescale 1ns / 1ps

module flare_classifier_top (
    input  wire         clk,
    input  wire         rst_n,
    
    input  wire         s_valid,
    input  wire [15:0]  s_data,   // 2 channels x 8 bits
    output wire         s_ready,
    
    output wire         m_valid,
    output wire [2:0]   m_class,  // 0=Bg, 1=A, 2=B, 3=C, 4=M, 5=X
    input  wire         m_ready
);

    // =========================================================================
    // Wires for pipeline connections
    // =========================================================================
    wire c1_valid, c1_ready;
    wire [16*8-1:0] c1_data;
    
    wire mp1_valid, mp1_ready;
    wire [16*8-1:0] mp1_data;

    wire c2_valid, c2_ready;
    wire [32*8-1:0] c2_data;
    
    wire mp2_valid, mp2_ready;
    wire [32*8-1:0] mp2_data;

    wire c3_valid, c3_ready;
    wire [64*8-1:0] c3_data;

    wire gap_valid, gap_ready;
    wire [64*8-1:0] gap_data;

    wire d1_valid, d1_ready;
    wire [32*8-1:0] d1_data;
    
    wire d2_valid, d2_ready;
    wire [6*8-1:0] d2_data;

    // =========================================================================
    // Layer 1: Conv1D (in=2, out=16, k=5)
    // =========================================================================
    conv1d_layer #(
        .IN_CHANNELS(2), .OUT_CHANNELS(16), .KERNEL_SIZE(5),
        .WEIGHT_FILE("weight_mems/conv1_weights.mem"), .REQUANT_SHIFT(7)
    ) conv1 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid), .s_data(s_data), .s_ready(s_ready),
        .m_valid(c1_valid), .m_data(c1_data), .m_ready(c1_ready)
    );

    // =========================================================================
    // Layer 2: MaxPool1D (channels=16, k=2, s=2)
    // =========================================================================
    maxpool_layer #(.CHANNELS(16)) maxpool1 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(c1_valid), .s_data(c1_data), .s_ready(c1_ready),
        .m_valid(mp1_valid), .m_data(mp1_data), .m_ready(mp1_ready)
    );

    // =========================================================================
    // Layer 3: Conv1D (in=16, out=32, k=5)
    // =========================================================================
    conv1d_layer #(
        .IN_CHANNELS(16), .OUT_CHANNELS(32), .KERNEL_SIZE(5),
        .WEIGHT_FILE("weight_mems/conv2_weights.mem"), .REQUANT_SHIFT(6)
    ) conv2 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(mp1_valid), .s_data(mp1_data), .s_ready(mp1_ready),
        .m_valid(c2_valid), .m_data(c2_data), .m_ready(c2_ready)
    );

    // =========================================================================
    // Layer 4: MaxPool1D (channels=32, k=2, s=2)
    // =========================================================================
    maxpool_layer #(.CHANNELS(32)) maxpool2 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(c2_valid), .s_data(c2_data), .s_ready(c2_ready),
        .m_valid(mp2_valid), .m_data(mp2_data), .m_ready(mp2_ready)
    );

    // =========================================================================
    // Layer 5: Conv1D (in=32, out=64, k=3)
    // =========================================================================
    conv1d_layer #(
        .IN_CHANNELS(32), .OUT_CHANNELS(64), .KERNEL_SIZE(3),
        .WEIGHT_FILE("weight_mems/conv3_weights.mem"), .REQUANT_SHIFT(6)
    ) conv3 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(mp2_valid), .s_data(mp2_data), .s_ready(mp2_ready),
        .m_valid(c3_valid), .m_data(c3_data), .m_ready(c3_ready)
    );

    // =========================================================================
    // Layer 6: Global Average Pool (channels=64, over 15 steps)
    // =========================================================================
    global_avg_pool #(.CHANNELS(64), .TIME_STEPS(15)) gap (
        .clk(clk), .rst_n(rst_n),
        .s_valid(c3_valid), .s_data(c3_data), .s_ready(c3_ready),
        .m_valid(gap_valid), .m_data(gap_data), .m_ready(gap_ready)
    );

    // =========================================================================
    // Layer 7: Dense1 (in=64, out=32, ReLU=1)
    // =========================================================================
    dense_layer #(
        .IN_CHANNELS(64), .OUT_CHANNELS(32),
        .WEIGHT_FILE("weight_mems/dense1_weights.mem"), .REQUANT_SHIFT(8), .RELU_EN(1)
    ) dense1 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(gap_valid), .s_data(gap_data), .s_ready(gap_ready),
        .m_valid(d1_valid), .m_data(d1_data), .m_ready(d1_ready)
    );

    // =========================================================================
    // Layer 8: Dense2 (in=32, out=6, ReLU=0)
    // =========================================================================
    dense_layer #(
        .IN_CHANNELS(32), .OUT_CHANNELS(6),
        .WEIGHT_FILE("weight_mems/dense2_weights.mem"), .REQUANT_SHIFT(8), .RELU_EN(0)
    ) dense2 (
        .clk(clk), .rst_n(rst_n),
        .s_valid(d1_valid), .s_data(d1_data), .s_ready(d1_ready),
        .m_valid(d2_valid), .m_data(d2_data), .m_ready(d2_ready)
    );

    // =========================================================================
    // Layer 9: ArgMax (finds the class with highest probability)
    // =========================================================================
    reg [2:0] max_idx;
    reg signed [7:0] max_val;
    integer i;
    
    // Register the final output
    reg final_valid;
    reg [2:0] final_class;
    
    assign m_valid = final_valid;
    assign m_class = final_class;
    assign d2_ready = m_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            final_valid <= 0;
            final_class <= 0;
        end else begin
            if (m_ready && final_valid) begin
                final_valid <= 0;
            end
            
            if (d2_valid && d2_ready) begin
                max_val = -128;
                max_idx = 0;
                for (i=0; i<6; i=i+1) begin
                    if ($signed(d2_data[i*8 +: 8]) > max_val) begin
                        max_val = d2_data[i*8 +: 8];
                        max_idx = i;
                    end
                end
                final_class <= max_idx;
                final_valid <= 1'b1;
            end
        end
    end

endmodule

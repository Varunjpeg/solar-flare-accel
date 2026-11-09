// =============================================================================
// rtl/global_avg_pool.v
// Global Average Pooling Layer
//
// PURPOSE:
//   Accumulates CHANNELS values over TIME_STEPS, then divides by TIME_STEPS.
//   For this model, TIME_STEPS = 15.
//
// MATH:
//   Divide by 15 is approximated as multiply by 4369 and shift right by 16.
//   out = (acc * 4369 + 32768) >> 16
// =============================================================================

`timescale 1ns / 1ps

module global_avg_pool #(
    parameter CHANNELS   = 64,
    parameter TIME_STEPS = 15
)(
    input  wire                             clk,
    input  wire                             rst_n,
    
    input  wire                             s_valid,
    input  wire signed [8*CHANNELS-1:0]     s_data,
    output wire                             s_ready,
    
    output reg                              m_valid,
    output reg  signed [8*CHANNELS-1:0]     m_data,
    input  wire                             m_ready
);

    reg [$clog2(TIME_STEPS+1)-1:0] t_count;
    reg signed [15:0] acc [0:CHANNELS-1];
    
    integer i;
    
    assign s_ready = (!m_valid || m_ready);

    always @(posedge clk) begin
        if (!rst_n) begin
            t_count <= 0;
            m_valid <= 0;
            m_data <= 0;
            for (i=0; i<CHANNELS; i=i+1) acc[i] <= 0;
        end else begin
            if (m_ready && m_valid) begin
                m_valid <= 0;
            end
            
            if (s_valid && s_ready) begin
                if (t_count == TIME_STEPS - 1) begin
                    // Final accumulation and division
                    for (i=0; i<CHANNELS; i=i+1) begin
                        // acc + current_data
                        wire signed [15:0] final_sum = acc[i] + s_data[i*8 +: 8];
                        
                        // Divide by 15 using (x * 4369 + 32768) >> 16
                        // Note: 4369 = 1/15 in Q16
                        wire signed [31:0] div = (final_sum * 32'sd4369 + 32'sd32768) >>> 16;
                        
                        m_data[i*8 +: 8] <= div[7:0];
                        
                        // Reset accumulator for next window
                        acc[i] <= 0;
                    end
                    m_valid <= 1'b1;
                    t_count <= 0;
                end else begin
                    // Accumulate
                    for (i=0; i<CHANNELS; i=i+1) begin
                        if (t_count == 0) begin
                            acc[i] <= s_data[i*8 +: 8];
                        end else begin
                            acc[i] <= acc[i] + s_data[i*8 +: 8];
                        end
                    end
                    t_count <= t_count + 1;
                end
            end
        end
    end

endmodule

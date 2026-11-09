// =============================================================================
// rtl/maxpool_layer.v
// 1D Max Pooling Layer
//
// PURPOSE:
//   Performs max pooling with kernel_size=2 and stride=2 over a stream
//   of time steps.
//
// OPERATION:
//   - Receives time step T=0 and stores it in a register.
//   - Receives time step T=1, compares element-wise with T=0, and outputs 
//     the maximum of the two.
//   - Downsamples the stream temporally by a factor of 2.
// =============================================================================

`timescale 1ns / 1ps

module maxpool_layer #(
    parameter CHANNELS = 16
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

    reg phase; // 0 = collecting first element, 1 = collecting second element
    reg signed [7:0] buf_data [0:CHANNELS-1];
    
    integer i;
    
    assign s_ready = (!m_valid || m_ready); // Ready if not stalled by output

    always @(posedge clk) begin
        if (!rst_n) begin
            phase <= 0;
            m_valid <= 0;
            m_data <= 0;
            for (i=0; i<CHANNELS; i=i+1) buf_data[i] <= 0;
        end else begin
            // Output handshake
            if (m_ready && m_valid) begin
                m_valid <= 0;
            end
            
            // Input handshake
            if (s_valid && s_ready) begin
                if (phase == 0) begin
                    // Store first element
                    for (i=0; i<CHANNELS; i=i+1) begin
                        buf_data[i] <= s_data[i*8 +: 8];
                    end
                    phase <= 1;
                end else begin
                    // Compare second element and output
                    for (i=0; i<CHANNELS; i=i+1) begin
                        if ($signed(s_data[i*8 +: 8]) > buf_data[i]) begin
                            m_data[i*8 +: 8] <= s_data[i*8 +: 8];
                        end else begin
                            m_data[i*8 +: 8] <= buf_data[i];
                        end
                    end
                    m_valid <= 1'b1;
                    phase <= 0;
                end
            end
        end
    end

endmodule

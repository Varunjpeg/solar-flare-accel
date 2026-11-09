// =============================================================================
// rtl/dense_layer.v
// Sequential Dense (Fully Connected) Layer
//
// PURPOSE:
//   Matrix-vector multiplier using a time-multiplexed MAC unit.
//   Takes a 1D vector of IN_CHANNELS and produces OUT_CHANNELS.
// =============================================================================

`timescale 1ns / 1ps

module dense_layer #(
    parameter IN_CHANNELS  = 64,
    parameter OUT_CHANNELS = 32,
    parameter WEIGHT_FILE  = "",
    parameter REQUANT_SHIFT = 8,
    parameter RELU_EN      = 1
)(
    input  wire                                clk,
    input  wire                                rst_n,
    
    input  wire                                s_valid,
    input  wire signed [8*IN_CHANNELS-1:0]     s_data,
    output wire                                s_ready,
    
    output reg                                 m_valid,
    output reg  signed [8*OUT_CHANNELS-1:0]    m_data,
    input  wire                                m_ready
);

    // =========================================================================
    // 1. Input Buffer
    // =========================================================================
    reg signed [7:0] in_buf [0:IN_CHANNELS-1];
    integer i;

    // =========================================================================
    // 2. FSM Definitions
    // =========================================================================
    localparam S_IDLE   = 3'd0;
    localparam S_LOAD   = 3'd1;
    localparam S_CALC   = 3'd2;
    localparam S_REQ    = 3'd3;
    localparam S_OUTPUT = 3'd4;
    
    reg [2:0] state;
    reg [$clog2(OUT_CHANNELS+1)-1:0] c_out;
    reg [$clog2(IN_CHANNELS+1)-1:0]  c_in;
    
    // =========================================================================
    // 3. Weight ROM
    // =========================================================================
    localparam NUM_WEIGHTS = OUT_CHANNELS * IN_CHANNELS;
    wire [$clog2(NUM_WEIGHTS)-1:0] rom_addr;
    wire signed [7:0]              rom_data;
    
    assign rom_addr = c_out * IN_CHANNELS + c_in;
    
    weight_rom #(
        .MEM_FILE(WEIGHT_FILE),
        .DEPTH(NUM_WEIGHTS),
        .DATA_WIDTH(8)
    ) w_rom (
        .clk(clk),
        .addr(rom_addr),
        .data_out(rom_data)
    );

    // =========================================================================
    // 4. MAC Unit
    // =========================================================================
    wire mac_clr = (state == S_CALC && c_in == 0);
    wire signed [31:0] mac_acc;
    
    reg signed [7:0] act_reg;
    reg              mac_en_reg;
    reg              mac_clr_reg;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            act_reg     <= 0;
            mac_en_reg  <= 0;
            mac_clr_reg <= 0;
        end else begin
            act_reg     <= in_buf[c_in];
            mac_en_reg  <= (state == S_CALC);
            mac_clr_reg <= mac_clr;
        end
    end
    
    mac_unit u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .en(mac_en_reg),
        .clr(mac_clr_reg),
        .weight_in(rom_data),
        .act_in(act_reg),
        .acc_out(mac_acc),
        .acc_valid() // unused, we track state manually
    );

    // =========================================================================
    // 5. Requantization
    // =========================================================================
    wire signed [7:0] req_out;
    wire req_valid;
    
    requant_unit #(
        .SHIFT_DEFAULT(REQUANT_SHIFT)
    ) u_req (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(state == S_REQ),
        .acc_in(mac_acc),
        .shift(REQUANT_SHIFT[4:0]),
        .bias(32'sd0),
        .relu_en(RELU_EN),
        .out_int8(req_out),
        .valid_out(req_valid)
    );

    // =========================================================================
    // 6. FSM Logic
    // =========================================================================
    assign s_ready = (state == S_IDLE);
    
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE;
            c_out <= 0;
            c_in  <= 0;
            m_valid <= 0;
            m_data <= 0;
            for (i=0; i<IN_CHANNELS; i=i+1) in_buf[i] <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (s_valid) begin
                        for (i=0; i<IN_CHANNELS; i=i+1) begin
                            in_buf[i] <= s_data[i*8 +: 8];
                        end
                        state <= S_CALC;
                        c_out <= 0;
                        c_in  <= 0;
                    end
                end
                
                S_CALC: begin
                    if (c_in == IN_CHANNELS-1) begin
                        c_in <= 0;
                        state <= S_REQ;
                    end else begin
                        c_in <= c_in + 1;
                    end
                end
                
                S_REQ: begin
                    if (req_valid) begin
                        m_data[c_out*8 +: 8] <= req_out;
                        
                        if (c_out == OUT_CHANNELS-1) begin
                            state <= S_OUTPUT;
                        end else begin
                            c_out <= c_out + 1;
                            state <= S_CALC;
                        end
                    end
                end
                
                S_OUTPUT: begin
                    m_valid <= 1'b1;
                    if (m_ready && m_valid) begin
                        m_valid <= 1'b0;
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule

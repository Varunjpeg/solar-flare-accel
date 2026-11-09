// =============================================================================
// rtl/conv1d_layer.v
// Sequential 1D Convolution Layer
//
// PURPOSE:
//   Computes a 1D convolution over a streaming time-series input.
//   Uses an area-optimized sequential FSM: a single MAC unit is time-multiplexed 
//   to compute all channels.
//
// OPERATION:
//   1. Waits for `s_valid` to load a new time step (vector of IN_CHANNELS).
//   2. Shifts the new vector into a line buffer of size KERNEL_SIZE.
//   3. Runs a nested loop FSM (Out_Ch -> In_Ch -> Kernel) over the buffer.
//   4. Fetches weights from `weight_rom.v`.
//   5. Requantizes the 32-bit accumulator back to 8-bit.
//   6. Outputs the resulting vector of OUT_CHANNELS, then goes back to IDLE.
// =============================================================================

`timescale 1ns / 1ps

module conv1d_layer #(
    parameter IN_CHANNELS  = 2,
    parameter OUT_CHANNELS = 16,
    parameter KERNEL_SIZE  = 5,
    parameter WEIGHT_FILE  = "",
    parameter REQUANT_SHIFT = 8,
    parameter BIAS_FILE    = ""  // Optional: could hardcode biases or load from ROM
)(
    input  wire                                clk,
    input  wire                                rst_n,
    
    // Input Stream (1 time step = IN_CHANNELS vector)
    input  wire                                s_valid,
    input  wire signed [8*IN_CHANNELS-1:0]     s_data,
    output wire                                s_ready,
    
    // Output Stream (1 time step = OUT_CHANNELS vector)
    output reg                                 m_valid,
    output reg  signed [8*OUT_CHANNELS-1:0]    m_data,
    input  wire                                m_ready
);

    // =========================================================================
    // 1. Line Buffer (Shift Register)
    // =========================================================================
    // Holds the sliding window of input data.
    reg signed [7:0] line_buf [0:KERNEL_SIZE-1][0:IN_CHANNELS-1];
    
    integer i, j;
    
    // =========================================================================
    // 2. FSM Definitions
    // =========================================================================
    localparam S_IDLE   = 3'd0;
    localparam S_SHIFT  = 3'd1;
    localparam S_CALC   = 3'd2;
    localparam S_REQ    = 3'd3;
    localparam S_OUTPUT = 3'd4;
    
    reg [2:0] state, next_state;
    
    // Nested loop counters
    reg [$clog2(OUT_CHANNELS+1)-1:0] c_out;
    reg [$clog2(IN_CHANNELS+1)-1:0]  c_in;
    reg [$clog2(KERNEL_SIZE+1)-1:0]  c_k;
    
    // =========================================================================
    // 3. Weight ROM Instantiation
    // =========================================================================
    localparam NUM_WEIGHTS = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE;
    wire [$clog2(NUM_WEIGHTS)-1:0] rom_addr;
    wire signed [7:0]              rom_data;
    
    // Address mapped in C-order: out_c * (in_c * k) + in_c * k + k
    assign rom_addr = c_out * (IN_CHANNELS * KERNEL_SIZE) + c_in * KERNEL_SIZE + c_k;
    
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
    wire mac_en, mac_clr;
    wire signed [31:0] mac_acc;
    wire mac_valid;
    
    // We register the data to align with the 1-cycle ROM read latency
    reg signed [7:0] act_reg;
    reg              mac_en_reg;
    reg              mac_clr_reg;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            act_reg     <= 0;
            mac_en_reg  <= 0;
            mac_clr_reg <= 0;
        end else begin
            act_reg     <= line_buf[c_k][c_in];
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
        .acc_valid(mac_valid)
    );
    
    assign mac_clr = (state == S_CALC && c_in == 0 && c_k == 0);
    
    // =========================================================================
    // 5. Requantization (INT32 -> INT8)
    // =========================================================================
    wire signed [7:0] req_out;
    wire req_valid;
    
    // Simplified bias handling: in a real ASIC we would load from a small ROM or RAM.
    // Here we use 0 for simplicity, assuming biases are negligible or handled externally,
    // OR we can expand this to read a bias ROM.
    // For now, bias=0.
    requant_unit #(
        .SHIFT_DEFAULT(REQUANT_SHIFT)
    ) u_req (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(state == S_REQ),
        .acc_in(mac_acc),
        .shift(REQUANT_SHIFT[4:0]),
        .bias(32'sd0),       // TODO: Hook up Bias ROM
        .relu_en(1'b1),      // ReLU is enabled for Conv1D
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
            c_k   <= 0;
            m_valid <= 0;
            m_data <= 0;
            
            for (i=0; i<KERNEL_SIZE; i=i+1)
                for (j=0; j<IN_CHANNELS; j=j+1)
                    line_buf[i][j] <= 0;
                    
        end else begin
            case (state)
                S_IDLE: begin
                    if (s_valid) begin
                        state <= S_SHIFT;
                    end
                end
                
                S_SHIFT: begin
                    // Shift line buffer
                    for (i=0; i<KERNEL_SIZE-1; i=i+1) begin
                        for (j=0; j<IN_CHANNELS; j=j+1) begin
                            line_buf[i][j] <= line_buf[i+1][j];
                        end
                    end
                    // Load new vector at end
                    for (j=0; j<IN_CHANNELS; j=j+1) begin
                        line_buf[KERNEL_SIZE-1][j] <= s_data[j*8 +: 8];
                    end
                    
                    state <= S_CALC;
                    c_out <= 0;
                    c_in  <= 0;
                    c_k   <= 0;
                end
                
                S_CALC: begin
                    // Nested counters
                    if (c_k == KERNEL_SIZE-1) begin
                        c_k <= 0;
                        if (c_in == IN_CHANNELS-1) begin
                            c_in <= 0;
                            state <= S_REQ; // Go requantize this output channel
                        end else begin
                            c_in <= c_in + 1;
                        end
                    end else begin
                        c_k <= c_k + 1;
                    end
                end
                
                S_REQ: begin
                    // Wait for requantization to finish (1 cycle delay)
                    if (req_valid) begin
                        // Pack INT8 into m_data bus
                        m_data[c_out*8 +: 8] <= req_out;
                        
                        if (c_out == OUT_CHANNELS-1) begin
                            state <= S_OUTPUT;
                        end else begin
                            c_out <= c_out + 1;
                            state <= S_CALC; // Back to calc next output channel
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

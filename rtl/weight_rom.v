// =============================================================================
// rtl/weight_rom.v
// Weight Read-Only Memory
//
// PURPOSE:
//   Instantiates a block RAM (or standard-cell ROM in ASIC) initialized with 
//   INT8 weights from a hex file. Provides a 1-cycle read latency.
// =============================================================================

`timescale 1ns / 1ps

module weight_rom #(
    parameter MEM_FILE = "",
    parameter DEPTH = 160,
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = $clog2(DEPTH)
)(
    input  wire                  clk,
    input  wire [ADDR_WIDTH-1:0] addr,
    output reg  [DATA_WIDTH-1:0] data_out
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (MEM_FILE != "") begin
            $readmemh(MEM_FILE, mem);
        end
    end

    // Synchronous read (standard BRAM inference)
    always @(posedge clk) begin
        data_out <= mem[addr];
    end

endmodule

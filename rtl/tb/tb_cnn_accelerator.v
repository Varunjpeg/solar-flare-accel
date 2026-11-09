// =============================================================================
// rtl/tb/tb_cnn_accelerator.v
// Testbench for 1D-CNN Solar Flare Accelerator Top Module
// =============================================================================

`timescale 1ns / 1ps

module tb_cnn_accelerator;

    reg clk;
    reg rst_n;
    reg s_axis_tvalid;
    reg signed [15:0] s_axis_tdata;
    wire s_axis_tready;

    wire m_axis_tvalid;
    wire [2:0] m_axis_tclass;
    wire signed [7:0] m_axis_tlogit;
    reg m_axis_tready;

    // Instantiate Top Design
    cnn_accelerator_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tready(s_axis_tready),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tclass(m_axis_tclass),
        .m_axis_tlogit(m_axis_tlogit),
        .m_axis_tready(m_axis_tready)
    );

    // Clock Generation (50 MHz = 20 ns period)
    always #10 clk = ~clk;

    integer i;
    initial begin
        clk = 0;
        rst_n = 0;
        s_axis_tvalid = 0;
        s_axis_tdata = 0;
        m_axis_tready = 1;

        #40;
        rst_n = 1;
        #20;

        $display("------------------------------------------------------------");
        $display("STARTING ACCELERATOR TESTBENCH SIMULATION");
        $display("------------------------------------------------------------");

        // Stream 60 sample timestamps into the accelerator
        for (i = 0; i < 60; i = i + 1) begin
            @(posedge clk);
            s_axis_tvalid <= 1;
            // Channel B: 8'sd10, Channel A: 8'sd5
            s_axis_tdata  <= {8'sd10, 8'sd5};
        end

        @(posedge clk);
        s_axis_tvalid <= 0;

        // Wait for classification output valid
        wait(m_axis_tvalid);
        $display("[SUCCESS] Received Classification Output!");
        $display("Predicted Flare Class Index: %d", m_axis_tclass);
        $display("Output Logit Score:          %d", m_axis_tlogit);
        $display("------------------------------------------------------------");

        #100;
        $finish;
    end

endmodule

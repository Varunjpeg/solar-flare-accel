`timescale 1ns / 1ps

// ====================================================================
// Testbench for flare_classifier_top
// Description: Feeds a 60-minute window of fake solar flux data 
//              into the RTL and checks the output class.
// ====================================================================

module tb_flare_classifier_top;

    // Clock and Reset
    reg clk;
    reg rst_n;

    // AXI-Stream-like Input signals
    reg         s_valid;
    reg  [15:0] s_data;
    wire        s_ready;

    // AXI-Stream-like Output signals
    wire        m_valid;
    wire [2:0]  m_class;
    reg         m_ready;

    // Instantiate the Top Level Module
    flare_classifier_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .s_valid(s_valid),
        .s_data(s_data),
        .s_ready(s_ready),
        .m_valid(m_valid),
        .m_class(m_class),
        .m_ready(m_ready)
    );

    // 100 MHz Clock Generation (10ns period)
    always #5 clk = ~clk;

    integer i;

    initial begin
        // Initialize signals
        clk = 0;
        rst_n = 0;
        s_valid = 0;
        s_data = 0;
        m_ready = 1; // Always ready to receive output

        // Dump waves for GTKWave viewing
        $dumpfile("tb_flare_classifier_top.vcd");
        $dumpvars(0, tb_flare_classifier_top);

        // Assert Reset for 20ns
        #20;
        rst_n = 1;
        #10;

        $display("===========================================");
        $display("Starting Testbench: Streaming 60 time-steps");
        $display("===========================================");

        // Stream 60 minutes of data into the chip
        // In a real testbench, we would load an array from a text file containing an X-class flare.
        // Here we just feed a rising slope of numbers to simulate an incoming flare.
        for (i = 0; i < 60; i = i + 1) begin
            // Wait until the chip is ready to accept data
            wait (s_ready == 1'b1);
            
            // Generate a fake X-ray flux (INT8 signed values)
            // Channel A (top 8 bits) and Channel B (bottom 8 bits)
            // We'll simulate a rising curve: 8-bit values from 0 up to ~60
            s_data = { (i[7:0] + 8'd5), i[7:0] }; 
            s_valid = 1;
            
            #10; // Wait 1 clock cycle
        end

        // Stop streaming
        s_valid = 0;

        // Wait for the pipeline to finish calculating and emit the final class
        $display("Streaming complete. Waiting for Neural Network inference to finish...");
        wait (m_valid == 1'b1);
        
        $display("\n===========================================");
        $display("              HARDWARE RESULT              ");
        $display("===========================================");
        $display("Predicted Flare Class (0=Bg, 1=A, 2=B, 3=C, 4=M, 5=X): %0d", m_class);
        $display("===========================================\n");

        // Wait a few cycles before ending simulation
        #50;
        $finish;
    end

endmodule

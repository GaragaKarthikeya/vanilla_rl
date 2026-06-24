// custom_5ch_mac: a hand-written design for this project's zero-shot
// generalization testing. Not derived from any existing benchmark suite.
//
// Structure: 5 independent channels, each with its own write port (enable +
// address + data) into its own 32-entry sample buffer, and its own weighted
// multiply against that channel's buffered sample. The 5 products are summed
// into a single accumulated output. Five independent write ports (rather
// than one shared buffer with multiple read taps) keep each channel's memory
// array functionally distinct, so synthesis can't collapse them into fewer
// BRAMs -- five 32-bit multiplies (DSP-inference) and five distinct 32-deep
// memory arrays (BRAM-inference), at the upper bound of 5 DSP / 5 BRAM.
module custom_5ch_mac (
    input clk,
    input reset,
    input wen0, wen1, wen2, wen3, wen4,
    input [31:0] data_in0, data_in1, data_in2, data_in3, data_in4,
    input [4:0] waddr0, waddr1, waddr2, waddr3, waddr4,
    input [4:0] raddr,
    output reg [31:0] result_out
);

    reg [31:0] buf0 [0:31];
    reg [31:0] buf1 [0:31];
    reg [31:0] buf2 [0:31];
    reg [31:0] buf3 [0:31];
    reg [31:0] buf4 [0:31];

    reg [31:0] w0, w1, w2, w3, w4;
    reg [31:0] r0, r1, r2, r3, r4;

    wire [31:0] p0, p1, p2, p3, p4;
    wire [31:0] mac_sum;

    assign p0 = r0 * w0;
    assign p1 = r1 * w1;
    assign p2 = r2 * w2;
    assign p3 = r3 * w3;
    assign p4 = r4 * w4;
    assign mac_sum = p0 + p1 + p2 + p3 + p4;

    always @(posedge clk) begin
        if (reset == 1'b1) begin
            w0 <= 32'd3;
            w1 <= 32'd5;
            w2 <= 32'd7;
            w3 <= 32'd11;
            w4 <= 32'd13;
            r0 <= 32'd0;
            r1 <= 32'd0;
            r2 <= 32'd0;
            r3 <= 32'd0;
            r4 <= 32'd0;
            result_out <= 32'd0;
        end else begin
            if (wen0)
                buf0[waddr0] <= data_in0;
            if (wen1)
                buf1[waddr1] <= data_in1;
            if (wen2)
                buf2[waddr2] <= data_in2;
            if (wen3)
                buf3[waddr3] <= data_in3;
            if (wen4)
                buf4[waddr4] <= data_in4;

            r0 <= buf0[raddr];
            r1 <= buf1[raddr];
            r2 <= buf2[raddr];
            r3 <= buf3[raddr];
            r4 <= buf4[raddr];

            result_out <= mac_sum;
        end
    end

endmodule

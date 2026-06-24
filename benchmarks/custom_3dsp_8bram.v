// custom_3dsp_8bram: a hand-written design targeting a gap in the training
// pool's resource diversity -- no existing training-pool benchmark has
// DSP in {2,3,4} or BRAM in {6,7,8,9,10}. Not derived from any existing
// benchmark suite.
//
// Structure: 8 independent channels, each with its own write port into its
// own 32-entry buffer (8 distinct memory arrays -> 8 BRAM-inference). Only
// the first 3 channels are weighted and multiplied (3 distinct 32-bit
// multiplies -> 3 DSP-inference); the remaining 5 channels are summed in
// unweighted, so every buffer is genuinely used (and therefore can't be
// optimized away) without adding more multipliers. Independent write ports
// per channel keep synthesis from collapsing any of the 8 buffers together.
module custom_3dsp_8bram (
    input clk,
    input reset,
    input wen0, wen1, wen2, wen3, wen4, wen5, wen6, wen7,
    input [31:0] data_in0, data_in1, data_in2, data_in3,
    input [31:0] data_in4, data_in5, data_in6, data_in7,
    input [4:0] waddr0, waddr1, waddr2, waddr3,
    input [4:0] waddr4, waddr5, waddr6, waddr7,
    input [4:0] raddr,
    output reg [31:0] result_out
);

    reg [31:0] buf0 [0:31];
    reg [31:0] buf1 [0:31];
    reg [31:0] buf2 [0:31];
    reg [31:0] buf3 [0:31];
    reg [31:0] buf4 [0:31];
    reg [31:0] buf5 [0:31];
    reg [31:0] buf6 [0:31];
    reg [31:0] buf7 [0:31];

    reg [31:0] w0, w1, w2;
    reg [31:0] r0, r1, r2, r3, r4, r5, r6, r7;

    wire [31:0] p0, p1, p2;
    wire [31:0] aux_sum;
    wire [31:0] mac_sum;

    assign p0 = r0 * w0;
    assign p1 = r1 * w1;
    assign p2 = r2 * w2;
    assign aux_sum = r3 + r4 + r5 + r6 + r7;
    assign mac_sum = p0 + p1 + p2 + aux_sum;

    always @(posedge clk) begin
        if (reset == 1'b1) begin
            w0 <= 32'd3;
            w1 <= 32'd5;
            w2 <= 32'd7;
            r0 <= 32'd0;
            r1 <= 32'd0;
            r2 <= 32'd0;
            r3 <= 32'd0;
            r4 <= 32'd0;
            r5 <= 32'd0;
            r6 <= 32'd0;
            r7 <= 32'd0;
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
            if (wen5)
                buf5[waddr5] <= data_in5;
            if (wen6)
                buf6[waddr6] <= data_in6;
            if (wen7)
                buf7[waddr7] <= data_in7;

            r0 <= buf0[raddr];
            r1 <= buf1[raddr];
            r2 <= buf2[raddr];
            r3 <= buf3[raddr];
            r4 <= buf4[raddr];
            r5 <= buf5[raddr];
            r6 <= buf6[raddr];
            r7 <= buf7[raddr];

            result_out <= mac_sum;
        end
    end

endmodule

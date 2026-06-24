// custom_8dsp_3bram: a hand-written design with the inverse resource ratio
// of custom_3dsp_8bram -- DSP-heavy instead of BRAM-heavy, at the upper
// bound of DSP count for this architecture. Not derived from any existing
// benchmark suite.
//
// Structure: 8 independent weighted multiply channels (8 distinct 32-bit
// multiplies -> 8 DSP-inference). Only 3 of the 8 operands come from
// independent 32-entry memory buffers with their own write ports (3 distinct
// memory arrays -> 3 BRAM-inference); the other 5 operands are plain
// registered inputs (flip-flops, not arrays), so they add multiplies
// without adding more memories. All 8 products are summed into one output.
module custom_8dsp_3bram (
    input clk,
    input reset,
    input wen0, wen1, wen2,
    input [31:0] data_in0, data_in1, data_in2,
    input [4:0] waddr0, waddr1, waddr2,
    input [4:0] raddr,
    input [31:0] direct_in3, direct_in4, direct_in5, direct_in6, direct_in7,
    output reg [31:0] result_out
);

    reg [31:0] buf0 [0:31];
    reg [31:0] buf1 [0:31];
    reg [31:0] buf2 [0:31];

    reg [31:0] w0, w1, w2, w3, w4, w5, w6, w7;
    reg [31:0] r0, r1, r2;
    reg [31:0] d3, d4, d5, d6, d7;

    wire [31:0] p0, p1, p2, p3, p4, p5, p6, p7;
    wire [31:0] mac_sum;

    assign p0 = r0 * w0;
    assign p1 = r1 * w1;
    assign p2 = r2 * w2;
    assign p3 = d3 * w3;
    assign p4 = d4 * w4;
    assign p5 = d5 * w5;
    assign p6 = d6 * w6;
    assign p7 = d7 * w7;
    assign mac_sum = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;

    always @(posedge clk) begin
        if (reset == 1'b1) begin
            w0 <= 32'd3;
            w1 <= 32'd5;
            w2 <= 32'd7;
            w3 <= 32'd11;
            w4 <= 32'd13;
            w5 <= 32'd17;
            w6 <= 32'd19;
            w7 <= 32'd23;
            r0 <= 32'd0;
            r1 <= 32'd0;
            r2 <= 32'd0;
            d3 <= 32'd0;
            d4 <= 32'd0;
            d5 <= 32'd0;
            d6 <= 32'd0;
            d7 <= 32'd0;
            result_out <= 32'd0;
        end else begin
            if (wen0)
                buf0[waddr0] <= data_in0;
            if (wen1)
                buf1[waddr1] <= data_in1;
            if (wen2)
                buf2[waddr2] <= data_in2;

            r0 <= buf0[raddr];
            r1 <= buf1[raddr];
            r2 <= buf2[raddr];

            d3 <= direct_in3;
            d4 <= direct_in4;
            d5 <= direct_in5;
            d6 <= direct_in6;
            d7 <= direct_in7;

            result_out <= mac_sum;
        end
    end

endmodule

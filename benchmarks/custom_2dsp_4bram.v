// custom_2dsp_4bram: a hand-written design with a "normal" (not deliberately
// extreme) resource mix that is still absent from the current 13-benchmark
// training pool -- no trained benchmark uses 2 DSP or 4 BRAM. Created to
// test whether the multi13_ambitious model's zero-shot generalization
// regression is specific to extreme/unseen resource combinations, or a
// general effect that shows up on any genuinely novel design.
//
// Structure: 4 independent channels, each with its own write port into its
// own 32-entry buffer (4 distinct memory arrays -> 4 BRAM-inference). Only
// the first 2 channels are weighted and multiplied (2 distinct 32-bit
// multiplies -> 2 DSP-inference); the other 2 channels are summed in
// unweighted, so every buffer is genuinely used without adding more
// multipliers. Independent write ports keep synthesis from collapsing any
// of the 4 buffers together.
module custom_2dsp_4bram (
    input clk,
    input reset,
    input wen0, wen1, wen2, wen3,
    input [31:0] data_in0, data_in1, data_in2, data_in3,
    input [4:0] waddr0, waddr1, waddr2, waddr3,
    input [4:0] raddr,
    output reg [31:0] result_out
);

    reg [31:0] buf0 [0:31];
    reg [31:0] buf1 [0:31];
    reg [31:0] buf2 [0:31];
    reg [31:0] buf3 [0:31];

    reg [31:0] w0, w1;
    reg [31:0] r0, r1, r2, r3;

    wire [31:0] p0, p1;
    wire [31:0] aux_sum;
    wire [31:0] mac_sum;

    assign p0 = r0 * w0;
    assign p1 = r1 * w1;
    assign aux_sum = r2 + r3;
    assign mac_sum = p0 + p1 + aux_sum;

    always @(posedge clk) begin
        if (reset == 1'b1) begin
            w0 <= 32'd3;
            w1 <= 32'd5;
            r0 <= 32'd0;
            r1 <= 32'd0;
            r2 <= 32'd0;
            r3 <= 32'd0;
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

            r0 <= buf0[raddr];
            r1 <= buf1[raddr];
            r2 <= buf2[raddr];
            r3 <= buf3[raddr];

            result_out <= mac_sum;
        end
    end

endmodule

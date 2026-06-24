// custom_4dsp_6bram_big: a hand-written design built to test the scale-
// mismatch hypothesis directly. Uses a clean, still-untested DSP/BRAM
// combination (4 DSP, 6 BRAM -- neither count appears anywhere in the
// current 13-benchmark training pool), but adds a deliberately large block
// of unrelated NONLINEAR logic (a 48-stage 128-bit AND/XOR mixing chain --
// pure XOR/rotate chains are GF(2)-linear and collapse to a single matrix
// under synthesis no matter how many stages, so this uses an AND term to
// force genuine nonlinearity that can't be algebraically simplified away)
// so the auto-sized grid lands closer to the scale of softmax/
// reduction_layer/arm_core (35-44 wide) instead of the pool's largest
// member (mkPktMerge, 26 wide). Separate named registers, not a reg array,
// so this cannot be inferred as another memory/BRAM -- CLB count only.
module custom_4dsp_6bram_big (
    input clk,
    input reset,
    input wen0, wen1, wen2, wen3, wen4, wen5,
    input [31:0] data_in0, data_in1, data_in2, data_in3, data_in4, data_in5,
    input [4:0] waddr0, waddr1, waddr2, waddr3, waddr4, waddr5,
    input [4:0] raddr,
    input [127:0] scramble_in,
    output reg [31:0] result_out,
    output reg [127:0] scramble_out
);

    reg [31:0] buf0 [0:31];
    reg [31:0] buf1 [0:31];
    reg [31:0] buf2 [0:31];
    reg [31:0] buf3 [0:31];
    reg [31:0] buf4 [0:31];
    reg [31:0] buf5 [0:31];

    reg [31:0] w0, w1, w2, w3;
    reg [31:0] r0, r1, r2, r3, r4, r5;

    wire [31:0] p0, p1, p2, p3;
    wire [31:0] aux_sum;
    wire [31:0] mac_sum;

    assign p0 = r0 * w0;
    assign p1 = r1 * w1;
    assign p2 = r2 * w2;
    assign p3 = r3 * w3;
    assign aux_sum = r4 + r5;
    assign mac_sum = p0 + p1 + p2 + p3 + aux_sum;

    reg [127:0] stage0, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12, stage13, stage14, stage15, stage16, stage17, stage18, stage19, stage20, stage21, stage22, stage23, stage24, stage25, stage26, stage27, stage28, stage29, stage30, stage31, stage32, stage33, stage34, stage35, stage36, stage37, stage38, stage39, stage40, stage41, stage42, stage43, stage44, stage45, stage46, stage47;

    always @(posedge clk) begin
        if (reset == 1'b1) begin
            w0 <= 32'd3;
            w1 <= 32'd5;
            w2 <= 32'd7;
            w3 <= 32'd11;
            r0 <= 32'd0;
            r1 <= 32'd0;
            r2 <= 32'd0;
            r3 <= 32'd0;
            r4 <= 32'd0;
            r5 <= 32'd0;
            result_out <= 32'd0;
            stage0 <= 128'd0;
            stage1 <= 128'd0;
            stage2 <= 128'd0;
            stage3 <= 128'd0;
            stage4 <= 128'd0;
            stage5 <= 128'd0;
            stage6 <= 128'd0;
            stage7 <= 128'd0;
            stage8 <= 128'd0;
            stage9 <= 128'd0;
            stage10 <= 128'd0;
            stage11 <= 128'd0;
            stage12 <= 128'd0;
            stage13 <= 128'd0;
            stage14 <= 128'd0;
            stage15 <= 128'd0;
            stage16 <= 128'd0;
            stage17 <= 128'd0;
            stage18 <= 128'd0;
            stage19 <= 128'd0;
            stage20 <= 128'd0;
            stage21 <= 128'd0;
            stage22 <= 128'd0;
            stage23 <= 128'd0;
            stage24 <= 128'd0;
            stage25 <= 128'd0;
            stage26 <= 128'd0;
            stage27 <= 128'd0;
            stage28 <= 128'd0;
            stage29 <= 128'd0;
            stage30 <= 128'd0;
            stage31 <= 128'd0;
            stage32 <= 128'd0;
            stage33 <= 128'd0;
            stage34 <= 128'd0;
            stage35 <= 128'd0;
            stage36 <= 128'd0;
            stage37 <= 128'd0;
            stage38 <= 128'd0;
            stage39 <= 128'd0;
            stage40 <= 128'd0;
            stage41 <= 128'd0;
            stage42 <= 128'd0;
            stage43 <= 128'd0;
            stage44 <= 128'd0;
            stage45 <= 128'd0;
            stage46 <= 128'd0;
            stage47 <= 128'd0;
            scramble_out <= 128'd0;
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

            r0 <= buf0[raddr];
            r1 <= buf1[raddr];
            r2 <= buf2[raddr];
            r3 <= buf3[raddr];
            r4 <= buf4[raddr];
            r5 <= buf5[raddr];

            result_out <= mac_sum;

            stage0 <= ({scramble_in[110:0], scramble_in[127:111]} & {scramble_in[66:0], scramble_in[127:67]}) ^ scramble_in;
            stage1 <= ({stage0[74:0], stage0[127:75]} & {stage0[104:0], stage0[127:105]}) ^ stage0;
            stage2 <= ({stage1[98:0], stage1[127:99]} & {stage1[30:0], stage1[127:31]}) ^ stage1;
            stage3 <= ({stage2[56:0], stage2[127:57]} & {stage2[116:0], stage2[127:117]}) ^ stage2;
            stage4 <= ({stage3[122:0], stage3[127:123]} & {stage3[48:0], stage3[127:49]}) ^ stage3;
            stage5 <= ({stage4[36:0], stage4[127:37]} & {stage4[84:0], stage4[127:85]}) ^ stage4;
            stage6 <= ({stage5[14:0], stage5[127:15]} & {stage5[20:0], stage5[127:21]}) ^ stage5;
            stage7 <= ({stage6[90:0], stage6[127:91]} & {stage6[60:0], stage6[127:61]}) ^ stage6;
            stage8 <= ({stage7[110:0], stage7[127:111]} & {stage7[66:0], stage7[127:67]}) ^ stage7;
            stage9 <= ({stage8[74:0], stage8[127:75]} & {stage8[104:0], stage8[127:105]}) ^ stage8;
            stage10 <= ({stage9[98:0], stage9[127:99]} & {stage9[30:0], stage9[127:31]}) ^ stage9;
            stage11 <= ({stage10[56:0], stage10[127:57]} & {stage10[116:0], stage10[127:117]}) ^ stage10;
            stage12 <= ({stage11[122:0], stage11[127:123]} & {stage11[48:0], stage11[127:49]}) ^ stage11;
            stage13 <= ({stage12[36:0], stage12[127:37]} & {stage12[84:0], stage12[127:85]}) ^ stage12;
            stage14 <= ({stage13[14:0], stage13[127:15]} & {stage13[20:0], stage13[127:21]}) ^ stage13;
            stage15 <= ({stage14[90:0], stage14[127:91]} & {stage14[60:0], stage14[127:61]}) ^ stage14;
            stage16 <= ({stage15[110:0], stage15[127:111]} & {stage15[66:0], stage15[127:67]}) ^ stage15;
            stage17 <= ({stage16[74:0], stage16[127:75]} & {stage16[104:0], stage16[127:105]}) ^ stage16;
            stage18 <= ({stage17[98:0], stage17[127:99]} & {stage17[30:0], stage17[127:31]}) ^ stage17;
            stage19 <= ({stage18[56:0], stage18[127:57]} & {stage18[116:0], stage18[127:117]}) ^ stage18;
            stage20 <= ({stage19[122:0], stage19[127:123]} & {stage19[48:0], stage19[127:49]}) ^ stage19;
            stage21 <= ({stage20[36:0], stage20[127:37]} & {stage20[84:0], stage20[127:85]}) ^ stage20;
            stage22 <= ({stage21[14:0], stage21[127:15]} & {stage21[20:0], stage21[127:21]}) ^ stage21;
            stage23 <= ({stage22[90:0], stage22[127:91]} & {stage22[60:0], stage22[127:61]}) ^ stage22;
            stage24 <= ({stage23[110:0], stage23[127:111]} & {stage23[66:0], stage23[127:67]}) ^ stage23;
            stage25 <= ({stage24[74:0], stage24[127:75]} & {stage24[104:0], stage24[127:105]}) ^ stage24;
            stage26 <= ({stage25[98:0], stage25[127:99]} & {stage25[30:0], stage25[127:31]}) ^ stage25;
            stage27 <= ({stage26[56:0], stage26[127:57]} & {stage26[116:0], stage26[127:117]}) ^ stage26;
            stage28 <= ({stage27[122:0], stage27[127:123]} & {stage27[48:0], stage27[127:49]}) ^ stage27;
            stage29 <= ({stage28[36:0], stage28[127:37]} & {stage28[84:0], stage28[127:85]}) ^ stage28;
            stage30 <= ({stage29[14:0], stage29[127:15]} & {stage29[20:0], stage29[127:21]}) ^ stage29;
            stage31 <= ({stage30[90:0], stage30[127:91]} & {stage30[60:0], stage30[127:61]}) ^ stage30;
            stage32 <= ({stage31[110:0], stage31[127:111]} & {stage31[66:0], stage31[127:67]}) ^ stage31;
            stage33 <= ({stage32[74:0], stage32[127:75]} & {stage32[104:0], stage32[127:105]}) ^ stage32;
            stage34 <= ({stage33[98:0], stage33[127:99]} & {stage33[30:0], stage33[127:31]}) ^ stage33;
            stage35 <= ({stage34[56:0], stage34[127:57]} & {stage34[116:0], stage34[127:117]}) ^ stage34;
            stage36 <= ({stage35[122:0], stage35[127:123]} & {stage35[48:0], stage35[127:49]}) ^ stage35;
            stage37 <= ({stage36[36:0], stage36[127:37]} & {stage36[84:0], stage36[127:85]}) ^ stage36;
            stage38 <= ({stage37[14:0], stage37[127:15]} & {stage37[20:0], stage37[127:21]}) ^ stage37;
            stage39 <= ({stage38[90:0], stage38[127:91]} & {stage38[60:0], stage38[127:61]}) ^ stage38;
            stage40 <= ({stage39[110:0], stage39[127:111]} & {stage39[66:0], stage39[127:67]}) ^ stage39;
            stage41 <= ({stage40[74:0], stage40[127:75]} & {stage40[104:0], stage40[127:105]}) ^ stage40;
            stage42 <= ({stage41[98:0], stage41[127:99]} & {stage41[30:0], stage41[127:31]}) ^ stage41;
            stage43 <= ({stage42[56:0], stage42[127:57]} & {stage42[116:0], stage42[127:117]}) ^ stage42;
            stage44 <= ({stage43[122:0], stage43[127:123]} & {stage43[48:0], stage43[127:49]}) ^ stage43;
            stage45 <= ({stage44[36:0], stage44[127:37]} & {stage44[84:0], stage44[127:85]}) ^ stage44;
            stage46 <= ({stage45[14:0], stage45[127:15]} & {stage45[20:0], stage45[127:21]}) ^ stage45;
            stage47 <= ({stage46[90:0], stage46[127:91]} & {stage46[60:0], stage46[127:61]}) ^ stage46;
            scramble_out <= stage47;
        end
    end

endmodule

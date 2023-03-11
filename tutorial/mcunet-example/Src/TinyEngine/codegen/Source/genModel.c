/* Automatically generated source file */
#include <float.h>
#include "arm_nnfunctions.h"

#include "genNN.h"
#include "genModel.h"

#include "tinyengine_function.h"
#include "tinyengine_function_fp.h"

#include "profile.h"

/* Variables used by all ops */
ADD_params add_params;
//Conv_Params conv_params;
//Depthwise_Params dpconv_params;
int i;
int8_t *int8ptr,*int8ptr2;
int32_t *int32ptr;
float *fptr,*fptr2,*fptr3;

signed char* getInput() {
    return &buffer0[123904];
}
signed char* getOutput() {
    return NNoutput;
}
void end2endinference(q7_t* img){
    invoke(NULL);
}
void invoke(float* labels){
/* layer 0:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[123904],176,176,3,(const q7_t*) weight0,bias0,shift0,multiplier0,-128,0,-128,127,&buffer0[0],88,88,16,sbuf,kbuf,0);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r176x176x3_88x88x16 3345408 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],88,88,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-128,128,-128,127,&buffer0[0],88,88,16,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r88x88x16_88x88x16 1115136 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],88,88,16,(const q7_t*) weight2,bias2,shift2,multiplier2,0,128,-128,127,&buffer0[185856],88,88,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r88x88x16_88x88x8 991232 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[185856],88,88,8,(const q7_t*) weight3,bias3,shift3,multiplier3,-128,0,-128,127,&buffer0[0],88,88,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r88x88x8_88x88x24 1486848 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride2_inplace_CHW(&buffer0[0],88,88,24,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,-128,128,-128,127,&buffer0[0],44,44,24,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r88x88x24_44x44x24 2276736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[0],44,44,24,(const q7_t*) weight5,bias5,shift5,multiplier5,0,128,-128,127,&buffer0[154880],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x24_44x44x16 743424 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[154880],44,44,16,(const q7_t*) weight6,bias6,shift6,multiplier6,-128,0,-128,127,&buffer0[0],44,44,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x80 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],44,44,80,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-128,128,-128,127,&buffer0[0],44,44,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r44x44x80_44x44x80 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],44,44,80,(const q7_t*) weight8,bias8,shift8,multiplier8,0,128,-128,127,&buffer0[185856],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x80_44x44x16 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(30976, &buffer0[185856],0.07058085501194,0,&buffer0[154880],0.07173234224319458,0,0.06555090844631195,0,&buffer0[216832]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r44x44x16_44x44x16 30976 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[216832],44,44,16,(const q7_t*) weight9,bias9,shift9,multiplier9,-128,0,-128,127,&buffer0[0],44,44,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x80 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],44,44,80,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-128,128,-128,127,&buffer0[0],44,44,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r44x44x80_44x44x80 7589120 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],44,44,80,(const q7_t*) weight11,bias11,shift11,multiplier11,0,128,-128,127,&buffer0[154880],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x80_44x44x16 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(30976, &buffer0[154880],0.021758602932095528,0,&buffer0[216832],0.06555090844631195,0,0.1004064679145813,0,&buffer0[123904]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r44x44x16_44x44x16 30976 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[123904],44,44,16,(const q7_t*) weight12,bias12,shift12,multiplier12,-128,0,-128,127,&buffer0[0],44,44,64,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x64 1982464 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],44,44,64,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,-128,128,-128,127,&buffer0[0],44,44,64,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r44x44x64_44x44x64 3097600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],44,44,64,(const q7_t*) weight14,bias14,shift14,multiplier14,0,128,-128,127,&buffer0[154880],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x64_44x44x16 1982464 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(30976, &buffer0[154880],0.013444670476019382,0,&buffer0[123904],0.1004064679145813,0,0.14225490391254425,0,&buffer0[185856]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r44x44x16_44x44x16 30976 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[185856],44,44,16,(const q7_t*) weight15,bias15,shift15,multiplier15,-128,0,-128,127,&buffer0[0],44,44,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x80 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride2_inplace_CHW(&buffer0[0],44,44,80,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,-128,128,-128,127,&buffer0[0],22,22,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r44x44x80_22x22x80 968000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],22,22,80,(const q7_t*) weight17,bias17,shift17,multiplier17,0,128,-128,127,&buffer0[58080],22,22,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x80_22x22x24 929280 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[58080],22,22,24,(const q7_t*) weight18,bias18,shift18,multiplier18,-128,0,-128,127,&buffer0[0],22,22,120,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x24_22x22x120 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],22,22,120,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,-128,128,-128,127,&buffer0[0],22,22,120,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r22x22x120_22x22x120 1452000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],22,22,120,(const q7_t*) weight20,bias20,shift20,multiplier20,0,128,-128,127,&buffer0[69696],22,22,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x120_22x22x24 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(11616, &buffer0[69696],0.03606591001152992,0,&buffer0[58080],0.08175227791070938,0,0.05693759769201279,0,&buffer0[81312]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r22x22x24_22x22x24 11616 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[81312],22,22,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-128,0,-128,127,&buffer0[0],22,22,120,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x24_22x22x120 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],22,22,120,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,-128,128,-128,127,&buffer0[0],22,22,120,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r22x22x120_22x22x120 1452000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],22,22,120,(const q7_t*) weight23,bias23,shift23,multiplier23,0,128,-128,127,&buffer0[58080],22,22,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x120_22x22x24 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(11616, &buffer0[58080],0.010867958888411522,0,&buffer0[81312],0.05693759769201279,0,0.0646895095705986,0,&buffer0[69696]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r22x22x24_22x22x24 11616 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[69696],22,22,24,(const q7_t*) weight24,bias24,shift24,multiplier24,-128,0,-128,127,&buffer0[0],22,22,120,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x24_22x22x120 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],22,22,120,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,-128,128,-128,127,&buffer0[0],11,11,120,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r22x22x120_11x11x120 130680 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,120,(const q7_t*) weight26,bias26,shift26,multiplier26,0,128,-128,127,&buffer0[29040],11,11,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x120_11x11x40 580800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[29040],11,11,40,(const q7_t*) weight27,bias27,shift27,multiplier27,-128,0,-128,127,&buffer0[0],11,11,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x40_11x11x240 1161600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],11,11,240,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-128,128,-128,127,&buffer0[0],11,11,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r11x11x240_11x11x240 1422960 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,240,(const q7_t*) weight29,bias29,shift29,multiplier29,0,128,-128,127,&buffer0[33880],11,11,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x240_11x11x40 1161600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4840, &buffer0[33880],0.02520747110247612,0,&buffer0[29040],0.06708217412233353,0,0.04652569070458412,0,&buffer0[19360]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x40_11x11x40 4840 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[19360],11,11,40,(const q7_t*) weight30,bias30,shift30,multiplier30,-128,0,-128,127,&buffer0[0],11,11,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x40_11x11x160 774400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],11,11,160,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-128,128,-128,127,&buffer0[0],11,11,160,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r11x11x160_11x11x160 484000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,160,(const q7_t*) weight32,bias32,shift32,multiplier32,0,128,-128,127,&buffer0[24200],11,11,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x160_11x11x40 774400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4840, &buffer0[24200],0.010844066739082336,0,&buffer0[19360],0.04652569070458412,0,0.04837863892316818,0,&buffer0[29040]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x40_11x11x40 4840 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[29040],11,11,40,(const q7_t*) weight33,bias33,shift33,multiplier33,-128,0,-128,127,&buffer0[0],11,11,200,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x40_11x11x200 968000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],11,11,200,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,-128,128,-128,127,&buffer0[0],11,11,200,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r11x11x200_11x11x200 605000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,200,(const q7_t*) weight35,bias35,shift35,multiplier35,0,128,-128,127,&buffer0[29040],11,11,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x200_11x11x48 1161600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[29040],11,11,48,(const q7_t*) weight36,bias36,shift36,multiplier36,-128,0,-128,127,&buffer0[0],11,11,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x48_11x11x240 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],11,11,240,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,-128,128,-128,127,&buffer0[0],11,11,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r11x11x240_11x11x240 1422960 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,240,(const q7_t*) weight38,bias38,shift38,multiplier38,0,128,-128,127,&buffer0[34848],11,11,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x240_11x11x48 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5808, &buffer0[34848],0.02392001822590828,0,&buffer0[29040],0.05639925226569176,0,0.04785430058836937,0,&buffer0[40656]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x48_11x11x48 5808 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[40656],11,11,48,(const q7_t*) weight39,bias39,shift39,multiplier39,-128,0,-128,127,&buffer0[0],11,11,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x48_11x11x240 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],11,11,240,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,-128,128,-128,127,&buffer0[0],11,11,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r11x11x240_11x11x240 261360 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,240,(const q7_t*) weight41,bias41,shift41,multiplier41,0,128,-128,127,&buffer0[29040],11,11,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x240_11x11x48 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5808, &buffer0[29040],0.01157377753406763,0,&buffer0[40656],0.04785430058836937,0,0.052783895283937454,0,&buffer0[34848]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x48_11x11x48 5808 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[34848],11,11,48,(const q7_t*) weight42,bias42,shift42,multiplier42,-128,0,-128,127,&buffer0[0],11,11,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x48_11x11x288 1672704 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],11,11,288,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,-128,128,-128,127,&buffer0[0],6,6,288,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r11x11x288_6x6x288 93312 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,288,(const q7_t*) weight44,bias44,shift44,multiplier44,0,128,-128,127,&buffer0[17280],6,6,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x288_6x6x96 995328 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[17280],6,6,96,(const q7_t*) weight45,bias45,shift45,multiplier45,-128,0,-128,127,&buffer0[0],6,6,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x96_6x6x480 1658880 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],6,6,480,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,-128,128,-128,127,&buffer0[0],6,6,480,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r6x6x480_6x6x480 846720 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,480,(const q7_t*) weight47,bias47,shift47,multiplier47,0,128,-128,127,&buffer0[20736],6,6,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x480_6x6x96 1658880 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3456, &buffer0[20736],0.027496028691530228,0,&buffer0[17280],0.06507851183414459,0,0.043968942016363144,0,&buffer0[13824]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r6x6x96_6x6x96 3456 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[13824],6,6,96,(const q7_t*) weight48,bias48,shift48,multiplier48,-128,0,-128,127,&buffer0[0],6,6,384,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x96_6x6x384 1327104 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],6,6,384,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,-128,128,-128,127,&buffer0[0],6,6,384,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r6x6x384_6x6x384 124416 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,384,(const q7_t*) weight50,bias50,shift50,multiplier50,0,128,-128,127,&buffer0[17280],6,6,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x384_6x6x96 1327104 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3456, &buffer0[17280],0.013334653340280056,0,&buffer0[13824],0.043968942016363144,0,0.05323520675301552,0,&buffer0[20736]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r6x6x96_6x6x96 3456 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20736],6,6,96,(const q7_t*) weight51,bias51,shift51,multiplier51,-128,0,-128,127,&buffer0[0],6,6,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x96_6x6x480 1658880 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 63:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],6,6,480,(const q7_t*) CHWweight52,offsetBias52,offsetRBias52,shift52,multiplier52,-128,128,-128,127,&buffer0[0],6,6,480,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r6x6x480_6x6x480 846720 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 64:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,480,(const q7_t*) weight53,bias53,shift53,multiplier53,0,128,-128,127,&buffer0[17280],6,6,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x480_6x6x160 2764800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 65:AVERAGE_POOL_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    avg_pooling(&buffer0[17280],6,6,160,6,6,1,1,-128,127,&buffer0[1000]);
}
end = HAL_GetTick();
sprintf(buf, "AVERAGE_POOL_2D r6x6x160_1x1x160 0 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 66:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1000],1,1,160,(const q7_t*) weight54,bias54,shift54,multiplier54,0,0,-128,127,&buffer0[0],1,1,1000,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x160_1x1x1000 160000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[123904],176,176,3,(const q7_t*) weight0,bias0,shift0,multiplier0,-128,0,-128,127,&buffer0[0],88,88,16,sbuf,kbuf,0);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r176x176x3_88x88x16 3345408 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],88,88,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-128,128,-128,127,&buffer0[0],88,88,16,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r88x88x16_88x88x16 1115136 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],88,88,16,(const q7_t*) weight2,bias2,shift2,multiplier2,0,128,-128,127,&buffer0[185856],88,88,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r88x88x16_88x88x8 991232 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[185856],88,88,8,(const q7_t*) weight3,bias3,shift3,multiplier3,-128,0,-128,127,&buffer0[0],88,88,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r88x88x8_88x88x24 1486848 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride2_inplace_CHW(&buffer0[0],88,88,24,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,-128,128,-128,127,&buffer0[0],44,44,24,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r88x88x24_44x44x24 2276736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[0],44,44,24,(const q7_t*) weight5,bias5,shift5,multiplier5,0,128,-128,127,&buffer0[154880],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x24_44x44x16 743424 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[154880],44,44,16,(const q7_t*) weight6,bias6,shift6,multiplier6,-128,0,-128,127,&buffer0[0],44,44,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x80 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],44,44,80,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-128,128,-128,127,&buffer0[0],44,44,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r44x44x80_44x44x80 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],44,44,80,(const q7_t*) weight8,bias8,shift8,multiplier8,0,128,-128,127,&buffer0[185856],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x80_44x44x16 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(30976, &buffer0[185856],0.07058085501194,0,&buffer0[154880],0.07173234224319458,0,0.06555090844631195,0,&buffer0[216832]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r44x44x16_44x44x16 30976 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[216832],44,44,16,(const q7_t*) weight9,bias9,shift9,multiplier9,-128,0,-128,127,&buffer0[0],44,44,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x80 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],44,44,80,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-128,128,-128,127,&buffer0[0],44,44,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r44x44x80_44x44x80 7589120 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],44,44,80,(const q7_t*) weight11,bias11,shift11,multiplier11,0,128,-128,127,&buffer0[154880],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x80_44x44x16 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(30976, &buffer0[154880],0.021758602932095528,0,&buffer0[216832],0.06555090844631195,0,0.1004064679145813,0,&buffer0[123904]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r44x44x16_44x44x16 30976 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[123904],44,44,16,(const q7_t*) weight12,bias12,shift12,multiplier12,-128,0,-128,127,&buffer0[0],44,44,64,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x64 1982464 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],44,44,64,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,-128,128,-128,127,&buffer0[0],44,44,64,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r44x44x64_44x44x64 3097600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],44,44,64,(const q7_t*) weight14,bias14,shift14,multiplier14,0,128,-128,127,&buffer0[154880],44,44,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x64_44x44x16 1982464 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(30976, &buffer0[154880],0.013444670476019382,0,&buffer0[123904],0.1004064679145813,0,0.14225490391254425,0,&buffer0[185856]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r44x44x16_44x44x16 30976 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[185856],44,44,16,(const q7_t*) weight15,bias15,shift15,multiplier15,-128,0,-128,127,&buffer0[0],44,44,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r44x44x16_44x44x80 2478080 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride2_inplace_CHW(&buffer0[0],44,44,80,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,-128,128,-128,127,&buffer0[0],22,22,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r44x44x80_22x22x80 968000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],22,22,80,(const q7_t*) weight17,bias17,shift17,multiplier17,0,128,-128,127,&buffer0[58080],22,22,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x80_22x22x24 929280 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[58080],22,22,24,(const q7_t*) weight18,bias18,shift18,multiplier18,-128,0,-128,127,&buffer0[0],22,22,120,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x24_22x22x120 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],22,22,120,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,-128,128,-128,127,&buffer0[0],22,22,120,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r22x22x120_22x22x120 1452000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],22,22,120,(const q7_t*) weight20,bias20,shift20,multiplier20,0,128,-128,127,&buffer0[69696],22,22,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x120_22x22x24 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(11616, &buffer0[69696],0.03606591001152992,0,&buffer0[58080],0.08175227791070938,0,0.05693759769201279,0,&buffer0[81312]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r22x22x24_22x22x24 11616 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[81312],22,22,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-128,0,-128,127,&buffer0[0],22,22,120,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x24_22x22x120 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],22,22,120,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,-128,128,-128,127,&buffer0[0],22,22,120,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r22x22x120_22x22x120 1452000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],22,22,120,(const q7_t*) weight23,bias23,shift23,multiplier23,0,128,-128,127,&buffer0[58080],22,22,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x120_22x22x24 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(11616, &buffer0[58080],0.010867958888411522,0,&buffer0[81312],0.05693759769201279,0,0.0646895095705986,0,&buffer0[69696]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r22x22x24_22x22x24 11616 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[69696],22,22,24,(const q7_t*) weight24,bias24,shift24,multiplier24,-128,0,-128,127,&buffer0[0],22,22,120,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r22x22x24_22x22x120 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],22,22,120,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,-128,128,-128,127,&buffer0[0],11,11,120,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r22x22x120_11x11x120 130680 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,120,(const q7_t*) weight26,bias26,shift26,multiplier26,0,128,-128,127,&buffer0[29040],11,11,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x120_11x11x40 580800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[29040],11,11,40,(const q7_t*) weight27,bias27,shift27,multiplier27,-128,0,-128,127,&buffer0[0],11,11,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x40_11x11x240 1161600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],11,11,240,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-128,128,-128,127,&buffer0[0],11,11,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r11x11x240_11x11x240 1422960 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,240,(const q7_t*) weight29,bias29,shift29,multiplier29,0,128,-128,127,&buffer0[33880],11,11,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x240_11x11x40 1161600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4840, &buffer0[33880],0.02520747110247612,0,&buffer0[29040],0.06708217412233353,0,0.04652569070458412,0,&buffer0[19360]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x40_11x11x40 4840 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[19360],11,11,40,(const q7_t*) weight30,bias30,shift30,multiplier30,-128,0,-128,127,&buffer0[0],11,11,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x40_11x11x160 774400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],11,11,160,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-128,128,-128,127,&buffer0[0],11,11,160,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r11x11x160_11x11x160 484000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,160,(const q7_t*) weight32,bias32,shift32,multiplier32,0,128,-128,127,&buffer0[24200],11,11,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x160_11x11x40 774400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4840, &buffer0[24200],0.010844066739082336,0,&buffer0[19360],0.04652569070458412,0,0.04837863892316818,0,&buffer0[29040]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x40_11x11x40 4840 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[29040],11,11,40,(const q7_t*) weight33,bias33,shift33,multiplier33,-128,0,-128,127,&buffer0[0],11,11,200,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x40_11x11x200 968000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],11,11,200,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,-128,128,-128,127,&buffer0[0],11,11,200,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r11x11x200_11x11x200 605000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,200,(const q7_t*) weight35,bias35,shift35,multiplier35,0,128,-128,127,&buffer0[29040],11,11,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x200_11x11x48 1161600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[29040],11,11,48,(const q7_t*) weight36,bias36,shift36,multiplier36,-128,0,-128,127,&buffer0[0],11,11,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x48_11x11x240 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],11,11,240,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,-128,128,-128,127,&buffer0[0],11,11,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r11x11x240_11x11x240 1422960 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,240,(const q7_t*) weight38,bias38,shift38,multiplier38,0,128,-128,127,&buffer0[34848],11,11,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x240_11x11x48 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5808, &buffer0[34848],0.02392001822590828,0,&buffer0[29040],0.05639925226569176,0,0.04785430058836937,0,&buffer0[40656]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x48_11x11x48 5808 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[40656],11,11,48,(const q7_t*) weight39,bias39,shift39,multiplier39,-128,0,-128,127,&buffer0[0],11,11,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x48_11x11x240 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],11,11,240,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,-128,128,-128,127,&buffer0[0],11,11,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r11x11x240_11x11x240 261360 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],11,11,240,(const q7_t*) weight41,bias41,shift41,multiplier41,0,128,-128,127,&buffer0[29040],11,11,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x240_11x11x48 1393920 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5808, &buffer0[29040],0.01157377753406763,0,&buffer0[40656],0.04785430058836937,0,0.052783895283937454,0,&buffer0[34848]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r11x11x48_11x11x48 5808 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[34848],11,11,48,(const q7_t*) weight42,bias42,shift42,multiplier42,-128,0,-128,127,&buffer0[0],11,11,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r11x11x48_11x11x288 1672704 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],11,11,288,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,-128,128,-128,127,&buffer0[0],6,6,288,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r11x11x288_6x6x288 93312 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,288,(const q7_t*) weight44,bias44,shift44,multiplier44,0,128,-128,127,&buffer0[17280],6,6,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x288_6x6x96 995328 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[17280],6,6,96,(const q7_t*) weight45,bias45,shift45,multiplier45,-128,0,-128,127,&buffer0[0],6,6,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x96_6x6x480 1658880 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],6,6,480,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,-128,128,-128,127,&buffer0[0],6,6,480,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r6x6x480_6x6x480 846720 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,480,(const q7_t*) weight47,bias47,shift47,multiplier47,0,128,-128,127,&buffer0[20736],6,6,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x480_6x6x96 1658880 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3456, &buffer0[20736],0.027496028691530228,0,&buffer0[17280],0.06507851183414459,0,0.043968942016363144,0,&buffer0[13824]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r6x6x96_6x6x96 3456 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[13824],6,6,96,(const q7_t*) weight48,bias48,shift48,multiplier48,-128,0,-128,127,&buffer0[0],6,6,384,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x96_6x6x384 1327104 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],6,6,384,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,-128,128,-128,127,&buffer0[0],6,6,384,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r6x6x384_6x6x384 124416 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,384,(const q7_t*) weight50,bias50,shift50,multiplier50,0,128,-128,127,&buffer0[17280],6,6,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x384_6x6x96 1327104 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3456, &buffer0[17280],0.013334653340280056,0,&buffer0[13824],0.043968942016363144,0,0.05323520675301552,0,&buffer0[20736]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r6x6x96_6x6x96 3456 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20736],6,6,96,(const q7_t*) weight51,bias51,shift51,multiplier51,-128,0,-128,127,&buffer0[0],6,6,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x96_6x6x480 1658880 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 63:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],6,6,480,(const q7_t*) CHWweight52,offsetBias52,offsetRBias52,shift52,multiplier52,-128,128,-128,127,&buffer0[0],6,6,480,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r6x6x480_6x6x480 846720 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 64:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],6,6,480,(const q7_t*) weight53,bias53,shift53,multiplier53,0,128,-128,127,&buffer0[17280],6,6,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r6x6x480_6x6x160 2764800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 65:AVERAGE_POOL_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    avg_pooling(&buffer0[17280],6,6,160,6,6,1,1,-128,127,&buffer0[1000]);
}
end = HAL_GetTick();
sprintf(buf, "AVERAGE_POOL_2D r6x6x160_1x1x160 0 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 66:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1000],1,1,160,(const q7_t*) weight54,bias54,shift54,multiplier54,0,0,-128,127,&buffer0[0],1,1,1000,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x160_1x1x1000 160000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}

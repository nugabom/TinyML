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
    return &buffer0[16384];
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
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[16384],64,64,3,(const q7_t*) weight0,bias0,shift0,multiplier0,4,128,-128,127,&buffer0[0],32,32,16,sbuf,kbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r64x64x3_32x32x16 442368 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel1x1_stride1_inplace_CHW(&buffer0[0],32,32,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-15,-4,-128,127,&buffer0[0],32,32,16,sbuf,4);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k1x1_r32x32x16_32x32x16 16384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],32,32,16,(const q7_t*) weight2,bias2,shift2,multiplier2,-27,15,-128,127,&buffer0[49152],32,32,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r32x32x16_32x32x8 131072 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[49152],32,32,8,(const q7_t*) weight3,bias3,shift3,multiplier3,-18,27,-128,127,&buffer0[0],32,32,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r32x32x8_32x32x48 393216 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],32,32,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,-29,18,-128,127,&buffer0[0],16,16,48,sbuf,-18);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r32x32x48_16x16x48 110592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight5,bias5,shift5,multiplier5,-18,29,-128,127,&buffer0[12288],16,16,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x48_16x16x8 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[12288],16,16,8,(const q7_t*) weight6,bias6,shift6,multiplier6,-15,18,-128,127,&buffer0[0],16,16,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x8_16x16x48 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,33,15,-128,127,&buffer0[0],16,16,48,sbuf,-15);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r16x16x48_16x16x48 110592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight8,bias8,shift8,multiplier8,-54,-33,-128,127,&buffer0[14336],16,16,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x48_16x16x8 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2048, &buffer0[14336],0.002026086673140526,-54,&buffer0[12288],0.0029037040658295155,-18,0.003416722873225808,-74,&buffer0[16384]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r16x16x8_16x16x8 2048 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[16384],16,16,8,(const q7_t*) weight9,bias9,shift9,multiplier9,-5,74,-128,127,&buffer0[0],16,16,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x8_16x16x48 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-30,5,-128,127,&buffer0[0],8,8,48,sbuf,-5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r16x16x48_8x8x48 27648 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],8,8,48,(const q7_t*) weight11,bias11,shift11,multiplier11,40,30,-128,127,&buffer0[6144],8,8,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x48_8x8x16 49152 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[6144],8,8,16,(const q7_t*) weight12,bias12,shift12,multiplier12,2,-40,-128,127,&buffer0[0],8,8,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x16_8x8x96 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,10,-2,-128,127,&buffer0[0],8,8,96,sbuf,2);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r8x8x96_8x8x96 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight14,bias14,shift14,multiplier14,18,-10,-128,127,&buffer0[7168],8,8,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x96_8x8x16 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(1024, &buffer0[7168],0.0024348075967282057,18,&buffer0[6144],0.00245222682133317,40,0.0037478625308722258,56,&buffer0[8192]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r8x8x16_8x8x16 1024 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[8192],8,8,16,(const q7_t*) weight15,bias15,shift15,multiplier15,9,-56,-128,127,&buffer0[0],8,8,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x16_8x8x96 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,1,-9,-128,127,&buffer0[0],8,8,96,sbuf,9);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r8x8x96_8x8x96 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight17,bias17,shift17,multiplier17,9,-1,-128,127,&buffer0[6144],8,8,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x96_8x8x16 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(1024, &buffer0[6144],0.0020680378656834364,9,&buffer0[8192],0.0037478625308722258,56,0.0047828820534050465,33,&buffer0[7168]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r8x8x16_8x8x16 1024 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[7168],8,8,16,(const q7_t*) weight18,bias18,shift18,multiplier18,-7,-33,-128,127,&buffer0[0],8,8,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x16_8x8x96 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,6,7,-128,127,&buffer0[0],4,4,96,sbuf,-7);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r8x8x96_4x4x96 13824 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,96,(const q7_t*) weight20,bias20,shift20,multiplier20,21,-6,-128,127,&buffer0[2304],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x96_4x4x24 36864 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[2304],4,4,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-6,-21,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,-5,6,-128,127,&buffer0[0],4,4,144,sbuf,-6);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight23,bias23,shift23,multiplier23,21,5,-128,127,&buffer0[2688],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x24 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(384, &buffer0[2688],0.0017378657357767224,21,&buffer0[2304],0.001899592811241746,21,0.002666281769052148,-8,&buffer0[3072]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x24_4x4x24 384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight24,bias24,shift24,multiplier24,-2,8,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,13,2,-128,127,&buffer0[0],4,4,144,sbuf,-2);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight26,bias26,shift26,multiplier26,-19,-13,-128,127,&buffer0[2304],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x24 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(384, &buffer0[2304],0.002509584417566657,-19,&buffer0[3072],0.002666281769052148,-8,0.004062060732394457,-29,&buffer0[2688]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x24_4x4x24 384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[2688],4,4,24,(const q7_t*) weight27,bias27,shift27,multiplier27,9,29,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,15,-9,-128,127,&buffer0[0],4,4,144,sbuf,9);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight29,bias29,shift29,multiplier29,-14,-15,-128,127,&buffer0[2304],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x24 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(384, &buffer0[2304],0.0027209443505853415,-14,&buffer0[2688],0.004062060732394457,-29,0.004572504200041294,0,&buffer0[3072]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x24_4x4x24 384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight30,bias30,shift30,multiplier30,-17,0,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-22,17,-128,127,&buffer0[0],4,4,144,sbuf,-17);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight32,bias32,shift32,multiplier32,24,22,-128,127,&buffer0[3072],4,4,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x32 73728 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[3072],4,4,32,(const q7_t*) weight33,bias33,shift33,multiplier33,-9,-24,-128,127,&buffer0[0],4,4,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x32_4x4x192 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,-4,9,-128,127,&buffer0[0],4,4,192,sbuf,-9);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x192_4x4x192 27648 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight35,bias35,shift35,multiplier35,37,4,-128,127,&buffer0[3584],4,4,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x192_4x4x32 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(512, &buffer0[3584],0.0022200383245944977,37,&buffer0[3072],0.0018070171354338527,24,0.0031476770527660847,50,&buffer0[4096]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x32_4x4x32 512 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[4096],4,4,32,(const q7_t*) weight36,bias36,shift36,multiplier36,10,-50,-128,127,&buffer0[0],4,4,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x32_4x4x192 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,4,-10,-128,127,&buffer0[0],4,4,192,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x192_4x4x192 27648 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight38,bias38,shift38,multiplier38,-4,-4,-128,127,&buffer0[3072],4,4,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x192_4x4x32 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(512, &buffer0[3072],0.002144323196262121,-4,&buffer0[4096],0.0031476770527660847,50,0.003769106464460492,37,&buffer0[3584]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x32_4x4x32 512 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[3584],4,4,32,(const q7_t*) weight39,bias39,shift39,multiplier39,-6,-37,-128,127,&buffer0[0],4,4,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x32_4x4x192 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,1,6,-128,127,&buffer0[0],2,2,192,sbuf,-6);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x192_2x2x192 6912 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,192,(const q7_t*) weight41,bias41,shift41,multiplier41,-24,-1,-128,127,&buffer0[1344],2,2,56,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x192_2x2x56 43008 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1344],2,2,56,(const q7_t*) weight42,bias42,shift42,multiplier42,5,24,-128,127,&buffer0[0],2,2,336,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x56_2x2x336 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,1,-5,-128,127,&buffer0[0],2,2,336,sbuf,5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r2x2x336_2x2x336 12096 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight44,bias44,shift44,multiplier44,6,-1,-128,127,&buffer0[1568],2,2,56,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x336_2x2x56 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(224, &buffer0[1568],0.0022980249486863613,6,&buffer0[1344],0.002694052876904607,-24,0.003149643074721098,-14,&buffer0[1792]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r2x2x56_2x2x56 224 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1792],2,2,56,(const q7_t*) weight45,bias45,shift45,multiplier45,3,14,-128,127,&buffer0[0],2,2,336,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x56_2x2x336 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,2,-3,-128,127,&buffer0[0],2,2,336,sbuf,3);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r2x2x336_2x2x336 12096 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight47,bias47,shift47,multiplier47,-11,-2,-128,127,&buffer0[1344],2,2,56,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x336_2x2x56 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(224, &buffer0[1344],0.002098551718518138,-11,&buffer0[1792],0.003149643074721098,-14,0.003843615995720029,-7,&buffer0[1568]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r2x2x56_2x2x56 224 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1568],2,2,56,(const q7_t*) weight48,bias48,shift48,multiplier48,14,7,-128,127,&buffer0[0],2,2,336,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x56_2x2x336 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,0,-14,-128,127,&buffer0[0],2,2,336,sbuf,14);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r2x2x336_2x2x336 12096 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight50,bias50,shift50,multiplier50,-2,0,-128,127,&buffer0[5120],2,2,112,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x336_2x2x112 150528 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[5120],2,2,112,(const q7_t*) weight51,bias51,shift51,multiplier51,-17,2,-128,127,&buffer0[0],2,2,1280,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x112_2x2x1280 573440 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[5120],1,1,1280,(const q7_t*) weight52,bias52,shift52,multiplier52,-19,19,-128,127,&buffer0[5120],1,1,2,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x1280_1x1x2 2560 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[16384],64,64,3,(const q7_t*) weight0,bias0,shift0,multiplier0,4,128,-128,127,&buffer0[0],32,32,16,sbuf,kbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r64x64x3_32x32x16 442368 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel1x1_stride1_inplace_CHW(&buffer0[0],32,32,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-15,-4,-128,127,&buffer0[0],32,32,16,sbuf,4);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k1x1_r32x32x16_32x32x16 16384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],32,32,16,(const q7_t*) weight2,bias2,shift2,multiplier2,-27,15,-128,127,&buffer0[49152],32,32,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r32x32x16_32x32x8 131072 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[49152],32,32,8,(const q7_t*) weight3,bias3,shift3,multiplier3,-18,27,-128,127,&buffer0[0],32,32,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r32x32x8_32x32x48 393216 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],32,32,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,-29,18,-128,127,&buffer0[0],16,16,48,sbuf,-18);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r32x32x48_16x16x48 110592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight5,bias5,shift5,multiplier5,-18,29,-128,127,&buffer0[12288],16,16,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x48_16x16x8 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[12288],16,16,8,(const q7_t*) weight6,bias6,shift6,multiplier6,-15,18,-128,127,&buffer0[0],16,16,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x8_16x16x48 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,33,15,-128,127,&buffer0[0],16,16,48,sbuf,-15);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r16x16x48_16x16x48 110592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight8,bias8,shift8,multiplier8,-54,-33,-128,127,&buffer0[14336],16,16,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x48_16x16x8 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2048, &buffer0[14336],0.002026086673140526,-54,&buffer0[12288],0.0029037040658295155,-18,0.003416722873225808,-74,&buffer0[16384]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r16x16x8_16x16x8 2048 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[16384],16,16,8,(const q7_t*) weight9,bias9,shift9,multiplier9,-5,74,-128,127,&buffer0[0],16,16,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r16x16x8_16x16x48 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-30,5,-128,127,&buffer0[0],8,8,48,sbuf,-5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r16x16x48_8x8x48 27648 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],8,8,48,(const q7_t*) weight11,bias11,shift11,multiplier11,40,30,-128,127,&buffer0[6144],8,8,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x48_8x8x16 49152 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[6144],8,8,16,(const q7_t*) weight12,bias12,shift12,multiplier12,2,-40,-128,127,&buffer0[0],8,8,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x16_8x8x96 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,10,-2,-128,127,&buffer0[0],8,8,96,sbuf,2);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r8x8x96_8x8x96 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight14,bias14,shift14,multiplier14,18,-10,-128,127,&buffer0[7168],8,8,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x96_8x8x16 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(1024, &buffer0[7168],0.0024348075967282057,18,&buffer0[6144],0.00245222682133317,40,0.0037478625308722258,56,&buffer0[8192]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r8x8x16_8x8x16 1024 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[8192],8,8,16,(const q7_t*) weight15,bias15,shift15,multiplier15,9,-56,-128,127,&buffer0[0],8,8,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x16_8x8x96 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,1,-9,-128,127,&buffer0[0],8,8,96,sbuf,9);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r8x8x96_8x8x96 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight17,bias17,shift17,multiplier17,9,-1,-128,127,&buffer0[6144],8,8,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x96_8x8x16 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(1024, &buffer0[6144],0.0020680378656834364,9,&buffer0[8192],0.0037478625308722258,56,0.0047828820534050465,33,&buffer0[7168]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r8x8x16_8x8x16 1024 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[7168],8,8,16,(const q7_t*) weight18,bias18,shift18,multiplier18,-7,-33,-128,127,&buffer0[0],8,8,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r8x8x16_8x8x96 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,6,7,-128,127,&buffer0[0],4,4,96,sbuf,-7);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r8x8x96_4x4x96 13824 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,96,(const q7_t*) weight20,bias20,shift20,multiplier20,21,-6,-128,127,&buffer0[2304],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x96_4x4x24 36864 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[2304],4,4,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-6,-21,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,-5,6,-128,127,&buffer0[0],4,4,144,sbuf,-6);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight23,bias23,shift23,multiplier23,21,5,-128,127,&buffer0[2688],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x24 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(384, &buffer0[2688],0.0017378657357767224,21,&buffer0[2304],0.001899592811241746,21,0.002666281769052148,-8,&buffer0[3072]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x24_4x4x24 384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight24,bias24,shift24,multiplier24,-2,8,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,13,2,-128,127,&buffer0[0],4,4,144,sbuf,-2);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight26,bias26,shift26,multiplier26,-19,-13,-128,127,&buffer0[2304],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x24 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(384, &buffer0[2304],0.002509584417566657,-19,&buffer0[3072],0.002666281769052148,-8,0.004062060732394457,-29,&buffer0[2688]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x24_4x4x24 384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[2688],4,4,24,(const q7_t*) weight27,bias27,shift27,multiplier27,9,29,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,15,-9,-128,127,&buffer0[0],4,4,144,sbuf,9);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight29,bias29,shift29,multiplier29,-14,-15,-128,127,&buffer0[2304],4,4,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x24 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(384, &buffer0[2304],0.0027209443505853415,-14,&buffer0[2688],0.004062060732394457,-29,0.004572504200041294,0,&buffer0[3072]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x24_4x4x24 384 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight30,bias30,shift30,multiplier30,-17,0,-128,127,&buffer0[0],4,4,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x24_4x4x144 55296 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-22,17,-128,127,&buffer0[0],4,4,144,sbuf,-17);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x144_4x4x144 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight32,bias32,shift32,multiplier32,24,22,-128,127,&buffer0[3072],4,4,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x144_4x4x32 73728 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[3072],4,4,32,(const q7_t*) weight33,bias33,shift33,multiplier33,-9,-24,-128,127,&buffer0[0],4,4,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x32_4x4x192 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,-4,9,-128,127,&buffer0[0],4,4,192,sbuf,-9);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x192_4x4x192 27648 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight35,bias35,shift35,multiplier35,37,4,-128,127,&buffer0[3584],4,4,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x192_4x4x32 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(512, &buffer0[3584],0.0022200383245944977,37,&buffer0[3072],0.0018070171354338527,24,0.0031476770527660847,50,&buffer0[4096]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x32_4x4x32 512 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[4096],4,4,32,(const q7_t*) weight36,bias36,shift36,multiplier36,10,-50,-128,127,&buffer0[0],4,4,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x32_4x4x192 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,4,-10,-128,127,&buffer0[0],4,4,192,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x192_4x4x192 27648 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight38,bias38,shift38,multiplier38,-4,-4,-128,127,&buffer0[3072],4,4,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x192_4x4x32 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(512, &buffer0[3072],0.002144323196262121,-4,&buffer0[4096],0.0031476770527660847,50,0.003769106464460492,37,&buffer0[3584]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r4x4x32_4x4x32 512 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[3584],4,4,32,(const q7_t*) weight39,bias39,shift39,multiplier39,-6,-37,-128,127,&buffer0[0],4,4,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r4x4x32_4x4x192 98304 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,1,6,-128,127,&buffer0[0],2,2,192,sbuf,-6);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r4x4x192_2x2x192 6912 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,192,(const q7_t*) weight41,bias41,shift41,multiplier41,-24,-1,-128,127,&buffer0[1344],2,2,56,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x192_2x2x56 43008 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1344],2,2,56,(const q7_t*) weight42,bias42,shift42,multiplier42,5,24,-128,127,&buffer0[0],2,2,336,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x56_2x2x336 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,1,-5,-128,127,&buffer0[0],2,2,336,sbuf,5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r2x2x336_2x2x336 12096 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight44,bias44,shift44,multiplier44,6,-1,-128,127,&buffer0[1568],2,2,56,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x336_2x2x56 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(224, &buffer0[1568],0.0022980249486863613,6,&buffer0[1344],0.002694052876904607,-24,0.003149643074721098,-14,&buffer0[1792]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r2x2x56_2x2x56 224 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1792],2,2,56,(const q7_t*) weight45,bias45,shift45,multiplier45,3,14,-128,127,&buffer0[0],2,2,336,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x56_2x2x336 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,2,-3,-128,127,&buffer0[0],2,2,336,sbuf,3);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r2x2x336_2x2x336 12096 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight47,bias47,shift47,multiplier47,-11,-2,-128,127,&buffer0[1344],2,2,56,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x336_2x2x56 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(224, &buffer0[1344],0.002098551718518138,-11,&buffer0[1792],0.003149643074721098,-14,0.003843615995720029,-7,&buffer0[1568]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r2x2x56_2x2x56 224 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1568],2,2,56,(const q7_t*) weight48,bias48,shift48,multiplier48,14,7,-128,127,&buffer0[0],2,2,336,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x56_2x2x336 75264 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,0,-14,-128,127,&buffer0[0],2,2,336,sbuf,14);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r2x2x336_2x2x336 12096 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight50,bias50,shift50,multiplier50,-2,0,-128,127,&buffer0[5120],2,2,112,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x336_2x2x112 150528 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[5120],2,2,112,(const q7_t*) weight51,bias51,shift51,multiplier51,-17,2,-128,127,&buffer0[0],2,2,1280,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r2x2x112_2x2x1280 573440 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[5120],1,1,1280,(const q7_t*) weight52,bias52,shift52,multiplier52,-19,19,-128,127,&buffer0[5120],1,1,2,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x1280_1x1x2 2560 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}

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
    return &buffer0[102400];
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
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[102400],160,160,3,(const q7_t*) weight0,bias0,shift0,multiplier0,-128,1,-128,127,&buffer0[0],80,80,16,sbuf,kbuf,-1);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r160x160x3_80x80x16 2764800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],80,80,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-128,128,-128,127,&buffer0[0],80,80,16,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r80x80x16_80x80x16 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],80,80,16,(const q7_t*) weight2,bias2,shift2,multiplier2,-6,128,-128,127,&buffer0[153600],80,80,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r80x80x16_80x80x8 819200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[153600],80,80,8,(const q7_t*) weight3,bias3,shift3,multiplier3,-128,6,-128,127,&buffer0[0],80,80,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r80x80x8_80x80x24 1228800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride2_inplace_CHW(&buffer0[0],80,80,24,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,-128,128,-128,127,&buffer0[0],40,40,24,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r80x80x24_40x40x24 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[0],40,40,24,(const q7_t*) weight5,bias5,shift5,multiplier5,15,128,-128,127,&buffer0[153600],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x24_40x40x16 614400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[153600],40,40,16,(const q7_t*) weight6,bias6,shift6,multiplier6,-128,-15,-128,127,&buffer0[0],40,40,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x96 2457600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],40,40,96,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-128,128,-128,127,&buffer0[0],40,40,96,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r40x40x96_40x40x96 7526400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],40,40,96,(const q7_t*) weight8,bias8,shift8,multiplier8,-24,128,-128,127,&buffer0[179200],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x96_40x40x16 2457600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(25600, &buffer0[179200],0.13589729368686676,-24,&buffer0[153600],0.08921323716640472,15,0.11609018594026566,0,&buffer0[128000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r40x40x16_40x40x16 25600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[128000],40,40,16,(const q7_t*) weight9,bias9,shift9,multiplier9,-128,0,-128,127,&buffer0[0],40,40,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x80 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],40,40,80,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-128,128,-128,127,&buffer0[0],40,40,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r40x40x80_40x40x80 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],40,40,80,(const q7_t*) weight11,bias11,shift11,multiplier11,23,128,-128,127,&buffer0[153600],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x80_40x40x16 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(25600, &buffer0[153600],0.07197478413581848,23,&buffer0[128000],0.11609018594026566,0,0.173182412981987,17,&buffer0[179200]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r40x40x16_40x40x16 25600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[179200],40,40,16,(const q7_t*) weight12,bias12,shift12,multiplier12,-128,-17,-128,127,&buffer0[0],40,40,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x80 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],40,40,80,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,-128,128,-128,127,&buffer0[0],40,40,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r40x40x80_40x40x80 3200000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],40,40,80,(const q7_t*) weight14,bias14,shift14,multiplier14,-4,128,-128,127,&buffer0[128000],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x80_40x40x16 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(25600, &buffer0[128000],0.04377514868974686,-4,&buffer0[179200],0.173182412981987,17,0.18031957745552063,18,&buffer0[153600]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r40x40x16_40x40x16 25600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[153600],40,40,16,(const q7_t*) weight15,bias15,shift15,multiplier15,-128,-18,-128,127,&buffer0[0],40,40,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x80 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],40,40,80,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,-128,128,-128,127,&buffer0[0],20,20,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r40x40x80_20x20x80 288000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],20,20,80,(const q7_t*) weight17,bias17,shift17,multiplier17,-5,128,-128,127,&buffer0[57600],20,20,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x80_20x20x24 768000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[57600],20,20,24,(const q7_t*) weight18,bias18,shift18,multiplier18,-128,5,-128,127,&buffer0[0],20,20,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x24_20x20x144 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],20,20,144,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,-128,128,-128,127,&buffer0[0],20,20,144,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r20x20x144_20x20x144 2822400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],20,20,144,(const q7_t*) weight20,bias20,shift20,multiplier20,8,128,-128,127,&buffer0[67200],20,20,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x144_20x20x24 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(9600, &buffer0[67200],0.06724858283996582,8,&buffer0[57600],0.056585509330034256,-5,0.09515953809022903,10,&buffer0[76800]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r20x20x24_20x20x24 9600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[76800],20,20,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-128,-10,-128,127,&buffer0[0],20,20,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x24_20x20x144 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],20,20,144,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,-128,128,-128,127,&buffer0[0],20,20,144,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r20x20x144_20x20x144 1440000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],20,20,144,(const q7_t*) weight23,bias23,shift23,multiplier23,0,128,-128,127,&buffer0[57600],20,20,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x144_20x20x24 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(9600, &buffer0[57600],0.040072161704301834,0,&buffer0[76800],0.09515953809022903,10,0.10984193533658981,12,&buffer0[38400]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r20x20x24_20x20x24 9600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[38400],20,20,24,(const q7_t*) weight24,bias24,shift24,multiplier24,-128,-12,-128,127,&buffer0[0],20,20,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x24_20x20x96 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride2_inplace_CHW(&buffer0[0],20,20,96,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,-128,128,-128,127,&buffer0[0],10,10,96,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r20x20x96_10x10x96 470400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,96,(const q7_t*) weight26,bias26,shift26,multiplier26,-3,128,-128,127,&buffer0[20000],10,10,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x96_10x10x40 384000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20000],10,10,40,(const q7_t*) weight27,bias27,shift27,multiplier27,-128,3,-128,127,&buffer0[0],10,10,200,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x40_10x10x200 800000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],10,10,200,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-128,128,-128,127,&buffer0[0],10,10,200,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r10x10x200_10x10x200 500000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,200,(const q7_t*) weight29,bias29,shift29,multiplier29,-4,128,-128,127,&buffer0[24000],10,10,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x200_10x10x40 800000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4000, &buffer0[24000],0.0487569235265255,-4,&buffer0[20000],0.054970744997262955,-3,0.05753161758184433,0,&buffer0[28000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r10x10x40_10x10x40 4000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[28000],10,10,40,(const q7_t*) weight30,bias30,shift30,multiplier30,-128,0,-128,127,&buffer0[0],10,10,200,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x40_10x10x200 800000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],10,10,200,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-128,128,-128,127,&buffer0[0],10,10,200,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r10x10x200_10x10x200 180000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,200,(const q7_t*) weight32,bias32,shift32,multiplier32,-1,128,-128,127,&buffer0[24000],10,10,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x200_10x10x48 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[24000],10,10,48,(const q7_t*) weight33,bias33,shift33,multiplier33,-128,1,-128,127,&buffer0[0],10,10,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x48_10x10x240 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],10,10,240,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,-128,128,-128,127,&buffer0[0],10,10,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r10x10x240_10x10x240 600000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,240,(const q7_t*) weight35,bias35,shift35,multiplier35,10,128,-128,127,&buffer0[28800],10,10,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x240_10x10x48 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4800, &buffer0[28800],0.06045204773545265,10,&buffer0[24000],0.0628800168633461,-1,0.06692639738321304,-6,&buffer0[19200]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r10x10x48_10x10x48 4800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[19200],10,10,48,(const q7_t*) weight36,bias36,shift36,multiplier36,-128,6,-128,127,&buffer0[0],10,10,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x48_10x10x192 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],10,10,192,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,-128,128,-128,127,&buffer0[0],10,10,192,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r10x10x192_10x10x192 172800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,192,(const q7_t*) weight38,bias38,shift38,multiplier38,27,128,-128,127,&buffer0[24000],10,10,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x192_10x10x48 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4800, &buffer0[24000],0.04233753681182861,27,&buffer0[19200],0.06692639738321304,-6,0.07153769582509995,-4,&buffer0[28800]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r10x10x48_10x10x48 4800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[28800],10,10,48,(const q7_t*) weight39,bias39,shift39,multiplier39,-128,4,-128,127,&buffer0[0],10,10,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x48_10x10x288 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride2_inplace_CHW(&buffer0[0],10,10,288,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,-128,128,-128,127,&buffer0[0],5,5,288,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r10x10x288_5x5x288 180000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,288,(const q7_t*) weight41,bias41,shift41,multiplier41,-3,128,-128,127,&buffer0[9600],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x288_5x5x96 691200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[9600],5,5,96,(const q7_t*) weight42,bias42,shift42,multiplier42,-128,3,-128,127,&buffer0[0],5,5,384,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x384 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],5,5,384,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,-128,128,-128,127,&buffer0[0],5,5,384,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r5x5x384_5x5x384 240000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,384,(const q7_t*) weight44,bias44,shift44,multiplier44,17,128,-128,127,&buffer0[12000],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x384_5x5x96 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2400, &buffer0[12000],0.03900395706295967,17,&buffer0[9600],0.04019002616405487,-3,0.05485454946756363,4,&buffer0[7200]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x96_5x5x96 2400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[7200],5,5,96,(const q7_t*) weight45,bias45,shift45,multiplier45,-128,-4,-128,127,&buffer0[0],5,5,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x288 691200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],5,5,288,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,-128,128,-128,127,&buffer0[0],5,5,288,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r5x5x288_5x5x288 180000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,288,(const q7_t*) weight47,bias47,shift47,multiplier47,-13,128,-128,127,&buffer0[9600],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x288_5x5x96 691200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2400, &buffer0[9600],0.03754161298274994,-13,&buffer0[7200],0.05485454946756363,4,0.06763854622840881,-11,&buffer0[12000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x96_5x5x96 2400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[12000],5,5,96,(const q7_t*) weight48,bias48,shift48,multiplier48,-128,11,-128,127,&buffer0[0],5,5,384,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x384 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,384,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,-128,128,-128,127,&buffer0[0],5,5,384,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x384_5x5x384 86400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,384,(const q7_t*) weight50,bias50,shift50,multiplier50,-19,128,-128,127,&buffer0[9600],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x384_5x5x96 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2400, &buffer0[9600],0.02234080247581005,-19,&buffer0[12000],0.06763854622840881,-11,0.07348241657018661,-8,&buffer0[14400]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x96_5x5x96 2400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[14400],5,5,96,(const q7_t*) weight51,bias51,shift51,multiplier51,-128,8,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x480 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 63:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight52,offsetBias52,offsetRBias52,shift52,multiplier52,-128,128,-128,127,&buffer0[0],5,5,480,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r5x5x480_5x5x480 300000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 64:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight53,bias53,shift53,multiplier53,3,128,-128,127,&buffer0[12000],5,5,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x160 1920000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 65:AVERAGE_POOL_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    avg_pooling(&buffer0[12000],5,5,160,5,5,1,1,-128,127,&buffer0[1000]);
}
end = HAL_GetTick();
sprintf(buf, "AVERAGE_POOL_2D r5x5x160_1x1x160 0 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 66:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1000],1,1,160,(const q7_t*) weight54,bias54,shift54,multiplier54,-35,-3,-128,127,&buffer0[0],1,1,1000,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x160_1x1x1000 160000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[102400],160,160,3,(const q7_t*) weight0,bias0,shift0,multiplier0,-128,1,-128,127,&buffer0[0],80,80,16,sbuf,kbuf,-1);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r160x160x3_80x80x16 2764800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],80,80,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-128,128,-128,127,&buffer0[0],80,80,16,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r80x80x16_80x80x16 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],80,80,16,(const q7_t*) weight2,bias2,shift2,multiplier2,-6,128,-128,127,&buffer0[153600],80,80,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r80x80x16_80x80x8 819200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[153600],80,80,8,(const q7_t*) weight3,bias3,shift3,multiplier3,-128,6,-128,127,&buffer0[0],80,80,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r80x80x8_80x80x24 1228800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride2_inplace_CHW(&buffer0[0],80,80,24,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,-128,128,-128,127,&buffer0[0],40,40,24,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r80x80x24_40x40x24 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[0],40,40,24,(const q7_t*) weight5,bias5,shift5,multiplier5,15,128,-128,127,&buffer0[153600],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x24_40x40x16 614400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[153600],40,40,16,(const q7_t*) weight6,bias6,shift6,multiplier6,-128,-15,-128,127,&buffer0[0],40,40,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x96 2457600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],40,40,96,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-128,128,-128,127,&buffer0[0],40,40,96,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r40x40x96_40x40x96 7526400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],40,40,96,(const q7_t*) weight8,bias8,shift8,multiplier8,-24,128,-128,127,&buffer0[179200],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x96_40x40x16 2457600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(25600, &buffer0[179200],0.13589729368686676,-24,&buffer0[153600],0.08921323716640472,15,0.11609018594026566,0,&buffer0[128000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r40x40x16_40x40x16 25600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[128000],40,40,16,(const q7_t*) weight9,bias9,shift9,multiplier9,-128,0,-128,127,&buffer0[0],40,40,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x80 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],40,40,80,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-128,128,-128,127,&buffer0[0],40,40,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r40x40x80_40x40x80 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],40,40,80,(const q7_t*) weight11,bias11,shift11,multiplier11,23,128,-128,127,&buffer0[153600],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x80_40x40x16 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(25600, &buffer0[153600],0.07197478413581848,23,&buffer0[128000],0.11609018594026566,0,0.173182412981987,17,&buffer0[179200]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r40x40x16_40x40x16 25600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[179200],40,40,16,(const q7_t*) weight12,bias12,shift12,multiplier12,-128,-17,-128,127,&buffer0[0],40,40,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x80 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],40,40,80,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,-128,128,-128,127,&buffer0[0],40,40,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r40x40x80_40x40x80 3200000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],40,40,80,(const q7_t*) weight14,bias14,shift14,multiplier14,-4,128,-128,127,&buffer0[128000],40,40,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x80_40x40x16 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(25600, &buffer0[128000],0.04377514868974686,-4,&buffer0[179200],0.173182412981987,17,0.18031957745552063,18,&buffer0[153600]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r40x40x16_40x40x16 25600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[153600],40,40,16,(const q7_t*) weight15,bias15,shift15,multiplier15,-128,-18,-128,127,&buffer0[0],40,40,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r40x40x16_40x40x80 2048000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],40,40,80,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,-128,128,-128,127,&buffer0[0],20,20,80,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r40x40x80_20x20x80 288000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],20,20,80,(const q7_t*) weight17,bias17,shift17,multiplier17,-5,128,-128,127,&buffer0[57600],20,20,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x80_20x20x24 768000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[57600],20,20,24,(const q7_t*) weight18,bias18,shift18,multiplier18,-128,5,-128,127,&buffer0[0],20,20,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x24_20x20x144 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride1_inplace_CHW(&buffer0[0],20,20,144,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,-128,128,-128,127,&buffer0[0],20,20,144,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r20x20x144_20x20x144 2822400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],20,20,144,(const q7_t*) weight20,bias20,shift20,multiplier20,8,128,-128,127,&buffer0[67200],20,20,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x144_20x20x24 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(9600, &buffer0[67200],0.06724858283996582,8,&buffer0[57600],0.056585509330034256,-5,0.09515953809022903,10,&buffer0[76800]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r20x20x24_20x20x24 9600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[76800],20,20,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-128,-10,-128,127,&buffer0[0],20,20,144,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x24_20x20x144 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],20,20,144,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,-128,128,-128,127,&buffer0[0],20,20,144,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r20x20x144_20x20x144 1440000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],20,20,144,(const q7_t*) weight23,bias23,shift23,multiplier23,0,128,-128,127,&buffer0[57600],20,20,24,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x144_20x20x24 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(9600, &buffer0[57600],0.040072161704301834,0,&buffer0[76800],0.09515953809022903,10,0.10984193533658981,12,&buffer0[38400]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r20x20x24_20x20x24 9600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch24(&buffer0[38400],20,20,24,(const q7_t*) weight24,bias24,shift24,multiplier24,-128,-12,-128,127,&buffer0[0],20,20,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r20x20x24_20x20x96 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel7x7_stride2_inplace_CHW(&buffer0[0],20,20,96,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,-128,128,-128,127,&buffer0[0],10,10,96,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k7x7_r20x20x96_10x10x96 470400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,96,(const q7_t*) weight26,bias26,shift26,multiplier26,-3,128,-128,127,&buffer0[20000],10,10,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x96_10x10x40 384000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20000],10,10,40,(const q7_t*) weight27,bias27,shift27,multiplier27,-128,3,-128,127,&buffer0[0],10,10,200,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x40_10x10x200 800000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],10,10,200,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-128,128,-128,127,&buffer0[0],10,10,200,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r10x10x200_10x10x200 500000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,200,(const q7_t*) weight29,bias29,shift29,multiplier29,-4,128,-128,127,&buffer0[24000],10,10,40,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x200_10x10x40 800000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4000, &buffer0[24000],0.0487569235265255,-4,&buffer0[20000],0.054970744997262955,-3,0.05753161758184433,0,&buffer0[28000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r10x10x40_10x10x40 4000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[28000],10,10,40,(const q7_t*) weight30,bias30,shift30,multiplier30,-128,0,-128,127,&buffer0[0],10,10,200,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x40_10x10x200 800000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],10,10,200,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-128,128,-128,127,&buffer0[0],10,10,200,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r10x10x200_10x10x200 180000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,200,(const q7_t*) weight32,bias32,shift32,multiplier32,-1,128,-128,127,&buffer0[24000],10,10,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x200_10x10x48 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[24000],10,10,48,(const q7_t*) weight33,bias33,shift33,multiplier33,-128,1,-128,127,&buffer0[0],10,10,240,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x48_10x10x240 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],10,10,240,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,-128,128,-128,127,&buffer0[0],10,10,240,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r10x10x240_10x10x240 600000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,240,(const q7_t*) weight35,bias35,shift35,multiplier35,10,128,-128,127,&buffer0[28800],10,10,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x240_10x10x48 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4800, &buffer0[28800],0.06045204773545265,10,&buffer0[24000],0.0628800168633461,-1,0.06692639738321304,-6,&buffer0[19200]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r10x10x48_10x10x48 4800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[19200],10,10,48,(const q7_t*) weight36,bias36,shift36,multiplier36,-128,6,-128,127,&buffer0[0],10,10,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x48_10x10x192 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],10,10,192,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,-128,128,-128,127,&buffer0[0],10,10,192,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r10x10x192_10x10x192 172800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],10,10,192,(const q7_t*) weight38,bias38,shift38,multiplier38,27,128,-128,127,&buffer0[24000],10,10,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x192_10x10x48 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(4800, &buffer0[24000],0.04233753681182861,27,&buffer0[19200],0.06692639738321304,-6,0.07153769582509995,-4,&buffer0[28800]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r10x10x48_10x10x48 4800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[28800],10,10,48,(const q7_t*) weight39,bias39,shift39,multiplier39,-128,4,-128,127,&buffer0[0],10,10,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r10x10x48_10x10x288 1382400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride2_inplace_CHW(&buffer0[0],10,10,288,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,-128,128,-128,127,&buffer0[0],5,5,288,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r10x10x288_5x5x288 180000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,288,(const q7_t*) weight41,bias41,shift41,multiplier41,-3,128,-128,127,&buffer0[9600],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x288_5x5x96 691200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[9600],5,5,96,(const q7_t*) weight42,bias42,shift42,multiplier42,-128,3,-128,127,&buffer0[0],5,5,384,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x384 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],5,5,384,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,-128,128,-128,127,&buffer0[0],5,5,384,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r5x5x384_5x5x384 240000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,384,(const q7_t*) weight44,bias44,shift44,multiplier44,17,128,-128,127,&buffer0[12000],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x384_5x5x96 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2400, &buffer0[12000],0.03900395706295967,17,&buffer0[9600],0.04019002616405487,-3,0.05485454946756363,4,&buffer0[7200]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x96_5x5x96 2400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[7200],5,5,96,(const q7_t*) weight45,bias45,shift45,multiplier45,-128,-4,-128,127,&buffer0[0],5,5,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x288 691200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],5,5,288,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,-128,128,-128,127,&buffer0[0],5,5,288,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r5x5x288_5x5x288 180000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,288,(const q7_t*) weight47,bias47,shift47,multiplier47,-13,128,-128,127,&buffer0[9600],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x288_5x5x96 691200 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2400, &buffer0[9600],0.03754161298274994,-13,&buffer0[7200],0.05485454946756363,4,0.06763854622840881,-11,&buffer0[12000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x96_5x5x96 2400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[12000],5,5,96,(const q7_t*) weight48,bias48,shift48,multiplier48,-128,11,-128,127,&buffer0[0],5,5,384,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x384 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,384,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,-128,128,-128,127,&buffer0[0],5,5,384,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x384_5x5x384 86400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,384,(const q7_t*) weight50,bias50,shift50,multiplier50,-19,128,-128,127,&buffer0[9600],5,5,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x384_5x5x96 921600 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2400, &buffer0[9600],0.02234080247581005,-19,&buffer0[12000],0.06763854622840881,-11,0.07348241657018661,-8,&buffer0[14400]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x96_5x5x96 2400 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[14400],5,5,96,(const q7_t*) weight51,bias51,shift51,multiplier51,-128,8,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x96_5x5x480 1152000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 63:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel5x5_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight52,offsetBias52,offsetRBias52,shift52,multiplier52,-128,128,-128,127,&buffer0[0],5,5,480,sbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k5x5_r5x5x480_5x5x480 300000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 64:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight53,bias53,shift53,multiplier53,3,128,-128,127,&buffer0[12000],5,5,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x160 1920000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 65:AVERAGE_POOL_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    avg_pooling(&buffer0[12000],5,5,160,5,5,1,1,-128,127,&buffer0[1000]);
}
end = HAL_GetTick();
sprintf(buf, "AVERAGE_POOL_2D r5x5x160_1x1x160 0 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 66:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[1000],1,1,160,(const q7_t*) weight54,bias54,shift54,multiplier54,-35,-3,-128,127,&buffer0[0],1,1,1000,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x160_1x1x1000 160000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}

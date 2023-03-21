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
    return &buffer0[82944];
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
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[82944],144,144,3,(const q7_t*) weight0,bias0,shift0,multiplier0,16,128,-128,127,&buffer0[0],72,72,16,sbuf,kbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r144x144x3_72x72x16 2239488 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],72,72,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,11,-16,-128,127,&buffer0[0],72,72,16,sbuf,16);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r72x72x16_72x72x16 746496 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],72,72,16,(const q7_t*) weight2,bias2,shift2,multiplier2,8,-11,-128,127,&buffer0[248832],72,72,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r72x72x16_72x72x8 663552 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[248832],72,72,8,(const q7_t*) weight3,bias3,shift3,multiplier3,20,-8,-128,127,&buffer0[0],72,72,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r72x72x8_72x72x48 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],72,72,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,13,-20,-128,127,&buffer0[0],36,36,48,sbuf,20);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r72x72x48_36x36x48 559872 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],36,36,48,(const q7_t*) weight5,bias5,shift5,multiplier5,-14,-13,-128,127,&buffer0[124416],36,36,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x48_36x36x16 995328 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[124416],36,36,16,(const q7_t*) weight6,bias6,shift6,multiplier6,11,14,-128,127,&buffer0[0],36,36,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x16_36x36x96 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],36,36,96,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-11,-11,-128,127,&buffer0[0],36,36,96,sbuf,11);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r36x36x96_36x36x96 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],36,36,96,(const q7_t*) weight8,bias8,shift8,multiplier8,15,11,-128,127,&buffer0[145152],36,36,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x96_36x36x16 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(20736, &buffer0[145152],0.0019896484445780516,15,&buffer0[124416],0.0016799638979136944,-14,0.002958007389679551,18,&buffer0[165888]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r36x36x16_36x36x16 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[165888],36,36,16,(const q7_t*) weight9,bias9,shift9,multiplier9,4,-18,-128,127,&buffer0[0],36,36,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x16_36x36x96 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],36,36,96,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-2,-4,-128,127,&buffer0[0],18,18,96,sbuf,4);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r36x36x96_18x18x96 279936 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],18,18,96,(const q7_t*) weight11,bias11,shift11,multiplier11,56,2,-128,127,&buffer0[31104],18,18,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x96_18x18x16 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[31104],18,18,16,(const q7_t*) weight12,bias12,shift12,multiplier12,3,-56,-128,127,&buffer0[0],18,18,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x16_18x18x96 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],18,18,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,-1,-3,-128,127,&buffer0[0],18,18,96,sbuf,3);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r18x18x96_18x18x96 279936 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],18,18,96,(const q7_t*) weight14,bias14,shift14,multiplier14,24,1,-128,127,&buffer0[36288],18,18,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x96_18x18x16 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5184, &buffer0[36288],0.0020296196453273296,24,&buffer0[31104],0.0020231460221111774,56,0.0025077250320464373,69,&buffer0[41472]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r18x18x16_18x18x16 5184 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[41472],18,18,16,(const q7_t*) weight15,bias15,shift15,multiplier15,-7,-69,-128,127,&buffer0[0],18,18,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x16_18x18x96 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],18,18,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,27,7,-128,127,&buffer0[0],18,18,96,sbuf,-7);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r18x18x96_18x18x96 279936 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],18,18,96,(const q7_t*) weight17,bias17,shift17,multiplier17,9,-27,-128,127,&buffer0[31104],18,18,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x96_18x18x16 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5184, &buffer0[31104],0.002151512075215578,9,&buffer0[41472],0.0025077250320464373,69,0.003665225114673376,28,&buffer0[36288]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r18x18x16_18x18x16 5184 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[36288],18,18,16,(const q7_t*) weight18,bias18,shift18,multiplier18,-8,-28,-128,127,&buffer0[0],18,18,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x16_18x18x96 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],18,18,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,30,8,-128,127,&buffer0[0],9,9,96,sbuf,-8);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r18x18x96_9x9x96 69984 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,96,(const q7_t*) weight20,bias20,shift20,multiplier20,-14,-30,-128,127,&buffer0[15552],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x96_9x9x32 248832 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[15552],9,9,32,(const q7_t*) weight21,bias21,shift21,multiplier21,5,14,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,23,-5,-128,127,&buffer0[0],9,9,192,sbuf,5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight23,bias23,shift23,multiplier23,9,-23,-128,127,&buffer0[18144],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x32 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2592, &buffer0[18144],0.0015853564254939556,9,&buffer0[15552],0.0018689772114157677,-14,0.0027132427785545588,8,&buffer0[20736]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x32_9x9x32 2592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20736],9,9,32,(const q7_t*) weight24,bias24,shift24,multiplier24,-8,-8,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,16,8,-128,127,&buffer0[0],9,9,192,sbuf,-8);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight26,bias26,shift26,multiplier26,13,-16,-128,127,&buffer0[15552],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x32 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2592, &buffer0[15552],0.0015302314423024654,13,&buffer0[20736],0.0027132427785545588,8,0.0032858115155249834,-6,&buffer0[18144]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x32_9x9x32 2592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[18144],9,9,32,(const q7_t*) weight27,bias27,shift27,multiplier27,10,6,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-11,-10,-128,127,&buffer0[0],9,9,192,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight29,bias29,shift29,multiplier29,-22,11,-128,127,&buffer0[15552],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x32 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2592, &buffer0[15552],0.003072245279327035,-22,&buffer0[18144],0.0032858115155249834,-6,0.0037072228733450174,1,&buffer0[20736]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x32_9x9x32 2592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20736],9,9,32,(const q7_t*) weight30,bias30,shift30,multiplier30,-4,-1,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-12,4,-128,127,&buffer0[0],9,9,192,sbuf,-4);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight32,bias32,shift32,multiplier32,-4,12,-128,127,&buffer0[23328],9,9,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x48 746496 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[23328],9,9,48,(const q7_t*) weight33,bias33,shift33,multiplier33,10,4,-128,127,&buffer0[0],9,9,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x48_9x9x288 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,288,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,13,-10,-128,127,&buffer0[0],9,9,288,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x288_9x9x288 209952 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,288,(const q7_t*) weight35,bias35,shift35,multiplier35,-31,-13,-128,127,&buffer0[27216],9,9,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x288_9x9x48 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3888, &buffer0[27216],0.0022798595018684864,-31,&buffer0[23328],0.0022294726222753525,-4,0.003075397340580821,-6,&buffer0[31104]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x48_9x9x48 3888 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[31104],9,9,48,(const q7_t*) weight36,bias36,shift36,multiplier36,-7,6,-128,127,&buffer0[0],9,9,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x48_9x9x288 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,288,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,4,7,-128,127,&buffer0[0],9,9,288,sbuf,-7);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x288_9x9x288 209952 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,288,(const q7_t*) weight38,bias38,shift38,multiplier38,14,-4,-128,127,&buffer0[23328],9,9,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x288_9x9x48 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3888, &buffer0[23328],0.0023156763054430485,14,&buffer0[31104],0.003075397340580821,-6,0.003238589968532324,-34,&buffer0[27216]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x48_9x9x48 3888 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[27216],9,9,48,(const q7_t*) weight39,bias39,shift39,multiplier39,10,34,-128,127,&buffer0[0],9,9,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x48_9x9x288 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],9,9,288,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,6,-10,-128,127,&buffer0[0],5,5,288,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x288_5x5x288 64800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,288,(const q7_t*) weight41,bias41,shift41,multiplier41,22,-6,-128,127,&buffer0[12000],5,5,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x288_5x5x80 576000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[12000],5,5,80,(const q7_t*) weight42,bias42,shift42,multiplier42,3,-22,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x80_5x5x480 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,-10,-3,-128,127,&buffer0[0],5,5,480,sbuf,3);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x480_5x5x480 108000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight44,bias44,shift44,multiplier44,-7,10,-128,127,&buffer0[14000],5,5,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x80 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2000, &buffer0[14000],0.0028855614364147186,-7,&buffer0[12000],0.0028024734929203987,22,0.0038166821468621492,10,&buffer0[16000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x80_5x5x80 2000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[16000],5,5,80,(const q7_t*) weight45,bias45,shift45,multiplier45,-1,-10,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x80_5x5x480 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,12,1,-128,127,&buffer0[0],5,5,480,sbuf,-1);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x480_5x5x480 108000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight47,bias47,shift47,multiplier47,11,-12,-128,127,&buffer0[12000],5,5,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x80 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2000, &buffer0[12000],0.0028687447775155306,11,&buffer0[16000],0.0038166821468621492,10,0.00556891830638051,5,&buffer0[14000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x80_5x5x80 2000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[14000],5,5,80,(const q7_t*) weight48,bias48,shift48,multiplier48,-5,-5,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x80_5x5x480 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,7,5,-128,127,&buffer0[0],5,5,480,sbuf,-5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x480_5x5x480 108000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight50,bias50,shift50,multiplier50,-7,-7,-128,127,&buffer0[32000],5,5,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x160 1920000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[32000],5,5,160,(const q7_t*) weight51,bias51,shift51,multiplier51,-3,7,-128,127,&buffer0[0],5,5,1280,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x160_5x5x1280 5120000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:AVERAGE_POOL_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    avg_pooling(&buffer0[32000],5,5,1280,5,5,1,1,-128,127,&buffer0[32000]);
}
end = HAL_GetTick();
sprintf(buf, "AVERAGE_POOL_2D r5x5x1280_1x1x1280 0 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 63:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[32000],1,1,1280,(const q7_t*) weight52,bias52,shift52,multiplier52,48,3,-128,127,&buffer0[64000],1,1,2,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x1280_1x1x2 2560 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[82944],144,144,3,(const q7_t*) weight0,bias0,shift0,multiplier0,16,128,-128,127,&buffer0[0],72,72,16,sbuf,kbuf,-128);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k3x3_r144x144x3_72x72x16 2239488 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 1:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],72,72,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,11,-16,-128,127,&buffer0[0],72,72,16,sbuf,16);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r72x72x16_72x72x16 746496 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 2:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[0],72,72,16,(const q7_t*) weight2,bias2,shift2,multiplier2,8,-11,-128,127,&buffer0[248832],72,72,8,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r72x72x16_72x72x8 663552 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 3:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch8(&buffer0[248832],72,72,8,(const q7_t*) weight3,bias3,shift3,multiplier3,20,-8,-128,127,&buffer0[0],72,72,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r72x72x8_72x72x48 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 4:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],72,72,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,13,-20,-128,127,&buffer0[0],36,36,48,sbuf,20);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r72x72x48_36x36x48 559872 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 5:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[0],36,36,48,(const q7_t*) weight5,bias5,shift5,multiplier5,-14,-13,-128,127,&buffer0[124416],36,36,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x48_36x36x16 995328 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 6:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[124416],36,36,16,(const q7_t*) weight6,bias6,shift6,multiplier6,11,14,-128,127,&buffer0[0],36,36,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x16_36x36x96 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 7:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],36,36,96,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-11,-11,-128,127,&buffer0[0],36,36,96,sbuf,11);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r36x36x96_36x36x96 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 8:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],36,36,96,(const q7_t*) weight8,bias8,shift8,multiplier8,15,11,-128,127,&buffer0[145152],36,36,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x96_36x36x16 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 9:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(20736, &buffer0[145152],0.0019896484445780516,15,&buffer0[124416],0.0016799638979136944,-14,0.002958007389679551,18,&buffer0[165888]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r36x36x16_36x36x16 20736 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 10:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[165888],36,36,16,(const q7_t*) weight9,bias9,shift9,multiplier9,4,-18,-128,127,&buffer0[0],36,36,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r36x36x16_36x36x96 1990656 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 11:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],36,36,96,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-2,-4,-128,127,&buffer0[0],18,18,96,sbuf,4);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r36x36x96_18x18x96 279936 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 12:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],18,18,96,(const q7_t*) weight11,bias11,shift11,multiplier11,56,2,-128,127,&buffer0[31104],18,18,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x96_18x18x16 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 13:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[31104],18,18,16,(const q7_t*) weight12,bias12,shift12,multiplier12,3,-56,-128,127,&buffer0[0],18,18,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x16_18x18x96 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 14:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],18,18,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,-1,-3,-128,127,&buffer0[0],18,18,96,sbuf,3);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r18x18x96_18x18x96 279936 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 15:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],18,18,96,(const q7_t*) weight14,bias14,shift14,multiplier14,24,1,-128,127,&buffer0[36288],18,18,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x96_18x18x16 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 16:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5184, &buffer0[36288],0.0020296196453273296,24,&buffer0[31104],0.0020231460221111774,56,0.0025077250320464373,69,&buffer0[41472]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r18x18x16_18x18x16 5184 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 17:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[41472],18,18,16,(const q7_t*) weight15,bias15,shift15,multiplier15,-7,-69,-128,127,&buffer0[0],18,18,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x16_18x18x96 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 18:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],18,18,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,27,7,-128,127,&buffer0[0],18,18,96,sbuf,-7);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r18x18x96_18x18x96 279936 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 19:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],18,18,96,(const q7_t*) weight17,bias17,shift17,multiplier17,9,-27,-128,127,&buffer0[31104],18,18,16,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x96_18x18x16 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 20:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(5184, &buffer0[31104],0.002151512075215578,9,&buffer0[41472],0.0025077250320464373,69,0.003665225114673376,28,&buffer0[36288]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r18x18x16_18x18x16 5184 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 21:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch16(&buffer0[36288],18,18,16,(const q7_t*) weight18,bias18,shift18,multiplier18,-8,-28,-128,127,&buffer0[0],18,18,96,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r18x18x16_18x18x96 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 22:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],18,18,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,30,8,-128,127,&buffer0[0],9,9,96,sbuf,-8);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r18x18x96_9x9x96 69984 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 23:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,96,(const q7_t*) weight20,bias20,shift20,multiplier20,-14,-30,-128,127,&buffer0[15552],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x96_9x9x32 248832 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 24:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[15552],9,9,32,(const q7_t*) weight21,bias21,shift21,multiplier21,5,14,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 25:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,23,-5,-128,127,&buffer0[0],9,9,192,sbuf,5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 26:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight23,bias23,shift23,multiplier23,9,-23,-128,127,&buffer0[18144],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x32 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 27:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2592, &buffer0[18144],0.0015853564254939556,9,&buffer0[15552],0.0018689772114157677,-14,0.0027132427785545588,8,&buffer0[20736]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x32_9x9x32 2592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 28:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20736],9,9,32,(const q7_t*) weight24,bias24,shift24,multiplier24,-8,-8,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 29:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,16,8,-128,127,&buffer0[0],9,9,192,sbuf,-8);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 30:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight26,bias26,shift26,multiplier26,13,-16,-128,127,&buffer0[15552],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x32 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 31:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2592, &buffer0[15552],0.0015302314423024654,13,&buffer0[20736],0.0027132427785545588,8,0.0032858115155249834,-6,&buffer0[18144]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x32_9x9x32 2592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 32:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[18144],9,9,32,(const q7_t*) weight27,bias27,shift27,multiplier27,10,6,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 33:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-11,-10,-128,127,&buffer0[0],9,9,192,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 34:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight29,bias29,shift29,multiplier29,-22,11,-128,127,&buffer0[15552],9,9,32,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x32 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 35:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2592, &buffer0[15552],0.003072245279327035,-22,&buffer0[18144],0.0032858115155249834,-6,0.0037072228733450174,1,&buffer0[20736]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x32_9x9x32 2592 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 36:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[20736],9,9,32,(const q7_t*) weight30,bias30,shift30,multiplier30,-4,-1,-128,127,&buffer0[0],9,9,192,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x32_9x9x192 497664 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 37:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,192,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,-12,4,-128,127,&buffer0[0],9,9,192,sbuf,-4);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x192_9x9x192 139968 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 38:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,192,(const q7_t*) weight32,bias32,shift32,multiplier32,-4,12,-128,127,&buffer0[23328],9,9,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x192_9x9x48 746496 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 39:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[23328],9,9,48,(const q7_t*) weight33,bias33,shift33,multiplier33,10,4,-128,127,&buffer0[0],9,9,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x48_9x9x288 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 40:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,288,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,13,-10,-128,127,&buffer0[0],9,9,288,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x288_9x9x288 209952 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 41:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,288,(const q7_t*) weight35,bias35,shift35,multiplier35,-31,-13,-128,127,&buffer0[27216],9,9,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x288_9x9x48 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 42:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3888, &buffer0[27216],0.0022798595018684864,-31,&buffer0[23328],0.0022294726222753525,-4,0.003075397340580821,-6,&buffer0[31104]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x48_9x9x48 3888 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 43:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[31104],9,9,48,(const q7_t*) weight36,bias36,shift36,multiplier36,-7,6,-128,127,&buffer0[0],9,9,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x48_9x9x288 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 44:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],9,9,288,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,4,7,-128,127,&buffer0[0],9,9,288,sbuf,-7);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x288_9x9x288 209952 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 45:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],9,9,288,(const q7_t*) weight38,bias38,shift38,multiplier38,14,-4,-128,127,&buffer0[23328],9,9,48,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x288_9x9x48 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 46:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(3888, &buffer0[23328],0.0023156763054430485,14,&buffer0[31104],0.003075397340580821,-6,0.003238589968532324,-34,&buffer0[27216]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r9x9x48_9x9x48 3888 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 47:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8_ch48(&buffer0[27216],9,9,48,(const q7_t*) weight39,bias39,shift39,multiplier39,10,34,-128,127,&buffer0[0],9,9,288,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r9x9x48_9x9x288 1119744 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 48:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],9,9,288,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,6,-10,-128,127,&buffer0[0],5,5,288,sbuf,10);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r9x9x288_5x5x288 64800 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 49:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,288,(const q7_t*) weight41,bias41,shift41,multiplier41,22,-6,-128,127,&buffer0[12000],5,5,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x288_5x5x80 576000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 50:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[12000],5,5,80,(const q7_t*) weight42,bias42,shift42,multiplier42,3,-22,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x80_5x5x480 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 51:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,-10,-3,-128,127,&buffer0[0],5,5,480,sbuf,3);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x480_5x5x480 108000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 52:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight44,bias44,shift44,multiplier44,-7,10,-128,127,&buffer0[14000],5,5,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x80 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 53:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2000, &buffer0[14000],0.0028855614364147186,-7,&buffer0[12000],0.0028024734929203987,22,0.0038166821468621492,10,&buffer0[16000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x80_5x5x80 2000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 54:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[16000],5,5,80,(const q7_t*) weight45,bias45,shift45,multiplier45,-1,-10,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x80_5x5x480 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 55:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,12,1,-128,127,&buffer0[0],5,5,480,sbuf,-1);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x480_5x5x480 108000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 56:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight47,bias47,shift47,multiplier47,11,-12,-128,127,&buffer0[12000],5,5,80,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x80 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 57:ADD */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    add_fpreq(2000, &buffer0[12000],0.0028687447775155306,11,&buffer0[16000],0.0038166821468621492,10,0.00556891830638051,5,&buffer0[14000]);
}
end = HAL_GetTick();
sprintf(buf, "ADD r5x5x80_5x5x80 2000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 58:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[14000],5,5,80,(const q7_t*) weight48,bias48,shift48,multiplier48,-5,-5,-128,127,&buffer0[0],5,5,480,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x80_5x5x480 960000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 59:DEPTHWISE_CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],5,5,480,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,7,5,-128,127,&buffer0[0],5,5,480,sbuf,-5);
}
end = HAL_GetTick();
sprintf(buf, "DEPTHWISE_CONV_2D  k3x3_r5x5x480_5x5x480 108000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 60:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[0],5,5,480,(const q7_t*) weight50,bias50,shift50,multiplier50,-7,-7,-128,127,&buffer0[32000],5,5,160,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x480_5x5x160 1920000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 61:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[32000],5,5,160,(const q7_t*) weight51,bias51,shift51,multiplier51,-3,7,-128,127,&buffer0[0],5,5,1280,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r5x5x160_5x5x1280 5120000 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 62:AVERAGE_POOL_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    avg_pooling(&buffer0[32000],5,5,1280,5,5,1,1,-128,127,&buffer0[32000]);
}
end = HAL_GetTick();
sprintf(buf, "AVERAGE_POOL_2D r5x5x1280_1x1x1280 0 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
/* layer 63:CONV_2D */
start = HAL_GetTick();
for(profile_i = 0; profile_i < RUNS; profile_i++) {
    convolve_1x1_s8(&buffer0[32000],1,1,1280,(const q7_t*) weight52,bias52,shift52,multiplier52,48,3,-128,127,&buffer0[64000],1,1,2,sbuf);
}
end = HAL_GetTick();
sprintf(buf, "CONV_2D  k1x1_r1x1x1280_1x1x2 2560 %.2f\r\n", (float)(end - start)/(float)RUNS);printLog(buf);
}

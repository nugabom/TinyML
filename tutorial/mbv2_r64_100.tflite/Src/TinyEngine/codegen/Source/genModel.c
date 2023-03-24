/* Automatically generated source file */
#include <float.h>
#include "arm_nnfunctions.h"

#include "genNN.h"
#include "genModel.h"

#include "tinyengine_function.h"
#include "tinyengine_function_fp.h"


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
convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[16384],64,64,3,(const q7_t*) weight0,bias0,shift0,multiplier0,-2,128,-128,127,&buffer0[0],32,32,16,sbuf,kbuf,-128);
/* layer 1:DEPTHWISE_CONV_2D */
depthwise_kernel1x1_stride1_inplace_CHW(&buffer0[0],32,32,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-3,2,-128,127,&buffer0[0],32,32,16,sbuf,-2);
/* layer 2:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[0],32,32,16,(const q7_t*) weight2,bias2,shift2,multiplier2,-49,3,-128,127,&buffer0[49152],32,32,8,sbuf);
/* layer 3:CONV_2D */
convolve_1x1_s8_ch8(&buffer0[49152],32,32,8,(const q7_t*) weight3,bias3,shift3,multiplier3,5,49,-128,127,&buffer0[0],32,32,48,sbuf);
/* layer 4:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],32,32,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,12,-5,-128,127,&buffer0[0],16,16,48,sbuf,5);
/* layer 5:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight5,bias5,shift5,multiplier5,-23,-12,-128,127,&buffer0[12288],16,16,8,sbuf);
/* layer 6:CONV_2D */
convolve_1x1_s8_ch8(&buffer0[12288],16,16,8,(const q7_t*) weight6,bias6,shift6,multiplier6,4,23,-128,127,&buffer0[0],16,16,48,sbuf);
/* layer 7:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-3,-4,-128,127,&buffer0[0],16,16,48,sbuf,4);
/* layer 8:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight8,bias8,shift8,multiplier8,-69,3,-128,127,&buffer0[14336],16,16,8,sbuf);
/* layer 9:ADD */
add_fpreq(2048, &buffer0[14336],0.0018728891154751182,-69,&buffer0[12288],0.002600747859105468,-23,0.003197368001565337,-34,&buffer0[16384]);
/* layer 10:CONV_2D */
convolve_1x1_s8_ch8(&buffer0[16384],16,16,8,(const q7_t*) weight9,bias9,shift9,multiplier9,34,34,-128,127,&buffer0[0],16,16,48,sbuf);
/* layer 11:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-17,-34,-128,127,&buffer0[0],8,8,48,sbuf,34);
/* layer 12:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[0],8,8,48,(const q7_t*) weight11,bias11,shift11,multiplier11,13,17,-128,127,&buffer0[6144],8,8,16,sbuf);
/* layer 13:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[6144],8,8,16,(const q7_t*) weight12,bias12,shift12,multiplier12,4,-13,-128,127,&buffer0[0],8,8,96,sbuf);
/* layer 14:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,1,-4,-128,127,&buffer0[0],8,8,96,sbuf,4);
/* layer 15:CONV_2D */
convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight14,bias14,shift14,multiplier14,16,-1,-128,127,&buffer0[7168],8,8,16,sbuf);
/* layer 16:ADD */
add_fpreq(1024, &buffer0[7168],0.002639292273670435,16,&buffer0[6144],0.0024709308054298162,13,0.0038081894163042307,56,&buffer0[8192]);
/* layer 17:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[8192],8,8,16,(const q7_t*) weight15,bias15,shift15,multiplier15,21,-56,-128,127,&buffer0[0],8,8,96,sbuf);
/* layer 18:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,-9,-21,-128,127,&buffer0[0],8,8,96,sbuf,21);
/* layer 19:CONV_2D */
convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight17,bias17,shift17,multiplier17,3,9,-128,127,&buffer0[6144],8,8,16,sbuf);
/* layer 20:ADD */
add_fpreq(1024, &buffer0[6144],0.0023181517608463764,3,&buffer0[8192],0.0038081894163042307,56,0.0038906459230929613,36,&buffer0[7168]);
/* layer 21:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[7168],8,8,16,(const q7_t*) weight18,bias18,shift18,multiplier18,-11,-36,-128,127,&buffer0[0],8,8,96,sbuf);
/* layer 22:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,-28,11,-128,127,&buffer0[0],4,4,96,sbuf,-11);
/* layer 23:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,96,(const q7_t*) weight20,bias20,shift20,multiplier20,30,28,-128,127,&buffer0[2304],4,4,24,sbuf);
/* layer 24:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[2304],4,4,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-6,-30,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 25:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,2,6,-128,127,&buffer0[0],4,4,144,sbuf,-6);
/* layer 26:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight23,bias23,shift23,multiplier23,13,-2,-128,127,&buffer0[2688],4,4,24,sbuf);
/* layer 27:ADD */
add_fpreq(384, &buffer0[2688],0.0018711609300225973,13,&buffer0[2304],0.0021679559722542763,30,0.002957628108561039,8,&buffer0[3072]);
/* layer 28:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight24,bias24,shift24,multiplier24,5,-8,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 29:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,8,-5,-128,127,&buffer0[0],4,4,144,sbuf,5);
/* layer 30:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight26,bias26,shift26,multiplier26,7,-8,-128,127,&buffer0[2304],4,4,24,sbuf);
/* layer 31:ADD */
add_fpreq(384, &buffer0[2304],0.00264854752458632,7,&buffer0[3072],0.002957628108561039,8,0.003491876646876335,18,&buffer0[2688]);
/* layer 32:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[2688],4,4,24,(const q7_t*) weight27,bias27,shift27,multiplier27,0,-18,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 33:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-8,0,-128,127,&buffer0[0],4,4,144,sbuf,0);
/* layer 34:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight29,bias29,shift29,multiplier29,-18,8,-128,127,&buffer0[2304],4,4,24,sbuf);
/* layer 35:ADD */
add_fpreq(384, &buffer0[2304],0.002226592507213354,-18,&buffer0[2688],0.003491876646876335,18,0.0038974937051534653,31,&buffer0[3072]);
/* layer 36:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight30,bias30,shift30,multiplier30,-1,-31,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 37:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,9,1,-128,127,&buffer0[0],4,4,144,sbuf,-1);
/* layer 38:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight32,bias32,shift32,multiplier32,-29,-9,-128,127,&buffer0[3072],4,4,32,sbuf);
/* layer 39:CONV_2D */
convolve_1x1_s8(&buffer0[3072],4,4,32,(const q7_t*) weight33,bias33,shift33,multiplier33,-16,29,-128,127,&buffer0[0],4,4,192,sbuf);
/* layer 40:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,6,16,-128,127,&buffer0[0],4,4,192,sbuf,-16);
/* layer 41:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight35,bias35,shift35,multiplier35,-16,-6,-128,127,&buffer0[3584],4,4,32,sbuf);
/* layer 42:ADD */
add_fpreq(512, &buffer0[3584],0.0026444317772984505,-16,&buffer0[3072],0.0024075605906546116,-29,0.003780597588047385,-26,&buffer0[4096]);
/* layer 43:CONV_2D */
convolve_1x1_s8(&buffer0[4096],4,4,32,(const q7_t*) weight36,bias36,shift36,multiplier36,-8,26,-128,127,&buffer0[0],4,4,192,sbuf);
/* layer 44:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,2,8,-128,127,&buffer0[0],4,4,192,sbuf,-8);
/* layer 45:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight38,bias38,shift38,multiplier38,-9,-2,-128,127,&buffer0[3072],4,4,32,sbuf);
/* layer 46:ADD */
add_fpreq(512, &buffer0[3072],0.0018214043229818344,-9,&buffer0[4096],0.003780597588047385,-26,0.004385652486234903,-28,&buffer0[3584]);
/* layer 47:CONV_2D */
convolve_1x1_s8(&buffer0[3584],4,4,32,(const q7_t*) weight39,bias39,shift39,multiplier39,12,28,-128,127,&buffer0[0],4,4,192,sbuf);
/* layer 48:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,-3,-12,-128,127,&buffer0[0],2,2,192,sbuf,12);
/* layer 49:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,192,(const q7_t*) weight41,bias41,shift41,multiplier41,2,3,-128,127,&buffer0[1344],2,2,56,sbuf);
/* layer 50:CONV_2D */
convolve_1x1_s8(&buffer0[1344],2,2,56,(const q7_t*) weight42,bias42,shift42,multiplier42,14,-2,-128,127,&buffer0[0],2,2,336,sbuf);
/* layer 51:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,6,-14,-128,127,&buffer0[0],2,2,336,sbuf,14);
/* layer 52:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight44,bias44,shift44,multiplier44,-2,-6,-128,127,&buffer0[1568],2,2,56,sbuf);
/* layer 53:ADD */
add_fpreq(224, &buffer0[1568],0.002816106891259551,-2,&buffer0[1344],0.002431119093671441,2,0.003238544799387455,-17,&buffer0[1792]);
/* layer 54:CONV_2D */
convolve_1x1_s8(&buffer0[1792],2,2,56,(const q7_t*) weight45,bias45,shift45,multiplier45,5,17,-128,127,&buffer0[0],2,2,336,sbuf);
/* layer 55:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,-1,-5,-128,127,&buffer0[0],2,2,336,sbuf,5);
/* layer 56:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight47,bias47,shift47,multiplier47,-17,1,-128,127,&buffer0[1344],2,2,56,sbuf);
/* layer 57:ADD */
add_fpreq(224, &buffer0[1344],0.002762523479759693,-17,&buffer0[1792],0.003238544799387455,-17,0.004113158211112022,-33,&buffer0[1568]);
/* layer 58:CONV_2D */
convolve_1x1_s8(&buffer0[1568],2,2,56,(const q7_t*) weight48,bias48,shift48,multiplier48,-9,33,-128,127,&buffer0[0],2,2,336,sbuf);
/* layer 59:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,-8,9,-128,127,&buffer0[0],2,2,336,sbuf,-9);
/* layer 60:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight50,bias50,shift50,multiplier50,23,8,-128,127,&buffer0[5120],2,2,112,sbuf);
/* layer 61:CONV_2D */
convolve_1x1_s8(&buffer0[5120],2,2,112,(const q7_t*) weight51,bias51,shift51,multiplier51,-11,-23,-128,127,&buffer0[0],2,2,1280,sbuf);
/* layer 62:CONV_2D */
convolve_1x1_s8(&buffer0[5120],1,1,1280,(const q7_t*) weight52,bias52,shift52,multiplier52,-24,12,-128,127,&buffer0[5120],1,1,100,sbuf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
convolve_s8_kernel3_inputch3_stride2_pad1(&buffer0[16384],64,64,3,(const q7_t*) weight0,bias0,shift0,multiplier0,-2,128,-128,127,&buffer0[0],32,32,16,sbuf,kbuf,-128);
/* layer 1:DEPTHWISE_CONV_2D */
depthwise_kernel1x1_stride1_inplace_CHW(&buffer0[0],32,32,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,shift1,multiplier1,-3,2,-128,127,&buffer0[0],32,32,16,sbuf,-2);
/* layer 2:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[0],32,32,16,(const q7_t*) weight2,bias2,shift2,multiplier2,-49,3,-128,127,&buffer0[49152],32,32,8,sbuf);
/* layer 3:CONV_2D */
convolve_1x1_s8_ch8(&buffer0[49152],32,32,8,(const q7_t*) weight3,bias3,shift3,multiplier3,5,49,-128,127,&buffer0[0],32,32,48,sbuf);
/* layer 4:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],32,32,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,shift4,multiplier4,12,-5,-128,127,&buffer0[0],16,16,48,sbuf,5);
/* layer 5:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight5,bias5,shift5,multiplier5,-23,-12,-128,127,&buffer0[12288],16,16,8,sbuf);
/* layer 6:CONV_2D */
convolve_1x1_s8_ch8(&buffer0[12288],16,16,8,(const q7_t*) weight6,bias6,shift6,multiplier6,4,23,-128,127,&buffer0[0],16,16,48,sbuf);
/* layer 7:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,shift7,multiplier7,-3,-4,-128,127,&buffer0[0],16,16,48,sbuf,4);
/* layer 8:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[0],16,16,48,(const q7_t*) weight8,bias8,shift8,multiplier8,-69,3,-128,127,&buffer0[14336],16,16,8,sbuf);
/* layer 9:ADD */
add_fpreq(2048, &buffer0[14336],0.0018728891154751182,-69,&buffer0[12288],0.002600747859105468,-23,0.003197368001565337,-34,&buffer0[16384]);
/* layer 10:CONV_2D */
convolve_1x1_s8_ch8(&buffer0[16384],16,16,8,(const q7_t*) weight9,bias9,shift9,multiplier9,34,34,-128,127,&buffer0[0],16,16,48,sbuf);
/* layer 11:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],16,16,48,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,shift10,multiplier10,-17,-34,-128,127,&buffer0[0],8,8,48,sbuf,34);
/* layer 12:CONV_2D */
convolve_1x1_s8_ch48(&buffer0[0],8,8,48,(const q7_t*) weight11,bias11,shift11,multiplier11,13,17,-128,127,&buffer0[6144],8,8,16,sbuf);
/* layer 13:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[6144],8,8,16,(const q7_t*) weight12,bias12,shift12,multiplier12,4,-13,-128,127,&buffer0[0],8,8,96,sbuf);
/* layer 14:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,shift13,multiplier13,1,-4,-128,127,&buffer0[0],8,8,96,sbuf,4);
/* layer 15:CONV_2D */
convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight14,bias14,shift14,multiplier14,16,-1,-128,127,&buffer0[7168],8,8,16,sbuf);
/* layer 16:ADD */
add_fpreq(1024, &buffer0[7168],0.002639292273670435,16,&buffer0[6144],0.0024709308054298162,13,0.0038081894163042307,56,&buffer0[8192]);
/* layer 17:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[8192],8,8,16,(const q7_t*) weight15,bias15,shift15,multiplier15,21,-56,-128,127,&buffer0[0],8,8,96,sbuf);
/* layer 18:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,shift16,multiplier16,-9,-21,-128,127,&buffer0[0],8,8,96,sbuf,21);
/* layer 19:CONV_2D */
convolve_1x1_s8(&buffer0[0],8,8,96,(const q7_t*) weight17,bias17,shift17,multiplier17,3,9,-128,127,&buffer0[6144],8,8,16,sbuf);
/* layer 20:ADD */
add_fpreq(1024, &buffer0[6144],0.0023181517608463764,3,&buffer0[8192],0.0038081894163042307,56,0.0038906459230929613,36,&buffer0[7168]);
/* layer 21:CONV_2D */
convolve_1x1_s8_ch16(&buffer0[7168],8,8,16,(const q7_t*) weight18,bias18,shift18,multiplier18,-11,-36,-128,127,&buffer0[0],8,8,96,sbuf);
/* layer 22:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],8,8,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,shift19,multiplier19,-28,11,-128,127,&buffer0[0],4,4,96,sbuf,-11);
/* layer 23:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,96,(const q7_t*) weight20,bias20,shift20,multiplier20,30,28,-128,127,&buffer0[2304],4,4,24,sbuf);
/* layer 24:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[2304],4,4,24,(const q7_t*) weight21,bias21,shift21,multiplier21,-6,-30,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 25:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,shift22,multiplier22,2,6,-128,127,&buffer0[0],4,4,144,sbuf,-6);
/* layer 26:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight23,bias23,shift23,multiplier23,13,-2,-128,127,&buffer0[2688],4,4,24,sbuf);
/* layer 27:ADD */
add_fpreq(384, &buffer0[2688],0.0018711609300225973,13,&buffer0[2304],0.0021679559722542763,30,0.002957628108561039,8,&buffer0[3072]);
/* layer 28:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight24,bias24,shift24,multiplier24,5,-8,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 29:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,shift25,multiplier25,8,-5,-128,127,&buffer0[0],4,4,144,sbuf,5);
/* layer 30:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight26,bias26,shift26,multiplier26,7,-8,-128,127,&buffer0[2304],4,4,24,sbuf);
/* layer 31:ADD */
add_fpreq(384, &buffer0[2304],0.00264854752458632,7,&buffer0[3072],0.002957628108561039,8,0.003491876646876335,18,&buffer0[2688]);
/* layer 32:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[2688],4,4,24,(const q7_t*) weight27,bias27,shift27,multiplier27,0,-18,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 33:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,shift28,multiplier28,-8,0,-128,127,&buffer0[0],4,4,144,sbuf,0);
/* layer 34:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight29,bias29,shift29,multiplier29,-18,8,-128,127,&buffer0[2304],4,4,24,sbuf);
/* layer 35:ADD */
add_fpreq(384, &buffer0[2304],0.002226592507213354,-18,&buffer0[2688],0.003491876646876335,18,0.0038974937051534653,31,&buffer0[3072]);
/* layer 36:CONV_2D */
convolve_1x1_s8_ch24(&buffer0[3072],4,4,24,(const q7_t*) weight30,bias30,shift30,multiplier30,-1,-31,-128,127,&buffer0[0],4,4,144,sbuf);
/* layer 37:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,144,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,shift31,multiplier31,9,1,-128,127,&buffer0[0],4,4,144,sbuf,-1);
/* layer 38:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,144,(const q7_t*) weight32,bias32,shift32,multiplier32,-29,-9,-128,127,&buffer0[3072],4,4,32,sbuf);
/* layer 39:CONV_2D */
convolve_1x1_s8(&buffer0[3072],4,4,32,(const q7_t*) weight33,bias33,shift33,multiplier33,-16,29,-128,127,&buffer0[0],4,4,192,sbuf);
/* layer 40:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,shift34,multiplier34,6,16,-128,127,&buffer0[0],4,4,192,sbuf,-16);
/* layer 41:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight35,bias35,shift35,multiplier35,-16,-6,-128,127,&buffer0[3584],4,4,32,sbuf);
/* layer 42:ADD */
add_fpreq(512, &buffer0[3584],0.0026444317772984505,-16,&buffer0[3072],0.0024075605906546116,-29,0.003780597588047385,-26,&buffer0[4096]);
/* layer 43:CONV_2D */
convolve_1x1_s8(&buffer0[4096],4,4,32,(const q7_t*) weight36,bias36,shift36,multiplier36,-8,26,-128,127,&buffer0[0],4,4,192,sbuf);
/* layer 44:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,shift37,multiplier37,2,8,-128,127,&buffer0[0],4,4,192,sbuf,-8);
/* layer 45:CONV_2D */
convolve_1x1_s8(&buffer0[0],4,4,192,(const q7_t*) weight38,bias38,shift38,multiplier38,-9,-2,-128,127,&buffer0[3072],4,4,32,sbuf);
/* layer 46:ADD */
add_fpreq(512, &buffer0[3072],0.0018214043229818344,-9,&buffer0[4096],0.003780597588047385,-26,0.004385652486234903,-28,&buffer0[3584]);
/* layer 47:CONV_2D */
convolve_1x1_s8(&buffer0[3584],4,4,32,(const q7_t*) weight39,bias39,shift39,multiplier39,12,28,-128,127,&buffer0[0],4,4,192,sbuf);
/* layer 48:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW(&buffer0[0],4,4,192,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,shift40,multiplier40,-3,-12,-128,127,&buffer0[0],2,2,192,sbuf,12);
/* layer 49:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,192,(const q7_t*) weight41,bias41,shift41,multiplier41,2,3,-128,127,&buffer0[1344],2,2,56,sbuf);
/* layer 50:CONV_2D */
convolve_1x1_s8(&buffer0[1344],2,2,56,(const q7_t*) weight42,bias42,shift42,multiplier42,14,-2,-128,127,&buffer0[0],2,2,336,sbuf);
/* layer 51:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,shift43,multiplier43,6,-14,-128,127,&buffer0[0],2,2,336,sbuf,14);
/* layer 52:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight44,bias44,shift44,multiplier44,-2,-6,-128,127,&buffer0[1568],2,2,56,sbuf);
/* layer 53:ADD */
add_fpreq(224, &buffer0[1568],0.002816106891259551,-2,&buffer0[1344],0.002431119093671441,2,0.003238544799387455,-17,&buffer0[1792]);
/* layer 54:CONV_2D */
convolve_1x1_s8(&buffer0[1792],2,2,56,(const q7_t*) weight45,bias45,shift45,multiplier45,5,17,-128,127,&buffer0[0],2,2,336,sbuf);
/* layer 55:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,shift46,multiplier46,-1,-5,-128,127,&buffer0[0],2,2,336,sbuf,5);
/* layer 56:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight47,bias47,shift47,multiplier47,-17,1,-128,127,&buffer0[1344],2,2,56,sbuf);
/* layer 57:ADD */
add_fpreq(224, &buffer0[1344],0.002762523479759693,-17,&buffer0[1792],0.003238544799387455,-17,0.004113158211112022,-33,&buffer0[1568]);
/* layer 58:CONV_2D */
convolve_1x1_s8(&buffer0[1568],2,2,56,(const q7_t*) weight48,bias48,shift48,multiplier48,-9,33,-128,127,&buffer0[0],2,2,336,sbuf);
/* layer 59:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW(&buffer0[0],2,2,336,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,shift49,multiplier49,-8,9,-128,127,&buffer0[0],2,2,336,sbuf,-9);
/* layer 60:CONV_2D */
convolve_1x1_s8(&buffer0[0],2,2,336,(const q7_t*) weight50,bias50,shift50,multiplier50,23,8,-128,127,&buffer0[5120],2,2,112,sbuf);
/* layer 61:CONV_2D */
convolve_1x1_s8(&buffer0[5120],2,2,112,(const q7_t*) weight51,bias51,shift51,multiplier51,-11,-23,-128,127,&buffer0[0],2,2,1280,sbuf);
/* layer 62:CONV_2D */
convolve_1x1_s8(&buffer0[5120],1,1,1280,(const q7_t*) weight52,bias52,shift52,multiplier52,-24,12,-128,127,&buffer0[5120],1,1,100,sbuf);
}

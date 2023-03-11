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
convolve_s8_kernel3_inputch3_stride2_pad1_fpreq(&buffer0[82944],144,144,3,(const q7_t*) weight0,bias0,scales0,23,128,-128,127,&buffer0[0],72,72,16,sbuf,kbuf,-128);
/* layer 1:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],72,72,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,scales1,-4,-23,-128,127,&buffer0[0],72,72,16,sbuf,23);
/* layer 2:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[0],72,72,16,(const q7_t*) weight2,bias2,scales2,30,4,-128,127,&buffer0[248832],72,72,8,sbuf);
/* layer 3:CONV_2D */
convolve_1x1_s8_ch8_fpreq(&buffer0[248832],72,72,8,(const q7_t*) weight3,bias3,scales3,-11,-30,-128,127,&buffer0[0],72,72,48,sbuf);
/* layer 4:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],72,72,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,scales4,6,11,-128,127,&buffer0[0],36,36,48,sbuf,-11);
/* layer 5:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[0],36,36,48,(const q7_t*) weight5,bias5,scales5,-35,-6,-128,127,&buffer0[124416],36,36,16,sbuf);
/* layer 6:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[124416],36,36,16,(const q7_t*) weight6,bias6,scales6,-7,35,-128,127,&buffer0[0],36,36,96,sbuf);
/* layer 7:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],36,36,96,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,scales7,-2,7,-128,127,&buffer0[0],36,36,96,sbuf,-7);
/* layer 8:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],36,36,96,(const q7_t*) weight8,bias8,scales8,3,2,-128,127,&buffer0[145152],36,36,16,sbuf);
/* layer 9:ADD */
add_fpreq(20736, &buffer0[145152],0.002402611542493105,3,&buffer0[124416],0.0021880092099308968,-35,0.0029951613396406174,27,&buffer0[165888]);
/* layer 10:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[165888],36,36,16,(const q7_t*) weight9,bias9,scales9,-9,-27,-128,127,&buffer0[0],36,36,96,sbuf);
/* layer 11:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],36,36,96,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,scales10,12,9,-128,127,&buffer0[0],18,18,96,sbuf,-9);
/* layer 12:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],18,18,96,(const q7_t*) weight11,bias11,scales11,-3,-12,-128,127,&buffer0[31104],18,18,16,sbuf);
/* layer 13:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[31104],18,18,16,(const q7_t*) weight12,bias12,scales12,-15,3,-128,127,&buffer0[0],18,18,96,sbuf);
/* layer 14:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],18,18,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,scales13,13,15,-128,127,&buffer0[0],18,18,96,sbuf,-15);
/* layer 15:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],18,18,96,(const q7_t*) weight14,bias14,scales14,-1,-13,-128,127,&buffer0[36288],18,18,16,sbuf);
/* layer 16:ADD */
add_fpreq(5184, &buffer0[36288],0.0024562906473875046,-1,&buffer0[31104],0.0020290270913392305,-3,0.002654152689501643,-30,&buffer0[41472]);
/* layer 17:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[41472],18,18,16,(const q7_t*) weight15,bias15,scales15,9,30,-128,127,&buffer0[0],18,18,96,sbuf);
/* layer 18:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],18,18,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,scales16,2,-9,-128,127,&buffer0[0],18,18,96,sbuf,9);
/* layer 19:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],18,18,96,(const q7_t*) weight17,bias17,scales17,-21,-2,-128,127,&buffer0[31104],18,18,16,sbuf);
/* layer 20:ADD */
add_fpreq(5184, &buffer0[31104],0.002441603224724531,-21,&buffer0[41472],0.002654152689501643,-30,0.0028999417554587126,-36,&buffer0[36288]);
/* layer 21:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[36288],18,18,16,(const q7_t*) weight18,bias18,scales18,-5,36,-128,127,&buffer0[0],18,18,96,sbuf);
/* layer 22:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],18,18,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,scales19,15,5,-128,127,&buffer0[0],9,9,96,sbuf,-5);
/* layer 23:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,96,(const q7_t*) weight20,bias20,scales20,-20,-15,-128,127,&buffer0[15552],9,9,32,sbuf);
/* layer 24:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[15552],9,9,32,(const q7_t*) weight21,bias21,scales21,2,20,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 25:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,scales22,-3,-2,-128,127,&buffer0[0],9,9,192,sbuf,2);
/* layer 26:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight23,bias23,scales23,7,3,-128,127,&buffer0[18144],9,9,32,sbuf);
/* layer 27:ADD */
add_fpreq(2592, &buffer0[18144],0.0021028753835707903,7,&buffer0[15552],0.002326514106243849,-20,0.00291330274194479,-18,&buffer0[20736]);
/* layer 28:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[20736],9,9,32,(const q7_t*) weight24,bias24,scales24,-7,18,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 29:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,scales25,0,7,-128,127,&buffer0[0],9,9,192,sbuf,-7);
/* layer 30:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight26,bias26,scales26,-2,0,-128,127,&buffer0[15552],9,9,32,sbuf);
/* layer 31:ADD */
add_fpreq(2592, &buffer0[15552],0.0028268916066735983,-2,&buffer0[20736],0.00291330274194479,-18,0.005000795237720013,-3,&buffer0[18144]);
/* layer 32:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[18144],9,9,32,(const q7_t*) weight27,bias27,scales27,-3,3,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 33:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,scales28,28,3,-128,127,&buffer0[0],9,9,192,sbuf,-3);
/* layer 34:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight29,bias29,scales29,-44,-28,-128,127,&buffer0[15552],9,9,32,sbuf);
/* layer 35:ADD */
add_fpreq(2592, &buffer0[15552],0.001877687405794859,-44,&buffer0[18144],0.005000795237720013,-3,0.004993002861738205,-9,&buffer0[20736]);
/* layer 36:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[20736],9,9,32,(const q7_t*) weight30,bias30,scales30,-15,9,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 37:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,scales31,10,15,-128,127,&buffer0[0],9,9,192,sbuf,-15);
/* layer 38:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight32,bias32,scales32,-12,-10,-128,127,&buffer0[23328],9,9,48,sbuf);
/* layer 39:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[23328],9,9,48,(const q7_t*) weight33,bias33,scales33,14,12,-128,127,&buffer0[0],9,9,288,sbuf);
/* layer 40:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,288,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,scales34,16,-14,-128,127,&buffer0[0],9,9,288,sbuf,14);
/* layer 41:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,288,(const q7_t*) weight35,bias35,scales35,-1,-16,-128,127,&buffer0[27216],9,9,48,sbuf);
/* layer 42:ADD */
add_fpreq(3888, &buffer0[27216],0.002591863740235567,-1,&buffer0[23328],0.001980195054784417,-12,0.0031075673177838326,1,&buffer0[31104]);
/* layer 43:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[31104],9,9,48,(const q7_t*) weight36,bias36,scales36,-3,-1,-128,127,&buffer0[0],9,9,288,sbuf);
/* layer 44:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,288,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,scales37,19,3,-128,127,&buffer0[0],9,9,288,sbuf,-3);
/* layer 45:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,288,(const q7_t*) weight38,bias38,scales38,-17,-19,-128,127,&buffer0[23328],9,9,48,sbuf);
/* layer 46:ADD */
add_fpreq(3888, &buffer0[23328],0.0022428107913583517,-17,&buffer0[31104],0.0031075673177838326,1,0.0036475802771747112,-16,&buffer0[27216]);
/* layer 47:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[27216],9,9,48,(const q7_t*) weight39,bias39,scales39,-7,16,-128,127,&buffer0[0],9,9,288,sbuf);
/* layer 48:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],9,9,288,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,scales40,4,7,-128,127,&buffer0[0],5,5,288,sbuf,-7);
/* layer 49:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,288,(const q7_t*) weight41,bias41,scales41,1,-4,-128,127,&buffer0[12000],5,5,80,sbuf);
/* layer 50:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[12000],5,5,80,(const q7_t*) weight42,bias42,scales42,6,-1,-128,127,&buffer0[0],5,5,480,sbuf);
/* layer 51:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],5,5,480,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,scales43,-9,-6,-128,127,&buffer0[0],5,5,480,sbuf,6);
/* layer 52:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,480,(const q7_t*) weight44,bias44,scales44,-2,9,-128,127,&buffer0[14000],5,5,80,sbuf);
/* layer 53:ADD */
add_fpreq(2000, &buffer0[14000],0.0026811123825609684,-2,&buffer0[12000],0.0023005802650004625,1,0.00359399919398129,6,&buffer0[16000]);
/* layer 54:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[16000],5,5,80,(const q7_t*) weight45,bias45,scales45,-2,-6,-128,127,&buffer0[0],5,5,480,sbuf);
/* layer 55:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],5,5,480,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,scales46,8,2,-128,127,&buffer0[0],5,5,480,sbuf,-2);
/* layer 56:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,480,(const q7_t*) weight47,bias47,scales47,2,-8,-128,127,&buffer0[12000],5,5,80,sbuf);
/* layer 57:ADD */
add_fpreq(2000, &buffer0[12000],0.0026515808422118425,2,&buffer0[16000],0.00359399919398129,6,0.003986136522144079,6,&buffer0[14000]);
/* layer 58:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[14000],5,5,80,(const q7_t*) weight48,bias48,scales48,-10,-6,-128,127,&buffer0[0],5,5,480,sbuf);
/* layer 59:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],5,5,480,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,scales49,-27,10,-128,127,&buffer0[0],5,5,480,sbuf,-10);
/* layer 60:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,480,(const q7_t*) weight50,bias50,scales50,18,27,-128,127,&buffer0[32000],5,5,160,sbuf);
/* layer 61:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[32000],5,5,160,(const q7_t*) weight51,bias51,scales51,-2,-18,-128,127,&buffer0[0],5,5,1280,sbuf);
/* layer 62:AVERAGE_POOL_2D */
avg_pooling(&buffer0[32000],5,5,1280,5,5,1,1,-128,127,&buffer0[32000]);
/* layer 63:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[32000],1,1,1280,(const q7_t*) weight52,bias52,scales52,15,2,-128,127,&buffer0[64000],1,1,1000,sbuf);
}
void invoke_inf(){
/* layer 0:CONV_2D */
convolve_s8_kernel3_inputch3_stride2_pad1_fpreq(&buffer0[82944],144,144,3,(const q7_t*) weight0,bias0,scales0,23,128,-128,127,&buffer0[0],72,72,16,sbuf,kbuf,-128);
/* layer 1:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],72,72,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,scales1,-4,-23,-128,127,&buffer0[0],72,72,16,sbuf,23);
/* layer 2:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[0],72,72,16,(const q7_t*) weight2,bias2,scales2,30,4,-128,127,&buffer0[248832],72,72,8,sbuf);
/* layer 3:CONV_2D */
convolve_1x1_s8_ch8_fpreq(&buffer0[248832],72,72,8,(const q7_t*) weight3,bias3,scales3,-11,-30,-128,127,&buffer0[0],72,72,48,sbuf);
/* layer 4:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],72,72,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,scales4,6,11,-128,127,&buffer0[0],36,36,48,sbuf,-11);
/* layer 5:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[0],36,36,48,(const q7_t*) weight5,bias5,scales5,-35,-6,-128,127,&buffer0[124416],36,36,16,sbuf);
/* layer 6:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[124416],36,36,16,(const q7_t*) weight6,bias6,scales6,-7,35,-128,127,&buffer0[0],36,36,96,sbuf);
/* layer 7:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],36,36,96,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,scales7,-2,7,-128,127,&buffer0[0],36,36,96,sbuf,-7);
/* layer 8:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],36,36,96,(const q7_t*) weight8,bias8,scales8,3,2,-128,127,&buffer0[145152],36,36,16,sbuf);
/* layer 9:ADD */
add_fpreq(20736, &buffer0[145152],0.002402611542493105,3,&buffer0[124416],0.0021880092099308968,-35,0.0029951613396406174,27,&buffer0[165888]);
/* layer 10:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[165888],36,36,16,(const q7_t*) weight9,bias9,scales9,-9,-27,-128,127,&buffer0[0],36,36,96,sbuf);
/* layer 11:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],36,36,96,(const q7_t*) CHWweight10,offsetBias10,offsetRBias10,scales10,12,9,-128,127,&buffer0[0],18,18,96,sbuf,-9);
/* layer 12:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],18,18,96,(const q7_t*) weight11,bias11,scales11,-3,-12,-128,127,&buffer0[31104],18,18,16,sbuf);
/* layer 13:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[31104],18,18,16,(const q7_t*) weight12,bias12,scales12,-15,3,-128,127,&buffer0[0],18,18,96,sbuf);
/* layer 14:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],18,18,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,scales13,13,15,-128,127,&buffer0[0],18,18,96,sbuf,-15);
/* layer 15:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],18,18,96,(const q7_t*) weight14,bias14,scales14,-1,-13,-128,127,&buffer0[36288],18,18,16,sbuf);
/* layer 16:ADD */
add_fpreq(5184, &buffer0[36288],0.0024562906473875046,-1,&buffer0[31104],0.0020290270913392305,-3,0.002654152689501643,-30,&buffer0[41472]);
/* layer 17:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[41472],18,18,16,(const q7_t*) weight15,bias15,scales15,9,30,-128,127,&buffer0[0],18,18,96,sbuf);
/* layer 18:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],18,18,96,(const q7_t*) CHWweight16,offsetBias16,offsetRBias16,scales16,2,-9,-128,127,&buffer0[0],18,18,96,sbuf,9);
/* layer 19:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],18,18,96,(const q7_t*) weight17,bias17,scales17,-21,-2,-128,127,&buffer0[31104],18,18,16,sbuf);
/* layer 20:ADD */
add_fpreq(5184, &buffer0[31104],0.002441603224724531,-21,&buffer0[41472],0.002654152689501643,-30,0.0028999417554587126,-36,&buffer0[36288]);
/* layer 21:CONV_2D */
convolve_1x1_s8_ch16_fpreq(&buffer0[36288],18,18,16,(const q7_t*) weight18,bias18,scales18,-5,36,-128,127,&buffer0[0],18,18,96,sbuf);
/* layer 22:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],18,18,96,(const q7_t*) CHWweight19,offsetBias19,offsetRBias19,scales19,15,5,-128,127,&buffer0[0],9,9,96,sbuf,-5);
/* layer 23:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,96,(const q7_t*) weight20,bias20,scales20,-20,-15,-128,127,&buffer0[15552],9,9,32,sbuf);
/* layer 24:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[15552],9,9,32,(const q7_t*) weight21,bias21,scales21,2,20,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 25:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight22,offsetBias22,offsetRBias22,scales22,-3,-2,-128,127,&buffer0[0],9,9,192,sbuf,2);
/* layer 26:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight23,bias23,scales23,7,3,-128,127,&buffer0[18144],9,9,32,sbuf);
/* layer 27:ADD */
add_fpreq(2592, &buffer0[18144],0.0021028753835707903,7,&buffer0[15552],0.002326514106243849,-20,0.00291330274194479,-18,&buffer0[20736]);
/* layer 28:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[20736],9,9,32,(const q7_t*) weight24,bias24,scales24,-7,18,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 29:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,scales25,0,7,-128,127,&buffer0[0],9,9,192,sbuf,-7);
/* layer 30:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight26,bias26,scales26,-2,0,-128,127,&buffer0[15552],9,9,32,sbuf);
/* layer 31:ADD */
add_fpreq(2592, &buffer0[15552],0.0028268916066735983,-2,&buffer0[20736],0.00291330274194479,-18,0.005000795237720013,-3,&buffer0[18144]);
/* layer 32:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[18144],9,9,32,(const q7_t*) weight27,bias27,scales27,-3,3,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 33:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight28,offsetBias28,offsetRBias28,scales28,28,3,-128,127,&buffer0[0],9,9,192,sbuf,-3);
/* layer 34:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight29,bias29,scales29,-44,-28,-128,127,&buffer0[15552],9,9,32,sbuf);
/* layer 35:ADD */
add_fpreq(2592, &buffer0[15552],0.001877687405794859,-44,&buffer0[18144],0.005000795237720013,-3,0.004993002861738205,-9,&buffer0[20736]);
/* layer 36:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[20736],9,9,32,(const q7_t*) weight30,bias30,scales30,-15,9,-128,127,&buffer0[0],9,9,192,sbuf);
/* layer 37:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight31,offsetBias31,offsetRBias31,scales31,10,15,-128,127,&buffer0[0],9,9,192,sbuf,-15);
/* layer 38:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,192,(const q7_t*) weight32,bias32,scales32,-12,-10,-128,127,&buffer0[23328],9,9,48,sbuf);
/* layer 39:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[23328],9,9,48,(const q7_t*) weight33,bias33,scales33,14,12,-128,127,&buffer0[0],9,9,288,sbuf);
/* layer 40:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,288,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,scales34,16,-14,-128,127,&buffer0[0],9,9,288,sbuf,14);
/* layer 41:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,288,(const q7_t*) weight35,bias35,scales35,-1,-16,-128,127,&buffer0[27216],9,9,48,sbuf);
/* layer 42:ADD */
add_fpreq(3888, &buffer0[27216],0.002591863740235567,-1,&buffer0[23328],0.001980195054784417,-12,0.0031075673177838326,1,&buffer0[31104]);
/* layer 43:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[31104],9,9,48,(const q7_t*) weight36,bias36,scales36,-3,-1,-128,127,&buffer0[0],9,9,288,sbuf);
/* layer 44:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,288,(const q7_t*) CHWweight37,offsetBias37,offsetRBias37,scales37,19,3,-128,127,&buffer0[0],9,9,288,sbuf,-3);
/* layer 45:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],9,9,288,(const q7_t*) weight38,bias38,scales38,-17,-19,-128,127,&buffer0[23328],9,9,48,sbuf);
/* layer 46:ADD */
add_fpreq(3888, &buffer0[23328],0.0022428107913583517,-17,&buffer0[31104],0.0031075673177838326,1,0.0036475802771747112,-16,&buffer0[27216]);
/* layer 47:CONV_2D */
convolve_1x1_s8_ch48_fpreq(&buffer0[27216],9,9,48,(const q7_t*) weight39,bias39,scales39,-7,16,-128,127,&buffer0[0],9,9,288,sbuf);
/* layer 48:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],9,9,288,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,scales40,4,7,-128,127,&buffer0[0],5,5,288,sbuf,-7);
/* layer 49:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,288,(const q7_t*) weight41,bias41,scales41,1,-4,-128,127,&buffer0[12000],5,5,80,sbuf);
/* layer 50:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[12000],5,5,80,(const q7_t*) weight42,bias42,scales42,6,-1,-128,127,&buffer0[0],5,5,480,sbuf);
/* layer 51:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],5,5,480,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,scales43,-9,-6,-128,127,&buffer0[0],5,5,480,sbuf,6);
/* layer 52:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,480,(const q7_t*) weight44,bias44,scales44,-2,9,-128,127,&buffer0[14000],5,5,80,sbuf);
/* layer 53:ADD */
add_fpreq(2000, &buffer0[14000],0.0026811123825609684,-2,&buffer0[12000],0.0023005802650004625,1,0.00359399919398129,6,&buffer0[16000]);
/* layer 54:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[16000],5,5,80,(const q7_t*) weight45,bias45,scales45,-2,-6,-128,127,&buffer0[0],5,5,480,sbuf);
/* layer 55:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],5,5,480,(const q7_t*) CHWweight46,offsetBias46,offsetRBias46,scales46,8,2,-128,127,&buffer0[0],5,5,480,sbuf,-2);
/* layer 56:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,480,(const q7_t*) weight47,bias47,scales47,2,-8,-128,127,&buffer0[12000],5,5,80,sbuf);
/* layer 57:ADD */
add_fpreq(2000, &buffer0[12000],0.0026515808422118425,2,&buffer0[16000],0.00359399919398129,6,0.003986136522144079,6,&buffer0[14000]);
/* layer 58:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[14000],5,5,80,(const q7_t*) weight48,bias48,scales48,-10,-6,-128,127,&buffer0[0],5,5,480,sbuf);
/* layer 59:DEPTHWISE_CONV_2D */
depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],5,5,480,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,scales49,-27,10,-128,127,&buffer0[0],5,5,480,sbuf,-10);
/* layer 60:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[0],5,5,480,(const q7_t*) weight50,bias50,scales50,18,27,-128,127,&buffer0[32000],5,5,160,sbuf);
/* layer 61:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[32000],5,5,160,(const q7_t*) weight51,bias51,scales51,-2,-18,-128,127,&buffer0[0],5,5,1280,sbuf);
/* layer 62:AVERAGE_POOL_2D */
avg_pooling(&buffer0[32000],5,5,1280,5,5,1,1,-128,127,&buffer0[32000]);
/* layer 63:CONV_2D */
convolve_1x1_s8_fpreq(&buffer0[32000],1,1,1280,(const q7_t*) weight52,bias52,scales52,15,2,-128,127,&buffer0[64000],1,1,1000,sbuf);
}

q7_t* cols_8b_start = cols_8b_iterptr;
q7_t* cols_8b_left_iter = cols_8b_start;
q7_t* cols_8b_right_iter = cols_8b_start + column_x - 2;
q7_t* cols_8b_bot_iter = cols_8b_start + (output_y - 2) * column_x;
q31_t ch_bias = bias[0] + biasR[0];
q7_t* cols_8b_left = cols_8b_left_iter;
q7_t* cols_8b_right = cols_8b_right_iter;

q31_t Lsum00 = ch_bias;
q31_t Rsum00 = ch_bias;
q31_t Lsum01;
q31_t Lsum10;
q31_t Rsum10;

// X00 0
Lsum00 += cols_8b_left[0] * pred[0];

// X01 1
Lsum00 += cols_8b_left[1] * pred[1];

cols_8b_left += 2 * column_x - 2;

// Y00 2
Rsum00 += cols_8b_right[0] * pred[2];

// Y01 3
Rsum00 += cols_8b_right[1] * pred[3];

// X10 4
Lsum00 += cols_8b_right[2] * pred[4];

// X11 5
Lsum00 += cols_8b_right[3] * pred[5];

// Y10 6
Rsum00 += cols_8b_left[0] * pred[6];

// Y11 7
Rsum00 += cols_8b_left[1] * pred[7];

cols_8b_left = cols_8b_bot_iterptr;
cols_8b_right = cols_8b_bot_iterptr + column_x - 2;

Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
Lsum00 += output_offset;
Lsum00 = MAX(Lsum00, activation_min);
Lsum00 = MIN(Lsum00, activation_max);

Rsum00 = arm_nn_requantize(Rsum00, *multiplier, *shift);
Rsum00 += output_offset;
Rsum00 = MAX(Rsum00, activation_min);
Rsum00 = MIN(Rsum00, activation_max);

output[0] = Lsum00;
output[(output_x - 1) * channel_offset] = Rsum00;

Lsum00 = ch_bias;
Rsum00 = ch_bias;

// X00 8
Lsum00 += cols_8b_left[0] * pred[8];

// X01 9
Lsum00 += cols_8b_left[1] * pred[9];

cols_8b_left += 2 * column_x - 2;

// Y00 10
Rsum00 += cols_8b_right[0] * pred[10];

// Y01 11
Rsum00 += cols_8b_right[1] * pred[11];

// X10 12
Lsum00 += cols_8b_right[2] * pred[12];

// X11 13
Lsum00 += cols_8b_right[3] * pred[13];

// Y10 14
Rsum00 += cols_8b_left[0] * pred[14];

// Y11 15
Rsum00 += cols_8b_left[1] * pred[15];

Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
Lsum00 += output_offset;
Lsum00 = MAX(Lsum00, activation_min);
Lsum00 = MIN(Lsum00, activation_max);

Rsum00 = arm_nn_requantize(Rsum00, *multiplier, *shift);
Rsum00 += output_offset;
Rsum00 = MAX(Rsum00, activation_min);
Rsum00 = MIN(Rsum00, activation_max);

output[(output_y - 1) * output_x * channel_offset] = Lsum00;
output[((output_y - 1) * output_x  + output_x - 1)* channel_offset] = Rsum00;

// TOP
q7_t* cols_8b_iter = cols_8b_start;
for (int x = 1; x < output_x/2; x++) {
    q7_t* cols_8b = cols_8b_iter;
    Lsum00 = ch_bias;
    Lsum01 = ch_bias;

    // X00
    Lsum00 += cols_8b[0] * pred[16];

    // X01
    Lsum01 += cols_8b[1] * pred[16];
    Lsum00 += cols_8b[1] * pred[17];

    // X02
    Lsum01 += cols_8b[2] * pred[17];
    Lsum00 += cols_8b[2] * pred[18];

    // X03
    Lsum01 += cols_8b[3] * pred[18];

    cols_8b += column_x;
    // X10
    Lsum00 += cols_8b[0] * pred[19];

    // X11
    Lsum01 += cols_8b[1] * pred[19];
    Lsum00 += cols_8b[1] * pred[20];

    // X12
    Lsum01 += cols_8b[2] * pred[20];
    Lsum00 += cols_8b[2] * pred[21];

    // X13
    Lsum01 += cols_8b[3] * pred[21];

    cols_8b_iter += 2;

    Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
    Lsum00 += output_offset;
    Lsum00 = MAX(Lsum00, activation_min);
    Lsum00 = MIN(Lsum00, activation_max);

    Lsum01 = arm_nn_requantize(Lsum00, *multiplier, *shift);
    Lsum01 += output_offset;
    Lsum01 = MAX(Lsum00, activation_min);
    Lsum01 = MIN(Lsum00, activation_max);

    output[(2 * x - 1) * channel_offset] = Lsum00;
    output[(2 * x) * channel_offset] = Lsum01;
}
if(output_x & 1) {
    Lsum00 = ch_bias;

    // X00
    Lsum00 += cols_8b[0] * pred[16];

    // X01
    Lsum00 += cols_8b[1] * pred[17];

    // X02
    Lsum00 += cols_8b[2] * pred[18];

    cols_8b += column_x;
    // X10
    Lsum00 += cols_8b[0] * pred[19];

    // X11
    Lsum00 += cols_8b[1] * pred[20];

    // X12
    Lsum00 += cols_8b[2] * pred[21];

    Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
    Lsum00 += output_offset;
    Lsum00 = MAX(Lsum00, activation_min);
    Lsum00 = MIN(Lsum00, activation_max);

    output[(output_x - 2) * channel_offset] = Lsum00;
}

q7_t* cols_8b_left_iterptr = cols_8b_left_iter;
q7_t* cols_8b_right_iterptr = cols_8b_right_iter;
for(int y = 1; y < output_y/2; y++) {
    cols_8b_left = cols_8b_left_iterptr;
    cols_8b_right = cols_8b_right_iterptr;

    Lsum00 = ch_bias;
    Lsum10 = ch_bias;
    
    Rsum00 = ch_bias;
    Rsum10 = ch_bias;

    // X00
    Lsum00 += cols_8b_left[0] * pred[22];

    // X01
    Lsum00 += cols_8b_left[1] * pred[23];

    cols_8b_left += 2 * column_x - 2;
    
    // Y00
    Rsum00 += cols_8b_right[0] * pred[24];

    // Y01 
    Rsum00 += cols_8b_right[1] * pred[25];
    
    // X10
    Lsum10 += cols_8b_right[2] * pred[22];
    Lsum00 += cols_8b_right[2] * pred[26];

    // X11
    Lsum10 += cols_8b_right[3] * pred[23];
    Lsum00 += cols_8b_right[3] * pred[27];

    cols_8b_right += 2 * column_x;

    // Y10
    Rsum10 += cols_8b_left[0] * pred[24];
    Rsum00 += cols_8b_left[0] * pred[28];

    // Y11
    Rsum10 += cols_8b_left[1] * pred[25];
    Rsum00 += cols_8b_left[1] * pred[29];
    
    // X20
    Lsum10 += cols_8b_left[2] * pred[26];
    Lsum00 += cols_8b_left[2] * pred[30];

    // X21
    Lsum10 += cols_8b_left[3] * pred[27];
    Lsum00 += cols_8b_left[3] * pred[31];

    cols_8b_left += 2 * column_x;

    // Y20
    Rsum10 += cols_8b_right[0] * pred[28];
    Rsum00 += cols_8b_right[0] * pred[32];

    // Y21
    Rsum10 += cols_8b_right[1] * pred[29];
    Rsum00 += cols_8b_right[1] * pred[33];

    Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
    Lsum00 += output_offset;
    Lsum00 = MAX(Lsum00, activation_min);
    Lsum00 = MIN(Lsum00, activation_max);

    Rsum00 = arm_nn_requantize(Rsum00, *multiplier, *shift);
    Rsum00 += output_offset;
    Rsum00 = MAX(Rsum00, activation_min);
    Rsum00 = MIN(Rsum00, activation_max);  

    output[(2 * y - 1) * output_x * channel_offset] = Lsum00;
    output[((2 * y - 1) * output_x + output_x - 1)* channel_offset] = Rsum00;

    // X30
    Lsum10 += cols_8b_right[2] * pred[30];

    // X31
    Lsum10 += cols_8b_right[3] * pred[31];

    // Y30
    Rsum10 += cols_8b_left[0] * pred[32];

    // Y31
    Rsum10 += cols_8b_left[1] * pred[33];

    Lsum10 = arm_nn_requantize(Lsum10, *multiplier, *shift);
    Lsum10 += output_offset;
    Lsum10 = MAX(Lsum10, activation_min);
    Lsum10 = MIN(Lsum10, activation_max);

    Rsum10 = arm_nn_requantize(Rsum10, *multiplier, *shift);
    Rsum10 += output_offset;
    Rsum10 = MAX(Rsum10, activation_min);
    Rsum10 = MIN(Rsum10, activation_max);  

    output[(2 * y) * output_x * channel_offset] = Lsum10;
    output[((2 * y) * output_x + output_x - 1)* channel_offset] = Rsum10;

    cols_8b_left_iter += column_x;
    cols_8b_right_iter += column_x;
} if (output_y & 1){ 
    cols_8b_left = cols_8b_left_iterptr;
    cols_8b_right = cols_8b_right_iterptr;

    Lsum00 = ch_bias;
    
    Rsum00 = ch_bias;

    // X00
    Lsum00 += cols_8b_left[0] * pred[22];

    // X01
    Lsum00 += cols_8b_left[1] * pred[23];

    cols_8b_left += 2 * column_x - 2;
    
    // Y00
    Rsum00 += cols_8b_right[0] * pred[24];

    // Y01 
    Rsum00 += cols_8b_right[1] * pred[25];
    
    // X10
    Lsum00 += cols_8b_right[2] * pred[26];

    // X11
    Lsum00 += cols_8b_right[3] * pred[27];

    cols_8b_right += 2 * column_x;

    // Y10
    Rsum00 += cols_8b_left[0] * pred[28];

    // Y11
    Rsum00 += cols_8b_left[1] * pred[29];
    
    // X20
    Lsum00 += cols_8b_left[2] * pred[30];

    // X21
    Lsum00 += cols_8b_left[3] * pred[31];

    cols_8b_left += 2 * column_x;

    // Y20
    Rsum00 += cols_8b_right[0] * pred[32];

    // Y21
    Rsum00 += cols_8b_right[1] * pred[33];

    Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
    Lsum00 += output_offset;
    Lsum00 = MAX(Lsum00, activation_min);
    Lsum00 = MIN(Lsum00, activation_max);

    Rsum00 = arm_nn_requantize(Rsum00, *multiplier, *shift);
    Rsum00 += output_offset;
    Rsum00 = MAX(Rsum00, activation_min);
    Rsum00 = MIN(Rsum00, activation_max);  

    output[((output_y - 1) * output_x + 1) * channel_offset] = Lsum00;
    output[((output_y - 1) * output_x + output_x - 2)* channel_offset] = Rsum00;
}

for (int x = 1; x < output_x/2; x++) {
    cols_8b = cols_8b_bot_iter;
    Lsum00 = ch_bias;
    Lsum01 = ch_bias;

    // X00
    Lsum00 += cols_8b[0] * pred[34];

    // X01
    Lsum01 += cols_8b[1] * pred[34];
    Lsum00 += cols_8b[1] * pred[35];

    // X02
    Lsum01 += cols_8b[2] * pred[35];
    Lsum00 += cols_8b[2] * pred[36];

    // X03
    Lsum01 += cols_8b[3] * pred[36];

    cols_8b += column_x;
    // X10
    Lsum00 += cols_8b[0] * pred[37];

    // X11
    Lsum01 += cols_8b[1] * pred[37];
    Lsum00 += cols_8b[1] * pred[38];

    // X12
    Lsum01 += cols_8b[2] * pred[38];
    Lsum00 += cols_8b[2] * pred[39];

    // X13
    Lsum01 += cols_8b[3] * pred[39];

    Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
    Lsum00 += output_offset;
    Lsum00 = MAX(Lsum00, activation_min);
    Lsum00 = MIN(Lsum00, activation_max);

    Lsum01 = arm_nn_requantize(Lsum01, *multiplier, *shift);
    Lsum01 += output_offset;
    Lsum01 = MAX(Lsum01, activation_min);
    Lsum01 = MIN(Lsum01, activation_max);  

    output[((output_y - 1) * output_x + 2 * x - 1) * channel_offset] = Lsum00;
    output[((output_y - 1) * output_x + 2 * x) * channel_offset] = Lsum01;

    cols_8b_bot_iter += 2;
}
if(output_x & 1) {
    cols_8b = cols_8b_bot_iter;
    Lsum00 = ch_bias;

    // X00
    Lsum00 += cols_8b[0] * pred[34];

    // X01
    Lsum00 += cols_8b[1] * pred[35];

    // X02
    Lsum00 += cols_8b[2] * pred[36];

    cols_8b += column_x;
    // X10
    Lsum00 += cols_8b[0] * pred[37];

    // X11
    Lsum00 += cols_8b[1] * pred[38];

    Lsum00 = arm_nn_requantize(Lsum00, *multiplier, *shift);
    Lsum00 += output_offset;
    Lsum00 = MAX(Lsum00, activation_min);
    Lsum00 = MIN(Lsum00, activation_max);

    output[(output_y * output_x - 2) * channel_offset] = Lsum00;
}

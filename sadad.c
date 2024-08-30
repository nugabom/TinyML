void K5_S2_dw_inplace(
    int* temporal_buffer,
    int* output,
    int* ksrc,
    int runtime_buffer_width,
    int output_y, int padded_output_x, int output_x,int output_ch)
{
	int stride = 2;
	int i, j;
	int sum, sum0, sum1;
	int* cols_8b;

	for(i = 0; i < output_y; i++) {
		for(j = 0; j < output_x/2; j++) {
			cols_8b = temporal_buffer;

			sum0 = 0;
			sum1 = 0;

            /* computation */
            sum0 += cols_8b[0]*ksrc[0];
            sum1 += cols_8b[2]*ksrc[0];
            sum0 += cols_8b[1]*ksrc[1];
            sum1 += cols_8b[3]*ksrc[1];
            sum0 += cols_8b[2]*ksrc[2];
            sum1 += cols_8b[4]*ksrc[2];
            sum0 += cols_8b[3]*ksrc[3];
            sum1 += cols_8b[5]*ksrc[3];
            sum0 += cols_8b[4]*ksrc[4];
            sum1 += cols_8b[6]*ksrc[4];
            cols_8b += runtime_buffer_width;
            sum0 += cols_8b[0]*ksrc[5];
            sum1 += cols_8b[2]*ksrc[5];
            sum0 += cols_8b[1]*ksrc[6];
            sum1 += cols_8b[3]*ksrc[6];
            sum0 += cols_8b[2]*ksrc[7];
            sum1 += cols_8b[4]*ksrc[7];
            sum0 += cols_8b[3]*ksrc[8];
            sum1 += cols_8b[5]*ksrc[8];
            sum0 += cols_8b[4]*ksrc[9];
            sum1 += cols_8b[6]*ksrc[9];
            cols_8b += runtime_buffer_width;
            sum0 += cols_8b[0]*ksrc[10];
            sum1 += cols_8b[2]*ksrc[10];
            sum0 += cols_8b[1]*ksrc[11];
            sum1 += cols_8b[3]*ksrc[11];
            sum0 += cols_8b[2]*ksrc[12];
            sum1 += cols_8b[4]*ksrc[12];
            sum0 += cols_8b[3]*ksrc[13];
            sum1 += cols_8b[5]*ksrc[13];
            sum0 += cols_8b[4]*ksrc[14];
            sum1 += cols_8b[6]*ksrc[14];
            cols_8b += runtime_buffer_width;
            sum0 += cols_8b[0]*ksrc[15];
            sum1 += cols_8b[2]*ksrc[15];
            sum0 += cols_8b[1]*ksrc[16];
            sum1 += cols_8b[3]*ksrc[16];
            sum0 += cols_8b[2]*ksrc[17];
            sum1 += cols_8b[4]*ksrc[17];
            sum0 += cols_8b[3]*ksrc[18];
            sum1 += cols_8b[5]*ksrc[18];
            sum0 += cols_8b[4]*ksrc[19];
            sum1 += cols_8b[6]*ksrc[19];
            cols_8b += runtime_buffer_width;
            sum0 += cols_8b[0]*ksrc[20];
            sum1 += cols_8b[2]*ksrc[20];
            sum0 += cols_8b[1]*ksrc[21];
            sum1 += cols_8b[3]*ksrc[21];
            sum0 += cols_8b[2]*ksrc[22];
            sum1 += cols_8b[4]*ksrc[22];
            sum0 += cols_8b[3]*ksrc[23];
            sum1 += cols_8b[5]*ksrc[23];
            sum0 += cols_8b[4]*ksrc[24];
            sum1 += cols_8b[6]*ksrc[24];

			output[(i * padded_output_x + 2 * j) * output_ch] = sum0;
			output[(i * padded_output_x + 2 * j + 1) * output_ch] = sum1;

			temporal_buffer += 2 * stride;
		}
		if (output_x & 1) {
			cols_8b = temporal_buffer;
			sum = 0;
            sum += cols_8b[0]*ksrc[0];
            sum += cols_8b[1]*ksrc[1];
            sum += cols_8b[2]*ksrc[2];
            sum += cols_8b[3]*ksrc[3];
            sum += cols_8b[4]*ksrc[4];
            cols_8b += runtime_buffer_width;
            sum += cols_8b[0]*ksrc[5];
            sum += cols_8b[1]*ksrc[6];
            sum += cols_8b[2]*ksrc[7];
            sum += cols_8b[3]*ksrc[8];
            sum += cols_8b[4]*ksrc[9];
            cols_8b += runtime_buffer_width;
            sum += cols_8b[0]*ksrc[10];
            sum += cols_8b[1]*ksrc[11];
            sum += cols_8b[2]*ksrc[12];
            sum += cols_8b[3]*ksrc[13];
            sum += cols_8b[4]*ksrc[14];
            cols_8b += runtime_buffer_width;
            sum += cols_8b[0]*ksrc[15];
            sum += cols_8b[1]*ksrc[16];
            sum += cols_8b[2]*ksrc[17];
            sum += cols_8b[3]*ksrc[18];
            sum += cols_8b[4]*ksrc[19];
            cols_8b += runtime_buffer_width;
            sum += cols_8b[0]*ksrc[20];
            sum += cols_8b[1]*ksrc[21];
            sum += cols_8b[2]*ksrc[22];
            sum += cols_8b[3]*ksrc[23];
            sum += cols_8b[4]*ksrc[24];
			output[(i * padded_output_x + output_x - 1) * output_ch] = sum;
            // printf("\nfinal outputm);
			temporal_buffer += stride;
		}
        temporal_buffer += 2 * 2 - (runtime_buffer_width & 1);
        temporal_buffer += (stride - 1) * (runtime_buffer_width);
	}
}

void depthwise_k5_S2_inplace_CHW_Store_XY(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* new_x_stm, int* new_y_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);


    // store new x stream
    int* stm_x_loc_in_patch = input + (input_x - bufR) * input_ch;
    for (i = 0; i < input_y; i++) {
        memcpy(new_x_stm, stm_x_loc_in_patch, bufR * input_ch * BIT);
        new_x_stm += bufR * input_ch;
        stm_x_loc_in_patch += input_x * input_ch;
    }

    // store new y stream
    int* stm_y_loc_in_patch = input + (input_y - bufR) * input_x * input_ch;
    memcpy(new_y_stm, stm_y_loc_in_patch, bufR * input_x * input_ch * BIT); 

    for(i = 0; i < receptive_pad_t * receptive_field_x_length; i++) {
        *cols_8b++ = PAD8;
    }

    for (i = 0; i <  receptive_field_y_length - (receptive_pad_b + receptive_pad_t); i++) {
        for(j = 0; j < receptive_pad_l; j++) {
            cols_8b[j] = PAD8;
        }

        cols_8b += receptive_field_x_length;

        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b * receptive_field_x_length; i++) {
        *cols_8b++ = PAD8;
    }
    
    int *src;
    int *ksrc = kernel;
    for(c = 0; c < input_ch; c++) {
        cols_8b = (int*) (cols_8b_start + receptive_field_x_length * receptive_pad_t);
        src = (int*) (input + pad_t * input_x * input_ch + c); 
        for(j = 0; j < receptive_field_y_length - (receptive_pad_t+ receptive_pad_b); j++) {
            cols_8b += receptive_pad_l;
            src += pad_l * input_ch;
            for(i = 0; i < receptive_field_x_length - (receptive_pad_l + receptive_pad_r); i++) {
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += pad_r * input_ch;
        }
        
        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}

void depthwise_k5_S2_inplace_CHW_LOAD_X_Store_XY(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* x_stm, 
    int* new_x_stm, int* new_y_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);

    int ystm_h_in_patch = bufR;
    int xstm_w_in_patch = min(bufR, input_x);
    int xstm_w_in_xstm = bufR - xstm_w_in_patch;

    // store new x stream
    int* stm_x_loc_in_patch = input + (input_x - xstm_w_in_patch) * input_ch;
    int* stm_x_loc_in_xstm = x_stm + (bufR - xstm_w_in_xstm) * input_ch;
    for (i = 0; i < input_y; i++) {
        if (xstm_w_in_xstm > 0) {
            memcpy(new_x_stm, stm_x_loc_in_xstm, xstm_w_in_xstm * input_ch * BIT);
            new_x_stm += xstm_w_in_xstm * input_ch;
            stm_x_loc_in_xstm += input_ch * bufR;
        }
        memcpy(new_x_stm, stm_x_loc_in_patch, xstm_w_in_patch * input_ch * BIT);
        new_x_stm += xstm_w_in_patch * input_ch;
        stm_x_loc_in_patch += input_x * input_ch;
    }

    // store new y stream
    int* stm_y_loc_in_patch = input + (input_y - bufR) * input_x * input_ch;
    int* stm_y_loc_in_xstm = x_stm + (input_y - bufR) * bufR * input_ch;
    for (i = 0; i < bufR; i++) {
        memcpy(new_y_stm, stm_y_loc_in_xstm, bufR * input_ch * BIT);
        new_y_stm += bufR * input_ch;
        stm_y_loc_in_xstm += bufR * input_ch;
        memcpy(new_y_stm, stm_y_loc_in_patch, input_x * input_ch * BIT);
        new_y_stm += input_x * input_ch;
        stm_y_loc_in_patch += input_x * input_ch;
    }

    for(i = 0; i < receptive_pad_t; i++) {
        cols_8b += bufR;
        for(j = 0; j < receptive_field_x_length - bufR; j++) {
            *cols_8b++ = PAD8;
        }
    }

    for (i = 0; i <  receptive_field_y_length - (receptive_pad_t+ receptive_pad_b); i++) {
        cols_8b += receptive_field_x_length;
        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b; i++) {
        cols_8b += bufR;
        for(j = 0; j < receptive_field_x_length - bufR; j++) {
            *cols_8b++ = PAD8;
        }
    }
    
    int *src;
    int *ksrc = kernel;
    int *x_stm_load;
    for(c = 0; c < input_ch; c++) {
        // cols_8b = (int*) (cols_8b_start + receptive_field_x_length * use_pad_t);
        cols_8b = (int*) (cols_8b_start);
        x_stm_load = x_stm + c;
        src = (int*) (input + pad_t * input_x * input_ch + c);
        
        // fill pad_t
        for(i = 0; i < receptive_pad_t; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load+= input_ch; 
            }
            cols_8b += receptive_field_x_length - bufR;
        } 
        // fill mid part
        for(i = 0; i < receptive_field_y_length - (receptive_pad_t + receptive_pad_b); i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch; 
            }
            for(j = 0; j < receptive_field_x_length - bufR - receptive_pad_r; j++) {
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += input_ch * pad_r;
        }

        // fill bot part
        for(i = 0; i < receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            cols_8b += receptive_field_x_length - bufR;
        }
        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}

void depthwise_k5_S2_inplace_CHW_LOAD_X_Store_Y(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* x_stm, 
    int* new_y_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);

    int ystm_h_in_patch = bufR;

    // store new y stream
    int* stm_y_loc_in_patch = input + (input_y - bufR) * input_x * input_ch;
    int* stm_y_loc_in_xstm = x_stm + (input_y - bufR) * bufR * input_ch;
    for (i = 0; i < bufR; i++) {
        memcpy(new_y_stm, stm_y_loc_in_xstm, bufR * input_ch * BIT);
        new_y_stm += bufR * input_ch;
        stm_y_loc_in_xstm += bufR * input_ch;
        memcpy(new_y_stm, stm_y_loc_in_patch, input_x * input_ch * BIT);
        new_y_stm += input_x * input_ch;
        stm_y_loc_in_patch += input_x * input_ch;
    }

    // Load
    for(i = 0; i < receptive_pad_t; i++) {
        cols_8b += bufR;
        for(j = 0; j < receptive_field_x_length - bufR; j++) {
            *cols_8b++ = PAD8;
        }
    }

    for (i = 0; i <  receptive_field_y_length - (receptive_pad_t+ receptive_pad_b); i++) {
        cols_8b += receptive_field_x_length;
        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b; i++) {
        cols_8b += bufR;
        for(j = 0; j < receptive_field_x_length - bufR; j++) {
            *cols_8b++ = PAD8;
        }
    }
    
    // printf("temporal buffer\n");
    // printTable(temporal_buffer, receptive_field_y_length, receptive_field_x_length, 1);
    // printf("\n\n");
    int *src;
    int *ksrc = kernel;
    int *x_stm_load;
    for(c = 0; c < input_ch; c++) {
        // cols_8b = (int*) (cols_8b_start + receptive_field_x_length * use_pad_t);
        cols_8b = (int*) (cols_8b_start);
        x_stm_load = x_stm + c;
        src = (int*) (input + pad_t * input_x * input_ch + c);
        
        // fill pad_t
        for(i = 0; i < receptive_pad_t; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load+= input_ch; 
            }
            cols_8b += receptive_field_x_length - bufR;
        } 
        // fill mid part
        for(i = 0; i < receptive_field_y_length - (receptive_pad_t+ receptive_pad_b); i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch; 
            }
            for(j = 0; j < receptive_field_x_length - bufR - receptive_pad_r; j++) {
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += pad_r * input_ch;
        }

        // fill bot part
        for(i = 0; i < receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            cols_8b += receptive_field_x_length - bufR;
        }
        printf("temporal buffer\n");
        printTable(cols_8b_start, receptive_field_y_length, receptive_field_x_length, 1);
        printf("\n\n");
        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}

void depthwise_k5_S2_inplace_CHW_LOAD_Y_Store_XY(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* y_stm,
    int* new_x_stm,int* new_y_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);

    int xstm_w_in_patch = bufR;
    int ystm_h_in_patch = min(bufR, input_y);
    int ystm_h_in_ystm = bufR - ystm_h_in_patch;

    // store new x stream
    int* stm_x_loc_in_patch = input + (input_x - bufR) * input_ch;
    for (i = 0; i < input_y; i++) {
        memcpy(new_x_stm, stm_x_loc_in_patch, bufR * input_ch * BIT);
        new_x_stm += bufR * input_ch;
        stm_x_loc_in_patch += input_x * input_ch;
    }
    
    // store new y stream
    int* stm_y_loc_in_patch = input + (input_y - ystm_h_in_patch) * input_x * input_ch;
    int* stm_y_loc_in_ystm = y_stm + (bufR - ystm_h_in_ystm) * input_x * input_ch;
    if (stm_y_loc_in_ystm > 0) {
        memcpy(new_y_stm, stm_y_loc_in_ystm, ystm_h_in_ystm * input_x * input_ch * BIT);
        new_y_stm +=  ystm_h_in_ystm * input_x * input_ch;
    }
    memcpy(new_y_stm, stm_y_loc_in_patch, ystm_h_in_patch * input_x * input_ch * BIT);
    
    cols_8b += receptive_field_x_length * bufR;
    for (i = 0; i <  receptive_field_y_length - (bufR+ receptive_pad_b); i++) {
        for(j = 0; j < receptive_pad_l; j++) {
            cols_8b[j] = PAD8;
        }

        cols_8b += receptive_field_x_length;

        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b* receptive_field_x_length; i++) {
        *cols_8b++ = PAD8;
    }
    
    // printf("temporal buffer\n");
    // printTable(temporal_buffer, receptive_field_y_length, receptive_field_x_length, 1);
    // printf("\n\n");
    int *src;
    int *ksrc = kernel;
    int *y_stm_load;
    for(c = 0; c < input_ch; c++) {
        cols_8b = (int*) (cols_8b_start);
        src = (int*) (input + pad_t * input_x * input_ch + c);
        y_stm_load = y_stm + c;
        for(i = 0; i < receptive_field_x_length * bufR; i++) {
            *cols_8b++ = *y_stm_load;
            y_stm_load += input_ch;
        }
        for(i = 0; i < receptive_field_y_length - bufR - receptive_pad_b; i++) {
            cols_8b += receptive_pad_l;
            src += pad_l * input_ch;
            for(j = 0; j < receptive_field_x_length - (receptive_pad_l+ receptive_pad_r); j++) {
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += pad_r * input_ch ;
        }
        
        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}

void depthwise_k5_S2_inplace_CHW_LOAD_XY_Store_XY(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* x_stm, int* y_stm,
    int* new_x_stm, int* new_y_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);

    int xstm_w_in_patch = min(bufR, input_x);
    int xstm_w_in_xstm = bufR - xstm_w_in_patch;
    int ystm_h_in_xstm_patch = min(bufR, input_y);
    int ystm_h_in_ystm = bufR - ystm_h_in_xstm_patch;


    // store new x stream
    int* stm_x_loc_in_patch = input + (input_x - xstm_w_in_patch) * input_ch;
    int* stm_x_loc_in_xstm = x_stm + (bufR - xstm_w_in_xstm) * input_ch;
    for (i = 0; i < input_y; i++) {
        if (xstm_w_in_xstm > 0) {
            memcpy(new_x_stm, stm_x_loc_in_xstm, xstm_w_in_xstm * input_ch * BIT);
            new_x_stm += xstm_w_in_xstm* input_ch;
            stm_x_loc_in_xstm += bufR * input_ch;
        }
        memcpy(new_x_stm, stm_x_loc_in_patch, xstm_w_in_patch * input_ch * BIT);
        new_x_stm += xstm_w_in_patch * input_ch;
        stm_x_loc_in_patch += input_x * input_ch;
    }

    // store new y stream
    int* stm_y_loc_in_ystm = y_stm + (bufR - ystm_h_in_ystm) * (input_x + bufR) * input_ch; 
    int* stm_y_loc_in_patch = input + (input_y - ystm_h_in_xstm_patch) * input_x * input_ch;
    int* stm_y_loc_in_xstm = x_stm + (input_y - ystm_h_in_xstm_patch) * bufR * input_ch;

    if (ystm_h_in_ystm > 0) {
        memcpy(new_y_stm, stm_y_loc_in_ystm, ystm_h_in_ystm * (bufR + input_x) * input_ch * BIT);
        new_y_stm += ystm_h_in_ystm * (bufR + input_x) * input_ch;
    }
    for (i = 0; i < ystm_h_in_xstm_patch; i++) {
        memcpy(new_y_stm, stm_y_loc_in_xstm, bufR * input_ch * BIT);
        new_y_stm += bufR * input_ch;
        stm_y_loc_in_xstm += bufR * input_ch;
        memcpy(new_y_stm, stm_y_loc_in_patch, input_x * input_ch * BIT);
        new_y_stm += input_x * input_ch;
        stm_y_loc_in_patch += input_x * input_ch;
    }

    cols_8b += bufR * (input_x + bufR);
    for (i = 0; i <  receptive_field_y_length - bufR - receptive_pad_b; i++) {
        cols_8b += receptive_field_x_length;

        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b* receptive_field_x_length; i++) {
        *cols_8b++ = PAD8;
    }

    // printf("temporal buffer\n");
    // printTable(temporal_buffer, receptive_field_y_length, receptive_field_x_length, 1);
    // printf("\n\n");
    int *src;
    int *ksrc = kernel;
    int *y_stm_load;
    int *x_stm_load;
    for(c = 0; c < input_ch; c++) {
        cols_8b = (int*) (cols_8b_start);
        src = (int*) (input + pad_t * input_x * input_ch + c);
        x_stm_load = x_stm + c; 
        y_stm_load = y_stm + c;
        for(i = 0; i < receptive_field_x_length * bufR; i++) {
            *cols_8b++ = *y_stm_load;
            y_stm_load += input_ch;
        }
        for(i = 0; i < receptive_field_y_length - bufR - receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            for(j = 0; j < receptive_field_x_length - bufR - receptive_pad_r; j++) {
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += input_ch * pad_r;

        }
        for(i = 0; i < receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            cols_8b += receptive_field_x_length - bufR;
        }

        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}


void depthwise_k5_S2_inplace_CHW_LOAD_XY_Store_Y(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* x_stm, int* y_stm,
    int* new_y_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);

    int ystm_h_in_xstm_patch = min(bufR, input_y);
    int ystm_h_in_ystm = bufR - ystm_h_in_xstm_patch;


    // store new y stream
    int* stm_y_loc_in_ystm = y_stm + (bufR - ystm_h_in_ystm) * (input_x + bufR) * input_ch; 
    int* stm_y_loc_in_patch = input + (input_y - ystm_h_in_xstm_patch) * input_x * input_ch;
    int* stm_y_loc_in_xstm = x_stm + (input_y - ystm_h_in_xstm_patch) * bufR * input_ch;

    memcpy(new_y_stm, stm_y_loc_in_ystm, ystm_h_in_ystm * (bufR + input_x) * input_ch * BIT);
    new_y_stm += ystm_h_in_ystm * (bufR + input_x) * input_ch;
    for (i = 0; i < ystm_h_in_xstm_patch; i++) {
        if (ystm_h_in_ystm > 0) {
            memcpy(new_y_stm, stm_y_loc_in_xstm, bufR * input_ch * BIT);
            new_y_stm += bufR * input_ch;
            stm_y_loc_in_xstm += bufR * input_ch;
        }
        memcpy(new_y_stm, stm_y_loc_in_patch, input_x * input_ch * BIT);
        new_y_stm += input_x * input_ch;
        stm_y_loc_in_patch += input_x * input_ch;
    }

    cols_8b += receptive_field_x_length * bufR;
    for (i = 0; i <  receptive_field_y_length - (bufR+ receptive_pad_b); i++) {
        for(j = 0; j < receptive_pad_l; j++) {
            cols_8b[j] = PAD8;
        }

        cols_8b += receptive_field_x_length;

        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b * receptive_field_x_length; i++) {
        *cols_8b++ = PAD8;
    }
    
    // printf("temporal buffer\n");
    // printTable(temporal_buffer, receptive_field_y_length, receptive_field_x_length, 1);
    // printf("\n\n");
    int *src;
    int *ksrc = kernel;
    int *y_stm_load;
    int *x_stm_load;
    for(c = 0; c < input_ch; c++) {
        cols_8b = (int*) (cols_8b_start);
        src = (int*) (input + pad_t * input_x * input_ch + c);
        x_stm_load = x_stm + c; 
        y_stm_load = y_stm + c;
        for(i = 0; i < receptive_field_x_length * bufR; i++) {
            *cols_8b++ = *y_stm_load;
            y_stm_load += input_ch;
        }
        for(i = 0; i < receptive_field_y_length - bufR - receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
                // printf("\n%d\n", *x_stm_load);
            }
            for(j = 0; j < receptive_field_x_length - bufR - receptive_pad_r; j++) {
                *cols_8b++ = *src;
                src += input_ch;
                
            }
            cols_8b += receptive_pad_r;
            src += input_ch * pad_r;

        }
        for(i = 0; i < receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            cols_8b += receptive_field_x_length - bufR;
        }

        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}

void depthwise_k5_S2_inplace_CHW_LOAD_Y_Store_X(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* y_stm, 
    int* new_x_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);

    int xstm_w_in_patch = bufR;
    

    // store new x stream
    int* stm_x_loc_in_patch = input + (input_x - bufR) * input_ch;
    for (i = 0; i < input_y; i++) {
        memcpy(new_x_stm, stm_x_loc_in_patch, bufR * input_ch * BIT);
        new_x_stm += bufR * input_ch;
        stm_x_loc_in_patch += input_x * input_ch;
    }
    
    
    cols_8b += receptive_field_x_length * bufR;
    for (i = 0; i <  receptive_field_y_length - (bufR +  receptive_pad_b); i++) {
        for(j = 0; j < receptive_pad_l; j++) {
            cols_8b[j] = PAD8;
        }

        cols_8b += receptive_field_x_length;

        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b * receptive_field_x_length; i++) {
        *cols_8b++ = PAD8;
    }
    
    // printf("temporal buffer\n");
    // printTable(temporal_buffer, receptive_field_y_length, receptive_field_x_length, 1);
    // printf("\n\n");
    int *src;
    int *ksrc = kernel;
    int *y_stm_load;
    for(c = 0; c < input_ch; c++) {
        cols_8b = (int*) (cols_8b_start);
        src = (int*) (input + pad_t * input_x * input_ch + c);
        y_stm_load = y_stm + c;
        for(i = 0; i < receptive_field_x_length * bufR; i++) {
            *cols_8b++ = *y_stm_load;
            y_stm_load += input_ch;
        }
        for(i = 0; i < receptive_field_y_length - bufR - receptive_pad_b; i++) {
            cols_8b += receptive_pad_l;
            src += pad_l * input_ch;
            for(j = 0; j < receptive_field_x_length - (receptive_pad_l+ receptive_pad_r); j++) {
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += input_ch * pad_r;
        }
        // for(j = 0; j < receptive_field_y_length - (use_pad_t + use_pad_b); j++) {
        //     cols_8b += use_pad_l;
        //     src += pad_l * input_ch;
        //     for(i = 0; i < receptive_field_x_length - (use_pad_l + use_pad_r); i++) {
        //         *cols_8b++ = *src;
        //         src += input_ch;
        //     }
        //     cols_8b += use_pad_r;
        //     src += pad_r * input_ch;
        
        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}
void depthwise_k5_S2_inplace_CHW_LOAD_XY_Store_X(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* x_stm, int* y_stm,
    int* new_x_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);



    int xstm_w_in_patch = min(bufR, input_x);
    int xstm_w_in_xstm = bufR - xstm_w_in_patch;


    int* stm_x_loc_in_patch = input + (input_x - xstm_w_in_patch) * input_ch;
    int* stm_x_loc_in_xstm = x_stm + (bufR - xstm_w_in_xstm) * input_ch;

    // new store X Stream
    for(i = 0; i < input_y; i++) {
        if (xstm_w_in_xstm > 0) {
            memcpy(new_x_stm, stm_x_loc_in_xstm, xstm_w_in_xstm * input_ch * BIT);
            stm_x_loc_in_xstm += xstm_w_in_xstm * input_ch;
            new_x_stm += xstm_w_in_xstm * input_ch;
        }
        memcpy(new_x_stm, stm_x_loc_in_patch, xstm_w_in_patch * input_ch * BIT);
        new_x_stm += xstm_w_in_patch * input_ch;
        stm_x_loc_in_patch += input_x * input_ch;
    }

    cols_8b += receptive_field_x_length * bufR;
    for (i = 0; i <  receptive_field_y_length - (bufR+ receptive_pad_b); i++) {
        for(j = 0; j < receptive_pad_l; j++) {
            cols_8b[j] = PAD8;
        }

        cols_8b += receptive_field_x_length;

        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b; i++){
        cols_8b += bufR;
        for(j = 0; j < receptive_field_x_length - bufR; j++) {
            *cols_8b++ = PAD8;
        }
    }
    
    int *src;
    int *ksrc = kernel;
    int *y_stm_load;
    int *x_stm_load;
    for(c = 0; c < input_ch; c++) {
        cols_8b = (int*) (cols_8b_start);
        src = (int*) (input + pad_t * input_x * input_ch + c);
        x_stm_load = x_stm + c; 
        y_stm_load = y_stm + c;
        for(i = 0; i < receptive_field_x_length * bufR; i++) {
            *cols_8b++ = *y_stm_load;
            y_stm_load += input_ch;
        }
        for(i = 0; i < receptive_field_y_length - bufR - receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            for(j = 0; j < receptive_field_x_length - bufR - receptive_pad_r; j++) {
                
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += input_ch * pad_r;

        }
        for(i = 0; i < receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            cols_8b += receptive_field_x_length - bufR;
        }
            
        printf("temporal buffer12\n");
        printTable(cols_8b_start, receptive_field_y_length, receptive_field_x_length, 1);
        printf("\n\n");
        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}

void depthwise_k5_S2_inplace_CHW_LOAD_XY(
    int* input,
    int input_x, int input_y, int input_ch,
    int* kernel,
    int padded_output_x, int padded_output_y, int output_ch,
    int* temporal_buffer,
    int pad_value,
    int pad_t, int pad_b, int pad_l, int pad_r,
    int out_pad_t, int out_pad_b, int out_pad_l, int out_pad_r,
    int* x_stm, int* y_stm,
    int bufR
    ) 
{
    // TinyEngine values
    int c, j, i;
    int* cols_8b_start = temporal_buffer;
    int* cols_8b = cols_8b_start;
    int PAD8 = pad_value;

    // get receptive field except zero output region
    int receptive_field_y_length = 2 * (padded_output_y - out_pad_t - out_pad_b) + 3;
    int receptive_field_x_length = 2 * (padded_output_x - out_pad_l - out_pad_r) + 3;

    // get final output x
    int output_y = padded_output_y - (out_pad_t + out_pad_b);
    int output_x= padded_output_x - (out_pad_l + out_pad_r);

    int receptive_pad_t = min(pad_t, 2);
    int receptive_pad_b = min(pad_b, 1);
    int receptive_pad_l = min(pad_l, 2);
    int receptive_pad_r = min(pad_r, 1);

    for (i = 0; i <  receptive_field_y_length - (receptive_pad_t+ receptive_pad_b); i++) {
        for(j = 0; j < receptive_pad_l; j++) {
            cols_8b[j] = PAD8;
        }

        cols_8b += receptive_field_x_length;

        for(j = 0; j < receptive_pad_r; j++) {
            cols_8b[-j-1] = PAD8;
        }
    }

    for(i = 0; i < receptive_pad_b * receptive_field_x_length; i++) {
        *cols_8b++ = PAD8;
    }
    
    // printf("temporal buffer\n");
    // printTable(temporal_buffer, receptive_field_y_length, receptive_field_x_length, 1);
    // printf("\n\n");
    int *src;
    int *ksrc = kernel;
    int *y_stm_load;
    int *x_stm_load;
    for(c = 0; c < input_ch; c++) {
        cols_8b = (int*) (cols_8b_start);
        src = (int*) (input + pad_t * input_x * input_ch + c);
        x_stm_load = x_stm + c; 
        y_stm_load = y_stm + c;
        for(i = 0; i < receptive_field_x_length * bufR; i++) {
            *cols_8b++ = *y_stm_load;
            y_stm_load += input_ch;
        }
        for(i = 0; i < receptive_field_y_length - bufR - receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            for(j = 0; j < receptive_field_x_length - bufR - receptive_pad_r; j++) {
                *cols_8b++ = *src;
                src += input_ch;
            }
            cols_8b += receptive_pad_r;
            src += input_ch * pad_r;

        }
        for(i = 0; i < receptive_pad_b; i++) {
            for(j = 0; j < bufR; j++) {
                *cols_8b++ = *x_stm_load;
                x_stm_load += input_ch;
            }
            cols_8b += receptive_field_x_length - bufR;
        }
        // for(j = 0; j < receptive_field_y_length - (use_pad_t + use_pad_b); j++) {
        //     cols_8b += use_pad_l;
        //     src += pad_l * input_ch;
        //     for(i = 0; i < receptive_field_x_length - (use_pad_l + use_pad_r); i++) {
        //         *cols_8b++ = *src;
        //         src += input_ch;
        //     }
        //     cols_8b += use_pad_r;
        //     src += pad_r * input_ch;
        
            
        printf("temporal buffer\n");
        printTable(cols_8b_start, receptive_field_y_length, receptive_field_x_length, 1);
        printf("\n\n");
        int* inplace_out = (int*) (input + out_pad_t * padded_output_x * input_ch + out_pad_l * input_ch + c);
        K5_S2_dw_inplace(cols_8b_start, inplace_out, kernel, receptive_field_x_length, output_y, padded_output_x, output_x, output_ch);
        kernel += 25;
    }
    
    // bypass
    memset(input, pad_value, out_pad_t * padded_output_x * input_ch * BIT);
    input += out_pad_t * padded_output_x * input_ch;
    for (i = 0; i < output_y; i++) {
        memset(input, pad_value, out_pad_l * input_ch * BIT);
        input += (output_x + out_pad_l)* input_ch;
        memset(input, pad_value, out_pad_r * input_ch * BIT);
        input +=  out_pad_r * input_ch;
    }
    memset(input, pad_value, out_pad_b * padded_output_x * input_ch * BIT);
}

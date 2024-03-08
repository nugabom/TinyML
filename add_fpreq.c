/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   add_fpreq.c
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <math.h>
#include "arm_math.h"
#include "tinyengine_function.h"
#include <string.h>

tinyengine_status add_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      output_data[i] = (int8_t)(clamped_output);
  }
}

tinyengine_status add_fpreq_FStore(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* r_store, int8_t* b_store, int8_t* br_store,
		int pad
		) {

  int i, j;
  int store_size = pad * input_ch;
  int8_t* r_store_loc = input1_data + size;
  int8_t width = (size + input_ch); // (w - 1) * C + C
  int8_t* b_store_loc = input1_data + h * width;

  for (j = 0; j < h; j++) {
	  memcpy(r_store, r_store_loc, store_size);
	  r_store += store_size;
	  r_store_loc += width;
  }
  memcpy(br_store, r_store_loc, store_size);
  memcpy(b_store, b_store_loc, size);

  for(j = 0; j < h; j++) {
	  for(i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_FStore_RLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* r_store, int8_t* b_store, int8_t* br_store,
		int8_t* r_load,
		int pad
		) {
	  int i, j;
  int store_size = pad * input_ch;
  int8_t* r_store_loc = input1_data + size;
  int8_t width = (size + input_ch);
  int8_t* b_store_loc = input1_data + h * width;

  for (j = 0; j < h; j++) {
	  memcpy(r_store, r_store_loc, store_size);
	  r_store += store_size;
	  r_store_loc += width;
  }
  memcpy(br_store, r_store_loc, store_size);
  memcpy(b_store, b_store_loc, size);


  for(j = 0; j < h; j++) {
	  for(int i = 0; i < store_size; i++) {
		  float input1_fp = ((float)*r_load++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }

	  for(i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_BStore_RLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* b_store,
		int8_t* r_load,
		int pad
		) {
	  int i, j;
  int store_size = pad * input_ch;
  int8_t width = size;
  int8_t* b_store_loc = input1_data + h * width;
  memcpy(b_store, b_store_loc, size);


  for(j = 0; j < h; j++) {
	  for(int i = 0; i < store_size; i++) {
		  float input1_fp = ((float)*r_load++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }

	  for(i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_FStore_BLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* r_store, int8_t* b_store, int8_t* br_store,
		int8_t* b_load,
		int pad
		) {

	int i,j;
  int store_size = pad * input_ch;
  int8_t* r_store_loc = input1_data + size;
  int8_t width = (size + input_ch);
  int8_t* b_store_loc = input1_data + h * width;

  for (int j = 0; j < h; j++) {
	  memcpy(r_store, r_store_loc, store_size);
	  r_store += store_size;
	  r_store_loc += width;
  }
  memcpy(br_store, r_store_loc, store_size);
  memcpy(b_store, b_store_loc, size);


  for( i = 0; i < size; i++) {
	  float input1_fp = ((float)*b_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }
  for(j = 0; j < h; j++) {
	  for(int i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_FStore_FLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* r_store, int8_t* b_store, int8_t* br_store,
		int8_t* r_load, int8_t* b_load, int8_t* br_load,
		int pad
		) {
	  int i, j;
  int store_size = pad * input_ch;
  int8_t* r_store_loc = input1_data + size;
  int8_t width = (size + input_ch);
  int8_t* b_store_loc = input1_data + h * width;

  for (int j = 0; j < h; j++) {
	  memcpy(r_store, r_store_loc, store_size);
	  r_store += store_size;
	  r_store_loc += width;
  }
  memcpy(br_store, r_store_loc, store_size);
  memcpy(b_store, b_store_loc, size);


  for(i = 0; i < store_size; i++) {
	  float input1_fp = ((float)*br_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }

  for( i = 0; i < size ; i++) {
	  float input1_fp = ((float)*b_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }
  for(j = 0; j < h; j++) {
	  for(int i = 0; i < store_size; i++) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*r_load++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  for( i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_BStore_FLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* b_store,
		int8_t* r_load, int8_t* b_load, int8_t* br_load,
		int pad
		) {

  int store_size = pad * input_ch;
  q7_t* b_store_loc = input1_data + h * size;
  int i, j;
  memcpy(b_store, b_store_loc, size);


  for(i = 0; i < store_size; i++) {
	  float input1_fp = ((float)*br_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }

  for(i = 0; i < size; i++) {
	  float input1_fp = ((float)*b_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }
  for(j = 0; j < h; j++) {
	  for(int i = 0; i < store_size; i++) {
		  float input1_fp = ((float)*r_load++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  for(i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_RStore_BLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* r_store,
		int8_t* b_load,
		int pad
		) {
	  int i, j;
  int store_size = pad * input_ch;
  int8_t width = (size + input_ch);

  q7_t* r_store_loc = input1_data + size;
  for (j = 0; j < h; j++) {
	  memcpy(r_store, r_store_loc, store_size);
	  r_store += store_size;
	  r_store_loc += width;
  }

  for( i = 0; i < size; i++) {
	  float input1_fp = ((float)*b_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }
  for(j = 0; j < h; j++) {
	  for(int i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_RStore_FLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* r_store,
		int8_t* r_load, int8_t* b_load, int8_t* br_load,
		int pad
		) {
	  int i, j;
  int store_size = pad * input_ch;
  int8_t width = (size + input_ch);
  int8_t* r_store_loc = input1_data + size;
  for (int j = 0; j < h; j++) {
	  memcpy(r_store, r_store_loc, store_size);
	  r_store += store_size;
	  r_store_loc += width;
  }


  for(i = 0; i < store_size; i++) {
	  float input1_fp = ((float)*br_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }

  for(i = 0; i < size; i++) {
	  float input1_fp = ((float)*b_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }
  for(j = 0; j < h; j++) {
	  for(i = 0; i < store_size; i++) {
		  float input1_fp = ((float)*r_load++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  for(i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

tinyengine_status add_fpreq_FLoad(
		int h,
		int size,
		int input_ch,
		const int8_t* input1_data,
		const float input1_scale,
		const float input1_zero,
		const int8_t* input2_data,
		const float input2_scale,
		const float input2_zero,
		const float output_scale,
		const float zero_y,
		int8_t* output_data,
		int8_t* r_load, int8_t* b_load, int8_t* br_load,
		int pad
		) {
  int i, j;
  int store_size = pad * input_ch;

  for(i = 0; i < store_size; i++) {
	  float input1_fp = ((float)*br_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }

  for(i = 0; i < size; i++) {
	  float input1_fp = ((float)*b_load++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);
      *output_data++ = (int8_t)(clamped_output);
  }
  for(j = 0; j < h; j++) {
	  for(int i = 0; i < store_size; i++) {
		  float input1_fp = ((float)*r_load++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  for(i = 0; i < size; ++i) {
		  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
		  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
	      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
	      clamped_output = TN_MAX(clamped_output, -128);
	      clamped_output = TN_MIN(clamped_output, 127);
	      *output_data++ = (int8_t)(clamped_output);
	  }
	  input1_data += input_ch;
  }
}

const int activation_min = -128;
const int activation_max = 127;


tinyengine_status add_fpreq_mask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask) {
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      int8_t mask_value = 1;
	  if (clamped_output < activation_min){
		  clamped_output = activation_min;
		  mask_value = 0;
	  }
	  if (clamped_output > activation_max){
		  clamped_output = activation_max;
		  mask_value = 0;
	  }
      output_data[i] = (int8_t)(clamped_output);
      output_mask[i] = mask_value;
  }
}


tinyengine_status add_fpreq_bitmask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask) {
  int mask_idx = 0;
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      int8_t mask_value = 1;
	  if (clamped_output < activation_min){
		  clamped_output = activation_min;
		  mask_value = 0;
	  }
	  if (clamped_output > activation_max){
		  clamped_output = activation_max;
		  mask_value = 0;
	  }
      output_data[i] = (int8_t)(clamped_output);
	  if (mask_value == 1)
		  BIT_SET(*output_mask, mask_idx);
	  else
		  BIT_CLEAR(*output_mask, mask_idx);
	  mask_idx++;
	  if (mask_idx == 8){
		  mask_idx = 0;
		  output_mask++;
	  }
  }
}

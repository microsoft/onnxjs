// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "binary-op.h"
#include "common.h"
#include "utils/shape_utils.h"
#include "utils/broadcast_utils.h"

// Core op
float add_core(const float &a, const float &b) { return a+b; }
float sub_core(const float &a, const float &b) { return a-b; }
float mul_core(const float &a, const float &b) { return a*b; }
float div_core(const float &a, const float &b) { return a/b; }
float xor_core(const float &a, const float &b) { return static_cast<float>(static_cast<bool>(a) ^ static_cast<bool>(b)); }
float or_core(const float &a, const float &b) { return static_cast<float>(static_cast<bool>(a) || static_cast<bool>(b)); }
float and_core(const float &a, const float &b) { return static_cast<float>(static_cast<bool>(a) && static_cast<bool>(b)); }
float prelu_core(const float &a, const float &b) {  return a>= 0 ? a : a*b; }


// Core binary operator implementation
void binary_f32_imp(const float *input1,  const int32_t &length1, const int32_t &rank1, const std::vector<int32_t> &dims1,
                             const float *input2, const int32_t &length2, const int32_t &rank2, const std::vector<int32_t> &dims2,
							 float *output, const int32_t &output_length, const int32_t &output_rank, const std::vector<int32_t> &output_dims,
							 float(*core_op)(const float&, const float&)) {
	const std::vector<int32_t> &strides1 = ShapeUtils::compute_strides(dims1);
	const std::vector<int32_t> &strides2 = ShapeUtils::compute_strides(dims2);
	const std::vector<int32_t> &output_strides = ShapeUtils::compute_strides(output_dims);
	
	for (size_t i = 0; i < output_length; ++i) {
		auto broadcasted_indices = ShapeUtils::offset_to_indices(output_strides, i);
		auto indices1 = BroadcastUtils::broadcasted_to_original_indices(broadcasted_indices, dims1);
		auto offset1 = ShapeUtils::indices_to_offset(strides1, indices1);
		auto indices2 = BroadcastUtils::broadcasted_to_original_indices(broadcasted_indices, dims2);
		auto offset2 = ShapeUtils::indices_to_offset(strides2, indices2);
		output[i] = core_op(input1[offset1], input2[offset2]);
	} 
}

// Core binary operator wrapper (Does some pre-processing prior to calling the core function)
void binary_f32_imp_wrapper(void *data,  float(*core_op)(const float&, const float&)) {
	uint32_t *dataIndex = static_cast<uint32_t *>(data);
	uint32_t const argc = dataIndex[0];
	const float *input1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
	const int32_t length1 = PARAM_INT32(data, dataIndex[2]);
	const int32_t rank1 = PARAM_INT32(data, dataIndex[3]);
	const int32_t *dims1 = PARAM_INT32_PTR(data, dataIndex[4]);
	std::vector<int32_t> dims1_vector;
	if(rank1 > 0) {
		dims1_vector.resize(rank1);
		for(int i = 0; i < rank1; ++i) {
			dims1_vector[i] = dims1[i];
		}
	}
	const float *input2 = PARAM_FLOAT_PTR(data, dataIndex[5]);
	const int32_t length2 = PARAM_INT32(data, dataIndex[6]);
	const int32_t rank2 = PARAM_INT32(data, dataIndex[7]);
	const int32_t *dims2 = PARAM_INT32_PTR(data, dataIndex[8]);
	std::vector<int32_t> dims2_vector;
	if(rank2 > 0) {
		dims2_vector.resize(rank2);
		for(int i = 0; i < rank2; ++i) {
			dims2_vector[i] = dims2[i];
		}
	}
	float *output = PARAM_FLOAT_PTR(data, dataIndex[9]);
	const int32_t output_length = PARAM_INT32(data, dataIndex[10]);
	const int32_t output_rank = PARAM_INT32(data, dataIndex[11]);
	const int32_t *output_dims = PARAM_INT32_PTR(data, dataIndex[12]);
	std::vector<int32_t> output_dims_vector;
	if(output_rank != 0) {
		output_dims_vector.resize(output_rank);
		for(size_t i = 0; i < output_rank; ++i) {
			output_dims_vector[i] = output_dims[static_cast<int32_t>(i)];
		}
	}
	binary_f32_imp(input1, length1, rank1, dims1_vector,
                           input2, length2, rank2, dims2_vector,
						    output, output_length, output_rank, output_dims_vector,
						    core_op);
}

// Wasm interop methods
void add_f32(void *data) {
	binary_f32_imp_wrapper(data, add_core);
}
void sub_f32(void *data) {
	binary_f32_imp_wrapper(data, sub_core);
}
void mul_f32(void *data) {
	binary_f32_imp_wrapper(data, mul_core);
}
void div_f32(void *data) {
	binary_f32_imp_wrapper(data, div_core);
}
void xor_f32(void *data) {
	binary_f32_imp_wrapper(data, xor_core);
}
void or_f32(void *data) {
	binary_f32_imp_wrapper(data, or_core);
}
void and_f32(void *data) {
	binary_f32_imp_wrapper(data, and_core);
}
void prelu_f32(void *data) {
	binary_f32_imp_wrapper(data, prelu_core);
}
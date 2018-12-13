// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "shape_utils.h"
#include <math.h>

size_t ShapeUtils::size_from_dims(const std::vector<int32_t> &dims) {
  auto rank = static_cast<size_t>(dims.size());
  if (rank == static_cast<size_t>(0)) {
    return static_cast<size_t>(1);
  }
  if (rank == static_cast<size_t>(1)) {
    return static_cast<size_t>(dims[0]);
  }
  size_t size = 1;
  for (auto &e : dims) {
    size *= e;
  }
  return size;
}

std::vector<int32_t>
ShapeUtils::compute_strides(const std::vector<int32_t> &dims) {
  auto rank = static_cast<size_t>(dims.size());
  if (rank == static_cast<size_t>(0) || rank == static_cast<size_t>(1)) {
    std::vector<int32_t> strides(1, 1);
    return strides;
  }
  std::vector<int32_t> strides(rank);
  strides[rank - 1] = 1;
  strides[rank - 2] = dims[rank - 1];
  for (int32_t i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

size_t ShapeUtils::indices_to_offset(const std::vector<int32_t> &strides,
                                     const std::vector<int32_t> indices) {
  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += strides[i] * indices[i];
  }
  return offset;
}

std::vector<int32_t> ShapeUtils::offset_to_indices(const std::vector<int32_t> &strides,
                                               size_t offset) {
  auto rank = static_cast<size_t>(strides.size());
  if (rank == static_cast<size_t>(0)) {
    return std::vector<int32_t>();
  }
  if (rank == static_cast<size_t>(1)) {
    return std::vector<int32_t>(1, offset * strides[0]);
  }
  std::vector<int32_t> indices(rank);
  for (size_t i = 0; i < indices.size() - 1; ++i) {
    indices[i] = floor(offset / strides[i]);
    offset -= indices[i] * strides[i];
  }
  indices[indices.size() - 1] = offset;
  return indices;
}

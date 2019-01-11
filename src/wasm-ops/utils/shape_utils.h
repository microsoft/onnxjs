// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

namespace ShapeUtils {
  size_t size_from_dims(const std::vector<int32_t> &dims);
  std::vector<int32_t> compute_strides(const std::vector<int32_t> &dims);
  size_t indices_to_offset(const std::vector<int32_t> &strides,
                                  const std::vector<int32_t> &indices);
  std::vector<int32_t>
  offset_to_indices(const std::vector<int32_t> &strides, size_t offset);
};

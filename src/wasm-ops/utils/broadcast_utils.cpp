// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "broadcast_utils.h"

std::vector<int32_t> BroadcastUtils::broadcasted_to_original_indices(
    const std::vector<int32_t> &broadcasted_indices,
    const std::vector<int32_t> &dims) {
  const auto rank = dims.size();
  if (rank == 0) {
    return std::vector<int32_t>();
  }
  std::vector<int32_t> original_indices(rank);
  BroadcastUtils::broadcasted_to_original_indices(broadcasted_indices, dims,
                                                  original_indices);
  return original_indices;
}

void BroadcastUtils::broadcasted_to_original_indices(
    const std::vector<int32_t> &broadcasted_indices,
    const std::vector<int32_t> &dims, std::vector<int32_t> &original_indices) {
  const auto rank = dims.size();
  if (rank == 0) {
    return;
  }
  auto offset = broadcasted_indices.size() - dims.size();
  for (size_t i = 0; i < rank; ++i) {
    original_indices[i] = broadcasted_indices[offset + i] % dims[i];
  }
}

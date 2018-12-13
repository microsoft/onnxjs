// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "broadcast_utils.h"

std::vector<int32_t> BroadcastUtils::broadcasted_to_original_indices(
    const std::vector<int32_t> &broadcasted_indices,
    const std::vector<int32_t> &dims) {
  if (broadcasted_indices.size() < dims.size()) {
    // TODO: Throw error
  }
  const auto rank = static_cast<size_t>(dims.size());
  if (rank == static_cast<size_t>(0)) {
    return std::vector<int32_t>();
  }
  std::vector<int32_t> original_indices(rank, 0);
  auto offset = broadcasted_indices.size() - dims.size();
  for (size_t i = 0; i < rank; ++i) {
    original_indices[i] = broadcasted_indices[offset + i] % dims[i];
  }
  return original_indices;
}

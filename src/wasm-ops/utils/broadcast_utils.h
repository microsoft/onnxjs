// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

namespace BroadcastUtils {
std::vector<int32_t>
broadcasted_to_original_indices(const std::vector<int32_t> &broadcasted_indices,
                                const std::vector<int32_t> &dims);

// Fills in values in the original_indices vector. Assumes it is of the required
// size.
void broadcasted_to_original_indices(
    const std::vector<int32_t> &broadcasted_indices,
    const std::vector<int32_t> &dims, std::vector<int32_t> &original_indices);
}; // namespace BroadcastUtils

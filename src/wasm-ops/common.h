// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stddef.h>
#include <stdint.h>

static_assert(sizeof(int) == sizeof(int32_t),
              "'int' and 'int32_t' should be the same type");

#define PARAM_VALUE(data, offset, type)                                        \
  (*((type *)((((uint8_t *)(data)) + ((size_t)(offset))))))
#define PARAM_PTR(data, offset, type)                                          \
  ((type *)((offset == 0) ? nullptr                                            \
                          : (((uint8_t *)(data)) + ((size_t)(offset)))))

#define PARAM_BOOL(data, offset) (!!PARAM_VALUE(data, offset, uint8_t))
#define PARAM_INT32(data, offset) PARAM_VALUE(data, offset, int32_t)
#define PARAM_FLOAT(data, offset) PARAM_VALUE(data, offset, float)
#define PARAM_BOOL_PTR(data, offset) PARAM_PTR(data, offset, uint8_t)
#define PARAM_INT32_PTR(data, offset) PARAM_PTR(data, offset, int32_t)
#define PARAM_FLOAT_PTR(data, offset) PARAM_PTR(data, offset, float)

// Data is passed to the core operator as a void pointer
// The first argument is the number of arguments
// The subsequent parameters have the offset to the actual parameters
// The core operator implementation parse the actual parameters based on the
// type it expects

// Example:
//     BYTES      PARAMETER      TYPE       DESCRIPTION           EXAMPLE VALUE
// [0   ...   3]  argc         [uint32]  count of arguments             5
// [4   ...   7]  offset_arg0  [uint32]  offset in bytes of arg0        24
// [    ...    ]  ...
// [20  ...  23]  offset_arg4  [uint32]  offset in bytes of arg4        200
// [24  ... ...]  data_arg0    ...
// [    ...    ]  ...
// [200 ... ...]  data_arg4    ...

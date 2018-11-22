// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// tslint:disable:no-require-imports
if (typeof window !== 'undefined') {
  require('./backends/webgl/test_glsl_function_inliner');
  require('./backends/webgl/test_conv_new');
}
require('./api/tensor');
require('./api/inference-session');

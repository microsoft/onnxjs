// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// TODO: Inference session test cases
import {expect} from 'chai';

import {InferenceSession} from '../../../lib/api';

describe('#UnitTest# - API - Check Globals and Imports', () => {
  it('Compare Global InferenceSession and Imported InferenceSession', () => {
    expect(onnx.InferenceSession).is.equal(InferenceSession);
  });
});

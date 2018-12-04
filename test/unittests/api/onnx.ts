// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {expect} from 'chai';

// tslint:disable:no-require-imports
const apiRequireIndex = require('../../../lib/api');
const onnxImpl = require('../../../lib/api/onnx-impl');

import {InferenceSession, Tensor, backend} from '../../../lib/api';

describe('#UnitTest# - API - Check Globals and Imports', () => {
  it('Compare Global onnx and Imported onnx', () => {
    expect(typeof onnx).is.not.equal('undefined');
    expect(typeof apiRequireIndex).is.not.equal('undefined');
    expect(typeof onnxImpl).is.not.equal('undefined');
    expect(apiRequireIndex).is.equal(onnx);
    expect(apiRequireIndex).is.equal(onnxImpl);
  });

  it('Compare Global Tensor and Imported Tensor', () => {
    expect(onnx.Tensor).is.equal(Tensor);
  });

  it('Compare Global InferenceSession and Imported InferenceSession', () => {
    expect(onnx.InferenceSession).is.equal(InferenceSession);
  });

  it('Compare Global backend and Imported backend', () => {
    expect(onnx.backend).is.equal(backend);
  });

  it('Ensure no value exported from interface file', () => {
    const onnx = require('../../../lib/api/onnx');
    const onnxPropertyNames = Object.getOwnPropertyNames(onnx);
    const onnxExportedValues = onnxPropertyNames.filter(name => name !== '__esModule');
    expect(onnxExportedValues).to.have.lengthOf(0);

    const tensor = require('../../../lib/api/tensor');
    const tensorPropertyNames = Object.getOwnPropertyNames(tensor);
    const tensorExportedValues = tensorPropertyNames.filter(name => name !== '__esModule');
    expect(tensorExportedValues).to.have.lengthOf(1);
    expect(tensorExportedValues).to.contain('Tensor');

    const inferenceSession = require('../../../lib/api/inference-session');
    const inferenceSessionPropertyNames = Object.getOwnPropertyNames(inferenceSession);
    const inferenceSessionExportedValues = inferenceSessionPropertyNames.filter(name => name !== '__esModule');
    expect(inferenceSessionExportedValues).to.have.lengthOf(1);
    expect(inferenceSessionExportedValues).to.contain('InferenceSession');
  });

  it('Ensure value exported from implementation file', () => {
    const onnxImplPropertyNames = Object.getOwnPropertyNames(onnxImpl);
    expect(onnxImplPropertyNames).to.have.lengthOf.at.least(1);

    const tensorImpl = require('../../../lib/api/tensor-impl');
    const tensorImplPropertyNames = Object.getOwnPropertyNames(tensorImpl);
    expect(tensorImplPropertyNames).to.contain('Tensor');

    const inferenceSessionImpl = require('../../../lib/api/inference-session-impl');
    const inferenceSessionImplPropertyNames = Object.getOwnPropertyNames(inferenceSessionImpl);
    expect(inferenceSessionImplPropertyNames).to.contain('InferenceSession');
  });
});

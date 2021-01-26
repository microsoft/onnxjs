// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
import {expect} from 'chai';

import {Backend, InferenceHandler, SessionHandler} from '../../../../lib/backend';
import {WebGLInferenceHandler} from '../../../../lib/backends/webgl/inference-handler';
import {WebGLPack} from '../../../../lib/backends/webgl/ops/pack';
import {WebGLUnpack} from '../../../../lib/backends/webgl/ops/unpack';
import {Profiler} from '../../../../lib/instrument';
import {Tensor} from '../../../../lib/tensor';

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

function createRandomArray(size: number): Float32Array {
  const randomTable = [0, 3, 6, 9, 2, 5, 8, 1, 4, 7];
  return new Float32Array(
      Array.from({length: size}, (v, k) => randomTable[k % 10] * 0.1 + randomTable[Math.trunc(k / 10) % 10] * 0.01));
}

describe('#UnitTest# - pack - Tensor pack', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await Backend('webgl');
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });
  it('Test pack kernal', () => {
    const op = new WebGLPack();

    // create input data and tensor. The input data will be used to verify if the output tensor contains the same value
    // but possibly different order depending on our packing algorithm.
    const inputData = createRandomArray(4);
    const inputTensor = new Tensor([2, 2], 'float32', undefined, undefined, inputData);

    // compile shader code
    const programInfo = op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensor]);
    const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;
    const artifact = webglInferenceHandler.session.programManager.build(programInfo);
    webglInferenceHandler.session.programManager.setArtifact(op, artifact);

    // run kernal and get output
    const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensor]);
    webglInferenceHandler.session.programManager.run(artifact, runData);
    const result = runData.outputTextureData;

    // verify result.
    // TODO: add more verifications including output value and order
    // TODO: also different input dimensions for different pack code path
    expect(result).to.not.equal(null);
  });
  it('Test unpack', () => {
    const op = new WebGLUnpack();

    // create input data and tensor. The input data will be used to verify if the output tensor contains the same value
    // but possibly different order depending on our packing algorithm.
    const inputData = createRandomArray(4);
    const inputTensor = new Tensor([2, 2], 'float32', undefined, undefined, inputData);

    // compile shader code
    const programInfo = op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensor]);
    const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;
    const artifact = webglInferenceHandler.session.programManager.build(programInfo);
    webglInferenceHandler.session.programManager.setArtifact(op, artifact);

    // run kernal and get output
    const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensor]);
    webglInferenceHandler.session.programManager.run(artifact, runData);
    const result = runData.outputTextureData;

    // verify result.
    // TODO: add more verifications including output value and order
    expect(result).to.not.equal(null);
  });
});

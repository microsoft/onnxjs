// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
import {expect} from 'chai';

import {Attribute} from '../../../../lib/attribute';
import {Backend, InferenceHandler, SessionHandler} from '../../../../lib/backend';
import {WebGLInferenceHandler} from '../../../../lib/backends/webgl/inference-handler';
import {WebGLPackedConcat} from '../../../../lib/backends/webgl/ops/concat_packed';
import {Profiler} from '../../../../lib/instrument';
import {Tensor} from '../../../../lib/tensor';
import {ShapeUtil} from '../../../../lib/util';

import {createAscendingArray} from './test_utils';
import {createTextureFromArray} from './test_utils';

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

describe('#UnitTest# - packed concat - Tensor concat', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await Backend('webgl');
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });
  const testDataSet = getTestData();
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Test concat ${JSON.stringify(testData)}`, () => {});
    it(`Test packed concat kernel `, () => {
      const op = new WebGLPackedConcat();
      const attributes = new Attribute(undefined);
      const axis = testData.axis;
      attributes.set('axis', 'int', axis);

      op.initialize(attributes);
      const elementCount = testData.elementCount;
      const inputTensorShape = testData.inputShape;
      const inputTextureShape = testData.inputTextureShape;
      const outputTensorShape = testData.outputShape;

      // create input data and tensor. The input data will be used to verify if the output tensor contains the
      // same value but possibly different order depending on our packing algorithm.
      const inputData = createAscendingArray(elementCount);
      const inputTensorA = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);
      const inputTensorB = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      // manually creat packed texture from inputTensor, and insert in cache
      const gl = webglInferenceHandler.session.textureManager.glContext.gl;
      webglInferenceHandler.session.textureManager.glContext.checkError();
      const webglTextureA = createTextureFromArray(
          webglInferenceHandler.session.textureManager.glContext, testData.rawInput ? testData.rawInput : inputData,
          gl.RGBA, inputTextureShape[0], inputTextureShape[1]);
      const webglTextureB = createTextureFromArray(
          webglInferenceHandler.session.textureManager.glContext, testData.rawInput ? testData.rawInput : inputData,
          gl.RGBA, inputTextureShape[0], inputTextureShape[1]);

      webglInferenceHandler.session.textureManager.glContext.checkError();
      const packedShape = inputTextureShape;
      const textureDataA = {
        width: inputTextureShape[0],
        height: inputTextureShape[1],
        channels: 4 as const,
        isPacked: true,
        shape: packedShape,
        strides: ShapeUtil.computeStrides(packedShape),
        unpackedShape: outputTensorShape,
        tensor: inputTensorA,
        texture: webglTextureA!
      };
      const textureDataB = {
        width: inputTextureShape[0],
        height: inputTextureShape[1],
        channels: 4 as const,
        isPacked: true,
        shape: packedShape,
        strides: ShapeUtil.computeStrides(packedShape),
        unpackedShape: outputTensorShape,
        tensor: inputTensorB,
        texture: webglTextureB!
      };

      webglInferenceHandler.setTextureData(inputTensorA.dataId, textureDataA);
      webglInferenceHandler.setTextureData(inputTensorB.dataId, textureDataB);

      // compile shader code
      const programInfo =
          op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensorA, inputTensorB]);

      const artifact = webglInferenceHandler.session.programManager.build(programInfo);
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensorA, inputTensorB]);
      webglInferenceHandler.session.programManager.run(artifact, runData);
      const result = runData.outputTextureData.tensor.data;

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      const expectedOutput = testData.expectedOutput;
      console.log('result: ', result);
      expect(result).to.not.equal(null);

      expect(result).to.have.lengthOf(elementCount * 2);

      expect(result).to.deep.equal(expectedOutput);
    });
  }
});
interface TestData {
  elementCount: number;
  axis: number;
  inputShape: number[];
  outputShape: number[];
  inputTextureShape: number[];
  outputTextureShape: number[];
  expectedOutput: Float32Array;
  rawInput?: Float32Array;
}
function getTestData(): TestData[] {
  return [
    // // test 1D tensor
    // {
    //   elementCount: 4,
    //   axis: 0,
    //   inputShape: [4],
    //   outputShape: [8],
    //   inputTextureShape: [2, 1],
    //   outputTextureShape: [4, 1],
    //   expectedOutput: new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 1, 2, 5, 6, 3, 4, 7, 8]),
    //   rawInput: new Float32Array([1, 2, 0, 0, 5, 6, 0, 0, 3, 4, 0, 0, 7, 8, 0, 0]),
    // },

    // // test 2D tensor
    {
      elementCount: 16,
      axis: 0,
      inputShape: [4, 4],
      outputShape: [8, 4],
      inputTextureShape: [2, 2],
      outputTextureShape: [2, 4],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16
      ]),
    },
    {
      elementCount: 16,
      axis: 1,
      inputShape: [4, 4],
      outputShape: [8, 4],
      inputTextureShape: [2, 2],
      outputTextureShape: [2, 2],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 1, 2, 5, 6, 3, 4, 7, 8, 3, 4, 7, 8, 9, 10, 13, 14, 9, 10, 13, 14, 11, 12, 15, 16, 11, 12, 15, 16
      ]),
    },

    // {
    //   elementCount: 8,
    //   axis: 0,
    //   inputShape: [2, 4],
    //   outputShape: [2, 4],
    //   inputTextureShape: [2, 1],
    //   outputTextureShape: [2, 1],
    //   useGeneratedOutput: true,
    // },
    // {
    //   elementCount: 8,
    //   axis: 1,
    //   inputShape: [2, 4],
    //   outputShape: [2, 4],
    //   inputTextureShape: [2, 1],
    //   outputTextureShape: [2, 1],
    //   useGeneratedOutput: true,
    // },
    // {
    //   elementCount: 6,
    //   axis: 0,
    //   inputShape: [2, 3],
    //   outputShape: [2, 3],
    //   inputTextureShape: [2, 1],
    //   outputTextureShape: [2, 3],
    //   rawData: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0]),
    // },
    // {
    //   elementCount: 6,
    //   axis: 1,
    //   inputShape: [2, 3],
    //   outputShape: [2, 3],
    //   inputTextureShape: [2, 1],
    //   outputTextureShape: [2, 3],
    //   rawData: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0]),
    // },

    // // // test 3d tensor
    // {
    //   elementCount: 16,
    //   axis: 0,
    //   inputShape: [2, 2, 4],
    //   outputShape: [2, 2, 4],
    //   inputTextureShape: [2, 2],
    //   outputTextureShape: [4, 4],
    //   useGeneratedOutput: true,
    // },
    // {
    //   elementCount: 16,
    //   axis: 1,
    //   inputShape: [2, 2, 4],
    //   outputShape: [2, 2, 4],
    //   inputTextureShape: [2, 2],
    //   outputTextureShape: [4, 4],
    //   useGeneratedOutput: true,
    // },
    // {
    //   elementCount: 24,
    //   axis: 0,
    //   inputShape: [2, 3, 4],
    //   outputShape: [2, 3, 4],
    //   inputTextureShape: [2, 4],
    //   outputTextureShape: [6, 4],
    //   rawData: new Float32Array([
    //     1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 0, 0, 11, 12, 0, 0,
    //     13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0
    //   ])
    // },
    // {
    //   elementCount: 24,
    //   axis: 1,
    //   inputShape: [2, 3, 4],
    //   outputShape: [2, 3, 4],
    //   inputTextureShape: [2, 4],
    //   outputTextureShape: [6, 4],
    //   rawData: new Float32Array([
    //     1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 0, 0, 11, 12, 0, 0,
    //     13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0
    //   ])
    // },
    // // test 4d tensor
    // {
    //   elementCount: 32,
    //   axis: 0,
    //   inputShape: [2, 2, 2, 4],
    //   outputShape: [2, 2, 2, 4],
    //   inputTextureShape: [2, 4],
    //   outputTextureShape: [8, 4],
    //   useGeneratedOutput: true,
    // },
    // {
    //   elementCount: 32,
    //   axis: 1,
    //   inputShape: [2, 2, 2, 4],
    //   outputShape: [2, 2, 2, 4],
    //   inputTextureShape: [2, 4],
    //   outputTextureShape: [8, 4],
    //   useGeneratedOutput: true,
    // },
    // {
    //   elementCount: 64,
    //   axis: 0,
    //   inputShape: [2, 2, 4, 4],
    //   outputShape: [2, 2, 4, 4],
    //   inputTextureShape: [2, 8],
    //   outputTextureShape: [16, 4],
    //   useGeneratedOutput: true,
    // },
    // {
    //   elementCount: 64,
    //   axis: 1,
    //   inputShape: [2, 2, 4, 4],
    //   outputShape: [2, 2, 4, 4],
    //   inputTextureShape: [2, 8],
    //   outputTextureShape: [16, 4],
    //   useGeneratedOutput: true,
    // },
  ];
}

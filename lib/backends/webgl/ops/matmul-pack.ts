// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {MatMul} from '../../../ops/matmul';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLMatMulPacked extends MatMul implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const hasBias = inputs.length > 2;
    const processBias = hasBias ? `value += getBiasAtOutCoords();` : ``;
    const aShape = inputs[0].dims;
    const bShape = inputs[1].dims;
    const outputShape = BroadcastUtil.calcShape(aShape, bShape, true);
    if (!outputShape) {
      throw new Error('Can\'t use matmul on the given tensors');
    }
    const rank = outputShape.length;
    const arank = aShape.length;
    const brank = bShape.length;
    const sharedDim = aShape[aShape.length - 1];
    const shaderSource = `
      vec4 process(int indices[${rank}]) {
          int a[${arank}];
          int b[${brank}];
          bcastMatmulIndices_A(indices, a);
          bcastMatmulIndices_B(indices, b);

          vec4 value;
          for (int k=0; k<((${sharedDim}+1)/2); ++k) {
              a[${arank - 1}] = k;
              b[${brank - 2}] = k;
              value += _A_Pack(a).rrbb * _B_Pack(b).rgrg;
              value += _A_Pack(a).ggaa * _B_Pack(b).baba;
          }
          ${processBias}
          return value;
      }`;
    return {
      inputLayouts: inputs.map((t, i) => handler.getOrCreateTextureLayout(t, 4, true, inputs[i].dims, true)),
      outputLayout:
          handler.createTextureLayoutFromShape(outputShape, 4, outputShape, {isPacked: true, reverseWH: true}),
      samplers: hasBias ? ['A', 'B', 'Bias'] : ['A', 'B'],
      shaderSource,
      expectPackedInputs: true,
      expectPackedOutputs: true,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map(
        (t) => handler.getOrCreateTextureData(t, handler.getOrCreateTextureLayout(t, 1, false, [], true), true));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

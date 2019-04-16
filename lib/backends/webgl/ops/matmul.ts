// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {MatMul} from '../../../ops/matmul';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {WebGLOperator} from '../webgl-operator';

export class WebGLMatMul extends MatMul implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
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
      uniform sampler2D A;
      uniform sampler2D B;
      float process(int indices[${rank}]) {
          int a[${arank}];
          int b[${brank}];
          bcastMatmulIndices_A(indices, a);
          bcastMatmulIndices_B(indices, b);

          float value;
          for (int k=0; k<${sharedDim}; ++k) {
              a[${arank - 1}] = k;
              b[${brank - 2}] = k;
              value += _A(a) * _B(b);
          }
          return value;
      }`;
    return {
      hasMain: false,
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createBasicTextureLayout(outputShape),
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreate(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].dataType),
      uniformData: {}
    };
  }
}

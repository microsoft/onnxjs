// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Gemm} from '../../../ops/gemm';
import {Tensor} from '../../../tensor';
import {GemmUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLGemm extends Gemm implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const aShape = inputs[0].dims.slice();
    const bShape = inputs[1].dims.slice();
    const cShape = inputs[2].dims.slice();
    const oShape = GemmUtil.getShapeOfGemmResult(aShape, this.transA, bShape, this.transB, cShape);
    if (!oShape) {
      throw new Error('Can\'t use gemm on the given tensors');
    }
    let sharedDim = aShape[aShape.length - 1];
    let line = '';
    if (this.transA) {
      sharedDim = aShape[0];
    }
    if (this.transA && this.transB) {
      line = `value += _A_T(a) * _B_T(b);`;
    } else if (this.transA && !this.transB) {
      line = `value += _A_T(a) * _B(b);`;
    } else if (!this.transA && this.transB) {
      line = `value += _A(a) * _B_T(b);`;
    } else if (!this.transA && !this.transB) {
      line = `value += _A(a) * _B(b);`;
    }
    const rank = oShape.length;
    const cRank = cShape.length;
    const shaderSource = `
      float process(int indices[${rank}]) {
          int a[${rank}];
          int b[${rank}];
          int c[${cRank}];

          copyVec(indices, a);
          copyVec(indices, b);
          bcastIndices_C(indices, c);

          float value = 0.0;
          for (int k=0; k<${sharedDim}; ++k) {
              a[${rank - 1}] = k;
              b[${rank - 2}] = k;
              ${line}
          }

          value = value * alpha;
          value += beta * _C(c);
          return value;
      }`;
    const inputLayouts = inputs.map(t => inferenceHandler.getOrCreateTextureLayout(t));
    return {
      inputLayouts,
      outputLayout: inferenceHandler.createTextureLayoutFromShape(oShape),
      samplers: ['A', 'B', 'C'],
      variables: [{name: 'alpha', type: 'float'}, {name: 'beta', type: 'float'}],
      shaderSource,
    };
  }
  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => inferenceHandler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {'alpha': this.alpha, 'beta': this.beta}
    };
  }
}

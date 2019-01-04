// Licensed under the MIT license.

import {Gather} from '../../../ops/gather';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {WebGLOperator} from '../webgl-operator';
import {WebGLOperatorHelper} from '../webgl-operator-utils';

export class WebGLGather extends Gather implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return WebGLOperatorHelper.run(this, inferenceHandler, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShape = inputs[0].dims.slice();
    const indexDataShape = inputs[1].dims.slice();
    const outputShape = new Array(inputShape.length + indexDataShape.length - 1);
    const indexCopyOps: string[] = [];
    for (let i = 0; i < outputShape.length; i++) {
      // outputShape is divided into three parts: A, B, C
      // |0         this.axis|           this.axis + indexDataShape.length|          end|
      // |     A             |                     B                      |      C      |
      //
      // inputIdx: [A, inputs[1][B], C]
      if (i < this.axis) {  // A
        outputShape[i] = inputShape[i];
        indexCopyOps.push(`inputIdx[${i}] = outputIdx[${i}];`);
      } else {
        if (i < this.axis + indexDataShape.length) {  // B
          outputShape[i] = indexDataShape[i - this.axis];
          indexCopyOps.push(`indexDataIdx[${i - this.axis}] = outputIdx[${i}];`);
        } else {                                                       // C
          outputShape[i] = inputShape[i - indexDataShape.length + 1];  // skip 1 for this.axis
          indexCopyOps.push(`inputIdx[${i - indexDataShape.length + 1}] = outputIdx[${i}];`);
        }
      }
    }

    const orank = outputShape.length;
    const irank = inputShape.length;
    const iDrank = indexDataShape.length;
    const shaderSource = `
      uniform sampler2D A;
      uniform sampler2D B;
      float process(int outputIdx[${orank}]) {
        int inputIdx[${irank}];
        int indexDataIdx[${iDrank}];
        ${indexCopyOps.join('\n        ')}
        inputIdx[${this.axis}] = int(_B(indexDataIdx));
        return _A(inputIdx);
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

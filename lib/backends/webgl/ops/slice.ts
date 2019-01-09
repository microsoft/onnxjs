// Licensed under the MIT license.

import {Slice} from '../../../ops/slice';
import {Tensor} from '../../../tensor';
import {getActualAxisFromNegativeValue} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {WebGLOperator} from '../webgl-operator';
import {WebGLOperatorHelper} from '../webgl-operator-utils';

export class WebGLSlice extends Slice implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return WebGLOperatorHelper.run(this, inferenceHandler, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const x = inputs[0];
    let axes = this.axes;
    let starts = this.starts;
    let ends = this.ends;

    if (axes.length === 0) {
      axes = x.dims.slice(0).map((val, ind) => ind);
    }
    axes = axes.map(axis => getActualAxisFromNegativeValue(axis, x.dims.length));
    starts = starts.map((start, ind) => {
      if (start > x.dims[axes[ind]] - 1) {
        return x.dims[axes[ind]];
      }
      return getActualAxisFromNegativeValue(start, x.dims[axes[ind]]);
    });
    ends = ends.map((end, ind) => {
      if (end > x.dims[axes[ind]] - 1) {
        return x.dims[axes[ind]];
      }
      return getActualAxisFromNegativeValue(end, x.dims[axes[ind]]);
    });

    const outputShape = x.dims.slice();

    const sliceOps: string[] = [];
    for (let i = 0; i < axes.length; i++) {
      outputShape[axes[i]] = ends[i] - starts[i];
      if (starts[i] > 0) {
        sliceOps.push(`outputIdx[${axes[i]}] += ${starts[i]};`);
      }  // else { sliceOps.push(`outputIdx[${axes[i]}] += 0;`); }
    }

    const rank = outputShape.length;
    const shaderSource = `
    uniform sampler2D A;
    float process(int outputIdx[${rank}]) {
      ${sliceOps.join('\n      ')}
      return _A(outputIdx);
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

// Licensed under the MIT license.

import {Slice, SliceV10} from '../../../ops/slice';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
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
    return createProgramInfo(handler, inputs[0], this.starts, this.ends, this.axes);
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    return createRunData(handler, programInfo, inputs);
  }
}

export class WebGLSliceV10 extends SliceV10 implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return WebGLOperatorHelper.run(this, inferenceHandler, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (!handler.session.isInitializer(inputs[1]) || !handler.session.isInitializer(inputs[2]) ||
        (inputs.length >= 4 && !handler.session.isInitializer(inputs[3])) ||
        (inputs.length >= 5 && !handler.session.isInitializer(inputs[4]))) {
      throw new Error(`dynamic slice attributes are not allowed`);
    }
    if (inputs.length >= 5 && inputs[4].integerData.some((i: number) => i !== 1)) {
      throw new Error(`currently non-1 steps is not supported for Slice`);
    }
    const starts = Array.from(inputs[1].integerData);
    const ends = Array.from(inputs[2].integerData);
    const axes = inputs.length >= 4 ? Array.from(inputs[3].integerData) : [];

    return createProgramInfo(handler, inputs[0], starts, ends, axes);
  }

  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    return createRunData(handler, programInfo, inputs);
  }
}

function createProgramInfo(
    handler: WebGLInferenceHandler, x: Tensor, starts: ReadonlyArray<number>, ends: ReadonlyArray<number>,
    axes: ReadonlyArray<number>): ProgramInfo {
  if (axes.length === 0) {
    axes = x.dims.slice(0).map((val, ind) => ind);
  }
  axes = axes.map(axis => ShapeUtil.parseAxis(axis, x.dims.length));
  starts = starts.map((start, ind) => {
    if (start > x.dims[axes[ind]] - 1) {
      return x.dims[axes[ind]];
    }
    return ShapeUtil.parseAxis(start, x.dims[axes[ind]]);
  });
  ends = ends.map((end, ind) => {
    if (end > x.dims[axes[ind]] - 1) {
      return x.dims[axes[ind]];
    }
    return ShapeUtil.parseAxis(end, x.dims[axes[ind]]);
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
    inputLayouts: [handler.getOrCreateTextureLayout(x)],
    outputLayout: handler.createBasicTextureLayout(outputShape),
    shaderSource,
  };
}

function createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
  const inputTDs = [handler.getOrCreate(inputs[0], programInfo.inputLayouts[0])];
  return {
    inputTextureDatas: inputTDs,
    outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].dataType),
    uniformData: {}
  };
}

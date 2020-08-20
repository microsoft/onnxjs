// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {ReduceBase} from '../../../ops/reduce-op';
import {Tensor} from '../../../tensor';
import {ReduceUtil, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuReduceSum extends ReduceBase {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = reduceSum(inputs[0], ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length), this.keepDims);
    return [output];
  }
}

export class CpuReduceSumSquare extends ReduceBase {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = reduceSumSquare(inputs[0], ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length), this.keepDims);
    return [output];
  }
}

export class CpuReduceLogSum extends ReduceBase {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = reduceLogSum(inputs[0], ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length), this.keepDims);
    return [output];
  }
}

export class CpuReduceMax extends ReduceBase {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = reduceMax(inputs[0], ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length), this.keepDims);
    return [output];
  }
}

export class CpuReduceMin extends ReduceBase {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = reduceMin(inputs[0], ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length), this.keepDims);
    return [output];
  }
}

export class CpuReduceMean extends ReduceBase {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = reduceMean(inputs[0], ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length), this.keepDims);
    return [output];
  }
}

export class CpuReduceProd extends ReduceBase {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = reduceProd(inputs[0], ShapeUtil.normalizeAxes(this.axes, inputs[0].dims.length), this.keepDims);
    return [output];
  }
}

// Functions implementing specific reduce operations
export function reduceSum(input: Tensor, axes: number[], keepDims: boolean): Tensor {
  return ReduceUtil.calcReduce(input, axes, keepDims, b => b, (a, b) => a + b);
}

export function reduceSumSquare(input: Tensor, axes: number[], keepDims: boolean): Tensor {
  return ReduceUtil.calcReduce(input, axes, keepDims, b => b * b, (a, b) => a + b);
}

export function reduceLogSum(input: Tensor, axes: number[], keepDims: boolean): Tensor {
  const output = ReduceUtil.calcReduce(input, axes, keepDims, b => b, (a, b) => a + b);
  const length = output.floatData.length;
  for (let i = 0; i < length; i++) {
    output.floatData[i] = Math.log(output.floatData[i]);
  }
  return output;
}

export function reduceMax(input: Tensor, axes: number[], keepDims: boolean): Tensor {
  return ReduceUtil.calcReduce(input, axes, keepDims, b => b, (a, b) => Math.max(a, b));
}

export function reduceMin(input: Tensor, axes: number[], keepDims: boolean): Tensor {
  return ReduceUtil.calcReduce(input, axes, keepDims, b => b, (a, b) => Math.min(a, b));
}

export function reduceMean(input: Tensor, axes: number[], keepDims: boolean): Tensor {
  const output = ReduceUtil.calcReduce(input, axes, keepDims, b => b, (a, b) => a + b);
  const outputDims = ReduceUtil.calcReduceShape(input.dims as number[], axes, keepDims);
  const inputSize = ShapeUtil.size(input.dims);
  const outputSize = ShapeUtil.size(outputDims);
  const numItems = inputSize / outputSize;
  const length = output.floatData.length;
  for (let i = 0; i < length; i++) {
    output.floatData[i] = output.floatData[i] / numItems;
  }
  return output;
}

export function reduceProd(input: Tensor, axes: number[], keepDims: boolean): Tensor {
  return ReduceUtil.calcReduce(input, axes, keepDims, b => b, (a, b) => a * b);
}

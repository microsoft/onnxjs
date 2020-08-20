// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Slice, SliceV10} from '../../../ops/slice';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuSlice extends Slice {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = slice(inputs[0], this.starts, this.ends, this.axes);
    return [output];
  }
}

export class CpuSliceV10 extends SliceV10 {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    if (inputs.length >= 5 && inputs[4].integerData.some((i: number) => i !== 1)) {
      throw new Error(`currently non-1 steps is not supported for Slice`);
    }
    const starts = Array.from(inputs[1].integerData);
    const ends = Array.from(inputs[2].integerData);
    const axes = inputs.length >= 4 ? Array.from(inputs[3].integerData) : [];
    const output = slice(inputs[0], starts, ends, axes);
    return [output];
  }
}

export function slice(
    x: Tensor, starts: ReadonlyArray<number>, ends: ReadonlyArray<number>, axes: ReadonlyArray<number>): Tensor {
  if (axes.length === 0) {
    axes = x.dims.map((val, ind) => ind);
  }
  axes = ShapeUtil.normalizeAxes(axes, x.dims.length);
  starts = starts.map((start, ind) => {
    if (start > x.dims[axes[ind]] - 1) {
      return x.dims[axes[ind]];
    }
    return ShapeUtil.normalizeAxis(start, x.dims[axes[ind]]);
  });
  ends = ends.map((end, ind) => {
    if (end > x.dims[axes[ind]] - 1) {
      return x.dims[axes[ind]];
    }
    return ShapeUtil.normalizeAxis(end, x.dims[axes[ind]]);
  });
  const size: number[] = [];
  const adjustedStarts: number[] = [];
  axes.forEach((val, ind) => {
    size[val] = ends[ind] - starts[ind];
    adjustedStarts[val] = starts[ind];
  });
  for (let i = 0; i < x.dims.length; i++) {
    size[i] = size[i] || x.dims[i];
    adjustedStarts[i] = adjustedStarts[i] || 0;
  }

  const newDimsStride = ShapeUtil.computeStrides(size);
  const oldDimsStride = ShapeUtil.computeStrides(x.dims ? x.dims : [x.data.length]);
  const X = x.data;
  const output = new Tensor(size, x.type);
  const Y = output.data;
  for (let i = 0; i < Y.length; ++i) {
    const newLogicalIndex = ShapeUtil.offsetToIndices(i, newDimsStride);
    const oldLogicalIndex = newLogicalIndex.map((idx, j) => idx + adjustedStarts[j]);
    const oldOffset = ShapeUtil.indicesToOffset(oldLogicalIndex, oldDimsStride);
    Y[i] = X[oldOffset];
  }
  return output;
}

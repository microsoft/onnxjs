// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Slice} from '../../../ops/slice';
import {Tensor} from '../../../tensor';
import {getActualAxisFromNegativeValue, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuSlice extends Slice {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = slice(inputs[0], this.starts, this.ends, this.axes);
    return [output];
  }
}

export function slice(x: Tensor, starts: number[], ends: number[], axes: number[]): Tensor {
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
  const size: number[] = [];
  const adjustedStarts: number[] = [];
  axes.map((val, ind) => {
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
  const Y = output.numberData;
  for (let i = 0; i < Y.length; ++i) {
    const newLogicalIndex = ShapeUtil.offsetToIndices(i, newDimsStride);
    const oldLogicalIndex = newLogicalIndex.map((idx, j) => idx + adjustedStarts[j]);
    const oldOffset = ShapeUtil.indicesToOffset(oldLogicalIndex, oldDimsStride);
    Y[i] = X[oldOffset] as number;
  }
  return output;
}

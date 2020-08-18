import {Gather} from '../../../ops/gather';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuGather extends Gather {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = gather(inputs[0], inputs[1], this.axis);
    return [output];
  }
}

export function gather(x: Tensor, indices: Tensor, axis: number): Tensor {
  axis = ShapeUtil.normalizeAxis(axis, x.dims.length);
  const dims = x.dims.slice();
  const newDims = dims.slice();
  const indicesData = indices.data;
  newDims[axis] = indicesData.length;
  const dimsStrides = ShapeUtil.computeStrides(dims);
  const newDimsStrides = ShapeUtil.computeStrides(newDims);
  const output = new Tensor(newDims, x.type);
  const Y = output.numberData;
  const X = x.data;
  for (let i = 0; i < Y.length; ++i) {
    const newLogicalIndex = ShapeUtil.offsetToIndices(i, newDimsStrides);
    const oldLogicalIndex = newLogicalIndex.slice();
    const idx = indicesData[newLogicalIndex[axis]] as number;
    oldLogicalIndex[axis] = idx < 0 ? idx + dims[axis] : idx;
    const oldOffset = ShapeUtil.indicesToOffset(oldLogicalIndex, dimsStrides);
    Y[i] = X[oldOffset] as number;
  }
  // calculate the output dims
  const outputDims = dims.slice(0, axis).concat(indices.dims).concat(dims.slice(axis + 1));
  return new Tensor(outputDims, x.type, undefined, undefined, Y);
}

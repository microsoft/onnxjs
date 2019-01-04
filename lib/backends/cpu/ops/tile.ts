import {Tile} from '../../../ops/tile';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuTile extends Tile {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = tile(inputs[0], inputs[1]);
    return [output];
  }
}

export function tile(x: Tensor, repeats: Tensor): Tensor {
  const dims = x.dims ? x.dims : [x.data.length];
  const rank = dims.length;
  const newDims = new Array(rank);
  for (let i = 0; i < rank; i++) {
    newDims[i] = dims[i] * repeats.numberData[i];
  }
  const dimsStrides = ShapeUtil.computeStrides(dims);
  const newDimsStrides = ShapeUtil.computeStrides(newDims);
  const output = new Tensor(newDims, x.type);
  const Y = output.numberData;
  // TensorTransformUtils.createTypedArray(x.type, ShapeUtil.size(newDims));
  const X = x.data;
  for (let i = 0; i < Y.length; ++i) {
    const newLogicalIndex = ShapeUtil.offsetToIndices(i, newDimsStrides);
    const oldLogicalIndex = new Array(rank);
    for (let j = 0; j < rank; ++j) {
      oldLogicalIndex[j] = newLogicalIndex[j] % x.dims[j];
    }
    const oldOffset = ShapeUtil.indicesToOffset(oldLogicalIndex, dimsStrides);
    Y[i] = X[oldOffset] as number;
  }
  return output;
}

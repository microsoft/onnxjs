import {ArgMax} from '../../../ops/argMax';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, ReduceUtil, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuArgMax extends ArgMax {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = argMax(inputs[0], this.axis, this.keepDims);
    return [output];
  }
}

export function argMax(x: Tensor, axis: number, keepdims: boolean): Tensor {
  const rank = x.dims ? x.dims.length : 1;
  axis = ShapeUtil.normalizeAxis(axis, rank);
  const outputDims = ReduceUtil.calcReduceShape(x.dims, [axis], true);
  const X = x.data;
  const Y = new Int32Array(ShapeUtil.size(outputDims));
  const blockSize = ShapeUtil.sizeFromDimension(x.dims, axis + 1);
  const strides = ShapeUtil.computeStrides(outputDims);
  const inputStrides = ShapeUtil.computeStrides(x.dims);
  const indicesY = new Array(x.dims.length);
  for (let i = 0; i < Y.length; i++) {
    const indices = ShapeUtil.offsetToIndices(i, strides);
    // map index
    BroadcastUtil.fillIndex(indices, x.dims, indicesY);
    const offset = ShapeUtil.indicesToOffset(indicesY, inputStrides);
    let max = x.data[offset];
    let index = 0;
    for (let j = 0; j < x.dims[axis]; ++j) {
      const value = X[offset + j * blockSize];
      if (value > max) {
        max = value;
        index = j;
      }
    }
    Y[i] = index;
  }

  return new Tensor(
      keepdims ? outputDims : ReduceUtil.calcReduceShape(x.dims, [axis], keepdims), 'int32', undefined, undefined, Y);
}

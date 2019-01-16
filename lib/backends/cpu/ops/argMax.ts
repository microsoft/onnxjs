import {ArgMax} from '../../../ops/argMax';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, getActualAxisFromNegativeValue, ReduceUtil, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuArgMax extends ArgMax {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = argMax(inputs[0], this.axis, this.keepDims);
    return [output];
  }
}

export function argMax(x: Tensor, axis: number, keepdims: number): Tensor {
  const rank = x.dims ? x.dims.length : 1;
  axis = getActualAxisFromNegativeValue(axis, rank);
  const outputDims = ReduceUtil.calcReduceShape(x.dims.slice(0), [axis], 1);
  const X = x.data;
  const Y = new Int32Array(ShapeUtil.size(outputDims));
  const blockSize = axis >= x.dims.length ? 1 : ShapeUtil.size(x.dims.slice(axis + 1));
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
      keepdims ? outputDims : ReduceUtil.calcReduceShape(x.dims.slice(0), [axis], keepdims), 'int32', undefined,
      undefined, Y);
}

import {TopK} from '../../../ops/topK';
import {Tensor} from '../../../tensor';
import {assert, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuTopK extends TopK {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = tensorTopK(inputs[0], this.k, this.axis);
    return output;
  }
}

export function tensorTopK(x: Tensor, K: number, axis: number): Tensor[] {
  assert(K === Math.round(K), () => 'K should be an integer');
  K = Math.round(K);

  const rank = x.dims ? x.dims.length : 1;
  axis = ShapeUtil.normalizeAxis(axis, rank);
  const outputDims = [...x.dims];
  outputDims[axis] = K;
  const dimsWithoutAxis = [...x.dims];
  dimsWithoutAxis[axis] = 1;

  const values = new Tensor(outputDims, x.type);
  const indices = new Tensor(outputDims, 'int32');

  const blockSize = ShapeUtil.sizeFromDimension(x.dims, axis + 1);
  const array = Array<number>(x.dims[axis]);
  const elems = ShapeUtil.size(outputDims);
  const xStrides = ShapeUtil.computeStrides(x.dims);
  const withoutAxisStrides = ShapeUtil.computeStrides(dimsWithoutAxis);
  const outputStrides = ShapeUtil.computeStrides(outputDims);
  for (let i = 0; i < elems; ++i) {
    const startIdx = ShapeUtil.offsetToIndices(i, withoutAxisStrides);
    const offset = ShapeUtil.indicesToOffset(startIdx, xStrides);
    for (let j = 0; j < x.dims[axis]; ++j) {
      array[j] = x.numberData[offset + j * blockSize];
    }

    const topk = listTopK(array, K);
    const dstOffset = ShapeUtil.indicesToOffset(startIdx, outputStrides);
    for (let j = 0; j < K; ++j) {
      values.numberData[dstOffset + j * blockSize] = topk[j][0];
      indices.numberData[dstOffset + j * blockSize] = topk[j][1];
    }
  }

  return [values, indices];
}

function listTopK(list: number[], K: number): Array<[number, number]> {
  const enumList: Array<[number, number]> = list.map((el, idx) => [el, idx]);
  let start = 0;
  let end = list.length;

  let iters = 0;
  while (start !== K && iters < list.length) {
    let curr = 0;
    for (let i = 0; i < end; ++i) {
      // compare with lower indices having preference in being topk
      if (enumList[i][0] > enumList[end - 1][0] ||
          (enumList[i][0] === enumList[end - 1][0] && enumList[i][1] <= enumList[end - 1][1])) {
        [enumList[i], enumList[curr]] = [enumList[curr], enumList[i]];
        curr++;
      }
    }

    if (curr === K) {
      break;
    } else if (curr > K) {
      end = curr - 1;
    } else {
      start = curr + 1;
    }
    iters++;
  }

  return enumList.slice(0, K);
}

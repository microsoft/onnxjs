// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Transpose} from '../../../ops/transpose';
import {Tensor} from '../../../tensor';
import {arrayCopyHelper, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuTranspose extends Transpose {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = transpose(inputs[0], this.perm);
    return [output];
  }
}

export function transpose(x: Tensor, perm: number[]): Tensor {
  const inputDims = x.dims;
  const rank = inputDims.length;

  // determine permutation to use
  // if no permutation was specified in the attributes,
  // the default is [rank-1, ..., 0]
  let finalPerm = new Array<number>(rank);
  if (perm.length === rank) {
    finalPerm = perm;
  } else {
    for (let i = 0; i < rank; i++) {
      finalPerm[i] = rank - i - 1;
    }
  }

  const outputDims = new Array<number>(rank);
  const stride = new Array<number>(rank);

  // determine shape of output, as well as stride to be used
  // stride[i] indicates the stride for the input-tensor dimension
  // corresponding to the i-th dimension of the output
  for (let i = 0; i < rank; i++) {
    const inpDim = finalPerm[i];
    outputDims[i] = inputDims[inpDim];
    if (inpDim + 1 < rank) {
      stride[i] = ShapeUtil.sizeFromDimension(inputDims, inpDim + 1);
    } else {
      stride[i] = 1;
    }
  }

  const output = new Tensor(outputDims, x.type);

  const X = x.floatData;
  const Y = output.floatData;

  // partition the permutation into a prefix and the largest suffix such that
  // every axis i in the suffix is mapped to i.
  let numAxesInPrefix = 0;  // number of axes in prefix
  let suffixBlocksize = 1;  // product of dimensions in the suffix
  let prefixBlocksize = 1;  // product of dimensions in the prefix
  let isSuffix = true;
  for (let i = rank - 1; i >= 0; --i) {
    const inpAxis = finalPerm[i];
    if (isSuffix && (inpAxis === i)) {
      suffixBlocksize *= inputDims[inpAxis];
    } else {
      isSuffix = false;
      prefixBlocksize *= inputDims[inpAxis];
      ++numAxesInPrefix;
    }
  }

  if (prefixBlocksize === 1) {
    doTransposeSingleBlock(suffixBlocksize, Y, X);
  } else if (suffixBlocksize === 1) {
    doTransposeEltWise(numAxesInPrefix, outputDims, prefixBlocksize, stride, Y, X);
  } else {
    doTranspose(numAxesInPrefix, outputDims, prefixBlocksize, suffixBlocksize, stride, Y, X);
  }

  return output;
}

// doTranspose: copies source tensor to target, transposing elements.
// the stride vector indicates the transposition.
function doTranspose(
    numAxes: number, targetDims: number[], numBlocks: number, numElementsInBlock: number, stride: number[],
    target: Tensor.FloatType, source: Tensor.FloatType) {
  const targetIndex = new Array<number>(numAxes).fill(0);

  const startSourceIndex = 0;
  let startTargetIndex = 0;

  for (let i = 0; i < numBlocks; ++i) {
    const sizeOffset = ShapeUtil.indicesToOffset(targetIndex, stride, numAxes);
    arrayCopyHelper(target, source, startTargetIndex, startSourceIndex + sizeOffset, numElementsInBlock);

    ShapeUtil.incrementIndex(targetIndex, targetDims, numAxes);
    startTargetIndex += numElementsInBlock;
  }
}

// doTransposeEltWise: specialization of DoTranspose for the
// num_elts_in_block=1 case. copies source tensor to target, transposing
// elements. The stride vector indicates the transposition.
function doTransposeEltWise(
    numAxes: number, targetDims: number[], numBlocks: number, stride: number[], target: Tensor.FloatType,
    source: Tensor.FloatType) {
  const targetIndex = new Array<number>(numAxes).fill(0);

  let startTargetIndex = 0;

  for (let i = 0; i < numBlocks; ++i) {
    const sourceOffset = ShapeUtil.indicesToOffset(targetIndex, stride, numAxes);
    target[startTargetIndex++] = source[sourceOffset];
    ShapeUtil.incrementIndex(targetIndex, targetDims, numAxes);
  }
}

// doTransposeSingleBlock: specialization of DoTranspose for the num_blocks=1
// case. copies source tensor to target, transposing elements. The stride
// vector indicates the transposition.
function doTransposeSingleBlock(numElementsInBlock: number, target: Tensor.FloatType, source: Tensor.FloatType) {
  arrayCopyHelper(target, source, 0, 0, numElementsInBlock);
}

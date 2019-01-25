// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {AveragePool, GlobalAveragePool, GlobalMaxPool, MaxPool} from '../../../ops/pool';
import {Tensor} from '../../../tensor';
import {PoolConvUtil, ShapeUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {PerformanceData} from '../../../wasm-binding-core';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmAveragePool extends AveragePool {
  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    return checkInputTypes(inputs);
  }

  async run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    return averagePool(inputs[0], this.autoPad, this.countIncludePad, this.kernelShape, this.pads, this.strides);
  }
}

export class WasmGlobalAveragePool extends GlobalAveragePool {
  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    return checkInputTypes(inputs);
  }

  async run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    return globalAveragePool(inputs[0]);
  }
}

export class WasmMaxPool extends MaxPool {
  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    return checkInputTypes(inputs);
  }

  async run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    return maxPool(inputs[0], this.autoPad, this.kernelShape, this.pads, this.strides);
  }
}

export class WasmGlobalMaxPool extends GlobalMaxPool {
  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    return checkInputTypes(inputs);
  }

  async run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    return globalMaxPool(inputs[0]);
  }
}

// type check function
function checkInputTypes(inputs: Tensor[]): boolean {
  // currently Wasm backend only supports 'float32' input type
  if (inputs[0].type !== 'float32') {
    return false;
  }

  return true;
}

// functions implementing specific pooling operations
async function averagePool(
    input: Tensor, autoPad: string, countIncludePad: boolean, kernelShape: number[], pads: number[],
    strides: number[]): Promise<Tensor[]> {
  return pool(false, 1, input, autoPad, countIncludePad, kernelShape, pads, strides);
}

async function globalAveragePool(input: Tensor): Promise<Tensor[]> {
  return pool(true, 1, input, 'NOTSET', false, [], [], []);
}

async function maxPool(
    input: Tensor, autoPad: string, kernelShape: number[], pads: number[], strides: number[]): Promise<Tensor[]> {
  return pool(false, 2, input, autoPad, false, kernelShape, pads, strides);
}

async function globalMaxPool(input: Tensor): Promise<Tensor[]> {
  return pool(true, 2, input, 'NOTSET', false, [], [], []);
}

/**
 * Perform pooling operations based on input
 * @param isGlobalOperator If true, perform global pooling.
 * @param poolType 1 if averagepool, 2 for maxpool.
 * @param input The input tensor.
 * @param autoPad DEPRECATED attribute supported for legacy models. Specifies how to implicitly calculate pads in each
 *     dimension. Can take values NOTSET, SAME_UPPER, SAME_LOWER, or VALID
 * @param countIncludePad Whether include pad pixels when calculating values for the edges.
 * @param kernelShape The size of the kernel along each axis.
 * @param pads Padding for the beginning and ending along each axis. `pads` format should be as follow [x1_begin,
 *       x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and
 *       xi_end, the number of pixels added at the end of axis `i`.
 * @param strides Stride along each axis.
 */
async function pool(
    isGlobalOperator: boolean, poolType: number, input: Tensor, autoPad: string, countIncludePad: boolean,
    kernelShape: number[], pads: number[], strides: number[]): Promise<Tensor[]> {
  // determine pool function name in wasm
  let poolFunc = '';
  switch (poolType) {
    case 1:
      poolFunc = '_average_pool_f32';
      break;
    case 2:
      poolFunc = '_max_pool_f32';
      break;
    default:
      throw new Error(`unknown pool type`);
  }
  // adjust the shapes of input attributes
  PoolConvUtil.adjustPoolAttributes(isGlobalOperator, input.dims, kernelShape, strides, pads);

  // calculate output shape based on input attributes
  const outputDims =
      PoolConvUtil.computePoolOutputShape(isGlobalOperator, input.dims, strides, kernelShape, pads, autoPad);

  // create output
  const y = new Tensor(outputDims, input.type);

  // determine number of threads needed to process
  const numThreads = determineNumThreads(input.dims[0], input.dims[1], WasmBinding.workerNumber);

  // no multi-threading
  if (numThreads === 1) {
    WasmBinding.getInstance().ccall(
        poolFunc, [kernelShape.length, 'int32'], [isGlobalOperator, 'bool'], [input.floatData, 'float32ptr'],
        [input.dims, 'int32ptr'], [y.floatData, 'float32ptr', 'out'], [y.dims, 'int32ptr'], [kernelShape, 'int32ptr'],
        [pads, 'int32ptr'], [strides, 'int32ptr'], [countIncludePad, 'bool']);
  }

  // multi-threaded using web-workers
  else {
    // data pre-processing
    const xDimsSp = input.dims.slice(0);
    xDimsSp[1] = Math.floor(input.dims[1] / numThreads);
    const xSizeSp = ShapeUtil.size(xDimsSp);

    const xDimsFinal = input.dims.slice(0);
    xDimsFinal[1] = input.dims[1] - (numThreads - 1) * xDimsSp[1];

    const yDimsSp = outputDims.slice(0);
    yDimsSp[1] = xDimsSp[1];
    const ySizeSp = ShapeUtil.size(yDimsSp);

    const yDimsFinal = outputDims.slice(0);
    yDimsFinal[1] = xDimsFinal[1];

    const workerTasks = new Array<Promise<PerformanceData>>(numThreads - 1);

    const X = input.floatData;
    const Y = y.floatData;

    // function calls
    for (let i = 0; i < numThreads; ++i) {
      if (i !== numThreads - 1) {
        workerTasks[i] = WasmBinding.getInstance().ccallRemote(
            i, poolFunc, [kernelShape.length, 'int32'], [isGlobalOperator, 'bool'],
            [X.subarray(i * xSizeSp, (i + 1) * xSizeSp), 'float32ptr'], [xDimsSp, 'int32ptr'],
            [Y.subarray(i * ySizeSp, (i + 1) * ySizeSp), 'float32ptr', 'out'], [yDimsSp, 'int32ptr'],
            [kernelShape, 'int32ptr'], [pads, 'int32ptr'], [strides, 'int32ptr'], [countIncludePad, 'bool']);
      } else {
        WasmBinding.getInstance().ccall(
            poolFunc, [kernelShape.length, 'int32'], [isGlobalOperator, 'bool'],
            [X.subarray((numThreads - 1) * xSizeSp), 'float32ptr'], [xDimsFinal, 'int32ptr'],
            [Y.subarray((numThreads - 1) * ySizeSp), 'float32ptr', 'out'], [yDimsFinal, 'int32ptr'],
            [kernelShape, 'int32ptr'], [pads, 'int32ptr'], [strides, 'int32ptr'], [countIncludePad, 'bool']);
      }
    }

    await Promise.all(workerTasks);
  }

  return [y];
}

// this function will determine the number of threads
// the strategy to parallelize is to parallelize on number of data channels
function determineNumThreads(batchSize: number, numChannels: number, numWebWorkers: number): number {
  // single threaded if:
  // 1) batch size is not 1 (data splitting logic across threads is specific to batch size being 1)
  // 2) if number of channels is 1
  // 3) number of web workers is 0
  if (batchSize !== 1 || numChannels === 1 || numWebWorkers <= 0) {
    return 1;
  }

  // multi-threaded:
  // determine number of threads
  return Math.min(numChannels, numWebWorkers + 1);
}

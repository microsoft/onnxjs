// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {PerformanceData} from '../../../wasm-binding-core';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmConv extends Conv {
  async run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];
    const w = inputs[1];
    const b = inputs.length === 3 ? inputs[2] : undefined;

    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      const wDims = inputs[1].dims;
      for (let i = 2; i < wDims.length; ++i) {
        this.kernelShape.push(wDims[i]);
      }
    }

    // create output Tensor after determining output size (after adjusting pads based on 'autoPad' attribute)
    const outputDims = PoolConvUtil.computeConvOutputShape(
        x.dims, w.dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);
    const y = new Tensor(outputDims, x.type);

    // determine number of threads needed to process
    const numThreads = determineNumThreads(x.dims[0], this.group, w.dims[0], WasmBinding.workerNumber);

    // no multi-threading
    if (numThreads === 1) {
      WasmBinding.getInstance().ccall(
          '_conv_f32', [x.floatData, 'float32ptr'], [x.dims, 'int32ptr'], [w.floatData, 'float32ptr'],
          [w.dims, 'int32ptr'], [y.floatData, 'float32ptr', 'out'], [y.dims, 'int32ptr'],
          [b ? b.floatData : null, 'float32ptr'], [this.dilations, 'int32ptr'], [this.group, 'int32'],
          [this.pads, 'int32ptr'], [this.strides, 'int32ptr']);
      return [y];
    }

    // multi-threaded using web-workers
    else {
      // data pre-processing
      const wDimsSp = w.dims.slice(0);
      wDimsSp[0] = Math.floor(w.dims[0] / numThreads);
      const wSizeSp = wDimsSp[0] * wDimsSp[1] * wDimsSp[2] * wDimsSp[3];

      const wDimsFinal = w.dims.slice(0);
      wDimsFinal[0] = w.dims[0] - (numThreads - 1) * wDimsSp[0];

      const yDimsSp = [1, wDimsSp[0], outputDims[2], outputDims[3]];
      const ySizeSp = wDimsSp[0] * outputDims[2] * outputDims[3];

      const yDimsFinal = [1, wDimsFinal[0], outputDims[2], outputDims[3]];

      const wArray = new Array<Float32Array>(numThreads);
      const yArray = new Array<Float32Array>(numThreads);
      const bArray = new Array<Float32Array>(numThreads);
      const workerTasks = new Array<Promise<PerformanceData>>(numThreads - 1);

      // function calls
      for (let i = 0; i < numThreads; ++i) {
        if (i !== numThreads - 1) {
          wArray[i] = w.floatData.subarray(i * wSizeSp, (i + 1) * wSizeSp) as Float32Array;
          yArray[i] = y.floatData.subarray(i * ySizeSp, (i + 1) * ySizeSp) as Float32Array;
          if (b) {
            bArray[i] = b.floatData.subarray(i * wDimsSp[0], (i + 1) * wDimsSp[0]) as Float32Array;
          }
          workerTasks[i] = WasmBinding.getInstance().ccallRemote(
              i, '_conv_f32', [x.floatData, 'float32ptr'], [x.dims, 'int32ptr'], [wArray[i], 'float32ptr'],
              [wDimsSp, 'int32ptr'], [yArray[i], 'float32ptr', 'out'], [yDimsSp, 'int32ptr'],
              [bArray.length > 0 ? bArray[i] : null, 'float32ptr'], [this.dilations, 'int32ptr'], [this.group, 'int32'],
              [this.pads, 'int32ptr'], [this.strides, 'int32ptr']);
        } else {
          wArray[i] = w.floatData.subarray(i * wSizeSp) as Float32Array;
          yArray[i] = y.floatData.subarray(i * ySizeSp) as Float32Array;
          if (b) {
            bArray[i] = b.floatData.subarray(i * wDimsSp[0]) as Float32Array;
          }
          WasmBinding.getInstance().ccall(
              '_conv_f32', [x.floatData, 'float32ptr'], [x.dims, 'int32ptr'], [wArray[i], 'float32ptr'],
              [wDimsFinal, 'int32ptr'], [yArray[i], 'float32ptr', 'out'], [yDimsFinal, 'int32ptr'],
              [bArray.length > 0 ? bArray[i] : null, 'float32ptr'], [this.dilations, 'int32ptr'], [this.group, 'int32'],
              [this.pads, 'int32ptr'], [this.strides, 'int32ptr']);
        }
      }

      await Promise.all(workerTasks);
      return [y];
    }
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32' || inputs[1].type !== 'float32') {
      return false;
    }

    if (inputs.length === 3 && inputs[2].type !== 'float32') {
      return false;
    }

    return true;
  }
}

// This function will determine the number of threads
// The strategy to parallelize is to parallelize on number of filter maps in the kernel
// (i.e.) number of output channels
function determineNumThreads(batchSize: number, group: number, numFilterMaps: number, numWebWorkers: number): number {
  // single threaded if:
  // 1) batch size is not 1 (data splitting logic across threads is specific to batch size being 1)
  // 2) multi-threading not supported yet for mulitple groups
  // 3) if number of filter maps is 1
  // 4) number of web workers is 0
  if (batchSize !== 1 || group !== 1 || numFilterMaps === 1 || numWebWorkers <= 0) {
    return 1;
  }

  // multi-threaded:
  // determine number of threads
  return Math.min(numFilterMaps, numWebWorkers + 1);
}

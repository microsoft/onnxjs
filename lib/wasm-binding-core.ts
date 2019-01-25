// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

declare interface OnnxWasmBindingJs {
  (self: OnnxWasmBindingJs): Promise<void>;

  _malloc: (ptr: number) => number;
  _free: (ptr: number) => void;

  buffer: ArrayBuffer;

  HEAP8: Int8Array;
  HEAP16: Int16Array;
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
  HEAPU16: Uint16Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;
}

// an interface to define argument handling
export interface WasmCallArgumentTypeMap {
  bool: boolean;
  int32: number;
  float32: number;
  float64: number;
  boolptr: ReadonlyArray<boolean>|Uint8Array;
  int32ptr: ReadonlyArray<number>|Uint32Array|Int32Array|null;
  float32ptr: ReadonlyArray<number>|Int32Array|Uint32Array|Float32Array|null;
  float64ptr: ReadonlyArray<number>|Float64Array|null;
}

// some types related to arguments
export type WasmCallArgumentType = keyof WasmCallArgumentTypeMap;
export type WasmCallArgumentDataType = WasmCallArgumentTypeMap[WasmCallArgumentType];

export type WasmCallArgumentPass = 'in'|'out'|'inout';

export type WasmCallArgument = [WasmCallArgumentDataType, WasmCallArgumentType, WasmCallArgumentPass?];

// an interface to deal with performance data related to Wasm function calls
export interface PerformanceData {
  startTime?: number;
  endTime?: number;

  startTimeFunc?: number;
  endTimeFunc?: number;
}

// some global parameters to deal with wasm binding initialization
let binding: OnnxWasmBindingJs|undefined;
let initialized = false;
let initializing = false;

/**
 * initialize the WASM instance.
 *
 * this function should be called before any other calls to the WASM binding.
 */
export function init(): Promise<void> {
  if (initialized) {
    return Promise.resolve();
  }
  if (initializing) {
    throw new Error(`multiple calls to 'init()' detected.`);
  }

  initializing = true;

  return new Promise<void>((resolve, reject) => {
    // tslint:disable-next-line:no-require-imports
    binding = require('../dist/onnx-wasm') as OnnxWasmBindingJs;
    binding(binding).then(
        () => {
          // resolve init() promise
          resolve();
          initializing = false;
          initialized = true;
        },
        err => {
          initializing = false;
          reject(err);
        });
  });
}

// class that deals with Wasm data interop and method calling
export class WasmBinding {
  protected ptr8: number;
  protected numBytesAllocated: number;
  protected constructor() {
    this.ptr8 = 0;
    this.numBytesAllocated = 0;
  }

  /**
   * ccall in current thread
   * @param functionName
   * @param params
   */
  ccall(functionName: string, ...params: WasmCallArgument[]): PerformanceData {
    if (!initialized) {
      throw new Error(`wasm not initialized. please ensure 'init()' is called.`);
    }
    const startTime = now();

    const offset: number[] = [];
    const size = WasmBinding.calculateOffsets(offset, params);
    if (size > this.numBytesAllocated) {
      this.expandMemory(size);
    }
    WasmBinding.ccallSerialize(binding!.HEAPU8.subarray(this.ptr8, this.ptr8 + size), offset, params);

    const startTimeFunc = now();
    this.func(functionName, this.ptr8);
    const endTimeFunc = now();

    WasmBinding.ccallDeserialize(binding!.HEAPU8.subarray(this.ptr8, this.ptr8 + size), offset, params);
    const endTime = now();

    return {startTime, endTime, startTimeFunc, endTimeFunc};
  }

  // raw ccall method  without invoking ccallSerialize() and ccallDeserialize()
  // user by ccallRemote() in the web-worker
  ccallRaw(functionName: string, data: Uint8Array): PerformanceData {
    if (!initialized) {
      throw new Error(`wasm not initialized. please ensure 'init()' is called.`);
    }
    const startTime = now();

    const size = data.byteLength;
    if (size > this.numBytesAllocated) {
      this.expandMemory(size);
    }

    // copy input memory (data) to WASM heap
    binding!.HEAPU8.subarray(this.ptr8, this.ptr8 + size).set(data);

    const startTimeFunc = now();
    this.func(functionName, this.ptr8);
    const endTimeFunc = now();

    // copy Wasm heap to output memory (data)
    data.set(binding!.HEAPU8.subarray(this.ptr8, this.ptr8 + size));
    const endTime = now();

    return {startTime, endTime, startTimeFunc, endTimeFunc};
  }

  protected func(functionName: string, ptr8: number): void {
    // tslint:disable-next-line:no-any
    const func = (binding as any)[functionName] as (data: number) => void;
    func(ptr8);
  }

  static calculateOffsets(offset: number[], params: WasmCallArgument[]): number {
    // calculate size and offset
    let size = 4 + 4 * params.length;
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = param[0];
      const paramType = param[1];
      const paramPass = param[2];

      let len = 0;
      switch (paramType) {
        case 'bool':
        case 'int32':
        case 'float32':
          len = 4;
          break;
        case 'float64':
          len = 8;
          break;
        case 'boolptr':
          if (!paramData) {
            // deal with nullptr
            offset.push(0);
            continue;
          } else if (Array.isArray(paramData) || ArrayBuffer.isView(paramData)) {
            len = 4 * Math.ceil(paramData.length / 4);
          } else {
            throw new Error(`boolptr requires boolean array or Uint8Array`);
          }
          break;
        case 'int32ptr':
        case 'float32ptr':
          if (!paramData) {
            // deal with nullptr
            offset.push(0);
            continue;
          } else if (Array.isArray(paramData)) {
            if (paramPass === 'inout' || paramPass === 'out') {
              throw new TypeError(`inout/out parameters must be ArrayBufferView for ptr types.`);
            }
            len = paramData.length * 4;
          } else if (ArrayBuffer.isView(paramData)) {
            len = paramData.byteLength;
          } else {
            throw new TypeError(`unsupported data type in 'ccall()'`);
          }
          break;
        default:
          throw new Error(`not supported parameter type: ${paramType}`);
      }

      offset.push(size);
      size += len;
    }

    return size;
  }

  // tranfer data parameters (in/inout) to emscripten heap for ccall()
  static ccallSerialize(heapU8: Uint8Array, offset: number[], params: WasmCallArgument[]) {
    const heap32 = new Int32Array(heapU8.buffer, heapU8.byteOffset);
    const heapU32 = new Uint32Array(heapU8.buffer, heapU8.byteOffset);
    const heapF32 = new Float32Array(heapU8.buffer, heapU8.byteOffset);

    heapU32[0] = params.length;

    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = param[0];
      const paramType = param[1];
      const paramPass = param[2];
      const offset8 = offset[i];
      const offset32 = offset8 >> 2;

      heapU32[i + 1] = offset8;

      if (paramPass === 'out' || offset8 === 0) {
        continue;
      }

      switch (paramType) {
        case 'bool':
          heapU8[offset8] = (paramData as WasmCallArgumentTypeMap['bool']) === true ? 1 : 0;
          break;
        case 'int32':
          heap32[offset32] = paramData as number;
          break;
        case 'float32':
          heapF32[offset32] = paramData as number;
          break;
        case 'boolptr':
          const boolArray = paramData as WasmCallArgumentTypeMap['boolptr'];
          // This will work for both Uint8Array as well as ReadonlyArray<boolean>
          heapU8.subarray(offset8, offset8 + boolArray.length).set(paramData as Uint8Array);
          break;
        case 'int32ptr':
          const int32Array = (paramData as WasmCallArgumentTypeMap['int32ptr'])!;
          heap32.subarray(offset32, offset32 + int32Array.length).set(int32Array);
          break;
        case 'float32ptr':
          const float32Array = (paramData as WasmCallArgumentTypeMap['float32ptr'])!;
          heapF32.subarray(offset32, offset32 + float32Array.length).set(float32Array);
          break;
        default:
          throw new Error(`not supported parameter type: ${paramType}`);
      }
    }
  }

  // retrieve data parameters (in/inout) from emscripten heap after ccall()
  static ccallDeserialize(buffer: Uint8Array, offset: number[], params: WasmCallArgument[]) {
    const heapF32 = new Float32Array(buffer.buffer, buffer.byteOffset);
    const heapU8 = new Uint8Array(buffer.buffer, buffer.byteOffset);

    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = param[0];
      const paramType = param[1];
      const paramPass = param[2];
      const offset8 = offset[i];
      // const offset16 = offset8 >> 1;
      const offset32 = offset8 >> 2;
      // const offset64 = offset8 >> 3;

      if (paramPass !== 'out' && paramPass !== 'inout') {
        continue;
      }

      switch (paramType) {
        case 'float32ptr':
          const float32Array = (paramData as Float32Array);
          float32Array.set(heapF32.subarray(offset32, offset32 + float32Array.length));
          break;
        case 'boolptr':
          const boolArray = (paramData as Uint8Array);
          boolArray.set(heapU8.subarray(offset8, offset8 + boolArray.length));
          break;
        default:
          throw new Error(`not supported parameter type: ${paramType}`);
      }
    }
  }

  // function for defining memory allocation strategy
  private expandMemory(minBytesRequired: number) {
    // free already held memory if applicable
    if (this.ptr8 !== 0) {
      binding!._free(this.ptr8);
    }
    // current simplistic strategy is to allocate 2 times the minimum bytes requested
    this.numBytesAllocated = 2 * minBytesRequired;
    this.ptr8 = binding!._malloc(this.numBytesAllocated);
    if (this.ptr8 === 0) {
      throw new Error('Unable to allocate requested amount of memory. Failing.');
    }
  }

  dispose(): void {
    if (!initialized) {
      throw new Error(`wasm not initialized. please ensure 'init()' is called.`);
    }
    if (this.ptr8 !== 0) {
      binding!._free(this.ptr8);
    }
  }
}

/**
 * returns a number to represent the current timestamp in a resolution as high as possible.
 */
export const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : Date.now;

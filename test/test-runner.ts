// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {expect} from 'chai';
import {readFile} from 'fs';
import {onnx as onnxProto} from 'onnx-proto';
import {extname} from 'path';
import {inspect, promisify} from 'util';

import * as api from '../lib/api';
import {fromInternalTensor, toInternalTensor} from '../lib/api/tensor-impl-utils';
import {Attribute} from '../lib/attribute';
import {Backend, InferenceHandler, SessionHandler} from '../lib/backend';
import {TextureData} from '../lib/backends/webgl/texture-data';
import {Logger, Profiler} from '../lib/instrument';
import {Operator} from '../lib/operators';
import {Tensor} from '../lib/tensor';

import {Test} from './test-types';

// the threshold that used to compare 2 float numbers. See above for TensorResultValidator.floatEqual().
const CPU_THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
const CPU_THRESHOLD_RELATIVE_ERROR = 1.000001;
const WEBGL_THRESHOLD_ABSOLUTE_ERROR = 1.0e-3;
const WEBGL_THRESHOLD_RELATIVE_ERROR = 1.00001;
const WASM_THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
const WASM_THRESHOLD_RELATIVE_ERROR = 1.000001;
const ONNXRUNTIME_THRESHOLD_ABSOLUTE_ERROR = 1.0e-3;
const ONNXRUNTIME_THRESHOLD_RELATIVE_ERROR = 1.00001;

/**
 * returns a number to represent the current timestamp in a resolution as high as possible.
 */
const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : Date.now;

/**
 * a ModelTestContext object contains all states in a ModelTest
 */
export class ModelTestContext {
  private constructor(
      readonly session: api.InferenceSession,
      readonly backend: string,
      readonly perfData: ModelTestContext.ModelTestPerfData,
      private readonly profile: boolean,
  ) {}

  /**
   * dump the current performance data
   */
  private logPerfData() {
    const data = this.perfData;
    Logger.verbose('TestRunner.Perf', `***Perf Data Start`);
    Logger.verbose('TestRunner.Perf', ` * Init          : ${data.init}`);
    Logger.verbose('TestRunner.Perf', ` * Running times : ${data.count}`);
    Logger.verbose('TestRunner.Perf', ` * FirstRun      : ${data.firstRun.toFixed(2)}`);
    const runs = data.runs;
    if (runs.length > 0) {
      Logger.verbose('TestRunner.Perf', ` * Runs          : ${runs.map(r => r.toFixed(2)).join(', ')}`);

      if (runs.length > 1) {
        const avg = runs.reduce((prev, current) => prev + current) / runs.length;
        Logger.verbose('TestRunner.Perf', ` * Runs Avg      : ${avg.toFixed(2)}`);
        const variance = runs.reduce((prev, current) => prev + (current - avg) * (current - avg));
        const sd = Math.sqrt(variance / (runs.length - 1));
        Logger.verbose('TestRunner.Perf', ` * Runs SD       : ${sd.toFixed(2)}`);
      }
    }
    Logger.verbose('TestRunner.Perf', `***Perf Data End`);
  }

  release() {
    if (this.profile) {
      this.session.endProfiling();
    }
    this.logPerfData();
  }

  /**
   * create a ModelTestContext object that used in every test cases in the given ModelTest.
   */
  static async create(modelTest: Test.ModelTest, profile: boolean): Promise<ModelTestContext> {
    if (this.initializing) {
      throw new Error(`cannot create a ModelTestContext object when the previous creation is not done`);
    }

    try {
      this.initializing = true;

      const initStart = now();
      const session = await initializeSession(modelTest.modelUrl, modelTest.backend!, profile);
      const initEnd = now();

      for (const testCase of modelTest.cases) {
        await loadTensors(testCase);
      }

      return new ModelTestContext(
          session,
          modelTest.backend!,
          {init: initEnd - initStart, firstRun: -1, runs: [], count: 0},
          profile,
      );
    } finally {
      this.initializing = false;
    }
  }

  private static initializing = false;
}

export declare namespace ModelTestContext {
  export interface ModelTestPerfData {
    init: number;
    firstRun: number;
    runs: number[];
    count: number;
  }
}

async function loadTensors(testCase: Test.ModelTestCase) {
  const inputs: Test.NamedTensor[] = [];
  const outputs: Test.NamedTensor[] = [];
  let dataFileType: 'none'|'pb'|'npy' = 'none';

  for (const dataFile of testCase.dataFiles) {
    const ext = extname(dataFile);
    if (ext.toLowerCase() === '.pb' || ext.toLowerCase() === '.tpb') {
      if (dataFileType === 'none') {
        dataFileType = 'pb';
      }
      if (dataFileType !== 'pb') {
        throw new Error(`cannot load data from test case "${testCase.name}", multiple types of files detected`);
      }

      const t = ext.toLowerCase() === '.pb' ? await loadTensorProto(dataFile) :  // onnx.TensorProto
          await loadMlProto(dataFile);                                           // (TBD)

      const dataFileBasename = dataFile.split(/[/\\]/).pop()!;

      if (dataFileBasename.indexOf('input') !== -1) {
        inputs.push(t);
      } else if (dataFileBasename.indexOf('output') !== -1) {
        outputs.push(t);
      }
    } else {
      throw new Error(`${ext} file is not supported now`);
    }
  }

  testCase.inputs = inputs;
  testCase.outputs = outputs;
}

async function loadTensorProto(uri: string): Promise<Test.NamedTensor> {
  const buf = await loadFile(uri);
  const tensorProto = onnxProto.TensorProto.decode(Buffer.from(buf));
  const tensor = Tensor.fromProto(tensorProto);
  // add property 'name' to the tensor object.
  const namedTensor = fromInternalTensor(tensor) as unknown as Test.NamedTensor;
  namedTensor.name = tensorProto.name;
  return namedTensor;
}

async function loadFile(uri: string): Promise<Uint8Array|ArrayBuffer> {
  if (typeof fetch === 'undefined') {
    // node
    return promisify(readFile)(uri);
  } else {
    // browser
    const response = await fetch(uri);
    return response.arrayBuffer();
  }
}

function loadMlProto(uri: string): Promise<Test.NamedTensor> {
  return Promise.reject('not supported');
}

async function initializeSession(modelFilePath: string, backendHint: string, profile: boolean) {
  Logger.verbose('TestRunner', `Start to load model from file: ${modelFilePath}`);

  const profilerConfig = profile ? {maxNumberEvents: 65536} : undefined;
  const sessionConfig: api.InferenceSession.Config = {backendHint, profiler: profilerConfig};
  const session = new onnx.InferenceSession(sessionConfig);

  if (profile) {
    session.startProfiling();
  }

  try {
    await session.loadModel(modelFilePath);
  } catch (e) {
    Logger.error('TestRunner', `Failed to load model from file: ${modelFilePath}. Error: ${inspect(e)}`);
    throw e;
  }

  Logger.verbose('TestRunner', `Finished loading model from file: ${modelFilePath}`);

  return session;
}

/**
 * run a single model test case. the inputs/outputs tensors should already been prepared.
 */
export async function runModelTestSet(context: ModelTestContext, testCase: Test.ModelTestCase): Promise<void> {
  Logger.verbose('TestRunner', `Start to run test data from folder: ${testCase.name}`);
  const validator = new TensorResultValidator(context.backend);
  try {
    const start = now();
    const outputs = await context.session.run(testCase.inputs!);
    const end = now();
    if (context.perfData.count === 0) {
      context.perfData.firstRun = end - start;
    } else {
      context.perfData.runs.push(end - start);
    }
    context.perfData.count++;

    Logger.verbose('TestRunner', `Finished running model from file: ${testCase.name}`);
    Logger.verbose('TestRunner', ` Stats:`);
    Logger.verbose('TestRunner', `  Input(s): ${testCase.inputs!.length}`);
    testCase.inputs!.forEach(i => {
      Logger.verbose('TestRunner', `   '${i.name}': ${i.type}[${i.dims.join(',')}]`);
    });
    Logger.verbose('TestRunner', `  Output(s): ${outputs.size}`);
    outputs.forEach((t, name) => {
      Logger.verbose('TestRunner', `   '${name}': ${t.type}[${t.dims.join(',')}]`);
    });

    validator.checkNamedTensorResult(outputs, testCase.outputs!);

    Logger.verbose('TestRunner', `  Result: PASS`);
  } catch (e) {
    Logger.error('TestRunner', `  Result: FAILED`);
    Logger.error('TestRunner', `Failed to run test data from folder: ${testCase.name}. Error: ${inspect(e)}`);
    throw e;
  }
}

/**
 * a OpTestContext object contains all states in a OpTest
 */
export class OpTestContext {
  static profiler = Profiler.create();

  readonly backendHint: string;
  sessionHandler: SessionHandler;
  inferenceHandler: InferenceHandler;

  constructor(protected opTest: Test.OperatorTest) {
    this.backendHint = opTest.backend === 'webgl' ? 'webgl' : 'cpu';
  }
  createOperator(): Operator {
    return initializeOperator(this.sessionHandler, this.opTest.operator, this.opTest.attributes);
  }

  dispose() {
    this.inferenceHandler.dispose();
    this.sessionHandler.dispose();
  }

  async init(): Promise<void> {
    const backend = await Backend(this.backendHint);
    this.sessionHandler = backend.createSessionHandler({profiler: OpTestContext.profiler});
    this.inferenceHandler = this.sessionHandler.createInferenceHandler();
  }
}

function initializeOperator(
    sessionHandler: SessionHandler, opType: string, attributeValues: ReadonlyArray<Test.AttributeValue>): Operator {
  const attributes = new Attribute(undefined);
  attributeValues.forEach(value => attributes.set(value.name, value.type, value.data));
  return sessionHandler.resolve({name: '', opType, inputs: [], outputs: [], attributes}, [{domain: '', version: 7}]);
}

/**
 * run a single operator test case.
 */
export async function runOpTest(testcase: Test.OperatorTestCase, context: OpTestContext): Promise<void> {
  await runOpTestcase(
      context.inferenceHandler, context.createOperator(), testcase, new TensorResultValidator(context.backendHint));
}

async function runOpTestcase<T extends TextureData|Tensor>(
    inferenceHandler: InferenceHandler, operator: Operator, testcase: Test.OperatorTestCase,
    validator: TensorResultValidator): Promise<void> {
  testcase.inputs.forEach((input, i) => {
    Logger.verbose('TestOpRunner', `   Input '${i}': ${input.type}[${input.dims.join(',')}]`);
  });
  const inputTensors = testcase.inputs.map(input => createTensor(input.dims, input.type, input.data));

  let results = operator.run(inferenceHandler, inputTensors);
  if ('then' in results) {
    results = await results;
  }

  results.forEach((output, i) => {
    Logger.verbose('TestOpRunner', `  Result'${i}': ${output.type}[${output.dims.join(',')}]`);
  });
  const expectedTensors = testcase.outputs.map(output => createTensor(output.dims, output.type, output.data));
  validator.checkTensorResult(results, expectedTensors);
}

function createTensor(dims: number[], type: Tensor.DataType, data: number[]): Tensor {
  const tensor = new Tensor(dims, type);
  for (let i = 0; i < data.length; ++i) {
    tensor.data[i] = data[i];
  }
  return tensor;
}

export class TensorResultValidator {
  private readonly absoluteThreshold: number;
  private readonly relativeThreshold: number;
  constructor(backend: string) {
    if (backend === 'cpu') {
      this.absoluteThreshold = CPU_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = CPU_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'webgl') {
      this.absoluteThreshold = WEBGL_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = WEBGL_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'wasm') {
      this.absoluteThreshold = WASM_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = WASM_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'onnxruntime') {
      this.absoluteThreshold = ONNXRUNTIME_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = ONNXRUNTIME_THRESHOLD_RELATIVE_ERROR;
    } else {
      throw new Error(`backend not supported: ${backend}`);
    }
  }

  checkTensorResult(actual: Tensor[], expected: Tensor[]): void {
    // check output size
    expect(actual.length, 'size of output tensors').to.equal(expected.length);

    // compare output one-by-one
    for (let i = 0; i < actual.length; ++i) {
      const match = this.areEqual(actual[i], expected[i]);
      if (!match) {
        Logger.error(
            'TestRunner',
            `Tensor mismatch: \nACTUAL: type=${actual[i].type}; dims=[${actual[i].dims}]; data=[${
                actual[i].data}]\nEXPECT: type=${expected[i].type}; dims=[${expected[i].dims}]; data=[${
                expected[i].data}]`);
      }
      expect(match, 'tensor data should match').to.be.true;
    }
  }

  checkApiTensorResult(actual: api.Tensor[], expected: api.Tensor[]): void {
    this.checkTensorResult(actual.map(toInternalTensor), expected.map(toInternalTensor));
  }

  checkNamedTensorResult(actual: ReadonlyMap<string, api.Tensor>, expected: Test.NamedTensor[]): void {
    // check output size
    expect(actual.size, 'size of output tensors').to.equal(expected.length);

    // check output mapping
    for (const expectedOneOutput of expected) {
      expect(actual, 'keys of output tensors').to.contain.keys(expectedOneOutput.name);
    }

    this.checkApiTensorResult(expected.map(i => actual.get(i.name)!), expected);
  }

  // This function check whether 2 tensors should be considered as 'match' or not
  areEqual(actual: Tensor, expected: Tensor): boolean {
    if (!actual || !expected) {
      return false;
    }
    if (!actual.dims || !expected.dims) {
      return false;
    }

    const actualDims = actual.dims;
    const actualType = actual.type;
    const expectedDims = expected.dims;
    const expectedType = expected.type;

    if (actualType !== expectedType) {
      return false;
    }
    if (actualDims.length !== expectedDims.length) {
      return false;
    }

    for (let i = 0; i < actualDims.length; i++) {
      if (actualDims[i] !== expectedDims[i]) {
        return false;
      }
    }

    switch (actualType) {
      case 'string':
        return this.strictEqual(actual.stringData, expected.stringData);

      case 'float32':
      case 'float64':
        return this.floatEqual(
            actual.numberData as number[] | Float32Array | Float64Array,
            expected.numberData as number[] | Float32Array | Float64Array);

      case 'uint8':
      case 'int8':
      case 'uint16':
      case 'int16':
      case 'int32':
      case 'uint32':
      case 'bool':
        return this.integerEqual(
            actual.numberData as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array |
                Int32Array,
            expected.numberData as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array |
                Int32Array);

      default:
        throw new Error('type not implemented or not supported');
    }
  }
  strictEqual<T>(actual: T, expected: T): boolean {
    try {
      expect(actual).to.deep.equal(expected);
      return true;
    } catch {
      return false;
    }
  }
  floatEqual(actual: number[]|Float32Array|Float64Array, expected: number[]|Float32Array|Float64Array): boolean {
    if (actual.length !== expected.length) {
      return false;
    }

    for (let i = actual.length - 1; i >= 0; i--) {
      const a = actual[i], b = expected[i];

      // check for NaN
      //
      if (Number.isNaN(a) && Number.isNaN(b)) {
        continue;  // 2 numbers are NaN, treat as equal
      }
      if (Number.isNaN(a) || Number.isNaN(b)) {
        Logger.error('Validator', `a or b isNan -- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
        return false;  // one is NaN and the other is not
      }

      // sign should be same if not equals to zero
      //
      if ((a > 0 && b < 0) || (a < 0 && b > 0)) {
        Logger.error('Validator', `signs are different -- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
        return false;  // sign is different
      }

      // Comparing 2 float numbers: (Suppose a >= b)
      //
      // if ( a - b < ABSOLUTE_ERROR || 1.0 < a / b < RELATIVE_ERROR)
      //   test pass
      // else
      //   test fail
      // endif
      //
      if (Math.abs(actual[i] - expected[i]) < this.absoluteThreshold) {
        continue;  // absolute error check pass
      }
      if (a !== 0 && b !== 0 && a / b < this.relativeThreshold && b / a < this.relativeThreshold) {
        continue;  // relative error check pass
      }

      // if code goes here, it means both (abs/rel) check failed.
      Logger.error('Validator', `abs/rel check failed-- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
      return false;
    }

    return true;
  }
  integerEqual(
      actual: number[]|Uint8Array|Int8Array|Uint16Array|Int16Array|Uint32Array|Int32Array,
      expected: number[]|Uint8Array|Int8Array|Uint16Array|Int16Array|Uint32Array|Int32Array): boolean {
    if (actual.length !== expected.length) {
      return false;
    }

    for (let i = actual.length - 1; i >= 0; i--) {
      if (actual[i] !== expected[i]) {
        return false;
      }
    }

    return true;
  }
}

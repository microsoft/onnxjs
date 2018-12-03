// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../lib/attribute';
import {Logger} from '../lib/instrument';
import {Tensor} from '../lib/tensor';

export declare namespace Test {
  export interface NamedTensor extends Tensor {
    name: string;
  }

  /**
   * This interface represent a value of Attribute. Should only be used in testing.
   */
  export interface AttributeValue {
    name: string;
    data: Attribute.DataTypeMap[Attribute.DataType];
    type: Attribute.DataType;
  }

  /**
   * This interface represent a value of Tensor. Should only be used in testing.
   */
  export interface TensorValue {
    data: number[];
    dims: number[];
    type: Tensor.DataType;
  }

  export interface ModelTestCase {
    name: string;
    dataFiles: ReadonlyArray<string>;
    inputs?: NamedTensor[];   // value should be populated at runtime
    outputs?: NamedTensor[];  // value should be populated at runtime
  }

  export interface ModelTest {
    name: string;
    modelUrl: string;
    backend?: string;  // value should be populated at build time
    cases: ReadonlyArray<ModelTestCase>;
  }

  export interface ModelTestGroup {
    name: string;
    tests: ReadonlyArray<ModelTest>;
  }

  export interface OperatorTestCase {
    name: string;
    inputs: ReadonlyArray<TensorValue>;
    outputs: ReadonlyArray<TensorValue>;
  }

  export interface OperatorTest {
    name: string;
    operator: string;
    backend?: string;  // value should be populated at build time
    attributes: ReadonlyArray<AttributeValue>;
    cases: ReadonlyArray<OperatorTestCase>;
  }

  export interface OperatorTestGroup {
    name: string;
    tests: ReadonlyArray<OperatorTest>;
  }

  /**
   * The data schema of a whitelist file.
   * A whitelist should only be applied when running suite test cases (suite0, suite1)
   */
  export interface WhiteList {
    [backend: string]: {[group: string]: ReadonlyArray<string>;};
  }

  /**
   * The data schema of a test config.
   */
  export interface Config {
    unittest: boolean;
    op: ReadonlyArray<OperatorTestGroup>;
    model: ReadonlyArray<ModelTestGroup>;

    log: ReadonlyArray<{category: string, config: Logger.Config}>;
    profile: boolean;
  }
}

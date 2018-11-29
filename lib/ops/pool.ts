// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

class PoolBase {
  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
      return false;
    }

    return true;
  }

  protected autoPad: string;
  protected countIncludePad: boolean;
  protected kernelShape: number[];
  protected strides: number[];
  protected pads: number[];
}

export abstract class AveragePool extends PoolBase implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.autoPad = attributes.getString('auto_pad', 'NOTSET');
    this.kernelShape = attributes.getInts('kernel_shape');
    this.strides = attributes.getInts('strides', []);
    this.pads = attributes.getInts('pads', []);
    this.countIncludePad = (attributes.getInt('count_include_pad', 0) === 0 ? false : true);
  }
}

export abstract class GlobalAveragePool extends PoolBase implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.countIncludePad = (attributes.getInt('count_include_pad', 0) === 0 ? false : true);
  }
}

export abstract class MaxPool extends PoolBase implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.autoPad = attributes.getString('auto_pad', 'NOTSET');
    this.kernelShape = attributes.getInts('kernel_shape');
    this.strides = attributes.getInts('strides', []);
    this.pads = attributes.getInts('pads', []);
  }
}

export abstract class GlobalMaxPool extends PoolBase implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {}
}

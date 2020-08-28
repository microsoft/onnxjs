// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Scatter implements Operator {
  // Inputs are {data_tensor->float32/float64, indices_tensor-> int32, update_data_tensor->float32/float64}
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {}

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 3) {
      return false;
    }
    const tensorRank = inputs[0].dims.length;
    if (tensorRank < 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
      return false;
    }
    if (inputs[1].type !== 'int32' && inputs[1].type !== 'int16') {
      return false;
    }
    if (inputs[2].type !== 'float32' && inputs[2].type !== 'float64') {
      return false;
    }
    return true;
  }
}

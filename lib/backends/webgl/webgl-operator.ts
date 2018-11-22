// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Operator} from '../../operators';
import {Tensor} from '../../tensor';

import {GlslPositionalFunction} from './glsl-definitions';
import {WebGLInferenceHandler} from './inference-handler';
import {ProgramInfo} from './program-info';
import {RunData} from './program-manager';

export interface PositionalSubOperator extends Operator {
  getPositionalFunction(handler: WebGLInferenceHandler, inputShape: number[], name?: string): GlslPositionalFunction;
}
export interface WebGLRunnable extends Operator {
  addPositionalSub(positionalSubOperator: PositionalSubOperator): void;
  positionalSubs: PositionalSubOperator[];
}
export interface WebGLOperator {
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo;
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData;
}

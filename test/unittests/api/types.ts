// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceSession, InferenceSessionConstructor, Tensor, TensorConstructor} from '../../../lib/api';

// This file check if interfaces and namespaces are exported correctly.
// If types are not exported correctly, a build breaks will happen

// tslint:disable:no-unnecessary-type-assertion

// those lines will fail if interface is not exported
type I1 = Tensor;
type I2 = InferenceSession;

// those lines will fail if namespace is not exported
type I3 = Tensor.Type;
type I4 = Tensor.DataType;
type I5 = InferenceSession.InputType;
type I6 = InferenceSession.OutputType;

// this will fail if Tensor interface does not have one of the members
type I7 = Pick<Tensor, 'data'|'dims'|'get'|'set'>;

// this will fail if InferenceSession interface does not have one of the members
type I8 = Pick<InferenceSession, 'loadModel'|'run'>;

// this will fail if onnx.Tensor is not of type TensorConstructor
type I9 = Exclude<typeof onnx.Tensor, TensorConstructor>;
type I10 = Exclude<TensorConstructor, typeof onnx.Tensor>;
let n: never = unused() as I9 | I10;

// this will fail if onnx.Tensor is not of type TensorConstructor
type I11 = Exclude<typeof onnx.InferenceSession, InferenceSessionConstructor>;
type I12 = Exclude<InferenceSessionConstructor, typeof onnx.InferenceSession>;
n = unused() as I11 | I12;

// this will fail if TensorConstructor does not create a Tensor instance
type I13 = Exclude<Tensor, InstanceType<TensorConstructor>>;
type I14 = Exclude<InstanceType<TensorConstructor>, Tensor>;
n = unused() as I13 | I14;

// this will fail if IConstructor does not create a Tensor instance
type I15 = Exclude<InferenceSession, InstanceType<InferenceSessionConstructor>>;
type I16 = Exclude<InstanceType<InferenceSessionConstructor>, InferenceSession>;
n = unused() as I15 | I16;

//
// typescript does not support to suppress --noUnusedLocals, --noUnusedParameters per file
// by-pass unused variable check
const u: I1|I2|I3|I4|I5|I6|I7|I8|{} = {};
unused(u, n);
function unused<U>(...t: unknown[]) {
  return t as unknown as U;
}

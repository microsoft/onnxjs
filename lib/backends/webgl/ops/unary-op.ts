// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
import {UnaryOp} from '../../../ops/unary-op';
import {Tensor} from '../../../tensor';
import {FunctionType, GlslValueFunction} from '../glsl-definitions';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {PositionalSubOperator, WebGLOperator} from '../webgl-operator';
import {WebGLOperatorHelper} from '../webgl-operator-utils';

export class WebGLUnaryOp extends UnaryOp implements WebGLOperator {
  constructor(protected typeConstraint: ReadonlyArray<Tensor.DataType>, protected glslFunc: GlslValueFunction) {
    super(typeConstraint);
  }
  initialize(attributes: Attribute): void {}
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return WebGLOperatorHelper.run(this, inferenceHandler, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = inputs[0].dims.slice();
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0]);
    const shaderSource = `
      uniform sampler2D A;
      ${this.glslFunc.body}
      void main() {
        vec4 v = texture2D(A, TexCoords);
        v = ${this.glslFunc.name}(v);
        gl_FragColor = v;
      }
      `;
    const outputLayout = handler.createBasicTextureLayout(outputShape);
    return {hasMain: true, inputLayouts: [inputLayout], outputLayout, shaderSource};
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreate(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].dataType),
      uniformData: {}
    };
  }
  addPositionalSub(positionalSubOperator: PositionalSubOperator): void {
    throw new Error('Unary ops don\'t use index-based functions or subops');
  }
}

export function glslAbs(): GlslValueFunction {
  return glslBuiltinUnary('abs');
}
export function glslAcos(): GlslValueFunction {
  return glslBuiltinUnary('acos');
}
export function glslAsin(): GlslValueFunction {
  return glslBuiltinUnary('asin');
}
export function glslAtan(): GlslValueFunction {
  return glslBuiltinUnary('atan');
}
export function glslCeil(): GlslValueFunction {
  return glslBuiltinUnary('ceil');
}
export function glslCos(): GlslValueFunction {
  return glslBuiltinUnary('cos');
}
export function glslExp(): GlslValueFunction {
  return glslBuiltinUnary('exp');
}
export function glslFloor(): GlslValueFunction {
  return glslBuiltinUnary('floor');
}
export function glslIdentity(): GlslValueFunction {
  const name = `indentity_`;
  const body = `
  float ${name}(float a) {
    return a;
  }
  vec4 ${name}(vec4 v) {
    return v;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslLog(): GlslValueFunction {
  return glslBuiltinUnary('log');
}
export function glslNeg(): GlslValueFunction {
  const name = `neg_`;
  const body = `
  float ${name}(float a) {
    return -a;
  }
  vec4 ${name}(vec4 v) {
    return -v;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslNot(): GlslValueFunction {
  const name = `not_`;
  const body = `
  float ${name}(float a) {
    return float( ! bool(a) );
  }
  bool ${name}(bool a) {
    return !a;
  }
  vec4 ${name}(vec4 v) {
    return vec4(!bool(v.x), !bool(v.y), !bool(v.z), !bool(v.w));
  }
  bvec4 ${name}(bvec4 v) {
    return bvec4(!v.x, !v.y, !v.z, !v.w);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslTanh(): GlslValueFunction {
  const name = `tanh_`;
  const body = `
  float ${name}(float a) {
    if (a < 0.0) {
      float t = exp(2.0*a);
      return (t - 1.0) / (t + 1.0);}
    else {
      float t = exp(-2.0*a);
      return (1.0 - t) / (1.0 + t);
    }
  }
  vec4 ${name}(vec4 v) {
    vec4  m = max(v, -v);  // to avoid overflow
    vec4 ep = exp( v - m); // exp(+v)
    vec4 em = exp(-v - m); // exp(-v)
    return (ep - em) / (ep + em);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSin(): GlslValueFunction {
  return glslBuiltinUnary('sin');
}
export function glslRelu(): GlslValueFunction {
  const name = `relu_`;
  const body = `
  float ${name}(float a) {
    return max( a, 0.0 );
  }
  vec4 ${name}(vec4 v) {
    return max( v, 0.0 );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSigmoid(): GlslValueFunction {
  const name = `sigmoid_`;
  const body = `
  float ${name}(float a) {
    if (a > 0.0) {
      return 1.0 / (1.0 + exp(-a));
    } else {
      float t = exp(a);
      return t / (1.0 + t);
    }
  }
  vec4 ${name}(vec4 v) {
    vec4  m = max(v, -v); // to avoid overflow
    vec4 ex = exp(v - m); // exp(x)
    vec4 e0 = exp(-m);    // exp(0)
    return ex / (ex + e0);// exp(x) / (exp(x) + 1.0)
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSqrt(): GlslValueFunction {
  return glslBuiltinUnary('sqrt');
}
export function glslTan(): GlslValueFunction {
  return glslBuiltinUnary('tan');
}
function glslBuiltinUnary(fname: string): GlslValueFunction {
  const name = `${fname}_`;
  const body = `
  float ${name}(float a) {
    return ${fname}(a);
  }
  vec4 ${name}(vec4 v) {
    return ${fname}(v);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}

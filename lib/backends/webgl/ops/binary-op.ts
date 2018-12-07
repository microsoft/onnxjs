// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
import {binaryOp as binaryCpuOp} from '../../../backends/cpu/ops/binary-op';
import {BinaryOp} from '../../../ops/binary-op';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../../util';
import {FunctionType, GlslValueFunction} from '../glsl-definitions';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {TextureData, TextureLayout} from '../texture-data';
import {WebGLOperator} from '../webgl-operator';
import {WebGLOperatorHelper} from '../webgl-operator-utils';

export class WebGLBinaryOp extends BinaryOp implements WebGLOperator {
  constructor(
      protected typeConstraint: ReadonlyArray<Tensor.DataType>, protected glslFunc: GlslValueFunction,
      protected opLambda: (e1: number, e2: number) => number, protected outputType?: Tensor.DataType) {
    super(typeConstraint);
  }
  initialize(attributes: Attribute): void {}
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    // both are scalars
    if (inputs[0].dims.length === 0 && inputs[1].dims.length === 0) {
      return [binaryCpuOp(inputs[0], inputs[1], this.opLambda, this.outputType)];
    }
    return WebGLOperatorHelper.run(this, inferenceHandler, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputLayouts: TextureLayout[] = [];
    if (inputs[0].dims.length !== 0) {
      inputLayouts.push(handler.getOrCreateTextureLayout(inputs[0]));
    }
    if (inputs[1].dims.length !== 0) {
      inputLayouts.push(handler.getOrCreateTextureLayout(inputs[1]));
    }
    const isBroadcast = !ShapeUtil.areEqual(inputs[0].dims.slice(), inputs[1].dims.slice());
    if (isBroadcast) {
      const outputShape = BroadcastUtil.calcShape(inputs[0].dims.slice(), inputs[1].dims.slice(), false);
      if (!outputShape) {
        throw new Error(`Can't perform binary op on the given tensors`);
      }
      const rank = outputShape.length;
      if (rank === 0) {
        throw new Error(`No WebGL support for scalar output generation`);
      }
      let shaderSource = ``;
      // no scalars involved
      if (inputs[0].dims.length !== 0 && inputs[1].dims.length !== 0) {
        shaderSource = `
        uniform sampler2D A;
        uniform sampler2D B;
        ${this.glslFunc.body}
        float process(int indices[${rank}]) {
          int aindices[${inputs[0].dims.slice().length}];
          int bindices[${inputs[1].dims.slice().length}];
          bcastIndices_A(indices, aindices);
          bcastIndices_B(indices, bindices);
          return ${this.glslFunc.name}(_A(aindices), _B(bindices));
        }`;
      }
      // one of them is a scalar
      else {
        let scalar = '';
        let texture = '';
        let textureCap = '';
        let scalarValue: number;
        let indicesRank: number;
        let scalarFirst = false;
        if (inputs[0].dims.length === 0) {
          scalarFirst = true;
          scalar = 'a';
          texture = 'b';
          textureCap = 'B';
          scalarValue = inputs[0].data[0] as number;
          indicesRank = inputs[1].dims.length;
        } else {
          scalar = 'b';
          texture = 'a';
          textureCap = 'A';
          scalarValue = inputs[1].data[0] as number;
          indicesRank = inputs[0].dims.length;
        }
        shaderSource = `
        uniform sampler2D ${textureCap};
        ${this.glslFunc.body}
        float process(int indices[${rank}]) {
          int ${texture}indices[${indicesRank}];
          float ${scalar} = float(${scalarValue});
          bcastIndices_${textureCap}(indices, ${texture}indices);`;

        if (scalarFirst) {
          shaderSource += `return ${this.glslFunc.name}(${scalar}, _${textureCap}(${texture}indices)); }`;
        } else {
          shaderSource += `return ${this.glslFunc.name}(_${textureCap}(${texture}indices), ${scalar}); }`;
        }
      }
      return {
        hasMain: false,
        inputLayouts,
        outputLayout: handler.createBasicTextureLayout(outputShape),
        shaderSource,
      };
    }
    const shaderSource = `
      uniform sampler2D A;
      uniform sampler2D B;
      ${this.glslFunc.body}
      void main() {
        vec4 v1 = texture2D(A, TexCoords);
        vec4 v2 = texture2D(B, TexCoords);
        vec4 result = ${this.glslFunc.name}(v1, v2);
        gl_FragColor = result;
      }
    `;
    return {
      hasMain: true,
      inputLayouts,
      outputLayout: handler.createBasicTextureLayout(inputs[0].dims.slice()),
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    let inputTDs: TextureData[] = [];
    // both scalars - no support in WebGL
    if (inputs[0].dims.length === 0 && inputs[1].dims.length === 0) {
      throw new Error(`No WebGL support for scalar output generation`);
    }
    // both non-scalars - process regularly
    if (inputs[0].dims.length !== 0 && inputs[1].dims.length !== 0) {
      inputTDs = inputs.map((t, i) => handler.getOrCreate(t, programInfo.inputLayouts[i]));
    }
    // one of them is a scalar
    else {
      if (inputs[0].dims.length !== 0) {
        inputTDs.push(handler.getOrCreate(inputs[0], programInfo.inputLayouts[0]));
      } else {
        inputTDs.push(handler.getOrCreate(inputs[1], programInfo.inputLayouts[0]));
      }
    }
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(
          programInfo.outputLayout, this.outputType ? this.outputType : inputs[0].type),
      uniformData: {}
    };
  }

  handleBothScalarTensors(input1: Tensor, input2: Tensor, opType: string) {
    switch (opType) {
      default:
        throw new Error('Unsupported binary op');
    }
  }
}

export function glslAdd(): GlslValueFunction {
  const name = `add_`;
  const body = `
  float ${name}(float a, float b) {
    return a + b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 + v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslDiv(): GlslValueFunction {
  const name = `div_`;
  const body = `
  float ${name}(float a, float b) {
    return a / b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 / v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslMul(): GlslValueFunction {
  const name = `mul_`;
  const body = `
  float ${name}(float a, float b) {
    return a * b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 * v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslSub(): GlslValueFunction {
  const name = `sub_`;
  const body = `
  float ${name}(float a, float b) {
    return a - b;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return v1 - v2;
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslEqual(): GlslValueFunction {
  const name = `equal_`;
  const body = `
  float ${name}(float a, float b) {
    return float(a == b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4( v1 == v2 );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslGreater(): GlslValueFunction {
  const name = `greater_`;
  const body = `
  float ${name}(float a, float b) {
    return float(a > b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4( v1.r > v2.r ,
      v1.g > v2.g,
      v1.b > v2.b,
      v1.a > v2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslLess(): GlslValueFunction {
  const name = `less_`;
  const body = `
  float ${name}(float a, float b) {
    return float(a < b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4( v1.r < v2.r ,
                v1.g < v2.g,
                v1.b < v2.b,
                v1.a < v2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslAnd(): GlslValueFunction {
  const name = `and_`;
  const body = `
  float ${name}(float a, float b) {
    return float( bool(a) && bool(b) );
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    bvec4 b1 = bvec4(v1);
    bvec4 b2 = bvec4(v2);
    return vec4( b1.r && b2.r ,
                b1.g && b2.g,
                b1.b && b2.b,
                b1.a && b2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslOr(): GlslValueFunction {
  const name = `or_`;
  const body = `
  float ${name}(float a, float b) {
    return float( bool(a) || bool(b) );
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    bvec4 b1 = bvec4(v1);
    bvec4 b2 = bvec4(v2);
    return vec4( b1.r || b2.r ,
                b1.g || b2.g,
                b1.b || b2.b,
                b1.a || b2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslXor(): GlslValueFunction {
  const name = `xor_`;
  const body = `
  float ${name}(float a, float b) {
    return float( bool(a) ^^ bool(b) );
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    bvec4 b1 = bvec4(v1);
    bvec4 b2 = bvec4(v2);
    return vec4( b1.r ^^ b2.r ,
                b1.g ^^ b2.g,
                b1.b ^^ b2.b,
                b1.a ^^ b2.a );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}
export function glslPow(): GlslValueFunction {
  return glslBuiltinBinary('pow');
}
export function glslPRelu(): GlslValueFunction {
  const name = `prelu_`;
  const body = `
  float ${name}(float a, float b) {
    return a < 0.0 ? a * b: a;
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return vec4(
      v1.r < 0.0 ? v1.r * v2.r: v1.r,
      v1.g < 0.0 ? v1.g * v2.g: v1.g,
      v1.b < 0.0 ? v1.b * v2.b: v1.b,
      v1.a < 0.0 ? v1.a * v2.a: v1.a
      );
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}

function glslBuiltinBinary(fname: string): GlslValueFunction {
  const name = `${fname}_`;
  const body = `
  float ${name}(float a, float b) {
    return ${fname}(a, b);
  }
  vec4 ${name}(vec4 v1, vec4 v2) {
    return ${fname}(v1, v2);
  }
  `;
  return {body, name, type: FunctionType.ValueBased};
}

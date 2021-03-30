// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {env} from '../../env';
import {Logger, Profiler} from '../../instrument';

import {GlslPreprocessor} from './glsl-preprocessor';
import {getVertexShaderSource} from './glsl-source';
import {TextureLayoutStrategy} from './texture-layout-strategy';
import {Artifact, ProgramInfo, RunData, TextureData, UniformData, VariableInfo} from './types';
import {WebGLContext} from './webgl-context';

/**
 * ProgramManager is the main class behind running computations
 * It builds ProgramInfo's into Artifacts
 * It compiles given ProgramInfo's into WebGL Prorams (cached as Artifacts)
 * Uses the artifact to run the computation by calling Draw on
 * the WebGL drawing buffer
 * ProgramManager automatically maps (binds) input variables to their
 * corresponding Location's in the binary program
 */
export class ProgramManager {
  repo: Map<{}, Artifact>;  // this should be per-session object
  vertexShader: WebGLShader;
  attributesBound: boolean;

  constructor(
      public profiler: Readonly<Profiler>, public glContext: WebGLContext,
      public textureLayoutStrategy: TextureLayoutStrategy) {
    this.repo = new Map();
    this.attributesBound = false;
  }
  getArtifact(key: {}): Artifact|undefined {
    return this.repo.get(key);
  }
  setArtifact(key: {}, artifact: Artifact): void {
    this.repo.set(key, artifact);
  }
  run(buildArtifact: Artifact, runData: RunData): void {
    const inputInfo = runData.inputTextureDatas.map((d, i) => `input${i}:[${d.shape}]`).join(', ');
    const outputInfo = `output: [${runData.outputTextureData.shape}]`;

    this.profiler.event('backend', `ProgramManager.run ${inputInfo} ; ${outputInfo}`, () => {
      const gl = this.glContext.gl;
      const program = buildArtifact.program;
      gl.useProgram(program);
      try {
        this.bindOutput(runData.outputTextureData);
        if (!this.attributesBound) {
          this.bindAttributes(buildArtifact.attribLocations);
        }
        this.bindUniforms(buildArtifact.uniformLocations, runData.uniformData, runData.inputTextureDatas);
      } catch (err) {
        Logger.error('ProgramManager', buildArtifact.programInfo.shaderSource);
        throw err;
      }
      this.profiler.event('backend', 'GlContext.draw()', () => {
        this.doDraw(buildArtifact, runData);
      });
    });
  }
  dispose(): void {
    if (this.vertexShader) {
      this.glContext.deleteShader(this.vertexShader);
    }
    this.repo.forEach(a => this.glContext.deleteProgram(a.program));
  }
  build(programInfo: ProgramInfo): Artifact {
    return this.profiler.event('backend', 'ProgramManager.build', () => {
      const preprocessor = new GlslPreprocessor(this.glContext, programInfo);
      const fragScript = preprocessor.preprocess();
      // const fragScript = `#version 300 es
      // precision highp float;
      // precision highp int;
      // precision highp sampler2D;
      // in vec2 TexCoords;
      // out vec4 outputColor;
      // const vec2 halfCR = vec2(0.5, 0.5);

      // // Custom vector types to handle higher dimenalities.
      // struct ivec5
      // {
      //   int x;
      //   int y;
      //   int z;
      //   int w;
      //   int u;
      // };

      // struct ivec6
      // {
      //   int x;
      //   int y;
      //   int z;
      //   int w;
      //   int u;
      //   int v;
      // };

      // int imod(int x, int y) {
      //   return x - y * (x / y);
      // }


      // uniform sampler2D A;
      // vec4 getA(int b2, int b, int row, int col) {
      //   int index = b2 * 3840 + b * 3840 + (row / 2) * 80 + (col / 2);
      //   int texR = index / 80;
      //   int texC = index - texR * 80;
      //   vec2 uv = (vec2(texC, texR) + halfCR) / vec2(80, 48);
      //   return texture(A, uv);
      // }

      //   ivec4 getOutputCoords() {
      //       ivec2 resTexRC = ivec2(TexCoords.xy *
      //                             vec2(96, 160));
      //       int index = resTexRC.y * 96 + resTexRC.x;
      //       int r = index / 15360; index -= r * 15360;
      //       int c = index / 15360; index -= c * 15360;
      //       int d = index / 160; int d2 = index - d * 160;
      //       return ivec4(r, c, d, d2);
      //     }



      // float getChannel(vec4 frag, int dim) {
      //   int modCoord = imod(dim, 2);
      //   return modCoord == 0 ? frag.r : frag.g;
      // }

      // float getChannel(vec4 frag, vec2 innerDims) {
      //   vec2 modCoord = mod(innerDims, 2.);
      //   return modCoord.x == 0. ?
      //     (modCoord.y == 0. ? frag.r : frag.g) :
      //     (modCoord.y == 0. ? frag.b : frag.a);
      // }

      //     void main() {
      //       ivec4 rc = getOutputCoords();

      //       // Sample the texture with the coords to get the rgba channel value.
      //       vec4 packedInput = getA(rc.x,rc.y,rc.z,rc.w);
      //       //outputColor = vec4(rc.z, 0, 0, 0);
      //       outputColor = vec4(getChannel(packedInput, vec2(rc.z,rc.w)), 0, 0, 0);
      //     }`;
      if (fragScript.indexOf('test place holder resize') !== -1) {
        console.log(fragScript);
      }
      const program = this.compile(fragScript);
      const artifact = {
        programInfo,
        program,
        uniformLocations: this.getUniformLocations(
            program, preprocessor.context.programInfo.samplers, preprocessor.context.programInfo.variables),
        attribLocations: this.getAttribLocations(program)
      };
      return artifact;
    });
  }
  protected doDraw(artifact: Artifact, runData: RunData): void {
    if (runData.draw) {
      Logger.verbose('ProgramManager', 'Custom draw function');
      runData.draw(this.glContext, artifact);
    } else {
      this.glContext.draw();
    }
  }
  protected compile(fragShaderScript: string): WebGLProgram {
    if (!this.vertexShader) {
      Logger.verbose('ProrgramManager', 'Compiling and caching Vertex shader for the first time');
      const vertexShaderScript = getVertexShaderSource(this.glContext.version);
      this.vertexShader = this.glContext.compileShader(vertexShaderScript, this.glContext.gl.VERTEX_SHADER);
    }
    if (env.debug) {
      Logger.verbose('ProrgramManager', `FragShader:
${fragShaderScript}
`);
    }
    const fragShader = this.glContext.compileShader(fragShaderScript, this.glContext.gl.FRAGMENT_SHADER);
    const program = this.glContext.createProgram(this.vertexShader, fragShader);
    this.glContext.deleteShader(fragShader);
    return program;
  }
  bindOutput(td: TextureData): void {
    const width = td.width;
    const height = td.height;
    Logger.verbose(
        'ProrgramManager',
        `Binding output texture to Framebuffer: w/h=${width}/${height}, shape=${td.shape}, type=${td.tensor.type}`);
    this.glContext.attachFramebuffer(td.texture, width, height);
  }
  bindAttributes(attribLocations: Artifact.AttribLocations): void {
    const positionHandle = attribLocations.position;
    const textureCoordHandle = attribLocations.textureCoord;
    this.glContext.setVertexAttributes(positionHandle, textureCoordHandle);
    this.attributesBound = true;
  }
  bindUniforms(uniformLocations: Artifact.UniformLocations, uniformData: UniformData, textures: TextureData[]): void {
    const gl = this.glContext.gl;
    let texturePosition = 0;
    for (const {name, type, location, arrayLength} of uniformLocations) {
      switch (type) {
        case 'sampler2D':
          this.bindTexture(textures[texturePosition], location, texturePosition);
          texturePosition++;
          break;
        case 'float':
          if (arrayLength) {
            gl.uniform1fv(location, uniformData[name] as number[]);
          } else {
            gl.uniform1f(location, uniformData[name] as number);
          }
          break;
        case 'int':
          if (arrayLength) {
            gl.uniform1iv(location, uniformData[name] as number[]);
          } else {
            gl.uniform1i(location, uniformData[name] as number);
          }
          break;
        default:
          throw new Error(`Uniform not implemented: ${type}`);
      }
    }
  }
  bindTexture(td: TextureData, uniformHandle: WebGLUniformLocation, position: number): void {
    this.glContext.bindTextureToUniform(td.texture, position, uniformHandle);
  }
  getAttribLocations(program: WebGLProgram): Artifact.AttribLocations {
    return {
      position: this.getAttribLocation(program, 'position'),
      textureCoord: this.getAttribLocation(program, 'textureCoord')
    };
  }
  getUniformLocations(program: WebGLProgram, samplers?: string[], variables?: VariableInfo[]):
      Artifact.UniformLocations {
    const uniformLocations: Artifact.UniformLocations = [];
    if (samplers) {
      for (const sampler of samplers) {
        uniformLocations.push({name: sampler, type: 'sampler2D', location: this.getUniformLocation(program, sampler)});
      }
    }
    if (variables) {
      for (const variable of variables) {
        uniformLocations.push({...variable, location: this.getUniformLocation(program, variable.name)});
      }
    }
    return uniformLocations;
  }
  getUniformLocation(program: WebGLProgram, name: string): WebGLUniformLocation {
    const gl = this.glContext.gl;
    const reference = gl.getUniformLocation(program, name);
    if (reference === null) {
      throw new Error(`Uniform ${name} not found.`);
    }
    return reference;
  }
  getAttribLocation(program: WebGLProgram, name: string): number {
    const gl = this.glContext.gl;
    const attributeLocation: number = gl.getAttribLocation(program, name);
    return attributeLocation;
  }
}

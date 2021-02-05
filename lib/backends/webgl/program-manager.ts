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
      /*const fragScript = `#version 300 es
      precision highp float;
      precision highp int;
      precision highp sampler2D;
      in vec2 TexCoords;
      out vec4 outputColor;
      const vec2 halfCR = vec2(0.5, 0.5);

      // Custom vector types to handle higher dimenalities.
      struct ivec5
      {
        int x;
        int y;
        int z;
        int w;
        int u;
      };

      struct ivec6
      {
        int x;
        int y;
        int z;
        int w;
        int u;
        int v;
      };

      int imod(int x, int y) {
        return x - y * (x / y);
      }


      uniform sampler2D A;
  uniform sampler2D B;

          void bcastMatmulIndices_A(int bcastedIndices[2], out int realIndices[2]) {

            realIndices[1] = bcastedIndices[1];
            realIndices[0] = bcastedIndices[0];
          }


          void bcastMatmulIndices_B(int bcastedIndices[2], out int realIndices[2]) {

            realIndices[1] = bcastedIndices[1];
            realIndices[0] = bcastedIndices[0];
          }


        int coordsToOffset(vec2 coords, int width, int height) {
          float s = coords.s * float(width);
          float t = coords.t * float(height);
          int offset = int(t) * width + int(s);
          return offset;
        }


        void toVec(vec2 texCoords, out int c[2]) {
          int offset = coordsToOffset(texCoords, 2, 2);

          c[0] = offset / 2;
          offset -= c[0] * 2;
          c[1] = offset;
        }
        void toVec(int offset, out int c[2]) {

          c[0] = offset / 2;
          offset -= c[0] * 2;
          c[1] = offset;
        }


        int indicesToOffset_A(int indices[2]) {
          int offset = 0;

          offset += indices[1] * 1;

          offset += indices[0] * 4;

          return offset;
        }


        vec2 offsetToCoords(int offset, int width, int height) {
          int t = offset / width;
          int s = offset - t*width;
          vec2 coords = (vec2(s,t) + vec2(0.5,0.5)) / vec2(width, height);
          return coords;
        }

  highp float decode(highp vec4 rgba) {
          return rgba.r;
        }


          float getColorAsFloat(vec4 color) {
              return decode(color);
          }


          float _A(int m[2]) {
            int offset = indicesToOffset_A(m);
            vec2 coords = offsetToCoords(offset, 2, 4);
            float value = getColorAsFloat(texture(A, coords));
            return value;
          }


          vec4 _A_Pack(int m[2]) {
            int offset = indicesToOffset_A(m);
            vec2 coords = offsetToCoords(offset, 2, 4);
            return texture(A, coords);
          }


        int indicesToOffset_B(int indices[2]) {
          int offset = 0;

          offset += indices[1] * 1;

          offset += indices[0] * 2;

          return offset;
        }


          float _B(int m[2]) {
            int offset = indicesToOffset_B(m);
            vec2 coords = offsetToCoords(offset, 4, 2);
            float value = getColorAsFloat(texture(B, coords));
            return value;
          }


          vec4 _B_Pack(int m[2]) {
            int offset = indicesToOffset_B(m);
            vec2 coords = offsetToCoords(offset, 4, 2);
            return texture(B, coords);
          }


            ivec2 getOutputCoords() {
              return ivec2(TexCoords.yx * vec2(2, 2));
            }



        vec4 process(int indices[2]) {
            ivec2 rc = getOutputCoords();
            int a[2];
            int b[2];
            bcastMatmulIndices_A(indices, a);
            bcastMatmulIndices_B(indices, b);

            vec4 value;
            for (int k=0; k<((4+1)/2); ++k) {
                a[1] = 0;
                b[0] = 0;
                value += _A_Pack(a).rrbb * _B_Pack(b).rgrg;
                value += _A_Pack(a).ggaa * _B_Pack(b).baba;
                value = _B_Pack(b).rgba;
            }
            return vec4(_B_Pack(b).a,2,3,4);
        }

    void main() {
      int indices[2];
      toVec(TexCoords, indices);
      vec4 result = vec4(process(indices));
      outputColor = result;
    }
    ` */
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

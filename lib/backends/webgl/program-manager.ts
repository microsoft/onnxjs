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
      // const fragScript = preprocessor.preprocess();
      const fragScript = `#version 300 es
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

      vec2 packedUVfrom2D(int texNumR, int texNumC, int texelsInLogicalRow, int row, int col) {
        int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
        int texC = texelIndex / texNumR;
        int texR = texelIndex - texC * texNumR;
        return (vec2(texR, texC) + halfCR) / vec2(texNumR, texNumC);
      }

vec4 getA(int row, int col) {
      vec2 uv = packedUVfrom2D(1, 2, 2, row, col);
      return texture(A, uv);
      // vec4 t = texture(A, vec2(0.25, 0.25));
      // return vec4(uv, 0, 0);
      // return t;
    }
      vec4 getA(int b, int row, int col) {
        return getA(row, col);
      }

        ivec3 getOutputCoords() {
          ivec2 resTexRC = ivec2(TexCoords.xy *
                                vec2(1, 2));
          int index = resTexRC.y * 1 + resTexRC.x;

          int b = index / 2;
          index -= b * 2;

          // reverse r and c order for packed texture
          int r = imod(index, 1) * 2;
          int c = 2 * (index / 1);

          return ivec3(b, r, c);
        }




    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      int r = index / 8; index -= r * 8;
      int c = index / 4; index -= c * 4;
      int d = index / 1; int undefined = index - d * 1;
      return ivec3(r, c, d);
    }


  int getFlatIndex(ivec3 coords) {
    // reverse y, z order
    return coords.x * 8 + coords.z * 2 + coords.y;
  }


    float getChannel(vec4 frag, int dim) {
      int modCoord = imod(dim, 2);
      return modCoord == 0 ? frag.r : frag.g;
    }

    float getChannel(vec4 frag, vec2 innerDims) {
      vec2 modCoord = mod(innerDims, 2.);
      return modCoord.x == 0. ?
        (modCoord.y == 0. ? frag.r : frag.g) :
        (modCoord.y == 0. ? frag.b : frag.a);
    }

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.0);

        ivec3 thisRC;
        int rows = 2;
        int cols = 4;


        thisRC = rc;

          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          //result[0] = getChannel(getA(inputRC.x, inputRC.z, inputRC.y), inputRCInnerDims);
          vec4 t = getA(inputRC.x, inputRC.y, inputRC.z);
          result = t;
          //result[0] = float(flatIndex);



        // thisRC = rc;thisRC.y += 1;
        // if(thisRC.y < rows && thisRC.z < cols){
        //   int flatIndex = getFlatIndex(thisRC);

        //   ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
        //   vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

        //   //result[1] = getChannel(getA(inputRC.x, inputRC.z, inputRC.y), inputRCInnerDims);
        //   vec4 t = getA(inputRC.x, inputRC.y, inputRC.z);
        //   result[1] = t[1];
        //   //result[1] = float(flatIndex);

        // }

        // thisRC = rc;thisRC.z += 1;
        // if(thisRC.y < rows && thisRC.z < cols){
        //   int flatIndex = getFlatIndex(thisRC);

        //   ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
        //   vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

        //   //result[2] = getChannel(getA(inputRC.x, inputRC.z, inputRC.y), inputRCInnerDims);
        //   vec4 t = getA(inputRC.x, inputRC.y, inputRC.z);
        //   result[2] = t[2];
        //   //result[2] = float(flatIndex);

        // }

        // thisRC = rc;thisRC.z += 1;thisRC.y += 1;
        // if(thisRC.y < rows && thisRC.z < cols){
        //   int flatIndex = getFlatIndex(thisRC);

        //   ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
        //   vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

        //   //result[3] = getChannel(getA(inputRC.x, inputRC.z, inputRC.y), inputRCInnerDims);
        //   vec4 t = getA(inputRC.x, inputRC.y, inputRC.z);
        //   result[3] = t[3];
        //   //result[3] = float(flatIndex);

        // }


        outputColor = result;
      }`;
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
    Logger.verbose(
        'ProrgramManager',
        `Binding output texture to Framebuffer: w/h=${td.width}/${td.height}, shape=${td.shape}, type=${
            td.tensor.type}`);
    this.glContext.attachFramebuffer(td.texture, td.width, td.height);
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

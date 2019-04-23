// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger, Profiler} from '../../instrument';

import {GlslPreprocessor} from './glsl-preprocessor';
import {Artifact, LocationInfo, ProgramInfo, RunData, TextureData, UniformData, VariableInfo} from './types';
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
  // tslint:disable-next-line:ban-types
  repo: Map<Object, Artifact>;  // this should be per-session object
  vertexShader: WebGLShader;
  attributesBound: boolean;

  constructor(public profiler: Readonly<Profiler>, public glContext: WebGLContext) {
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
    this.profiler.event('backend', 'ProgramManager.run', () => {
      if (runData.preRun) {
        Logger.verbose('ProgramManager', 'PreRun');
        runData.preRun(this.glContext, buildArtifact);
      }
      const gl = this.glContext.gl;
      const program = buildArtifact.program;
      gl.useProgram(program);
      try {
        this.bindOutput(runData.outputTextureData);
        if (!this.attributesBound) {
          this.bindAttributes(buildArtifact.attribLocations);
        }
        this.bindUniforms(buildArtifact.uniformLocations, runData.uniformData);
        this.bindTextures(buildArtifact.uniformLocations, runData.inputTextureDatas);
      } catch (err) {
        Logger.error('ProgramManager', buildArtifact.programInfo.shaderSource);
        throw err;
      }
      this.profiler.event('backend', 'GlContext.draw()', () => {
        if (buildArtifact.programInfo.blockSize) {
          this.doBlockDraw(buildArtifact, runData);
        } else {
          this.doDraw(buildArtifact, runData);
        }
        gl.flush();
      });
      if (runData.postRun) {
        Logger.verbose('ProgramManager', 'PostRun');
        runData.postRun(this.glContext, buildArtifact);
      }
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
      preprocessor.extractAttribInfo(this.getDefaultVertexShaderSource());
      const fragScript = preprocessor.preprocess();
      try {
        const attribInfos = preprocessor.context.attribInfo;
        const uniformInfos = preprocessor.context.uniformInfo;
        const program = this.compile(fragScript);
        const artifact = {
          programInfo,
          program,
          uniformLocations: this.getUniformLocations(program, uniformInfos),
          attribLocations: this.getAttribLocations(program, attribInfos)
        };
        return artifact;
      } catch (err) {
        Logger.error('ProgramManager', fragScript);
        throw err;
      }
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
  protected doBlockDraw(artifact: Artifact, runData: RunData): void {
    const gl = this.glContext.gl;
    const [blockWidth, blockHeight] = artifact.programInfo.blockSize;
    const widthLocation = artifact.uniformLocations.blockWidth.location;
    const heightLocation = artifact.uniformLocations.blockHeight.location;
    const yOffsetLocation = artifact.uniformLocations.blockYOffset.location;
    const xOffsetLocation = artifact.uniformLocations.blockXOffset.location;
    const height = runData.outputTextureData.height;
    const width = runData.outputTextureData.width;

    for (let col = 0; col < width; col += blockWidth) {
      const colCount = Math.min(blockWidth, width - col);
      gl.uniform1i(widthLocation, colCount);
      gl.uniform1i(xOffsetLocation, col);
      for (let row = 0; row < height; row += blockHeight) {
        const rowCount = Math.min(blockHeight, height - row);
        Logger.verbose('ProgramManager', `row=${row}, rowCount=${rowCount}, col=${col}, colCount=${colCount}`);
        gl.viewport(col, row, colCount, rowCount);
        gl.uniform1i(heightLocation, rowCount);
        gl.uniform1i(yOffsetLocation, row);
        this.doDraw(artifact, runData);
      }
    }
  }
  protected compile(fragShaderScript: string): WebGLProgram {
    if (!this.vertexShader) {
      Logger.verbose('ProrgramManager', 'Compiling and caching Vertex shader for the first time');
      this.vertexShader =
          this.glContext.compileShader(this.getDefaultVertexShaderSource(), this.glContext.gl.VERTEX_SHADER);
    }
    const fragShader = this.glContext.compileShader(fragShaderScript, this.glContext.gl.FRAGMENT_SHADER);
    const program = this.glContext.createProgram(this.vertexShader, fragShader);
    this.glContext.deleteShader(fragShader);
    return program;
  }
  bindOutput(td: TextureData): void {
    Logger.verbose('ProrgramManager', `Binding output texture to Framebuffer:
       w/h: ${td.width}/${td.height},
       shape: ${td.shape},
       type: ${td.dataType}
    `);
    this.glContext.attachFramebuffer(td.texture, td.width, td.height);
  }
  bindAttributes(attribLocations: {[name: string]: LocationInfo}): void {
    const positionHandle = attribLocations.position.location as number;
    const textureCoordHandle = attribLocations.textureCoord.location as number;
    this.glContext.setVertexAttributes(positionHandle, textureCoordHandle);
    this.attributesBound = true;
  }
  bindUniformArray(location: WebGLUniformLocation, type: string, value: number[]): void {
    const gl = this.glContext.gl;
    switch (type) {
      case 'float':
        gl.uniform1fv(location, value);
        break;
      case 'int':
        gl.uniform1iv(location, value);
        break;
      default:
        throw new Error('Uniform not implemented: ' + type);
    }
    this.glContext.checkError();
  }
  bindUniform(location: WebGLUniformLocation, type: string, value: number): void {
    const gl = this.glContext.gl;
    switch (type) {
      case 'float':
        gl.uniform1f(location, value);
        break;
      case 'int':
        gl.uniform1i(location, value);
        break;
      default:
        throw new Error('Uniform not implemented: ' + type);
    }
    this.glContext.checkError();
  }
  bindUniforms(uniformLocations: {[name: string]: LocationInfo}, inputScalars: UniformData): void {
    if (!inputScalars) {
      return;
    }
    Object.keys(uniformLocations).forEach(key => {
      const li = uniformLocations[key];
      if (!li.variable.type.startsWith('sampler')) {
        const value = inputScalars[li.variable.name];
        if (li.variable.isVec) {
          this.bindUniformArray(li.location, li.variable.type, value as number[]);
        } else {
          this.bindUniform(li.location, li.variable.type, value as number);
        }
      }
    });
  }
  bindTextures(uniformLocations: {[name: string]: LocationInfo}, textures: TextureData[]): void {
    if (!textures) {
      return;
    }
    Object.keys(uniformLocations).forEach((key, i) => {
      const li = uniformLocations[key];
      if (li.variable.type.startsWith('sampler')) {
        const tex = textures[i];
        this.bindTexture(tex, li.location, i++);
      }
    });
  }
  bindTexture(td: TextureData, uniformHandle: WebGLUniformLocation, position: number): void {
    this.glContext.bindTextureToUniform(td.texture, position, uniformHandle);
  }
  getAttribLocations(program: WebGLProgram, variableInfos: VariableInfo[]): {[name: string]: LocationInfo} {
    const locationInfos: {[name: string]: LocationInfo} = {};
    variableInfos.forEach(vi => {
      locationInfos[vi.name] = {variable: vi, location: this.getAttribLocation(program, vi.name)};
    });
    return locationInfos;
  }
  getUniformLocations(program: WebGLProgram, variableInfos: VariableInfo[]): {[name: string]: LocationInfo} {
    const locationInfos: {[name: string]: LocationInfo} = {};
    variableInfos.forEach(vi => {
      locationInfos[vi.name] = {variable: vi, location: this.getUniformLocation(program, vi.name)};
    });
    return locationInfos;
  }
  getUniformLocation(program: WebGLProgram, name: string): WebGLUniformLocation {
    const gl = this.glContext.gl;
    const reference = gl.getUniformLocation(program, name);
    if (reference === null) {
      throw new Error('Uniform ' + name + ' not found.');
    }
    return reference;
  }
  getAttribLocation(program: WebGLProgram, name: string): number {
    const gl = this.glContext.gl;
    const attributeLocation: number = gl.getAttribLocation(program, name);
    if (attributeLocation === -1) {
      throw new Error('Attribute ' + name + ' not found.');
    }
    return attributeLocation;
  }
  protected getDefaultVertexShaderSource(): string {
    return `
        precision highp float;
        attribute vec3 position;
        attribute vec2 textureCoord;

        varying vec2 TexCoords;

        void main()
        {
            gl_Position = vec4(position, 1.0);
            TexCoords = textureCoord;
        }`;
  }
}

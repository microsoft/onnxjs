import {LogSoftmax} from '../../../ops/log-softmax';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData, TextureLayout} from '../types';

export class WebGLLogSoftmax extends LogSoftmax {
  constructor() {
    super();
  }
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    if (!this.artifacts) {
      this.artifacts = [];
      const programInfos = this.createProgramInfos(inferenceHandler, inputs);
      programInfos.forEach((pi, i) => {
        const artifact = inferenceHandler.session.programManager.build(pi);
        this.artifacts.push(artifact);
      });
    }

    const runDatas = this.createRunDatas(inferenceHandler, this.artifacts.map(a => a.programInfo), inputs);
    runDatas.forEach((v, i) => inferenceHandler.session.programManager.run(this.artifacts[i], v));

    return [runDatas[runDatas.length - 1].outputTextureData.tensor];
  }
  createSoftMaxProgramInfo(
      inferenceHandler: WebGLInferenceHandler, input: Tensor, N: number, D: number,
      maxElementPerLogicalRow: TextureLayout, normalizationPerLogicalRow: TextureLayout): ProgramInfo {
    const inputShape = input.dims.slice();
    const inputLayout = inferenceHandler.createTextureLayoutFromShape(inputShape);
    const outputShape = inputShape;
    const rank = outputShape.length;
    const textureWidth = inputLayout.width;
    const textureHeight = inputLayout.height;

    if (N < 1 || D < 1) {
      throw new Error(`Logical row count N and feature count D must be greater than or equal to 1`);
    }

    if (maxElementPerLogicalRow.shape.length !== 1 || normalizationPerLogicalRow.shape.length !== 1) {
      throw new Error(`Dimensionality of the intermediate results should be 1`);
    }

    if (maxElementPerLogicalRow.shape[0] !== N || normalizationPerLogicalRow.shape[0] !== N) {
      throw new Error(`Shape of the intermediate results should be equal to logical row count`);
    }

    const shaderSource = `
    float process(int[${rank}] indices) {
      int offset = coordsToOffset(TexCoords, ${textureWidth}, ${textureHeight});
      int logical_row_index[1];
      logical_row_index[0] = offset / ${D};

      float norm_factor = _Norm(logical_row_index);

      if(norm_factor == 0.0)
        return 0.0;

      return exp(_A(indices) - _Max(logical_row_index)) / norm_factor;
    }`;
    return {
      inputLayouts: [inputLayout, maxElementPerLogicalRow, normalizationPerLogicalRow],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'Max', 'Norm'],
      shaderSource,
    };
  }

  createComputScaleProgramInfo(
      inferenceHandler: WebGLInferenceHandler, x: Tensor, N: number, D: number, maxElementPerLogicalRow: TextureLayout,
      outputShape: number[]): ProgramInfo {
    const xlayout = inferenceHandler.createTextureLayoutFromShape(x.dims.slice());
    const rank = outputShape.length;
    const textureWidth = xlayout.width;
    const textureHeight = xlayout.height;

    const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
    const shaderSource = `
    float process(int[${rank}] indices) {

      int logical_row_start_offset = indices[0] * ${D};

      float norm_factor = 0.0;
      float max = _Max(indices);
      for(int i=0; i<${D}; ++i)
      {
        norm_factor += exp(getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset + i, ${
        textureWidth}, ${textureHeight}))) - max);
      }

      return norm_factor;
    }`;
    return {
      inputLayouts: [xlayout, maxElementPerLogicalRow],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A', 'Max'],
      shaderSource,
    };
  }

  createComputeMaxProgramInfo(
      inferenceHandler: WebGLInferenceHandler, x: Tensor, N: number, D: number, outputShape: number[]): ProgramInfo {
    const xlayout = inferenceHandler.createTextureLayoutFromShape(x.dims.slice());
    const rank = outputShape.length;
    const textureWidth = xlayout.width;
    const textureHeight = xlayout.height;

    const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
    const shaderSource = `
        float process(int[${rank}] indices) {

          int logical_row_start_offset = indices[0] * ${D};

          float max = getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset, ${textureWidth}, ${
        textureHeight} )));
          for(int i=1; i<${D}; ++i)
          {
            float current = getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset + i, ${
        textureWidth}, ${textureHeight})));
            if(current > max)
              max = current;
          }

          return max;
        }`;
    return {
      inputLayouts: [xlayout],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
    };
  }
  createProgramInfos(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo[] {
    const inputShape = inputs[0].dims.slice();
    const axis = ShapeUtil.normalizeAxis(this.axis, inputShape.length);
    const N = ShapeUtil.sizeToDimension(inputShape, axis);
    const D = ShapeUtil.sizeFromDimension(inputShape, axis);
    const computeMaxProgramInfo = this.createComputeMaxProgramInfo(inferenceHandler, inputs[0], N, D, [N]);
    const computeScaleProgramInfo =
        this.createComputScaleProgramInfo(inferenceHandler, inputs[0], N, D, computeMaxProgramInfo.outputLayout, [N]);
    const softMaxProgramInfo = this.createSoftMaxProgramInfo(
        inferenceHandler, inputs[0], N, D, computeMaxProgramInfo.outputLayout, computeScaleProgramInfo.outputLayout);

    const programInfos: ProgramInfo[] = [computeMaxProgramInfo, computeScaleProgramInfo, softMaxProgramInfo];
    return programInfos;
  }
  createRunDatas(inferenceHandler: WebGLInferenceHandler, programInfos: ProgramInfo[], inputs: Tensor[]): RunData[] {
    const dataType = inputs[0].type;
    const inputTD = inferenceHandler.getOrCreateTextureData(inputs[0], programInfos[0].inputLayouts[0]);
    const runDatas: RunData[] = [];
    runDatas.push({
      inputTextureDatas: [inputTD],
      outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[0].outputLayout, dataType),
      uniformData: {}
    });
    for (let i = 1; i < programInfos.length; ++i) {
      runDatas.push({
        inputTextureDatas: [...runDatas[i - 1].inputTextureDatas, runDatas[i - 1].outputTextureData],
        outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[i].outputLayout, dataType),
        uniformData: {}
      });
    }
    return runDatas;
  }
  protected artifacts: Artifact[];
}

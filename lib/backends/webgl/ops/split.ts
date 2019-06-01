// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Split} from '../../../ops/split';
import {Tensor} from '../../../tensor';
import {SplitUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData} from '../types';

export class WebGLSplit extends Split {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const count = this.getProgramCount(inferenceHandler, inputs);
    if (!this.artifacts) {
      this.artifacts = [];
      for (let i = 0; i < count; ++i) {
        const programInfo = this.createProgramInfo(inferenceHandler, inputs[0], i);
        const artifact = inferenceHandler.session.programManager.build(programInfo);
        this.artifacts.push(artifact);
      }
    }
    const results: Tensor[] = [];

    this.artifacts.forEach(artifact => {
      const rundata = this.createRunData(inferenceHandler, artifact.programInfo, inputs);
      inferenceHandler.session.programManager.run(artifact, rundata);
      results.push(rundata.outputTextureData.tensor);
    });
    return results;
  }
  getProgramCount(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): number {
    const [, offsets] = SplitUtil.splitShape(inputs[0].dims, this.axis, this.split, this.numOutputs);
    return offsets.length;
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, input: Tensor, index: number): ProgramInfo {
    const [shapes, offsets] = SplitUtil.splitShape(input.dims, this.axis, this.split, this.numOutputs);
    const offset = offsets[index];
    const outputShape = shapes[index];
    const rank = outputShape.length;
    const shaderSource = `
      float process(int indices[${rank}]) {
        indices[${this.axis}] += ${offset};
        return _A(indices);
      }`;
    return {
      inputLayouts: [inferenceHandler.getOrCreateTextureLayout(input)],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape),
      samplers: ['A'],
      shaderSource,
    };
  }
  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [inferenceHandler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData:
          inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
  protected artifacts: Artifact[];
}

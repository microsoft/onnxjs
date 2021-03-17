import {Tensor} from '../../../lib/tensor';
import {WebGLInferenceHandler} from './inference-handler';
import {WebGLPack} from './ops/pack';

export function assert(expr: boolean, msg: () => string) {
  if (!expr) {
    throw new Error(typeof msg === 'string' ? msg : msg());
  }
}

// this is un-optimized version of reshape: unpack + create_new_tensor + pack_with_new_shape
// TODO: optimize it.
export function reshape(inferenceHandler: WebGLInferenceHandler, input: Tensor, shape: number[]): Tensor {
  // packing
  const reshapedTensor = new Tensor(shape, input.type, undefined, undefined, input.data);
  const key = `${shape}`;
  let op = inferenceHandler.session.packOpCache.get(key);
  if (!op) {
    op = new WebGLPack();
    inferenceHandler.session.packOpCache.set(key, op);
  }
  let artifact = inferenceHandler.session.programManager.getArtifact(op);
  if (!artifact) {
    const programInfo = op.createProgramInfo(inferenceHandler, [reshapedTensor]);

    artifact = inferenceHandler.session.programManager.build(programInfo);
    inferenceHandler.session.programManager.setArtifact(op, artifact);
  }
  const runData = op.createRunData(inferenceHandler, artifact.programInfo, [reshapedTensor]);
  inferenceHandler.runProgram(artifact, runData);
  return runData.outputTextureData.tensor;
}

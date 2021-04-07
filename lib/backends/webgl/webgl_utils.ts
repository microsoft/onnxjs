import {Tensor} from '../../../lib/tensor';
import {WebGLInferenceHandler} from './inference-handler';
import {WebGLReshapePacked} from './ops/reshape-packed';

export function reshape(inferenceHandler: WebGLInferenceHandler, input: Tensor, shape: number[]): Tensor {
  const op = new WebGLReshapePacked();
  const newShape = new Tensor([shape.length], 'int32');
  for (let i = 0; i < shape.length; i++) {
    newShape.data[i] = shape[i];
  }
  return op.run(inferenceHandler, [input, newShape])[0];
}

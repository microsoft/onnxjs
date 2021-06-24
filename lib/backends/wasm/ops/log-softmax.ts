import {LogSoftmax} from '../../../ops/log-softmax';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmLogSoftmax extends LogSoftmax {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const x = inputs[0];
    const axis = ShapeUtil.normalizeAxis(this.axis, x.dims.length);
    const N = ShapeUtil.sizeToDimension(x.dims, axis);
    const D = ShapeUtil.sizeFromDimension(x.dims, axis);
    const y = new Tensor(x.dims, x.type);
    WasmBinding.getInstance().ccall(
        '_logsoftmax_f32', [x.floatData, 'float32ptr'], [y.floatData, 'float32ptr', 'out'], [N, 'int32'], [D, 'int32']);

    return [y];
  }

  checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'float32') {
      return false;
    }

    return true;
  }
}

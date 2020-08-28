import {Scatter} from '../../../ops/scatter';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuScatter extends Scatter {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = scatter(inputs[0], inputs[1], inputs[2]);
    return [output];
  }
}

export function scatter(data: Tensor, indices: Tensor, updates: Tensor): Tensor {
  const datadims = data.dims;
  const indicedims = indices.dims;
  const updatedims = updates.dims;
  const datanew = new Tensor(datadims, data.type);
  const Y = datanew.data;
  const X = updates.data;
  let flatIndex = 0;
  let updateFlatIndex = 0;
  for (let i = 0; i < datadims[0]; ++i) {
    for (let j = 0; j < indicedims[1]; ++j) {
      flatIndex = i * datadims[1] + (indices.data[j] as number);
      updateFlatIndex = i * updatedims[1] + j;
      Y[flatIndex] = X[updateFlatIndex];
    }
  }
  return new Tensor(datadims, data.type, undefined, undefined, Y);
}

import {Attribute} from '../../attribute';
import {InferenceHandler} from '../../backend';
import {Operator} from '../../operators';
import {Tensor} from '../../tensor';
import {Graph} from '../../graph';
import {ShapeUtil} from '../../util';

export class NNSubgraph implements Operator {

  constructor (private _onnxNode: Graph.Node) {
    this._operandIndex = 0;
    this._inputsMapping = [];
    this._outputsMapping = [];
    this._nnOperands = [];
    this._operations = [];
    this._tensorTypes = [];
  }

  async run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {

    if (!this._execution) {
      // warm up
      await this.createCompiledModel(inputs);
    }

    // bind input tensors at runtime
    this._inputsMapping.forEach((input) => {
      // get runtime input tensor
      const {nnInputIndex, onnxInputIndex} = input;
      const tensor = inputs[onnxInputIndex];

      // reorder tensor
      tensor.toNHWC();

      // set inputs
      this._execution.setInput(nnInputIndex!, tensor.data as NNTensorType);

      // recover inputs
      tensor.toNCHW();
    });

    // run submodel
    await this._execution.startCompute();

    // create tensor from outputs
    const outputId = this._outputsMapping[0].nnTensorId;
    const outputDims = this._getTensorTypeById(outputId).dimensions!;
    const y = new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC');
    const nnOutput = this._nnOperands[outputId];
    y.floatData.set(nnOutput);
    // console.log('Conv');
    // console.log(y.toNCHW().data)
    return [y.toNCHW()];
  };

  initialize(attributes: Attribute): void {}

  checkInputs(inputs: Tensor[]): boolean { return true; }

  async createCompiledModel(inputTensors: Tensor[]) {
    this._nn = (navigator as NNNavigator).ml.getNeuralNetworkContext();
    this._model = await this._nn.createModel({backend: 'WASM'});
    this._addOpsAndParams(inputTensors);
    this._addInputsOutputs();
    await this._model.finish();

    this._compilation = await this._model.createCompilation();
    this._compilation.setPreference(this._nn.PREFER_FAST_SINGLE_ANSWER);
    await this._compilation.finish();

    this._execution = await this._compilation.createExecution();
    // bind output tensors at compile time
    this._outputsMapping.forEach((output) => {
      const {nnOutputIndex, nnTensorId} = output;
      const operand = this._nnOperands[nnTensorId];
      this._execution.setOutput(nnOutputIndex!, operand);
    });
  }

  _addInputsOutputs() {
    const modelInputs: number[] = [];
    const modelOutputs: number[] = [];
    this._inputsMapping.forEach((input, index) => {
      input.nnInputIndex = index;
      modelInputs.push(input.nnTensorId);
    });
    this._outputsMapping.forEach((output, index) => {
      output.nnOutputIndex = index;
      modelOutputs.push(output.nnTensorId);
    });
    this._model.identifyInputsAndOutputs(modelInputs, modelOutputs);
  }

  _addOpsAndParams(inputTensors: Tensor[]) {

    // reorder to NHWC
    inputTensors.forEach((t) => t.toNHWC());

    let opType = -1;
    let inputs = [];
    let outputs = [];

    switch(this._onnxNode.opType) {
      case 'Conv': {
        const input = inputTensors[0];
        const convFilter = inputTensors[1];
        const convBias = inputTensors[2];

        const inputId = this._addTensorFloat32(input.data as Float32Array, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});

        const attributes = this._onnxNode.attributes;
        const nGroups = attributes.getInt('group', 1);
        const dims = convFilter.dims;
        const nChannels = dims[0];
        const convFilterId = this._addTensorFloat32(convFilter.floatData, convFilter.dims);
        const convBiasId = typeof convBias !== 'undefined' ? // optional bias
          this._addTensorFloat32(convBias.floatData, convBias.dims):
          this._addTensorFloat32(new Float32Array(nChannels).fill(0), [nChannels]);

        inputs.push(inputId);
        inputs.push(convFilterId);
        inputs.push(convBiasId);

        const kernelShape = attributes.getInts('kernel_shape', []);
        if (!kernelShape || kernelShape.length !== 2) {
          throw new Error('Invalid kernelShape');
        }
        const kernelHeight = kernelShape[0];
        const kernelWidth = kernelShape[1];

        const pads = attributes.getInts('pads', [0, 0, 0, 0]);
        if (pads.length !== 4) {
          throw new Error('Invalid pads');
        }
        const paddingHeightBegin = pads[0];
        const paddingWidthBegin = pads[1];
        const paddingHeightEnd = pads[2];
        const paddingWidthEnd = pads[3];
        inputs.push(this._addScalarInt32(paddingWidthBegin));
        inputs.push(this._addScalarInt32(paddingWidthEnd));
        inputs.push(this._addScalarInt32(paddingHeightBegin));
        inputs.push(this._addScalarInt32(paddingHeightEnd));

        const strides = attributes.getInts('strides', [1, 1]);
        if (!strides || strides.length !== 2) {
          throw new Error('Invalid strides');
        }
        const strideY = strides[0];
        const strideX = strides[1];
        inputs.push(this._addScalarInt32(strideX));
        inputs.push(this._addScalarInt32(strideY));

        // reshape kernel for depthwise conv
        const [batch, inputHeight, inputWidth, inputChannels] = input.dims;
        let isDepthWiseConv = false;
        if (nGroups > 1) {
          if (nGroups !== inputChannels) {
            throw new Error('Group convolution is not supported.');
          } else {
            isDepthWiseConv = true;
            console.log(`  groups: ${nGroups} (depthwise convolution)`);
            let nhwc = convFilter.data as Float32Array;
            // NHWC -> CHWN where C === 1
            let chwnData = new Float32Array(nhwc.length);
            const N = dims[0];
            const H = dims[1];
            const W = dims[2];
            for (let n = 0; n < N; ++n) {
              for (let h = 0; h < H; ++h) {
                for (let w = 0; w < W; ++w) {
                  chwnData[h*W*N + w*N + n] = nhwc[n*H*W + h*W + w];
                }
              }
            }

            this._setOperandValue(convFilterId, chwnData);
            const convFilterType = this._getTensorTypeById(convFilterId);
            convFilterType.dimensions![0] = 1;
            convFilterType.dimensions![3] = nGroups;

            // set multiplier to 1, not used in onnx model
            inputs.push(this._addScalarInt32(1));
          }
        }

        inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

        // Add outputs
        const outputHeight = Math.floor((inputHeight - kernelHeight + paddingHeightBegin+paddingHeightEnd)/strideY + 1);
        const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
        const outputChannels = isDepthWiseConv ? nGroups : nChannels;
        const outputDims = [batch, outputHeight, outputWidth, outputChannels];
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        opType = isDepthWiseConv ? this._nn.DEPTHWISE_CONV_2D : this._nn.CONV_2D;
      } break;
    }

    this._addOperation(opType, inputs, outputs);

    // write back all cached operands and operations
    for (const type of this._tensorTypes) {
      this._model.addOperand(type);
    }
    for (const [index, value] of Object.entries(this._nnOperands)) {
      this._model.setOperandValue(parseInt(index), value);
    }
    for (const [opCode, inputs, outputs] of this._operations) {
      this._model.addOperation(opCode, inputs, outputs);
    }

    // recover to NCHW
    inputTensors.forEach((t) => t.toNCHW());

  }

  _getOperandValue(id: number) {
    const data = this._nnOperands[id]
    if (!data) {
      throw new Error('No tensor data');
    } else {
      return data;
    }
  }

  _setOperandValue(index: number, value: NNTensorType) {
    // Cache operand value. It could be modified later: BN fusion/Unsqueeze
    this._nnOperands[index] = value;
  }

  _getTensorTypeById(index: number) {
    return this._tensorTypes[index];
  }

  _addOperand(type: OperandOptions, value?: NNTensorType) {
    let index = this._operandIndex++;
    // Cache operand type. It could be modified later: Depthwise Conv
    this._tensorTypes.push(type);
    if (typeof value !== 'undefined') {
      this._setOperandValue(index, value);
    }
    return index;
  }

  _addOperation(opCode: number, inputs: number[], outputs: number[]) {
    this._operations.push([opCode, inputs, outputs]);
  }

  _addScalarInt32(value: number) {
    return this._addOperand({type: this._nn.INT32}, new Int32Array([value]));
  }

  _addScalarFloat32(value: number) {
    return this._addOperand({type: this._nn.FLOAT32}, new Float32Array([value]));
  }

  _addTensorFloat32(tensor: Tensor.FloatType, dims: number[]) {
    return this._addOperand({
      type: this._nn.TENSOR_FLOAT32,
      dimensions: dims
    }, new Float32Array(tensor));
  }

  private _operandIndex: number;
  private _inputsMapping: Array<{nnInputIndex?: number, onnxInputIndex: number, nnTensorId: number}>;
  private _outputsMapping: Array<{nnOutputIndex?: number, nnTensorId: number}>;
  private _nnOperands: NNTensorType[];
  private _operations: any[];
  private _tensorTypes: OperandOptions[];
  private _nn: NeuralNetworkContext;
  private _model: Model;
  private _compilation: Compilation;
  private _execution: Execution;
}

import {Attribute} from '../../attribute';
import {InferenceHandler} from '../../backend';
import {Operator} from '../../operators';
import {Tensor} from '../../tensor';
import {Graph} from '../../graph';
import {ShapeUtil} from '../../util';
import {Profiler} from '../../instrument';

export class NNSubgraph implements Operator {

  constructor (
      private _onnxNode: Graph.Node,
      private enablePseudoReorder = false,
      private profiler: Readonly<Profiler>) {
    this._operandIndex = 0;
    this._inputsMapping = [];
    this._outputsMapping = [];
    this._nnOperands = [];
    this._operations = [];
    this._tensorTypes = [];
    this._subgraphName = this._onnxNode.opType;
    this._outputTensors = [];
  }

  async run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {

    if (!this._execution) {
      // compile at first run
      await this.createCompiledModel(inputs);
    }

    // bind input tensors at runtime
    this._inputsMapping.forEach((input) => {
      // get runtime input tensor
      const {nnInputIndex, onnxInputIndex} = input;
      const tensor = inputs[onnxInputIndex];

      // reorder tensor
      tensor.toNHWC(this.enablePseudoReorder);

      this.profiler.event('backend', 'WebNN.Execution.setInput', () => {
        // set inputs
        this._execution.setInput(nnInputIndex!, tensor.data as NNTensorType);
      });

      // recover inputs
      tensor.toNCHW(this.enablePseudoReorder);
    });

    this._outputTensors.forEach((output) => {
      output.toNHWC(this.enablePseudoReorder);
    });

    // run submodel
    await this.profiler.event('backend', 'WebNN.Execution.startCompute', async () => {
      await this._execution.startCompute();
    });

    this._outputTensors.forEach((output) => {
      output.toNCHW(this.enablePseudoReorder);
    });

    return this._outputTensors;
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

      const outputDims = this._getTensorTypeById(nnTensorId).dimensions!;
      const tensor = new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', operand);
      this._outputTensors.push(tensor);
      tensor.toNCHW(this.enablePseudoReorder);
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

    const attributes = this._onnxNode.attributes;

    switch(this._onnxNode.opType) {
      case 'Conv': {
        const input = inputTensors[0];
        const convFilter = inputTensors[1];
        const convBias = inputTensors[2];

        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});

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
            let nhwc = convFilter.floatData;
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
      case 'BatchNormalization': {
        // Add inputs
        const input = inputTensors[0];
        const scale = inputTensors[1];
        const bnBias = inputTensors[2];
        const mean = inputTensors[3];
        const variance = inputTensors[4];
        const epsilon = attributes.getFloat('epsilon', 1e-5);

        const scaleTensor = scale.floatData;
        const meanTensor = mean.floatData;
        const varTensor = variance.floatData;
        const bnBiasTensor = bnBias.floatData;

        // Conv with identity kernel
        const nChannels = input.dims[3];
        const convFilterTensor = new Float32Array(nChannels * nChannels).fill(0);
        const convBiasTensor = new Float32Array(nChannels).fill(0);
        const convFilterDims = [nChannels, 1, 1, nChannels];
        const convBiasDims = [nChannels];

        for (let c = 0; c < nChannels; c++) {
          const w = scaleTensor[c] / Math.sqrt(varTensor[c] + epsilon);
          convFilterTensor[c * nChannels + c] = w;
          convBiasTensor[c] = bnBiasTensor[c] - w * meanTensor[c];
        }

        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});

        inputs.push(inputId);
        inputs.push(this._addTensorFloat32(convFilterTensor, convFilterDims));
        inputs.push(this._addTensorFloat32(convBiasTensor, convBiasDims));
        // paddings
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        // strides
        inputs.push(this._addScalarInt32(1));
        inputs.push(this._addScalarInt32(1));

        inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

        // Add outputs
        const outputDims = Array.from(input.dims);
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        opType = this._nn.CONV_2D;
      } break;
      case 'Relu': {
        // Add inputs
        const input = inputTensors[0];

        // Conv with identity kernel
        const nChannels = input.dims[3];
        const convFilterTensor = new Float32Array(nChannels * nChannels).fill(0);
        const convBiasTensor = new Float32Array(nChannels).fill(0);
        const convFilterDims = [nChannels, 1, 1, nChannels];
        const convBiasDims = [nChannels];

        for (let c = 0; c < nChannels; c++) {
          convFilterTensor[c * nChannels + c] = 1;
        }

        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});

        inputs.push(inputId);
        inputs.push(this._addTensorFloat32(convFilterTensor, convFilterDims));
        inputs.push(this._addTensorFloat32(convBiasTensor, convBiasDims));
        // paddings
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        // strides
        inputs.push(this._addScalarInt32(1));
        inputs.push(this._addScalarInt32(1));
        inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));

        // Add outputs
        const outputDims = Array.from(input.dims);
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        opType = this._nn.CONV_2D;
      } break;
      case 'Mul':
      case 'Sum':
      case 'Add': {

        if (this._onnxNode.opType === 'Sum' && inputTensors.length !== 2) {
          throw new Error(`Only support Sum with two inputs`);
        }

        const in1 = inputTensors[0];
        const in2 = inputTensors[1];
        const in1Id = this._addTensorFloat32(in1.floatData, in1.dims);
        const in2Id = this._addTensorFloat32(in2.floatData, in2.dims);
        this._inputsMapping.push({nnTensorId: in1Id, nnInputIndex: 0, onnxInputIndex: 0});
        this._inputsMapping.push({nnTensorId: in2Id, nnInputIndex: 1, onnxInputIndex: 1});
        inputs.push(in1Id);
        inputs.push(in2Id);

        inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

        // Add outputs
        const in1Dims = in1.dims;
        const in2Dims = in2.dims;

        // Compatible dims (multidirectional broadcasting)
        const outputDims = new Array(Math.max(in1Dims.length, in2Dims.length));
        for (let i = in1Dims.length - 1, j = in2Dims.length - 1, k = outputDims.length - 1; k >= 0;) {
          let dim1 = in1Dims[i--] || 1;
          let dim2 = in2Dims[j--] || 1;
          if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1)
            throw new Error(`Dimensions of ${in1} and ${in2} are not compatible`);
          outputDims[k--] = Math.max(dim1, dim2);
        }

        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        if (this._onnxNode.opType === 'Add' || this._onnxNode.opType === 'Sum') {
          opType = this._nn.ADD;
        } else if (this._onnxNode.opType === 'Mul') {
          opType = this._nn.MUL;
        }
      } break;
      case 'Gemm': {
        // Add inputs
        const input = inputTensors[0];    // A
        const weights = inputTensors[1];  // B
        const bias = inputTensors[2];     // C

        const alpha  = attributes.getInt('alpha',  1);
        const beta   = attributes.getInt('beta',   1);
        const transA = attributes.getInt('transA', 0);
        const transB = attributes.getInt('transB', 0);

        if (alpha !== 1 || beta !== 1 || transA || !transB) {
          throw new Error('Only support fc-like Gemm oprations, i.e. alpha == beta == 1 && !transA && transB');
        }

        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});
        const weightsId = this._addTensorFloat32(weights.floatData, weights.dims);
        const biasId = this._addTensorFloat32(bias.floatData, bias.dims);

        inputs.push(inputId);
        inputs.push(weightsId);
        inputs.push(biasId);
        inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));

        // Add outputs
        const nUnits = weights.dims[0];
        const batchSize = ShapeUtil.size(input.dims) / weights.dims[1];
        const outputDims = [batchSize, nUnits];
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        opType = this._nn.FULLY_CONNECTED;
      } break;
      case 'AveragePool':
      case 'MaxPool': {
        const input = inputTensors[0];
        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});
        inputs.push(inputId);

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

        const kernelShape = attributes.getInts('kernel_shape', []);
        if (!kernelShape || kernelShape.length !== 2) {
          throw new Error('Invalid kernelShape');
        }
        const kernelHeight = kernelShape[0];
        const kernelWidth = kernelShape[1];
        inputs.push(this._addScalarInt32(kernelWidth));
        inputs.push(this._addScalarInt32(kernelHeight));
        inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

        const [batch, inputHeight, inputWidth, inputChannels] = input.dims;
        const outputHeight = Math.floor((inputHeight - kernelHeight + paddingHeightBegin + paddingHeightEnd)/strideY+1);
        const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
        const outputDims = [batch, outputHeight, outputWidth, inputChannels];
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        if (this._onnxNode.opType === 'MaxPool') {
          opType = this._nn.MAX_POOL_2D;
        } else if (this._onnxNode.opType === 'AveragePool') {
          opType = this._nn.AVERAGE_POOL_2D;
        }
      } break;
      case 'Reshape': {
        const input = inputTensors[0];
        const shape = inputTensors[1];
        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});
        const shapeId = this._addTensorInt32(shape.integerData, shape.dims);
        inputs.push(inputId);
        inputs.push(shapeId);

        const inputDims = input.dims;
        let outputDims = Array.from(shape.integerData);
        // dim == 0 means actual dim is unchanged, i.e. taken from the inputDim
        outputDims = outputDims.map((d, i) => d === 0 ? inputDims[i] : d);
        // At most one dimension of the new shape can be -1
        const minusOneCnt = outputDims.filter(x => x === -1).length;
        if (minusOneCnt === 1) {
          const nonAdaptDim = outputDims.filter(x => x !== -1);
          const adaptDimIdx = outputDims.indexOf(-1);
          outputDims[adaptDimIdx] = ShapeUtil.size(inputDims) / ShapeUtil.size(nonAdaptDim);
        } else if (minusOneCnt !== 0) {
          throw new Error(`Invalid shape ${outputDims}`);
        }
        this._setOperandValue(shapeId, new Int32Array(outputDims));

        // Add outputs
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})

        outputs.push(outputId);
        opType = this._nn.RESHAPE;
      } break;
      case 'Concat': {
        for (let i = 0; i < inputTensors.length; ++i) {
          const input = inputTensors[i];
          const inputId = this._addTensorFloat32(input.floatData, input.dims);
          inputs.push(inputId);
          this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: i, onnxInputIndex: i});
        }

        const axis = attributes.getInt('axis');
        if (axis && axis !== 1) {
          throw new Error(`Invalid axis ${axis}`);
        }
        // C axis is 3 in NHWC layout
        const concatAxis = 3;
        inputs.push(this._addScalarInt32(concatAxis));

        // Add output
        let outputDims = Array.from(inputTensors[0].dims);
        for (let i = 1; i < inputTensors.length; ++i) {
          outputDims[concatAxis] += inputTensors[i].dims[concatAxis];
        }
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        opType = this._nn.CONCATENATION;
      } break;
      case 'GlobalAveragePool': {
        const input = inputTensors[0];
        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});
        inputs.push(inputId);
        // paddings
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        inputs.push(this._addScalarInt32(0));
        // strides
        inputs.push(this._addScalarInt32(1));
        inputs.push(this._addScalarInt32(1));
        // filters
        const [batch, inputHeight, inputWidth, inputChannels] = input.dims;
        inputs.push(this._addScalarInt32(inputWidth));
        inputs.push(this._addScalarInt32(inputHeight));
        inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

        // Add outputs
        const outputHeight = 1;
        const outputWidth = 1;
        const outputDims = [batch, outputHeight, outputWidth, inputChannels];
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0});
        outputs.push(outputId);

        opType = this._nn.AVERAGE_POOL_2D;
      } break;
      case 'Softmax': {
        const input = inputTensors[0];
        const inputId = this._addTensorFloat32(input.floatData, input.dims);
        this._inputsMapping.push({nnTensorId: inputId, nnInputIndex: 0, onnxInputIndex: 0});
        inputs.push(inputId);
        // Set beta to 1.0
        inputs.push(this._addScalarFloat32(1.0));

        const outputDims = input.dims;
        const outputData = new Float32Array(ShapeUtil.size(outputDims));
        const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
        this._outputsMapping.push({nnTensorId: outputId, nnOutputIndex: 0})
        outputs.push(outputId);

        opType = this._nn.SOFTMAX;
      } break;
      default: {
        throw new Error(`${this._onnxNode.opType} is not supported.}`);
      }
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

  _addTensorInt32(tensor: Tensor.IntegerType, dims: number[]) {
    return this._addOperand({
      type: this._nn.TENSOR_INT32,
      dimensions: dims
    }, new Int32Array(tensor));
  }

  // @ts-ignore
  private _subgraphName: string;
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
  private _outputTensors: Tensor[];
}

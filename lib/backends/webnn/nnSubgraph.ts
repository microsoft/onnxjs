import {Attribute} from '../../attribute';
import {InferenceHandler} from '../../backend';
import {Operator} from '../../operators';
import {Tensor} from '../../tensor';
import {NNSubgraphNode} from '../../graph';
import {ShapeUtil} from '../../util';
import {Profiler} from '../../instrument';

export class NNSubgraph implements Operator {

  constructor (
      private subgraph: NNSubgraphNode,
      initializers: Map<number, Tensor>,
      private enablePseudoReorder = false,
      private profiler: Readonly<Profiler>) {
    this._operandIndex = 0;
    this._nnOperands = [];
    this._operations = [];
    this._tensorTypes = [];
    this._tensorData = [];
    initializers.forEach((tensor, i) => {
      this._tensorData[i] = {tensor: tensor};
    });
  }

  async run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {

    if (!this._execution) {
      // compile at first run
      await this.createCompiledModel(inputs);
    }

    // bind input tensors at runtime
    inputs.forEach((tensor, i) => {
      // reorder input tensors
      tensor.toNHWC(this.enablePseudoReorder);

      this.profiler.event('backend', 'WebNN.Execution.setInput', () => {
        this._execution.setInput(i, tensor.data as NNTensorType);
      });

      // recover input tensors
      tensor.toNCHW(this.enablePseudoReorder);
    });

    const outputTensors = this.subgraph.outputs.map((tensorId) => this._getTensorByOnnxId(tensorId));
    // reorder output tensors
    outputTensors.forEach((tensor) => tensor.toNHWC(this.enablePseudoReorder));

    // run submodel
    await this.profiler.event('backend', 'WebNN.Execution.startCompute', async () => {
      await this._execution.startCompute();
    });

    // recover output tensors
    outputTensors.forEach((tensor) => tensor.toNCHW(this.enablePseudoReorder));
    return outputTensors;
  };

  initialize(attributes: Attribute): void {}

  checkInputs(inputs: Tensor[]): boolean { return true; }

  async createCompiledModel(graphInputTensors: Tensor[]) {
    this._nn = (navigator as NNNavigator).ml.getNeuralNetworkContext();
    this._model = await this._nn.createModel({backend: 'WASM'});

    graphInputTensors.forEach((tensor, i) => {
      tensor.toNHWC();
      const tensorId = this._addTensorFloat32(tensor.floatData as Float32Array, tensor.dims);
      this._tensorData[this.subgraph.inputs[i]] = {tensor: tensor, nnTensorId: tensorId};
    });

    // reorder to NHWC
    this._tensorData.forEach(({tensor}) => {
      tensor.toNHWC();
    });

    this._addOpsAndParams();
    this._addInputsOutputs();
    await this._model.finish();

    this._compilation = await this._model.createCompilation();
    this._compilation.setPreference(this._nn.PREFER_FAST_SINGLE_ANSWER);
    await this._compilation.finish();

    this._execution = await this._compilation.createExecution();
    // bind output tensors at compile time
    this.subgraph.outputs.forEach((tensorId, i) => {
      const tensor = this._getTensorByOnnxId(tensorId);
      // TODO: eliminate type casting
      this._execution.setOutput(i, tensor.floatData as NNTensorType);
    });

    // recover to NCHW
    this._tensorData.forEach(({tensor}) => {
      tensor.toNCHW();
    });
  }

  _addInputsOutputs() {
    const modelInputs = this.subgraph.inputs.map((onnxTensorId) => this._tensorData[onnxTensorId].nnTensorId!);
    const modelOutputs = this.subgraph.outputs.map((onnxTensorId) => this._tensorData[onnxTensorId].nnTensorId!);
    this._model.identifyInputsAndOutputs(modelInputs, modelOutputs);
  }

  _addOpsAndParams() {

    for (let i = 0; i < this.subgraph.nodes.length; i++) {

      let opType: number = -1;
      let inputs: number[] = [];
      let outputs: number[] = [];

      let node = this.subgraph.nodes[i];
      let attributes = node.attributes;

      switch(node.opType) {
        case 'Conv': {
          const input = this._getTensorByOnnxId(node.inputs[0]);
          const convFilter = this._getTensorByOnnxId(node.inputs[1]);
          const convBias = node.inputs[2] !== undefined ? this._getTensorByOnnxId(node.inputs[2]) : undefined;

          const nGroups = attributes.getInt('group', 1);
          const dims = convFilter.dims;
          const nChannels = dims[0];
          const convFilterId = this._addTensorFloat32(convFilter.floatData as Float32Array, convFilter.dims);
          const convBiasId = convBias !== undefined ? // optional bias
            this._addTensorFloat32(convBias.floatData as Float32Array, convBias.dims):
            this._addTensorFloat32(new Float32Array(nChannels).fill(0), [nChannels]);

          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
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

          let nextNode = this.subgraph.nodes[i + 1];
          // fuse batch norm preceded by a conv
          if (nextNode &&
              nextNode.opType === 'BatchNormalization' &&
              node.outputs[0] === nextNode.inputs[0]) {
            const bnNode = nextNode;
            const scale = this._getTensorByOnnxId(bnNode.inputs[1]);
            const bnBias = this._getTensorByOnnxId(bnNode.inputs[2]);
            const mean = this._getTensorByOnnxId(bnNode.inputs[3]);
            const variance = this._getTensorByOnnxId(bnNode.inputs[4]);
            const epsilon = bnNode.attributes.getFloat('epsilon', 1e-5);

            const scaleTensor = scale.floatData;
            const meanTensor = mean.floatData;
            const varTensor = variance.floatData;
            const bnBiasTensor = bnBias.floatData;
            const convFilterTensor = convFilter.floatData;
            const convBiasTensor = this._nnOperands[convBiasId];

            const nPixels = ShapeUtil.size(dims.slice(1));
            for (let c = 0; c < nChannels; c++) {
              const w = scaleTensor[c] / Math.sqrt(varTensor[c] + epsilon);
              convBiasTensor[c] = bnBiasTensor[c] + (convBiasTensor[c] - meanTensor[c]) * w;
              for (let p = c * nPixels; p < (c+1) * nPixels; p++) {
                convFilterTensor[p] *= w;
              }
            }

            i++;
            node = nextNode;
            nextNode = this.subgraph.nodes[i + 1];
          }

          if (nextNode &&
              nextNode.opType === 'Relu' &&
              node.outputs[0] === nextNode.inputs[0]) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));
            i++;
            node = nextNode;
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

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
              inputs.splice(9, 0, this._addScalarInt32(1));
            }
          }

          // Add outputs
          const outputHeight = Math.floor((inputHeight-kernelHeight + paddingHeightBegin+paddingHeightEnd)/strideY + 1);
          const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
          const outputChannels = isDepthWiseConv ? nGroups : nChannels;
          const outputDims = [batch, outputHeight, outputWidth, outputChannels];
          const outputData = new Float32Array(ShapeUtil.size(outputDims));
          const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = isDepthWiseConv ? this._nn.DEPTHWISE_CONV_2D : this._nn.CONV_2D;
        } break;
        case 'BatchNormalization': {
          // Add inputs
          const input = this._getTensorByOnnxId(node.inputs[0]);
          const scale = this._getTensorByOnnxId(node.inputs[1]);
          const bnBias = this._getTensorByOnnxId(node.inputs[2]);
          const mean = this._getTensorByOnnxId(node.inputs[3]);
          const variance = this._getTensorByOnnxId(node.inputs[4]);
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

          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
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

          let nextNode = this.subgraph.nodes[i + 1];
          if (nextNode &&
              nextNode.opType === 'Relu' &&
              node.outputs[0] === nextNode.inputs[0]) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));
            i++;
            node = nextNode;
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const outputDims = Array.from(input.dims);
          const outputData = new Float32Array(ShapeUtil.size(outputDims));
          const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = this._nn.CONV_2D;
        } break;
        case 'Relu': {
          // Add inputs
          const input = this._getTensorByOnnxId(node.inputs[0]);

          // Conv with identity kernel
          const nChannels = input.dims[3];
          const convFilterTensor = new Float32Array(nChannels * nChannels).fill(0);
          const convBiasTensor = new Float32Array(nChannels).fill(0);
          const convFilterDims = [nChannels, 1, 1, nChannels];
          const convBiasDims = [nChannels];

          for (let c = 0; c < nChannels; c++) {
            convFilterTensor[c * nChannels + c] = 1;
          }

          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
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
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = this._nn.CONV_2D;
        } break;
        case 'Mul':
        case 'Sum':
        case 'Add': {

          if (node.opType === 'Sum' && node.inputs.length !== 2) {
            throw new Error(`Only support Sum with two inputs`);
          }
          const in1 = this._getTensorByOnnxId(node.inputs[0]);
          const in2 = this._getTensorByOnnxId(node.inputs[1]);
          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
          inputs.push(this._tensorData[node.inputs[1]].nnTensorId!);
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
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          if (node.opType === 'Add' || node.opType === 'Sum') {
            opType = this._nn.ADD;
          } else if (node.opType === 'Mul') {
            opType = this._nn.MUL;
          }
        } break;
        case 'Gemm': {
          // Add inputs
          const input = this._getTensorByOnnxId(node.inputs[0]);    // A
          const weights = this._getTensorByOnnxId(node.inputs[1]);  // B
          const bias = this._getTensorByOnnxId(node.inputs[2]);     // C

          const alpha  = attributes.getFloat('alpha', 1.0);
          const beta   = attributes.getFloat('beta', 1.0);
          const transA = attributes.getInt('transA', 0);
          const transB = attributes.getInt('transB', 0);

          if (alpha !== 1 || beta !== 1 || transA || !transB) {
            throw new Error('Only support fc-like Gemm oprations, i.e. alpha == beta == 1 && !transA && transB');
          }

          const weightsId = this._addTensorFloat32(weights.floatData as Float32Array, weights.dims);
          const biasId = this._addTensorFloat32(bias.floatData as Float32Array, bias.dims);

          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
          inputs.push(weightsId);
          inputs.push(biasId);
          inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));

          // Add outputs
          const nUnits = weights.dims[0];
          const batchSize = ShapeUtil.size(input.dims) / weights.dims[1];
          const outputDims = [batchSize, nUnits];
          const outputData = new Float32Array(ShapeUtil.size(outputDims));
          const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = this._nn.FULLY_CONNECTED;
        } break;
        case 'AveragePool':
        case 'MaxPool': {
          const input = this._getTensorByOnnxId(node.inputs[0]);
          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);

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
          const outputHeight =
              Math.floor((inputHeight - kernelHeight + paddingHeightBegin + paddingHeightEnd)/strideY+1);
          const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
          const outputDims = [batch, outputHeight, outputWidth, inputChannels];
          const outputData = new Float32Array(ShapeUtil.size(outputDims));
          const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          if (node.opType === 'MaxPool') {
            opType = this._nn.MAX_POOL_2D;
          } else if (node.opType === 'AveragePool') {
            opType = this._nn.AVERAGE_POOL_2D;
          }
        } break;
        case 'Reshape': {
          const input = this._getTensorByOnnxId(node.inputs[0]);
          const shape = this._getTensorByOnnxId(node.inputs[1]);
          const shapeId = this._addTensorInt32(shape.integerData as Int32Array, shape.dims);
          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
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
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = this._nn.RESHAPE;
        } break;
        case 'Concat': {
          for (let i = 0; i < node.inputs.length; ++i) {
            inputs.push(this._tensorData[node.inputs[i]].nnTensorId!);
          }

          const axis = attributes.getInt('axis');
          if (axis && axis !== 1) {
            throw new Error(`Invalid axis ${axis}`);
          }
          // C axis is 3 in NHWC layout
          const concatAxis = 3;
          inputs.push(this._addScalarInt32(concatAxis));

          // Add output
          let outputDims = Array.from(this._getTensorByOnnxId(node.inputs[0]).dims);
          for (let i = 1; i < node.inputs.length; ++i) {
            outputDims[concatAxis] += this._getTensorByOnnxId(node.inputs[i]).dims[concatAxis];
          }
          const outputData = new Float32Array(ShapeUtil.size(outputDims));
          const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = this._nn.CONCATENATION;
        } break;
        case 'GlobalAveragePool': {
          const input = this._getTensorByOnnxId(node.inputs[0]);
          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
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
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = this._nn.AVERAGE_POOL_2D;
        } break;
        case 'Softmax': {
          const input = this._getTensorByOnnxId(node.inputs[0]);
          inputs.push(this._tensorData[node.inputs[0]].nnTensorId!);
          // Set beta to 1.0
          inputs.push(this._addScalarFloat32(1.0));

          const outputDims = input.dims;
          const outputData = new Float32Array(ShapeUtil.size(outputDims));
          outputData[0] = 1;
          const outputId = this._addTensorFloat32(outputData, outputDims);  // allocate output placehoder
          const outputTensor =
              new Tensor(outputDims, 'float32', undefined, undefined, undefined, undefined, 'NHWC', outputData);
          this._tensorData[node.outputs[0]] = {tensor: outputTensor, nnTensorId: outputId};
          outputs.push(outputId);

          opType = this._nn.SOFTMAX;
        } break;
        default: {
          throw new Error(`${node.opType} is not supported.}`);
        }
      }

      this._addOperation(opType, inputs, outputs);
    }

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
  }

  _getTensorByOnnxId(id: number) {
    const data = this._tensorData[id];
    if (data === undefined) {
      throw new Error(`Cannot find tensor ${id}`);
    }
    return data.tensor;
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

  _addTensorFloat32(tensor: Float32Array, dims: number[]) {
    return this._addOperand({
      type: this._nn.TENSOR_FLOAT32,
      dimensions: dims
    }, tensor);
  }

  _addTensorInt32(tensor: Int32Array, dims: number[]) {
    return this._addOperand({
      type: this._nn.TENSOR_INT32,
      dimensions: dims
    }, tensor);
  }

  private _operandIndex: number;
  private _nnOperands: NNTensorType[];
  private _operations: any[];
  private _tensorTypes: OperandOptions[];
  private _tensorData: {tensor: Tensor, nnTensorId?: number}[];
  private _nn: NeuralNetworkContext;
  private _model: Model;
  private _compilation: Compilation;
  private _execution: Execution;
}

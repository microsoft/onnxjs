interface NNNavigator extends Navigator {
  ml: {
    isPolyfill: boolean | undefined;
    getNeuralNetworkContext: () => NeuralNetworkContext;
  }
}

interface NeuralNetworkContext {
  // Operand types.
  FLOAT32: number;
  INT32: number;
  UINT32: number;
  TENSOR_FLOAT32: number;
  TENSOR_INT32: number;
  TENSOR_QUANT8_ASYMM: number;

  // Operation types.
  ADD: number;
  AVERAGE_POOL_2D: number;
  CONCATENATION: number;
  CONV_2D: number;
  DEPTHWISE_CONV_2D: number;
  DEPTH_TO_SPACE: number;
  DEQUANTIZE: number;
  EMBEDDING_LOOKUP: number;
  FLOOR: number;
  FULLY_CONNECTED: number;
  HASHTABLE_LOOKUP: number;
  L2_NORMALIZATION: number;
  L2_POOL_2D: number;
  LOCAL_RESPONSE_NORMALIZATION: number;
  LOGISTIC: number;
  LSH_PROJECTION: number;
  LSTM: number;
  MAX_POOL_2D: number;
  MUL: number;
  RELU: number;
  RELU1: number;
  RELU6: number;
  RESHAPE: number;
  RESIZE_BILINEAR: number;
  RNN: number;
  SOFTMAX: number;
  SPACE_TO_DEPTH: number;
  SVDF: number;
  TANH: number;
  ATROUS_CONV_2D: number;
  ATROUS_DEPTHWISE_CONV_2D: number;

  // Fused activation function types.
  FUSED_NONE: number;
  FUSED_RELU: number;
  FUSED_RELU1: number;
  FUSED_RELU6: number;

  // Implicit padding algorithms.
  PADDING_SAME: number;
  PADDING_VALID: number;

  // Execution preferences.
  PREFER_LOW_POWER: number;
  PREFER_FAST_SINGLE_ANSWER: number;
  PREFER_SUSTAINED_SPEED: number;

  createModel: (options?: {backend: string}) => Promise<Model>;
}

interface nnOperandTypeMap {
  // bool: number;
  [float32: string]: number;
  float64: number;
  // string: number;
  int8: number;
  uint8: number;
  int16: number;
  uint16: number;
  int32: number;
  uint32: number;
}

// Supported typed array
type NNTensorType = Int32Array | Float32Array;

interface OperandOptions {
  type: number;
  dimensions?: number[];
  // scale: an non-negative floating point value
  scale?: number;
  // zeroPoint: an integer, in range [0, 255]
  zeroPoint?: number;
}

interface Model {
  addOperand: (options: OperandOptions) => void;
  setOperandValue: (index: number, data: NNTensorType) => void;
  addOperation: (type: number, inputs: number[], outputs: number[]) => void;
  identifyInputsAndOutputs: (inputs: number[], outputs: number[]) => void;
  finish: () => Promise<number>;
  createCompilation: () => Promise<Compilation>;
}

interface Compilation {
  setPreference: (preference: number) => void;
  finish: () => Promise<number>;
  createExecution: () => Promise<Execution>;
}

interface Execution {
  setInput: (index: number, data: NNTensorType) => void;
  setOutput: (index: number, data: NNTensorType) => void;
  startCompute: () => Promise<number>;
}

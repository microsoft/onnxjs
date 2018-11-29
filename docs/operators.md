The following table lists the ONNX operators supported by each of the available backends.

The supported platforms are Windows 10 + Edge/Chrome/Firefox/Electron/Node.js.

|                                               Operator                                                 | Cpu Backend | Wasm Backend | WebGl Backend |
|:------------------------------------------------------------------------------------------------------:|:-----------:|:------------:|:-------------:|
|                [Abs](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Abs)                |      x      |              |       x       |
|               [Acos](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Acos)               |      x      |              |       x       |
|                [Add](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Add)                |      x      |              |       x       |
|                [And](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#And)                |      x      |              |       x       |
|               [Asin](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Asin)               |      x      |              |       x       |
|               [Atan](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Atan)               |      x      |              |       x       |
|        [AveragePool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AveragePool)        |      x      |       x      |       x       |
| [BatchNormalization](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#BatchNormalization) |      x      |       x      |       x       |
|               [Ceil](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Ceil)               |      x      |              |       x       |
|             [Concat](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Concat)             |      x      |              |       x       |
|               [Conv](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Conv)               |      x      |       x      |       x       |
|                [Cos](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Cos)                |      x      |              |       x       |
|                [Div](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Div)                |      x      |              |       x       |
|            [Dropout](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Dropout)            |      x      |              |       x       |
|              [Equal](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Equal)              |             |              |       x       |
|                [Exp](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Exp)                |      x      |              |       x       |
|              [Floor](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Floor)              |      x      |              |       x       |
|               [Gemm](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Gemm)               |             |       x      |       x       |
|  [GlobalAveragePool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#GlobalAveragePool)  |      x      |       x      |       x       |
|      [GlobalMaxPool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#GlobalMaxPool)      |      x      |       x      |       x       |
|            [Greater](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Greater)            |             |              |       x       |
|           [Identity](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Identity)           |             |              |       x       |
|        [ImageScaler](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ImageScaler)        |             |              |       x       |
|          [LeakyRelu](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#LeakyRelu)          |      x      |              |       x       |
|               [Less](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Less)               |             |              |       x       |
|                [Log](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Log)                |      x      |              |       x       |
|                [LRN](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#LRN)                |      x      |              |               |
|             [MatMul](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#MatMul)             |      x      |              |       x       |
|            [MaxPool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#MaxPool)            |      x      |       x      |       x       |
|                [Mul](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Mul)                |      x      |              |       x       |
|                [Neg](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Neg)                |      x      |              |       x       |
|                [Not](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Not)                |             |              |       x       |
|                 [Or](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Or)                 |      x      |              |       x       |
|                [Pad](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Pad)                |             |              |       x       |
|                [Pow](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Pow)                |             |              |       x       |
|              [PRelu](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#PRelu)              |      x      |              |       x       |
|       [ReduceLogSum](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ReduceLogSum)       |      x      |              |               |
|          [ReduceMax](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ReduceMax)          |      x      |              |               |
|         [ReduceMean](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ReduceMean)         |      x      |              |               |
|          [ReduceMin](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceMin)         |             |              |               |
|         [ReduceProd](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceProd)        |             |              |               |
|          [ReduceSum](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceSum)         |             |              |               |
|    [ReduceSumSquare](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceSumSquare)   |             |              |               |
|               [Relu](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Relu)               |      x      |              |       x       |
|            [Reshape](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Reshape)            |      x      |              |       x       |
|            [Sigmoid](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sigmoid)            |      x      |              |       x       |
|                [Sin](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sin)                |      x      |              |       x       |
|            [Softmax](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Softmax)            |      x      |       x      |       x       |
|              [Split](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Split)              |             |              |       x       |
|               [Sqrt](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sqrt)               |      x      |              |       x       |
|                [Sub](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sub)                |      x      |              |       x       |
|                [Sum](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sum)                |      x      |       x      |       x       |
|                [Tan](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Tan)                |      x      |              |       x       |
|               [Tanh](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Tanh)               |      x      |              |               |
|          [Transpose](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Transpose)          |      x      |              |       x       |
|          [Unsqueeze](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Unsqueeze)          |      x      |              |               |
|                [Xor](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Xor)                |      x      |              |       x       |

The following table lists the **ai.onnx** operators supported by each of the available backends.

See [Compatibility](../README.md#Compatibility) for a list of the supported platforms.

|                                               Operator                                                 | Cpu Backend | Wasm Backend | WebGl Backend |
|:------------------------------------------------------------------------------------------------------:|:-----------:|:------------:|:-------------:|
|                [Abs](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Abs)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|               [Acos](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Acos)               |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Add](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Add)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [And](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#And)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|               [Asin](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Asin)               |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|               [Atan](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Atan)               |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|        [AveragePool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AveragePool)        |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
| [BatchNormalization](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#BatchNormalization) |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
|               [Ceil](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Ceil)               |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|             [Concat](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Concat)             |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|               [Conv](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Conv)               |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
|                [Cos](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Cos)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Div](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Div)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|            [Dropout](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Dropout)            |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|              [Equal](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Equal)              |             |              |      :heavy_check_mark:      |
|                [Exp](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Exp)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|              [Floor](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Floor)              |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|               [Gemm](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Gemm)               |             |      :heavy_check_mark:     |      :heavy_check_mark:      |
|  [GlobalAveragePool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#GlobalAveragePool)  |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
|      [GlobalMaxPool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#GlobalMaxPool)      |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
|            [Greater](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Greater)            |             |              |      :heavy_check_mark:      |
|           [Identity](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Identity)           |             |              |      :heavy_check_mark:      |
|        [ImageScaler](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ImageScaler)        |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|          [LeakyRelu](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#LeakyRelu)          |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|               [Less](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Less)               |             |              |      :heavy_check_mark:      |
|                [Log](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Log)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [LRN](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#LRN)                |     :heavy_check_mark:     |              |               |
|             [MatMul](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#MatMul)             |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|            [MaxPool](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#MaxPool)            |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
|                [Mul](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Mul)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Neg](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Neg)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Not](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Not)                |             |              |      :heavy_check_mark:      |
|                 [Or](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Or)                 |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Pad](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Pad)                |             |              |      :heavy_check_mark:      |
|                [Pow](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Pow)                |             |              |      :heavy_check_mark:      |
|              [PRelu](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#PRelu)              |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|       [ReduceLogSum](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ReduceLogSum)       |     :heavy_check_mark:     |              |               |
|          [ReduceMax](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ReduceMax)          |     :heavy_check_mark:     |              |               |
|         [ReduceMean](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#ReduceMean)         |     :heavy_check_mark:     |              |               |
|          [ReduceMin](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceMin)         |             |              |               |
|         [ReduceProd](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceProd)        |             |              |               |
|          [ReduceSum](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceSum)         |             |              |               |
|    [ReduceSumSquare](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#AReduceSumSquare)   |             |              |               |
|               [Relu](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Relu)               |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|            [Reshape](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Reshape)            |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|            [Sigmoid](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sigmoid)            |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Sin](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sin)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|            [Softmax](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Softmax)            |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
|              [Split](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Split)              |             |              |      :heavy_check_mark:      |
|               [Sqrt](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sqrt)               |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Sub](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sub)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|                [Sum](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Sum)                |     :heavy_check_mark:     |      :heavy_check_mark:     |      :heavy_check_mark:      |
|                [Tan](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Tan)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|               [Tanh](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Tanh)               |     :heavy_check_mark:     |              |               |
|          [Transpose](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Transpose)          |     :heavy_check_mark:     |              |      :heavy_check_mark:      |
|          [Unsqueeze](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Unsqueeze)          |     :heavy_check_mark:     |              |               |
|                [Xor](https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md#Xor)                |     :heavy_check_mark:     |              |      :heavy_check_mark:      |

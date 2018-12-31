# **API Documentation**

## **Table of Contents**
### - [Onnx](#ref-Onnx)
  - [Tensor](#ref-Onnx-Tensor)
  - [InferenceHandler](#ref-Onnx-InferenceHandler)
  - [backend](#ref-Onnx-backend)
  - [ENV](#ref-Onnx-ENV)
  - [TensorTransform](#ref-Onnx-TensorTransform)

### - [Inference Session](#ref-InferenceSession)
  - [Creating an Inference Session](#Creating-an-Inference-Session)
  - [Run in Inference Session](#Run-in-Inference-Session)
  - [Profile a Session](#Profile-a-Session)

### - [Tensor](#ref-Tensor)
  - [Create a Tensor](#Create-a-Tensor)
  - [Tensor Properties](#Tensor-Properties)
  - [Access Tensor Elements](#Access-Tensor-Elements)

## <a name="ref-Onnx"></a>**Onnx**

The `onnx` object is the exported object of the module. It's available in global context (`window.onnx` in browser, `global.onnx` in Node.js) after require/import 'onnxjs' module, or imported from a `<script> tag`.

- ### <a name="ref-Onnx-Tensor"></a>**onnx.Tensor**
  See [Tensor](#ref-Tensor).

- ### <a name="ref-Onnx-InferenceSession"></a>**onnx.InferenceSession**
  See [InferenceSession](#ref-InferenceSession).

- ### <a name="ref-Onnx-backend"></a>**onnx.backend**
  Customizes settings for all available backends. `ONNX.js` currently supports three types of backend - *cpu* (pure JavaScript backend), *webgl* (WebGL backend), and *wasm*   (WebAssembly backend).

  ### `backend.cpu`
    An object specifying CPU backend settings. Available soon.
  ***
  ### `backend.webgl`
    An object specifying WebGL backend settings. Available soon.
  ***
  ### `backend.wasm`
    An object specifying WebAssembly backend settings. The supported member variables are:
    - **worker** (`number`)

      Optional. Specifies the number of [web workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers) to run in background threads. If not set, run with number of `(CPU cores - 1)` workers.
    - **cpuFallback** (`boolean`)

      Optional. Determines whether to fall back to use CPU backend if WebAssembly backend is missing certain ONNX operators. Default is set to true.

- ### <a name="ref-Onnx-backend"></a>**ENV**
  Represent runtime environment settings and status of ONNX.js
  ### `ENV.debug`
  A global flag to indicate whether to run `ONNX.js` in debug mode.

## <a name="ref-InferenceSession"></a>**Inference Session**
An `InferenceSession` encapsulates the environment for `ONNX.js` operations to execute. It loads and runs `ONNX` models with the desired configurations.

To configure an `InferenceSession`, use an object with the following parameters-
  - **backendHint** (`string`)
      Specify a preferred backend to start an `InferenceSession`. Current available backend hints are:
      - `'cpu'`: CPU backend
      - `'wasm'`: WebAssembly backend
      - `'webgl'`: WebGL backend
      If not set, the backend will be determined by the platform and environment.

  - **profiler** (`Config.Profiler`)
    An object specifying profiler configurations that used in an `InferenceSession`. If not set, run profiler in default config. Detailed settings are listed in [`Config.Profiler`] (#Config.Profiler).
  ***

  ### `Config.Profiler`
  Represents the configuration of the profiler that used in an `InferenceSession`. The supported member variables are:

    - **maxNumberEvents** (`number`)
      Optional. The max number of events to be recorded. Default = 10000.

    - **flushBatchSize** (`number`)
      Optional. The maximum size of a batch to flush. Default = 10.

    - **flushIntervalInMilliseconds** (`number`)
      Optional. The maximum interval in milliseconds to flush. Default = 5000.
  ***

  ### <a name="ref-Onnx-TensorTransform"></a>**onnx.TensorTransform**
  `onnx.TensorTransforms` contains a set of utility functions to quickly create and compute [Tensor](#ref-Tensor)
  ### Utility Tensor Creators
  #### `zeros(shape, type)`
  Creates a Tensor of given shape and type with elements all zeros.

  *parameters*
  - **shape**(`ReadonlyArray<number>`):

    The desired output Tensor shape.
  - **type** (`'int32'` | `'float32'`):

    The data type of the output Tensor.

  *returns*
  - **Tensor**

    A new Tensor of given shape with elements all zeros
  ***

  #### `linspace(start, stop, num)`
  Creates an evenly spaced sequence of numbers over the given interval.

  *parameters*
  - **start** (`number`):

    The desired output Tensor shape.

  - **stop** (`number`):

    The data type of the output Tensor.

  - **num** (`number`):

    The number of values to generate.

  *returns*
  - **Tensor**

    A new 1-D Tensor filled with an evenly spaced sequence of numbers over the given interval.

  ***

  #### `range(start, end, step?, type?)`
  Creates a 1-D Tensor filled with an arithmetic sequence.

  *parameters*
  - **start** (`number`):

    The start value of the sequence.
  - **end** (`number`):

    The end value of the sequence
  - **step** (`number`):

    The increment value. Optional. Default is 1.
  - **type** (`'int32'` |`'float32'`):

    The output Tensor data type. Optional. Default is "float32"

  *returns*
  - **Tensor**

    A new 1-D Tensor filled with the generated arithmetic sequence.

  ***

  #### `as1d(x)`
  Reshapes a Tensor to 1-D Tensor.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor.

  *returns*
  - **Tensor**

    A reshaped 1-D Tensor of the same data as the input Tensor.

  ***

  #### `scalar(value, type?)`
  Creates a scalar Tensor (rank = 0) with given value and type.

  *parameters*
  - **value**(`ReadonlyArray<number>`):

    The desired data value of the scalar Tensor

  - **type**  (`'int32'`|`'float32'`):

    The data type of the output Tensor. Optional. Default is "float32".

  *returns*
  - **Tensor**

    A new scalar Tensor of given data and type.

  ***
  ### Basic Math Tensor Transforms

  #### `exp(x)`
  Calculates the exponential of the given input Tensor, element-wise.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `sigmoid(x)`
  Calculates sigmoid of the given input Tensor. Sigmoid takes one input data (Tensor) and produces one output data (Tensor) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the tensor elementwise.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Arithmetic Tensor Transforms
  #### `add(a, b)`
  Performs element-wise binary addition (with Numpy-style broadcasting support).

  *parameters*
  - **a** (`Tensor`):

    The first operand.
  - **b** (`Tensor`):

    The second operand.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `sub(a, b)`
  Performs element-wise binary substraction (with Numpy-style broadcasting support).

  *parameters*
  - **a** (`Tensor`):

    The first operand.
  - **b** (`Tensor`):

    The second operand.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `mul(a, b)`
  Performs element-wise binary multiplication (with Numpy-style broadcasting support).

  *parameters*
  - **a** (`Tensor`):

    The first operand.
  - **b** (`Tensor`):

    The second operand.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `div(a, b)`
  Performs element-wise binary division (with Numpy-style broadcasting support).

  *parameters*
  - **a** (`Tensor`):

    The first operand.
  - **b** (`Tensor`):

    The second operand.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Normalization Tensor Transforms
  #### `softmax(x, axis?)`
  Computes the softmax (normalized exponential) values.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor.
  - **axis** (`number`):

    The axis of the inputs when coerced to 2D. Optional. Default to 1.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Slice and Join Tensor Transforms
  #### `concat(x, axis)`
  Concatenate a list of Tneosrs into a single Tensor.

  *parameters*
  - **x** (`Tensor[]`):

    List of Tensors for concatenation.
  - **axis**(`number`):

    Which axis to concat on.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `gather(x, indices, axis?)`
  Gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices, and concatenates them in an output tneosr of rank q + (r - 1).

  *parameters*
  - **x** (`Tensor`):

    Tensor of rank r >= 1.
  - **indices** (`Tensor`):

    Tensor of int32/int64 indices, of any rank q.
  - **axis** (`number`):

    The axis to gather on. Optional. Defaults to 0. Negative vaulue means counting dimensions from the back.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `slice(x, starts, ends, axes?)`
  Produces a slice of the input Tensor along multiple axes.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor to slice from.
  - **starts** (`number[]`):

    Starting indices of corresponding axis in "axes".
  - **ends** (`number[]`):

    Ending indices (exclusive) of corresponding axis in "axes".
  - **axes**(`number[]`):

    Axes that "starts" and "ends" apply to. Optional. Default = [0, 1, ..., len("start" - 1)].

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `stack(x, axis?)`
  Stack a list of Tensors into a single Tensor.

  *parameters*
  - **x** (`Tensor`):

    List of Tensors of the same shape and type.
  - **axis** (`number`):

    The axis to stack on. Optional. Defaults to 0 (first dim).

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `tile(x, repeats)`
  Constructs a Tensor by tiling a given Tensor

  *parameters*
  - **x** (`Tensor`):

    The input Tensor of any shape.
  - **repeats** (`ReadonlyArray<number>`):

    A number array of the same length as input's dimension number, specifying numbers of repeated copies along input's dimension.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Permutation Tensor Transforms
  #### `transpose(x, perm?)`
  Transpose the input tensor similar to numpy.transpose. For example, when perm=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape will be (2, 1, 3).

  *parameters*
  - **x** (`Tensor`):

    The input Tensor of any shape.
  - **perm** (`number[]`):

    A list of integers. Optional. By default, reverse the dimensions, otherwise permute the axes according to the values given.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Shape Tensor Transforms
  #### `expandDims(x, axis)`
  Creates a Tensor with rank expanded at the specified axis.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor of any shape.
  - **axis** (`number[]`):

    The axis at which to expand.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `reshape(x, shape)`
  Reshapes the input Tensor to a new shape.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor of any shape.
  - **shape** (`ReadonlyArray<number>`):

    The new shape.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Logical Tensor Transforms
  #### `greaterEqual(a, b)`
  Returns the tensor resulted from performing the greater logical operation elementwise on the input tensors A and B (with Numpy-style broadcasting support).

  *parameters*
  - **a** (`Tensor`):

    First input operand for the logical operator.
  - **b** (`Tensor`):

    Second input operand for the logical operator.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `where(condition, a, b)`
  Returns the elements, either a or b depending on the condition. If the condition is true, select from a, otherwise select from b.

  *parameters*
  - **condition** (`Tensor`):

    The input condition. Must be of data type bool.
  - **a** (`Tensor`):

    A Tensor. If condition is rank 1, it may have a higher rank but its first dimension must match the size of condition.
  - **b** (`Tensor`):

    A tensor with the same shape and type as "a".

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Cast Tensor Transforms
  #### `cast(x, type)`
  The operator casts the elements of a given input tensor to a data type specified by the "type" argument and returns an output tensor of the same size in the converted type. NOTE: casting to string is not supported yet.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor
  - **type** (`'int32'`|`'float32'`|`'bool'`):

    The data type to which the elements of the input tensor are cast. Strictly must be one of the types

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  ### Reduction Tensor Transforms
  #### `argMax(x, axis?, keepdims?)`
  Computes the indices of the max elements of the input tensor's element along the provided axis. The resulted tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor
  - **axis** (`number`):

    The axis in which to compute the arg indices. Optional. Default is 0.
  - **keepdims** (`number`):

    Keep the reduced dimension or not. Optional. Default 1 mean keep reduced dimension.

  *returns*
  - **Tensor**

    The resulting Tensor.

  ***

  #### `reduceMax(x, axis?, keepdims?)`
  Computes the max of the input tensor's element along the provided axes. The resulted tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.

  *parameters*
  - **x** (`Tensor`):

    The input Tensor
  - **axis** (`number`):

    The axis in which to compute the arg indices. Optional. Default is 0.
  - **keepdims** (`number`):

    Keep the reduced dimension or not. Optional. Default 1 mean keep reduced dimension.

  *returns*
  - **Tensor**

    The resulting Tensor.

***
- ### **Creating an Inference Session**
  ### `new InferenceSession(config?)`

  Construct a new `InferenceSession` object with configuration.

  *Parameters*

  - **config** (`{backendHint?: string, Profiler?: Config.profiler}`):

    Optional. Specify configuration for creating a new inference session. If not set, the session will run in default settings.

  *Returns*

  - `InferenceSession` object

  *Examples*

  ```ts
  // Imports
  import {InferenceSession, Tensor} from 'onnxjs';
  ```

  ```ts
  // Create an Inference Session with CPU backend
  const session = new InferenceSession({backendHint: 'cpu'});
  ```

- ### **Run in Inference Session**
  To use an `InferenceSession` to run `ONNX` models, start with loading a model from a `.onnx` model file, and then `run()` with your input data.
    ### `loadModel(arg0, byteOffset?, length?)`

    Load ONNX model from an onnx model source asynchronously.

    *Parameters*

    - **arg0** (`string` | `Blob` | `ArrayBuffer` | `Uint8Array`)

        The Onnx model representation types. Currently supports the following four types.
        - `string`: URI representation of the model.
        - `Blob`: Blob object representation.
        - `ArrayBuffer`: ArrayBuffer representation. If in this type, `byteOffset` and `length` can be set.
        - `Uint8Array`: Uint8Array representation.
    - **byteOffset** (`number`): Optional. Byte offset for `ArrayBuffer`
    - **length** (`number`): Optional. Length param for `ArrayBuffer`

    *Returns*

    - `Promise<void>`

    *Examples*

    ```ts
    // Create an Inference Session with default settings.
    const session = new InferenceSession();
    // load model from uri
    const outputData = await session.loadModel('https://some.domain.com/models/myModel.onnx');
    ```
    ***
    ### `run(inputs, options?)`

    Execute the model asynchronously with the given input data and output names.

    *Parameters*

    - **inputs** ( `ReadonlyMap<string,Tensor>`|`{readonly [name: string]: Tensor}`| `ReadonlyArray<Tensor>`):

        Representation of inputs in one of the accepted format, with or without input names. The input `Tensors` should be in the same shapes and orders as the inputs specified in the loaded `ONNX` model.


    - **options** (`ReadonlyArray<string>`): Optional. Contains a list of output names. See more in [`RunOptions`](#RunOptions). If not specified, use the model's output list.

    *Returns*
    - `Promise<ReadonlyMap<string, Tensor>>`

    *Examples*

    ```ts
    const inputs = [new Tensor([1, 2], 'float32')];
    const output1 = await session.run(inputs); // run with ReadonlyArray<Tensor>
    ```

  ### `RunOptions`
  Options for running inference.

  - **outputNames** (`ReadonlyArray<string>`)

      Optional. Represent a list of output names as an array of string. This must be a subset of the output list defined by the model. If not specified, use the model's output list.


- ### **Profile an Inference session**

  ### `startProfiling()`

  Start profiling for current session.
  ```ts
  // Create an Inference Session with default settings.
  const session = new InferenceSession();
  // start profiling
  session.startProfiling();
  ```
  ***
  ### `endProfiling()`

  Stop profiling for current session.
   ```ts
    // Create an Inference Session with default settings.
    const session = new InferenceSession();
    // end profiling
    session.endProfiling();
    ```


## <a name="ref-Tensor"></a>**Tensor**
Tensor is a representation of vectors, matrices and n-dimension data in `ONNX.js`. Tensors are used in `InferenceSession` as inputs for models to run.

- ### **Create a Tensor**

  ### `new Tensor(data, type, dims?)`

  Creates a new Tensor with the specified data, type and shape.

  *Parameters*

  - **data** (`Uint8Array` | `Float32Array` | `Int32Array` | `string` | `boolean[]` | `number[]`):

    Initial data to put in a Tensor.
  - **type** (`'bool'` | `'float32'` | `'int32'` | `'string'`):

    Type of the Tensor. Must be consistent with the `data` type or a `TypeError` will be thrown.

    String | Data Type
    ---: | :---
    `'bool'` |   `Uint8Array` \| `boolean[]`
    `'float32'` | `Float32Array` \| `number[]`
    `'int32'` | `Int32Array` \| `number[]`
    `'string'` | `string[]`

  - **dims** (`ReadonlyArray<number>`):

    Optional. The shape of the Tensor in a array of Integers. Current implementation supports up to 6 dimensions.  If not set, a 1-D Tensor will be created with length inferred from `data`.

  *Returns*
  - `Tensor` object.

  *Examples*

  ```ts
  // Create a int32-typed 1D tensor with 2 elements
  t = new Tensor([2, 3], 'int32');
  ```
  ```ts
  // Create a int32-typed 1D tensor with 10 elements
  t = new Tensor(new Int32Array(10), 'int32');
  ```
  ```ts
  // Create a string-typed 1D tensor with 3 elements
  t = new Tensor(['a', 'b', 'c'], 'string');
  ```
  ```ts
  // Create a 3*2 int32 matrix with specified values.
  t = new Tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 'float32', [3, 2]);
  ```


- ### **Tensor Properties**
  Each `Tensor` contains the following properties. These can be useful in calculations involving Tensors.
  - **data** (`Uint8Array` | `Float32Array` | `Int32Array` | `string`)

    Returns the underlying raw data in Tensor. Note that to modify the data inside Tensor, use the `get()`/`set()` functions.
  - **dims** (`ReadonlyArray<number>`)

    Returns the dimension/shape of the Tensor.
  - **type** (`'bool'` | `'float32'` | `'int32'` | `'string'`)

    Returns a string representing the underlying data type of the Tensor.

  - **size** (`int`)

    Returns the size of Tensor in logical elements.



- ### **Access Tensor Elements**
  To access elements in Tensor, use the `get()`/`set()` methods.

  ### Getters
  ### `get(...indices)`

  Gets a Tensor element by index from a `Tensor` object.

  *Parameters*

  - **...indices** (`i, j, k...`):

      Index in spread syntax. The index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  *Returns*
  - Tensor element.

  *Example*
  ```ts
  const t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
  const res = t.get(0, 1, 0); // res = 3
  ```
  ***
  ### `get(indices)`

  Gets a Tensor element by index from a `Tensor` object.

  *Parameters*

  - **indices** (`number[]`):

      Index in array. The index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  *Returns*
  - Tensor element.

  *Example*
  ```ts
  const t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
  const res = t.get([0, 1, 0]); // res = 3
  ```
  ***

  ### Setters

  ### `set(value, ...indices)`

  Sets a Tensor element by index to a `Tensor` object.

  *Parameters*

  - **value** (`boolean` | `number` | `string`):

    The value to set. The value type must be consistent with the data type in tensor or a `TypeError` will be thrown.
  - **...indices** (`i, j, k...`):

    Index of the element in spread syntax. TThe index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  *Returns*
  - `void`

  *Example*
  ```ts
  t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
  t.set(2, 0, 1, 0); // Sets element at [0, 1, 0] to 2. t.data = [1, 2, 2]
  t.set(3, 0, 2, 0); // Sets element at [0, 2, 0] to 3. t.data = [1, 2, 3]
  ```
  ***
  ### `set(value, indices)`

  Sets a Tensor element by index to a `Tensor` object.

  *Parameters*

  - **value** (`boolean` | `number` | `string`):

    The value to set. The value type must be consistent with the data type in tensor or a `TypeError` will be thrown.
  - **indices** (`number[]`):

    Index of the element to set. The index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  *Returns*
  - `void`

  *Example*
  ```ts
  t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
  t.set(2, [0, 1, 0]); // Sets element at [0, 1, 0] to 2. t.data = [1, 2, 2]
  t.set(3, [0, 2, 0]); // Sets element at [0, 2, 0] to 3. t.data = [1, 2, 3]
  ```



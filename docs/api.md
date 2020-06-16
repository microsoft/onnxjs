# **API Documentation**

## **Table of Contents**

### - [Onnx](#ref-Onnx)

- [Tensor](#ref-Onnx-Tensor)
- [InferenceHandler](#ref-Onnx-InferenceHandler)
- [backend](#ref-Onnx-backend)
- [ENV](#ref-Onnx-ENV)

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

  Customizes settings for all available backends. `ONNX.js` currently supports three types of backend - _cpu_ (pure JavaScript backend), _webgl_ (WebGL backend), and _wasm_ (WebAssembly backend).

  ### `backend.hint`

  A string or an array of strings that indicate a hint to use backend(s).

  ***

  ### `backend.cpu`

  An object specifying CPU backend settings. Currently no members are available.

  ***

  ### `backend.webgl`

  An object specifying WebGL backend settings. The supported member variables are:

  - **contextId** (`'webgl'|'webgl2'`)

    Optional. Force the WebGL Context ID.

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
  An object specifying the profiler configuration that is used in an `InferenceSession`. If not set, run profiler in default config. Detailed settings are listed in [`Config.Profiler`](#Config.Profiler).

---

### `Config.Profiler`

Represents the configuration of the profiler that is used in an `InferenceSession`. The supported member variables are:

    - **maxNumberEvents** (`number`)
      Optional. The max number of events to be recorded. Default = 10000.

    - **flushBatchSize** (`number`)
      Optional. The maximum size of a batch to flush. Default = 10.

    - **flushIntervalInMilliseconds** (`number`)
      Optional. The maximum interval in milliseconds to flush. Default = 5000.

---

- ### **Creating an Inference Session**

  ### `new InferenceSession(config?)`

  Construct a new `InferenceSession` object with configuration.

  _Parameters_

  - **config** (`{backendHint?: string, Profiler?: Config.profiler}`):

    Optional. Specify configuration for creating a new inference session. If not set, the session will run in default settings.

  _Returns_

  - `InferenceSession` object

  _Examples_

  ```ts
  // Imports
  import { InferenceSession, Tensor } from "onnxjs";
  ```

  ```ts
  // Create an Inference Session with CPU backend
  const session = new InferenceSession({ backendHint: "cpu" });
  ```

- ### **Run in Inference Session**

  To use an `InferenceSession` to run `ONNX` models, start with loading a model from a `.onnx` model file, and then `run()` with your input data.

  ### `loadModel(arg0, byteOffset?, length?)`

  Load ONNX model from an onnx model source asynchronously.

  _Parameters_

  - **arg0** (`string` | `Blob` | `ArrayBuffer` | `Uint8Array`)

    The Onnx model representation types. Currently supports the following four types.

    - `string`: URI representation of the model.
    - `Blob`: Blob object representation.
    - `ArrayBuffer`: ArrayBuffer representation. If in this type, `byteOffset` and `length` can be set.
    - `Uint8Array`: Uint8Array representation.

  - **byteOffset** (`number`): Optional. Byte offset for `ArrayBuffer`
  - **length** (`number`): Optional. Length param for `ArrayBuffer`

  _Returns_

  - `Promise<void>`

  _Examples_

  ```ts
  // Create an Inference Session with default settings.
  const session = new InferenceSession();
  // load model from uri
  const outputData = await session.loadModel(
    "https://some.domain.com/models/myModel.onnx"
  );
  ```

  ***

  ### `run(inputs, options?)`

  Execute the model asynchronously with the given input data and output names.

  _Parameters_

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

* ### **Profile an Inference session**

  ### `startProfiling()`

  Start profiling for current session.

  ```ts
  // Create an Inference Session with default settings.
  const session = new InferenceSession();
  // start profiling
  session.startProfiling();
  ```

  ---

  ### `endProfiling()`

  Stop profiling for current session.

  ```ts
  // Create an Inference Session with default settings.
  const session = new InferenceSession();
  // end profiling
  session.endProfiling();
  ```

## <a name="ref-Tensor"></a>**Tensor**

Tensor is a representation of vectors, matrices and n-dimensional data in `ONNX.js`. Tensors are used in `InferenceSession` as inputs for models to run.

- ### **Create a Tensor**

  ### `new Tensor(data, type, dims?)`

  Creates a new Tensor with the specified data, type and shape.

  _Parameters_

  - **data** (`Uint8Array` | `Float32Array` | `Int32Array` | `string` | `boolean[]` | `number[]`):

    Initial data to put in a Tensor.

  - **type** (`'bool'` | `'float32'` | `'int32'` | `'string'`):

    Type of the Tensor. Must be consistent with the `data` type or a `TypeError` will be thrown.

    |      String | Data Type                    |
    | ----------: | :--------------------------- |
    |    `'bool'` | `Uint8Array` \| `boolean[]`  |
    | `'float32'` | `Float32Array` \| `number[]` |
    |   `'int32'` | `Int32Array` \| `number[]`   |
    |  `'string'` | `string[]`                   |

  - **dims** (`ReadonlyArray<number>`):

    Optional. The shape of the Tensor in an array of Integers. Current implementation supports up to 6 dimensions. If not set, a 1-D Tensor will be created with length inferred from `data`.

  _Returns_

  - `Tensor` object.

  _Examples_

  ```ts
  // Create a int32-typed 1D tensor with 2 elements
  t = new Tensor([2, 3], "int32");
  ```

  ```ts
  // Create a int32-typed 1D tensor with 10 elements
  t = new Tensor(new Int32Array(10), "int32");
  ```

  ```ts
  // Create a string-typed 1D tensor with 3 elements
  t = new Tensor(["a", "b", "c"], "string");
  ```

  ```ts
  // Create a 3*2 int32 matrix with specified values.
  t = new Tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "float32", [3, 2]);
  ```

* ### **Tensor Properties**

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

  _Parameters_

  - **...indices** (`i, j, k...`):

    Index in spread syntax. The index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  _Returns_

  - Tensor element.

  _Example_

  ```ts
  const t = new Tensor([1, 3, 2], "int32", [1, 3, 1]);
  const res = t.get(0, 1, 0); // res = 3
  ```

  ***

  ### `get(indices)`

  Gets a Tensor element by index from a `Tensor` object.

  _Parameters_

  - **indices** (`number[]`):

    Index in array. The index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  _Returns_

  - Tensor element.

  _Example_

  ```ts
  const t = new Tensor([1, 3, 2], "int32", [1, 3, 1]);
  const res = t.get([0, 1, 0]); // res = 3
  ```

  ***

  ### Setters

  ### `set(value, ...indices)`

  Sets a Tensor element by index to a `Tensor` object.

  _Parameters_

  - **value** (`boolean` | `number` | `string`):

    The value to set. The value type must be consistent with the data type in tensor or a `TypeError` will be thrown.

  - **...indices** (`i, j, k...`):

    Index of the element in spread syntax. TThe index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  _Returns_

  - `void`

  _Example_

  ```ts
  t = new Tensor([1, 3, 2], "int32", [1, 3, 1]);
  t.set(2, 0, 1, 0); // Sets element at [0, 1, 0] to 2. t.data = [1, 2, 2]
  t.set(3, 0, 2, 0); // Sets element at [0, 2, 0] to 3. t.data = [1, 2, 3]
  ```

  ***

  ### `set(value, indices)`

  Sets a Tensor element by index to a `Tensor` object.

  _Parameters_

  - **value** (`boolean` | `number` | `string`):

    The value to set. The value type must be consistent with the data type in tensor or a `TypeError` will be thrown.

  - **indices** (`number[]`):

    Index of the element to set. The index should be within `Tensor`'s `dims` or a `RangeError` will be thrown.

  _Returns_

  - `void`

  _Example_

  ```ts
  t = new Tensor([1, 3, 2], "int32", [1, 3, 1]);
  t.set(2, [0, 1, 0]); // Sets element at [0, 1, 0] to 2. t.data = [1, 2, 2]
  t.set(3, [0, 2, 0]); // Sets element at [0, 2, 0] to 3. t.data = [1, 2, 3]
  ```

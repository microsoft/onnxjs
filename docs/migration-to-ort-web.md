## Migration ONNX.js to ONNX Runtime Web

This document demonstrates how to migrate your project that uses ONNX.js to use ONNX Runtime Web in a few simple steps. We assume that you are already using ONNX.js in your project. If you want to create a new web app, we strongly recommend to use ONNX Runtime Web (see also: [Get started](https://onnxruntime.ai/docs/get-started/with-javascript.html), [Tutorials](https://onnxruntime.ai/docs/tutorials/web/)).

### STEP.1 - consume the NPM package for ONNX Runtime Web

Uninstall package `onnxjs` and install `onnxruntime-web` instead.

Your project may be build broken at the moment - don't worry, we will fix it in later steps.

### STEP.2 - fix import

Import the whole package:

```js
//import * as onnx from 'onnxjs';
import * as ort from "onnxruntime-web";
```

Import separated exported objects:

```js
// import { Tensor, Session } from 'onnxjs';
import { Tensor, InferenceSession } from "onnxruntime-web";
```

### STEP.3 - fix usage of inference session

An inference session is represented as type `InferenceSession` in both [ONNX.js](./api.md#ref-InferenceSession) and [ONNX Runtime Web](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.html).

The creation of inference session in ONNX.js:

```js
import { InferenceSession } from "onnxjs";

const session = new InferenceSession({ backendHint: "webgl" });
await session.loadModel("https://example.com/models/myModel.onnx");
```

Equivalent code for that in ONNX Runtime Web:

```js
import { InferenceSession } from "onnxruntime-web";

const session = await InferenceSession.create(
  "https://example.com/models/myModel.onnx",
  {
    executionProviders: ["webgl"],
  }
);
```

Please note that both ONNX.js and ONNX Runtime Web require an async context to create and inference session instance.

Other part in inference session creation:

- backend: backend is specified as `backendHint` in ONNX.js and it becomes `executionProviders` in ONNX Runtime Web. Both "wasm" (for WebAssembly) and "webgl" (for WebGL) are supported.
- load from `ArrayBuffer`/`Uint8Array`: this is supported in both ONNX.js and ONNX Runtime Web. Just replace the URL string in the example above by a `Uint8Array` instance and it will work.
- `startProfiling()` and `endProfiling()`: they are supported in both ONNX.js and ONNX Runtime Web.

See also:

- [`InferenceSession` type](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.html)
- [`InferenceSession.create()`](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSessionFactory.html#create)
- [`SessionOptions` type](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html)

### STEP.4 - fix usage of tensor

A tensor instance is of type `Tensor` in both [ONNX.js](./api.md#ref-Tensor) and [ONNX Runtime Web](https://onnxruntime.ai/docs/api/js/interfaces/TensorConstructor.html).

The biggest difference is, in ONNX.js, tensor is created using `new Tensor(data, type, dims?)`, and in ONNX Runtime Web it's `new Tensor(type, data, dims?)` or `new Tensor(data, dims?)`, if the type can be inferred from the data. Please note the order of the parameters.

Other part in tensor type is the same, including property `dims`, `data`, `type`, `size`.

See also:

- [`Tensor` type](https://onnxruntime.ai/docs/api/js/interfaces/Tensor.html)
- [`Tensor` constructors](https://onnxruntime.ai/docs/api/js/interfaces/TensorConstructor.html#constructor)

### STEP.5 - fix usage of `ENV`

[`ENV`](./api.md#ref-Onnx-ENV) in ONNX.js is replaced by [`env`](https://onnxruntime.ai/docs/api/js/interfaces/Env.html) in ONNX Runtime Web. They can be accessed in a similar way:

```js
import { ENV } from "onnxjs";
```

```js
import { env } from "onnxruntime-web";
```

Please refer to API reference to check available flags: [ONNX.js](./api.md#ref-Onnx-ENV)/[ONNX Runtime Web](https://onnxruntime.ai/docs/api/js/interfaces/Env.html)

### STEP.6 - deploy

#### JavaScript bundling

In most use case, you don't need to do anything because the bundler is smart enough to figure out the correct file list to pack and bundle. If you want to customize it, make sure to bundle file `dist/ort.min.js` from `onnxruntime-web` instead of file `dist/onnx.min.js` from `onnxjs`.

#### WebAssembly deploy

If you are using WebAssembly backend, you need this step to deploy WebAssembly files so that they are correctly served on server.

In ONNX.js, you need to deploy file `onnx-wasm.wasm` to the same folder to your bundle file (or onnx.min.js, if you didn't bundle it into your web app). In ONNX Runtime Web, you need to deploy 4 files to the same folder to your bundle file:

- ort-wasm.wasm
- ort-wasm-threaded.wasm
- ort-wasm-simd.wasm
- ort-wasm-simd-threaded.wasm

You need to deploy 4 files because it's a combination of feature ON/OFF for multi-threading support and SIMD support. By default, when user's device support these features, we want to use them to make inferencing faster. However, those features are not available on devices of every user, so we need to fall back on those devices. ONNX Runtime Web detects availability for those features and try to use the best one for the current runtime environment.

If you don't need this feature, you can turn off them in the code explicitly. For example, you want to disable multi-threading. The following code will disable it forcely:

```js
// Set 'numThreads' to 1 to disable multi-threading. Set this before creating inference session.
ort.env.wasm.numThreads = 1;
```

Once multi-threading is disabled, it is safe not to deploy file ort-wasm-threaded.wasm and ort-wasm-simd-threaded.wasm.

This is same to SIMD feature. Specifically, if you disable both multi-threading and SIMD feature, you can serve only one WebAssembly file: ort-wasm.wasm.

See also:

- [`numThreads`](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#numThreads)
- [`simd`](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#simd)
- [use a custom path for .wasm files: `wasmPaths`](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html#wasmPaths)

# Examples
Welcome to ONNX.js Examples section

## Run `ONNX.js` in Browser
The following examples are to demonstrate how to run `ONNX.js` in browser using HTTP server. Details are provided with their corresponding README files:

1. **Add** (./browser/add)
    Simple example which adds two Tensors and validates the result.

2. **Resnet50** (./browser/resnet50)
    Loads and runs a [Resnet50](https://github.com/onnx/models/tree/master/models/image_classification/resnet) Model, which is a highly accurate image classification model train on ImageNet.

3. **Squeezenet** (./browser/squeezenet)
    Loads and runs a [Squeezenet](https://github.com/onnx/models/tree/master/models/image_classification/squeezenet) Model, which is a highly efficient image classification model trained on ImageNet.

## Run `ONNX.js` in Node
The following example shows how to run `ONNX.js` using `node`. Further details are provided with its README file:

1. **Add** (./node/add)
    Simple example which adds two Tensors and validates the result.

## Run-time dependency for WebAssembly backend

### onnx-wasm.wasm file
This file should be available to the browser whenever the usage of *WebAssembly backend* is desired.
It is suggested to place this file<sup>1</sup> in the same path containing the .html file

### onnx-worker.js file
This file should be available to the browser whenever the usage of *WebAssembly backend with Web Workers* is desired.
It is suggested to place this file<sup>1</sup> in the same path containing the .html file

1 -
There are several ways to get this file:
  * The file can be found at [https://cdn.jsdelivr.net/npm/onnxjs/dist/](https://cdn.jsdelivr.net/npm/onnxjs/dist/)
  * If consuming the module though NPM, this file can be found at `node_modules/onnxjs/dist`
  * By building the source code - after running `npm run build`, this file will be found in the `dist` folder of the repo



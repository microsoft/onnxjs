# Benchmarks
This sub project is to benchmark model zoo's super resolution model and compare ONNX.js peformance vs other leading in-browser AI Inference frameworks.

## Frameworks
- TensorFlow.js
- ONNX.js

## Backends
- WebGL

## Browsers
(not all framework/backend combinations are supported by all browsers)
- Chrome (WebGL 2)
- Edge (WebGL 1)

## Instructions

1. Ensure that the ONNX.js project (the parent) is already installed and built:
```bash
npm ci
npm run build
```
2. Change to `benchmark/super_resolution_model_zoo` subfolder and run npm ci and build in the benchmark folder
```bash
cd benchmark
npm install
npm run build
```
3. Run tests (Chrome)
```bash
npm run test
```

The test command supports enabling pack mode through environment variables, use:
```bash
PACK=1 npm run test
```
to enable webGL texture packing for onnxjs and tfjs.

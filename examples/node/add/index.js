require('onnxjs');

// uncomment the following line to enable ONNXRuntime node binding
// require('onnxjs-node');

const assert = require('assert');

async function main() {
  // Create an ONNX inference session with WebAssembly backend.
  const session = new onnx.InferenceSession({backendHint: 'wasm'});
  // Load an ONNX model. This model takes two tensors of the same size and return their sum. 
  await session.loadModel("./add.onnx");

  const x = new Float32Array(3 * 4 * 5).fill(1);
  const y = new Float32Array(3 * 4 * 5).fill(2);
  const tensorX = new onnx.Tensor(x, 'float32', [3, 4, 5]);
  const tensorY = new onnx.Tensor(y, 'float32', [3, 4, 5]);

  // Run model with Tensor inputs and get the result by output name defined in model.
  const outputMap = await session.run([tensorX, tensorY]);
  const outputData = outputMap.get('sum');

  // Check if result is expected.
  assert.deepEqual(outputData.dims, [3, 4, 5]);
  assert(outputData.data.every((value) => value === 3));
  console.log(`Got an Tensor of size ${outputData.data.length} with all elements being ${outputData.data[0]}`);
}

main();

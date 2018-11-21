# Node-Add Example

This example shows:
- How to create an InferenceSession using `Node`
- Load a simple ONNX model which has one Operator: Add
- Create two random Tensors of given shape
- Run the inference using the inputs
- Get output Tensor back
- Access raw data in the Tensor
- Match the results against the exptected values

## How to run
1. Inside the folder, run `npm install`.

2. run `node index.js`

## Files in folder
- **index.js**

    The main .js file that holds all `ONNX.js` logic to load and execute the model.

- **add.onnx**

    A simple ONNX model file that contains only one `Add` operator.

- **package.json**

    A package file that specifies `ONNX.js` dependency.

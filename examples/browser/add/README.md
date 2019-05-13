# Tensor-Add Example

This example shows:
- How to create an InferenceSession
- Load a simple ONNX model which has one Operator: Add
- Create two random Tensors of given shape
- Run the inference using the inputs
- Get output Tensor back
- Access raw data in the Tensor
- Match the results against the exptected values

## How to run
1. Download model file `add.onnx` from [examples/models](https://github.com/Microsoft/onnxjs-demo/tree/data/data/examples/models) and put it in the current folder.

2. Start an http server in this folder. You can install [`http-server`](https://github.com/indexzero/http-server) via
    ```
    npm install http-server -g
    ```
    Then start an http server by running
    ```
    http-server .. -c-1 -p 3000
    ```

    This will start the local http server with disabled cache and listens on port 3000

3. Open up the browser and access this URL:
http://localhost:3000/add/

4. Click on Run button to see results of the inference run

## Files in folder
- **index.html**

    The HTML file to render the UI in browser
- **index.js**

    The main .js file that holds all `ONNX.js` logic to load and execute the model.

- **add.onnx**

    A simple ONNX model file that contains only one `Add` operator.

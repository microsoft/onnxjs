# Resent50 Example

This example shows:
- How to create an InferenceSession
- Load Resnet50 model
- Use an image as input Tensor
- Run the inference using the input
- Get output Tensor back
- Access raw data in the Tensor
- Match the results with the predefined ImageNet vector

## How to run
1. Download model file `resnet50_8.onnx` from [examples/models](https://github.com/Microsoft/onnxjs-demo/tree/data/data/examples/models) and put it in the current folder.

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
http://localhost:3000/resnet50/

4. Click on Run button to see results of the inference run.


## Files in folder
- **index.html**

    The HTML file to render the UI in browser

- **index.js**

    The main .js file that holds all `ONNX.js` logic of how to load and execute the model.

- **resnet-cat.jpg**

    A sample image chosen from one of the 1000 categories. Could be replaced with any image of your choice.

- **resnet50_8.onnx**

    The ONNX model file that contains the ResNet50 model.

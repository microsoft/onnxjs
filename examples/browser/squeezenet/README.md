# Squeezenet Example

This example shows:
- How to create an InferenceSession
- Load SqueezeNet model
- Use an image as input Tensor
- Run the inference using the input
- Get output Tensor back
- Access raw data in the Tensor
- Match the results with the predefined ImageNet vector

## How to run
1. Start an http server in this folder. You can install [`http-server`](https://github.com/indexzero/http-server) via
    ```
    npm install http-server -g
    ```
    Then start an http server by running
    ```
    http-server . -c-1 -p 3000
    ```

    This will start the local http server with disabled cache and listens on port 3000

2. Open up the browser and access this URL:
http://localhost:3000

3. Click on Run button to see results of the inference run.

## Files in folder
- **index.html**

    The HTML file to render the UI in browser
- **index.js**

    The main .js file that holds all `ONNX.js` logic to load and execute the model.
- **squeezenet-piano.jpg**

    A sample image chosen from one of the 1000 categories. Could be replaced with any image of your choice.
- **squeezenetV1_8.onnx**

    The ONNX model file that contains the SqueezeNet model.

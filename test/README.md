# ONNX.js Tests
Unit and functional tests for ONNX.js can be found in this section.

## Basic Usage
Run `npm test` after having run the install/ci command
```
npm ci
npm test
```

All tests are run in a browser instance (Chrome by default -- see below for Test Options)

## Test Categories
There are currently three categories of tests:

### Unit Tests
These are located under the folder `test/lib` and are free-form tests written to test specific areas of concern within code:


To run these tests simply type:
```
npm test -- unittest
```

### Model Tests
These are tests based on Onnx models and their inputs and outputs are also in binary `ProtoBuf` formats. Data for these
tests is located under `test/test-data/models` and `test/test-data/nodes`.

To run individual Model tests for the included, well-known models (such as Resnet50) type the following:
```
npm test -- model test/test-data/onnx/v7/resnet
```
Replace `resnet` with the name of any other model located under `test/test-data/onnx/v7` folder.


To run individual Model tests for the unit-level models (such as `abs`) type the following:
```
npm test -- model test/test-data/node/test_abs
```
Substitute `test_abs` with any other sub-folder located under the `test/test-data/node` folder.

### Op Tests
These are specifically scripted to test the Operators in ONNX.js. The test data can be found under `test/test-data/ops`

To run an individual operator test use the following command:
```
npm test -- op test/test-data/ops/abs.jsonc
```
Replace `abs.jsonc` with any other operator data file located under the `test/test-data/ops` folder.

## More information
For further info type the following:
```
npm test -- --help
```
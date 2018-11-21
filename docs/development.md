## Prerequisites
- Node.js (10.13.0+): https://nodejs.org/en/
	- (Optional) Use nvm ([Windows](https://github.com/coreybutler/nvm-windows) / [Mac/Linux] (https://github.com/creationix/nvm)) to install Node.js

- Python (2.7 or 3.6+): https://www.python.org/downloads/
    - python should be added to the PATH environment variable

- Visual Studio Code: https://code.visualstudio.com/
    - **required** extension: [TSLint](https://marketplace.visualstudio.com/items?itemName=eg2.tslint)
    - **required** extension: [Clang-Format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
    - **required** extension: [Debugger for Chrome](https://marketplace.visualstudio.com/items?itemName=msjsdiag.debugger-for-chrome)

- Chrome Browser

- (Optional) Netron: https://lutzroeder.github.io/Netron/

## Development

Please follow the following steps to running tests:

1. run `npm ci` in the root folder of the repo.
2. (Optional) run `npm run build` in the root folder of the repo to enable WebAssebmly features.
3. run `npm test` to run suite0 test cases and check the console output.
	- if (2) is not run, please run `npm test -- -b=cpu,webgl` to skip WebAssebmly tests 
	
To debug the code from test-runner on Chrome:
- make sure `npm ci` executed at least once
- use **vscode** to open the root folder.
- run `npm test -- <your-args> --debug` to run one or more test cases.
- In the open Chrome browser, click the `DEBUG` button on the top-right of the page
- In VSCode, click [menu]->Debug->Start Debugging or press F5, select `Karma DEBUG RUNNER` to attach
- put breakpoints in source code, and Refresh the page to reload


*Note: run `npm test -- --help` for a full CLI instruction.*


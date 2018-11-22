## WASM-OPS

All source code under the sub-folder './wasm-ops' contains ONNX operator implementations that can be compiled and exported to a WebAssembly binary file (.wasm file extension)

## WASM-BUILD-CONFIG

'wasm-build-config.config' contains the configurations pertaining to building the source code under './ops' and which specific functions are to be exported into the .wasm file. Only functions exported into the .wasm file can be invoked from JavaScript.
 
	* The section 'src' in 'wasm-build-config.json' contains the pattern specifying the source code files to be used for compilation.
	* The section 'exported_functions' in 'wasm-build-config.json' specifies the functions that are to be exported into .wasm file. Please export only the functions that are needed so as to keep the .wasm file size small. Please do not remove functions from the list unless you are absolutely sure about what you are doing. There are some WebAssembly usage specific functions (for example - '_malloc', '_free') that are needed to use WebAssembly itself. Please make sure they remain exported.

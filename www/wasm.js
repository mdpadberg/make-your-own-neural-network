import init from './pkg/wasm.js';

async function wasmInit() {
    await init();
}

wasmInit();
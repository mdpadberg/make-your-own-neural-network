#!/bin/bash
version=$(wasm-bindgen -V)
if [[ $version != *"wasm-bindgen"* ]]; then
    cargo install -f wasm-bindgen-cli
fi
cargo build -p wasm --target wasm32-unknown-unknown --release && \
wasm-bindgen target/wasm32-unknown-unknown/release/wasm.wasm --target web --out-dir www/pkg
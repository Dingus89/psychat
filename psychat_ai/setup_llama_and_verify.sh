#!/bin/bash

# CONFIGURATION
MODEL_PATH=~/psychat/psychat_ai/models/llama3/llama3-psychiatrist-v1.3B-fp16.Q4_K_M.gguf
LLAMA_DIR=~/llama.cpp

echo "[1/5] Installing dependencies..."
sudo apt update && sudo apt install -y build-essential cmake git

echo "[2/5] Cloning llama.cpp (or updating if already exists)..."
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
else
    cd "$LLAMA_DIR"
    git pull origin master
fi

echo "[3/5] Building llama.cpp from source..."
cd "$LLAMA_DIR"
rm -rf build
cmake -B build
cmake --build build --config Release

# Check if build succeeded
if [ ! -f build/bin/main ]; then
    echo "[ERROR] Build failed: main binary not found at build/bin/main"
    exit 1
fi

echo "[4/5] Checking model info for:"
echo "$MODEL_PATH"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found at: $MODEL_PATH"
    exit 1
fi

echo "[5/5] Model metadata output:"
build/bin/main -m "$MODEL_PATH" --info
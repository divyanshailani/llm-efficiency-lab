#!/bin/bash
set -e

LLAMA_BENCH="../../tools/llama.cpp/build/bin/llama-bench"
MODEL_PATH="../../models/gemma-4-12b-it-qat-q4_0.gguf"

if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    echo "Please build llama.cpp first."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure gemma-4-12b-it-qat-q4_0.gguf is in the models directory."
    exit 1
fi

echo "===================================================================="
echo " Starting Gemma 4 12B Speed Benchmark in SAFE MODE (-b 512 -ub 512)"
echo "===================================================================="
echo "Note: Using q8_0 KV Cache and Flash Attention."
echo "--------------------------------------------------------------------"

$LLAMA_BENCH -m "$MODEL_PATH" \
  -p 512,2048,4096 \
  -n 128 \
  -r 3 \
  -b 512 \
  -ub 512 \
  -fa 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -t 4

echo "--------------------------------------------------------------------"
echo "Benchmark complete."

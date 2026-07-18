#!/usr/bin/env bash
# run_benchmark.sh
# 
# Description: Reproducible benchmark script for Qwythos-9B-v2 on Apple Silicon (M4).
# This script downloads the required GGUF files and runs llama-bench.

set -e

REPO_ID="mradermacher/Qwythos-9B-v2-GGUF"
MODELS=("Qwythos-9B-v2.Q4_K_M.gguf" "Qwythos-9B-v2.Q8_0.gguf")
MODELS_DIR="../../models"
RESULTS_DIR="../../results/mac_mini_m4"
LLAMA_BENCH="../../tools/llama.cpp/build/bin/llama-bench"

# Ensure directories exist
mkdir -p "$MODELS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== LLM Efficiency Lab: Mac M4 Baseline Benchmark ==="

# 1. Download Models
echo "[1/2] Downloading Models..."
for MODEL in "${MODELS[@]}"; do
    if [ ! -f "${MODELS_DIR}/${MODEL}" ]; then
        echo "Downloading ${MODEL}..."
        # Using curl to download directly from HuggingFace to avoid hf-transfer CDN rate limits (403s) on large files
        curl -C - -O -L "https://huggingface.co/${REPO_ID}/resolve/main/${MODEL}"
        mv "${MODEL}" "${MODELS_DIR}/"
    else
        echo "${MODEL} already exists, skipping download."
    fi
done

# 2. Run Benchmarks
echo "[2/2] Running Benchmarks..."

if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    echo "Please build llama.cpp first!"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="${MODELS_DIR}/${MODEL}"
    RESULT_FILE="${RESULTS_DIR}/$(echo $MODEL | tr '[:upper:]' '[:lower:]' | sed 's/\.gguf//' | sed 's/-/_/g').json"
    
    echo "Benchmarking ${MODEL}..."
    # n_batch = 2048, n_ubatch = 512, threads = 4, prompt sizes = 512..16384, gen = 128, repetitions = 5
    "$LLAMA_BENCH" -m "$MODEL_PATH" -p 512,2048,4096,8192,16384 -n 128 -r 5 -o json > "$RESULT_FILE"
    echo "Saved results to ${RESULT_FILE}"
done

echo "Benchmark suite complete!"

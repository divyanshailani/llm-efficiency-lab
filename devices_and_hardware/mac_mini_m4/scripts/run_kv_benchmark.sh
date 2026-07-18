#!/usr/bin/env bash
# run_kv_benchmark.sh
# 
# Description: Executes Phase 4 KV cache quantization and Flash Attention scaling 
# matrix on Qwythos-9B-v2 (Mac M4).

set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
REPO_ID="mradermacher/Qwythos-9B-v2-GGUF"
MODEL_Q4_K_M="Qwythos-9B-v2.Q4_K_M.gguf"
MODEL_Q4_K_S="Qwythos-9B-v2.Q4_K_S.gguf" 
MODELS_DIR="${ROOT_DIR}/models"
RESULTS_DIR="${ROOT_DIR}/results/mac_mini_m4/kv_scaling"
LLAMA_BENCH="${ROOT_DIR}/tools/llama.cpp/build/bin/llama-bench"

mkdir -p "$MODELS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== LLM Efficiency Lab: KV Scaling & Flash Attention ==="

# 1. Download Q4_K_S Model for control test
if [ ! -f "${MODELS_DIR}/${MODEL_Q4_K_S}" ]; then
    echo "Downloading ${MODEL_Q4_K_S}..."
    curl -C - -L -o "${MODELS_DIR}/${MODEL_Q4_K_S}" "https://huggingface.co/${REPO_ID}/resolve/main/${MODEL_Q4_K_S}"
fi

if [ ! -f "$LLAMA_BENCH" ]; then
    echo "Error: llama-bench not found at $LLAMA_BENCH"
    exit 1
fi

CONTEXTS="512,2048,4096,8192,16384"
REPS=5
GEN_TOKENS=128

echo "[2/3] Running KV Quantization Matrix (Matrix 1)..."

# Note: KV quantization on Metal REQUIRES Flash Attention (-fa 1)
KV_COMBINATIONS=(
    "f16 f16"
    "q8_0 q8_0"
    "q8_0 q4_0"
    "q4_0 q8_0"
    "q4_0 q4_0"
)

for kv in "${KV_COMBINATIONS[@]}"; do
    read -r k_type v_type <<< "$kv"
    RESULT_FILE="${RESULTS_DIR}/kv_${k_type}_${v_type}.json"
    echo "Benchmarking KV: K=${k_type}, V=${v_type} (FA ON)..."
    "$LLAMA_BENCH" -m "${MODELS_DIR}/${MODEL_Q4_K_M}" -p $CONTEXTS -n $GEN_TOKENS -r $REPS -t 4 --cache-type-k $k_type --cache-type-v $v_type -fa 1 -o json > "$RESULT_FILE"
done

echo "[3/3] Running Threads Matrix (Matrix 2 & 3)..."

# Thread Scaling Test (Using optimal symmetric K=q8_0 V=q8_0, FA=1)
for t in 6 8 10; do
    echo "Benchmarking Threads=${t}..."
    "$LLAMA_BENCH" -m "${MODELS_DIR}/${MODEL_Q4_K_M}" -p $CONTEXTS -n $GEN_TOKENS -r $REPS -t $t --cache-type-k q8_0 --cache-type-v q8_0 -fa 1 -o json > "${RESULTS_DIR}/threads_${t}_q8_q8.json"
done

# Q4_K_S vs Q4_K_M Control Test (Default settings, FA=1, t=4)
echo "Benchmarking Q4_K_S Control Model..."
"$LLAMA_BENCH" -m "${MODELS_DIR}/${MODEL_Q4_K_S}" -p $CONTEXTS -n $GEN_TOKENS -r $REPS -t 4 -fa 1 -o json > "${RESULTS_DIR}/model_q4_k_s_baseline.json"

echo "KV Benchmark suite complete! All JSONs saved to ${RESULTS_DIR}."

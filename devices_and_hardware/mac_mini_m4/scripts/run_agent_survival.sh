#!/bin/bash
set -e

# Path to the llama-server binary
LLAMA_SERVER="../../tools/llama.cpp/build/bin/llama-server"
MODEL_PATH="../../models/gemma-4-12b-it-qat-q4_0.gguf"
BENCHMARK_SCRIPT="../../benchmarks/micro_quality/agent_survival_benchmark.py"

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "Error: llama-server not found at $LLAMA_SERVER"
    echo "Please build llama.cpp first."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download gemma-4-12b-it-qat-q4_0.gguf to the models directory."
    exit 1
fi

echo "=========================================================="
echo " Starting Gemma 4 12B in SAFE MODE (-np 1 -b 512)"
echo " This strictly prevents Unified Memory OOM crashes on Mac."
echo "=========================================================="

# Start server in the background
$LLAMA_SERVER -m "$MODEL_PATH" -c 8192 -np 1 -fa 1 --cache-type-k q4_0 --cache-type-v q4_0 -b 512 -ub 512 -ngl 99 --no-mmap > server.log 2>&1 &
SERVER_PID=$!

echo "Waiting for server to load Metal weights (usually ~4 mins on M4 SSD)..."
while ! curl -s --max-time 1 http://127.0.0.1:8080/health | grep -q "ok"; do 
    sleep 10
done

echo "Server is UP! Running strict Quality Gates benchmark..."
echo "----------------------------------------------------------"

python3 "$BENCHMARK_SCRIPT"
BENCH_EXIT=$?

echo "----------------------------------------------------------"
echo "Cleaning up server process..."
kill -9 $SERVER_PID

exit $BENCH_EXIT

# Qwythos-9B-v2 Mac M4 Baseline Benchmark

This document provides reproducibility details for the local Apple Silicon baseline tests on the Qwythos 9B v2 model using a Mac mini (M4).

## Hardware Configuration
- **Device:** Mac mini (M4)
- **CPU Cores:** 10
- **Unified Memory:** 16 GB

## Software Versions
- **Test Harness:** `llama-bench` (from llama.cpp)
- **llama.cpp Version:** Commit `4937ca83f` (Build 10067)
- **Backend:** Metal (Apple Silicon native)

## Tested Models
We evaluated two quantizations of the Qwythos-9B-v2 model sourced from `mradermacher/Qwythos-9B-v2-GGUF`:
1. `Qwythos-9B-v2.Q4_K_M.gguf` (~5.4 GB)
2. `Qwythos-9B-v2.Q8_0.gguf` (~9.1 GB)

## Reproduction Steps

1. **Build llama.cpp:** Ensure you have compiled `llama.cpp` locally with Metal support enabled.
   ```bash
   cd tools/llama.cpp
   make clean && make LLAMA_METAL=1
   ```
2. **Run the Benchmark Script:** Navigate to the experiment directory and run the automated script. This script automatically handles downloading the model files via `curl` (to avoid HuggingFace hf-transfer CDN rate limits) and executing the benchmark suite.
   ```bash
   cd experiments/06_mac_m4_baseline
   ./run_benchmark.sh
   ```

### Exact Benchmark Command
For reference, the script executes the following parameters for each model:
```bash
./llama-bench -m <model_path> -p 512,2048,4096,8192,16384 -n 128 -r 5 -o json
```
- `n_batch` = 2048, `n_ubatch` = 512, `threads` = 4
- `prompt sizes` = 512, 2048, 4096, 8192, 16384
- `generation` = 128 tokens
- `repetitions` = 5

## Expected Outcomes
The output JSON files are saved to `results/mac_mini_m4/`. On a 16 GB Mac mini M4, you should observe:
- **Q4_K_M:** ~16.67 tokens/sec decode throughput, consuming ~7.4 GB unified memory without swapping.
- **Q8_0:** ~10.67 tokens/sec decode throughput, pushing the 16 GB RAM to its practical limit (~15.23 GB used, ~1.5 GB compressed) but operating without disk swapping.
- **Thermals:** Sustained temperatures around 67–75°C.

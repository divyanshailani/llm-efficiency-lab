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
2. **Run the Benchmark Script:** Navigate to the `scripts/` directory and run the automated script. This script automatically handles downloading the model files via `curl` (to avoid HuggingFace hf-transfer CDN rate limits) and executing the benchmark suite.
   ```bash
   cd scripts/
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

## Benchmark Results

Below are the aggregated results from `llama-bench` evaluating prompt processing (ingestion) and token generation on the M4 hardware. 

### Prompt Processing (Ingestion Speed)
*Measured in tokens per second (t/s)*

| Context Size (Tokens) | Qwythos-9B-v2 `Q4_K_M` | Qwythos-9B-v2 `Q8_0` |
|-----------------------|-------------------------|-----------------------|
| **512**               | 209.69 t/s             | 216.37 t/s            |
| **2048**              | 208.21 t/s             | 215.35 t/s            |
| **4096**              | 202.32 t/s             | 211.41 t/s            |
| **8192**              | 197.20 t/s             | 206.62 t/s            |
| **16384**             | 186.77 t/s             | 197.62 t/s            |

> [!NOTE]
> Interestingly, the `Q8_0` model shows slightly faster prompt ingestion speeds compared to `Q4_K_M`. This is a known phenomenon on Apple Silicon where 8-bit memory alignments can sometimes process faster through the Metal backend than asymmetric 4-bit block quantizations during the compute-heavy prompt evaluation phase.

### Text Generation (Decode Speed)
*Measured in tokens per second (t/s) for 128 tokens*

| Model Quantization | Decode Speed | 
|--------------------|--------------|
| **Q4_K_M** (~5.4 GB) | 16.67 t/s |
| **Q8_0** (~9.1 GB)   | 10.67 t/s |

> [!TIP]
> For interactive chat applications, `Q4_K_M` provides a much smoother reading experience (16+ t/s), while `Q8_0` (10.6 t/s) remains highly usable but pushes the memory limits of a 16GB system when combined with a large KV cache.

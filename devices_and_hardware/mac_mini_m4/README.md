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

## Phase 3: MLX OptiQ 4-Bit Baseline

To validate Apple's native `mlx` framework against `llama.cpp`, we benchmarked the official `mlx-community` mixed-precision quantization. 

**Configuration:**
- **Model:** `mlx-community/Qwythos-9B-v2-OptiQ-4bit`
- **MTP:** disabled (Language tower only)
- **Quantization:** OptiQ mixed precision
- **Effective BPW:** 5.211
- **Language tower size:** ~6.6 GB (Total HuggingFace cache ~8.2GB)

### Results vs llama.cpp

| Metric | `llama.cpp` (`Q4_K_M`, ~5.4 GB) | `mlx-lm` (`OptiQ 4-bit`, ~6.6 GB) |
| :--- | :--- | :--- |
| **Prompt Processing (PP)** | 209.69 tok/s (512 ctx) | **146.69 tok/s** (57 ctx) |
| **Decode Speed (TPS)** | **16.67 tok/s** | 15.17 tok/s |
| **Peak Memory Usage** | ~5.4 GB | 7.32 GB |

> [!NOTE]
> The MLX OptiQ model retains sensitive layers at 8-bit for maximum quality (yielding 5.211 bits per weight). This larger footprint (6.6 GB vs 5.4 GB) perfectly explains the slightly slower decode speed (15.17 TPS vs 16.67 TPS), validating that decode remains strictly bound by memory bandwidth on Apple Silicon.

## Phase 4: KV Cache Quantization & Flash Attention

We ran a comprehensive matrix testing `llama.cpp`'s KV cache quantization across 5 repetitions per configuration to see if we could push decode speeds closer to 20 t/s on the `Q4_K_M` baseline.

### Key Findings
1. **Flash Attention Requirement:** On the Apple Silicon (`MTL`) backend, quantized KV caches strictly require Flash Attention (`-fa 1`). Disabling it causes an immediate context creation failure.
2. **The Asymmetric Precision Trap:** Mixing Key/Value precisions (e.g., `K=q8_0, V=q4_0`) exposed a severe unoptimized fallback in the Metal kernel. Prompt processing plummeted to **53.6 t/s** (down from 193 t/s), and generation completely hung the GPU at 16k context depths. *Note: This is a known upstream `llama.cpp` issue describing this exact pattern (`q8_0/q4_0` failing on Metal while symmetric quantizations work). It is likely a backend-specific bug rather than a theoretical limitation.*
3. **Symmetric 8-Bit is Optimal:** Using `q8_0:q8_0` yielded a highly stable and measurable performance boost, reducing memory overhead while actually *increasing* decode speed slightly due to reduced bandwidth saturation.

### Symmetric `q8_0:q8_0` vs `f16:f16` (Q4_K_M)

| Metric | `f16:f16` (Baseline) | `q8_0:q8_0` |
| :--- | :--- | :--- |
| **Prompt Processing (512 ctx)** | 208.97 t/s | 207.65 t/s |
| **Prompt Processing (16k ctx)** | 186.78 t/s | 180.90 t/s |
| **Decode Speed (TPS)** | 15.75 t/s | **16.25 t/s** |

> [!TIP]
> The `q8_0:q8_0` cache provides a free 3.2% decode speedup by halving the memory bandwidth required to read the context cache during auto-regressive generation, with an imperceptible penalty to prompt ingestion speed. Always use `-fa 1 --cache-type-k q8_0 --cache-type-v q8_0` on Mac M4.

**The Main Lesson:** KV-cache quantization is primarily a memory and long-context scaling optimization. It is not automatically a major decode-speed optimization. Our decode speeds only improved from 15.75 to 16.25 t/s, demonstrating that generation remains heavily dominated by reading the 5.4 GB `Q4_K_M` model weights, not just the attention cache.

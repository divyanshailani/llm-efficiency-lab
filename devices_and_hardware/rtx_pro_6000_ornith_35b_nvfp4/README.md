# Ornith 1.0 35B NVFP4 MoE — NVIDIA RTX PRO 6000 Blackwell Benchmark

This directory contains the empirical evaluation, speed sweep, and 6-gate survival quality benchmark for **`sakamakismile/Ornith-1.0-35B-NVFP4`** (True NVFP4 W4A4 Mixture-of-Experts) deployed on an **NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)** hardware node.

## Hardware Configuration
- **GPU:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Architecture:** `sm_120` (Blackwell Compute Capability)
- **Peak Memory Bandwidth:** ~1,800 GB/s

## Software & Deployment Stack
- **Inference Engine:** `vLLM 0.22.0`
- **Attention Kernel Backend:** `FlashInfer` (CUDA 13.1 JIT compiled for Blackwell `sm_120`)
- **Quantization Format:** `compressed-tensors` (`nvfp4-pack-quantized` W4A4)

## ⚡ Context Speed Sweep Results

| Context Size (Target) | Median Ingestion Speed | Median Decode Speed | Median TTFT |
|-----------------------|------------------------|---------------------|-------------|
| **512 tokens**        | **509.73 t/s**         | **189.90 t/s**      | **1.044s**  |
| **2,048 tokens**      | **1,819.80 t/s**       | **192.13 t/s**      | **1.133s**  |
| **4,096 tokens**      | **2,375.36 t/s**       | **190.92 t/s**      | **1.731s**  |
| **8,192 tokens**      | **5,992.20 t/s**       | **191.24 t/s**      | **1.370s**  |
| **16,384 tokens**     | **8,970.23 t/s**       | **200.65 t/s**      | **1.828s**  |

## 🛡️ Survival Quality Scorecard: 6/6 GATES PASSED (100%)

- ✅ **Gate 1: Tool Calling (JSON payload)**: PASSED
- ✅ **Gate 2: Schema Validity**: PASSED
- ✅ **Gate 3: State Tracking**: PASSED
- ✅ **Gate 4: Executable Debugging**: PASSED
- ✅ **Gate 5: Edit-Plan Follow-Through**: PASSED
- ✅ **Gate 6: Long-Context Recall (4K Needle)**: PASSED (*Recalled `ORNITH_MOE_BLACKWELL_2026` cleanly*)

## 🔬 MoE Speed Mechanics
- **Active Parameters per Token**: Only 8 active experts per token out of 256 total experts (~6.5 GB active weight read per token).
- **Single-User Decode Speed**: **~190 to 200 tokens/second** (**6x faster** than dense 40B models!).
- **Hardware Efficiency**: $\frac{1800 \text{ GB/s}}{6.5 \text{ GB}} \approx 276.9 \text{ t/s}$ theoretical limit. Measured **~200 t/s** represents **~72.5% physical memory bandwidth utilization**.

## Directory Contents
- `scripts/local_nvfp4_speed_sweep.py`: Public CLI speed sweep benchmark script.
- `scripts/local_nvfp4_survival_benchmark.py`: Public CLI 6-gate survival benchmark script.
- `results/`: Official JSON benchmark results for speed sweep and survival gates.

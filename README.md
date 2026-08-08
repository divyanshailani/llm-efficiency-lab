# LLM Inference Efficiency Lab

Finding the best quality, memory, and speed trade-offs for quantized LLMs across Apple Silicon, Hosted GPUs, and Dedicated CPU Architectures, followed by selective mobile validation.

## Architecture & Tiers

We use a strict 3-tier testing pipeline to evaluate models before they ever touch physical edge hardware.

### Tier 1: Local Apple Silicon & Dedicated CPU Nodes
Our controlled research baseline, utilizing memory bandwidth and multi-core SIMD execution to measure prompt-processing throughput and decode speeds.
* **`Liquid AI LFM2.5`:** Hybrid 22 Double-Gated Short-Conv + 8 GQA Attention layers ($O(1)$ constant memory complexity on a 4-Core CPU Node / 8 execution threads).
* **`llama.cpp` + Metal:** The GGUF standard baseline.
* **`mlx-lm`:** Clean MLX-native single-request benchmarks.
* **`oMLX`:** Serving overhead, persistent KV caching, concurrency, and SSD-cache experiments.

### Tier 2: Hosted GPU Validation
For workloads that require massive parallel throughput or specialized tensor precision, we leverage hosted NVIDIA infrastructure to stress-test scaling limits.
* **Experiments:** True NVFP4 W4A4 MoE models (Ornith 35B, Agents-A1 35B, Qwopus 122B), aggressive KV-cache quantization, large batch size stress tests, and long-context scaling (4K, 8K, 16K, 32K, and 128K).

### Tier 3: Selective Android Validation
Once the optimal configurations are mapped on Tiers 1 and 2, we selectively deploy the strongest candidates to constrained physical edge devices (e.g., 6GB RAM Android phones) to validate real-world feasibility.

## Repository Structure

We organize evaluations into strict **device-and-hardware-centric** directories paired with exact reproduction scripts, sanitized results, and documentation:

* [`devices_and_hardware/cpu_4core_liquid_lfm2.5_2.6b/`](./devices_and_hardware/cpu_4core_liquid_lfm2.5_2.6b/README.md): 4-Core (8-Thread) CPU benchmark for Liquid AI LFM2.5 (100% 11/11 Master Gates, $O(1)$ memory).
* [`devices_and_hardware/mac_mini_m4_qwythos_9b/`](./devices_and_hardware/mac_mini_m4_qwythos_9b/README.md): M4 baseline scripts, JSON results, and benchmark guides.
* [`devices_and_hardware/rtx_pro_6000_ornith_35b_nvfp4/`](./devices_and_hardware/rtx_pro_6000_ornith_35b_nvfp4/README.md): Blackwell NVFP4 35B MoE testing (~200 tok/s).
* [`devices_and_hardware/samsung_f23_5g_qwen2.5_3b/`](./devices_and_hardware/samsung_f23_5g_qwen2.5_3b/README.md): Physical edge testing scripts, Termux setups, and Context Cliff metrics.
* `telemetry/`: Core profiling engine for capturing hardware metrics.
* `docs/`: Theory and general architecture documentation.

## Methodology and Limitations

> [!WARNING]
> Please note that all results documented in this repository are heavily **hardware-**, **runtime-**, **model-**, and **context-dependent**. 
> - **Hardware:** Variations in memory bandwidth, core counts, and thermal throttling (especially on mobile) significantly alter real-world metrics.
> - **Runtime:** Different inference engines (`llama.cpp`, `Transformers`, `MLX`, `vLLM`) utilize hardware accelerators (AVX-512, Metal, CUDA) differently.
> - **Model & Context:** Quantization formats and context lengths dictate whether an operation is compute-bound or memory-bandwidth bound.

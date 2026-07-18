# LLM Inference Efficiency Lab

Finding the best quality, memory, and speed trade-offs for quantized LLMs across Apple Silicon and Hosted GPUs, followed by selective mobile validation.

## Architecture & Tiers

We use a strict 3-tier testing pipeline to evaluate models (like Qwen 3B, 7B, 9B) before they ever touch an edge device.

### Tier 1: Local Apple Silicon
Our controlled research baseline, utilizing the massive memory bandwidth of Apple Silicon (Unified Memory) to measure prompt-processing throughput (PP) and decode speeds.
* **`llama.cpp` + Metal:** The GGUF standard baseline.
* **`mlx-lm`:** Clean MLX-native single-request benchmarks.
* **`oMLX`:** Serving overhead, persistent KV caching, concurrency, and SSD-cache experiments.

### Tier 2: Hosted GPU Validation
For workloads that exceed local memory constraints, we leverage hosted NVIDIA infrastructure to stress-test scaling limits.
* **Experiments:** TurboQuant implementations, aggressive KV-Cache quantization, large batch size stress tests, and long-context scaling (4K, 8K, 16K, 32K, and conditionally 128K).

### Tier 3: Selective Android Validation
Once the optimal configurations are mapped on Tiers 1 and 2, we selectively deploy the strongest candidates to constrained physical edge devices (e.g., 6GB RAM Android phones) to validate real-world feasibility.

## Repository Structure

We have moved to a strict **device-centric** repository structure so that anyone can easily navigate to their target hardware and find the exact scripts, results, and documentation.

* `devices_and_hardware/mac_mini_m4/`: Contains the M4 baseline scripts, JSON results, and benchmark guides.
* `devices_and_hardware/samsung_f23_5g/`: Contains the physical edge testing scripts, Termux setups, and results for the Samsung F23.
* `telemetry/`: Core profiling engine for capturing hardware metrics.
* `docs/`: Theory and general architecture documentation.

## Milestones
* **Phase 1 (Complete):** Established edge constraints (Context cliff and KV memory scaling) on Samsung Galaxy F23 5G via native `llama-bench`. See [devices_and_hardware/samsung_f23_5g/README.md](devices_and_hardware/samsung_f23_5g/README.md) and `devices_and_hardware/samsung_f23_5g/scripts` for reproducible scripts.
* **Phase 2 (Complete):** Local Apple Silicon baseline tests on Mac mini (M4, 16GB) for Qwythos-9B-v2 (`Q4_K_M` vs `Q8_0`) using `llama-bench`. See [devices_and_hardware/mac_mini_m4/README.md](devices_and_hardware/mac_mini_m4/README.md) and `devices_and_hardware/mac_mini_m4/scripts` for reproducible scripts.
* **Phase 3 (Active):** Local Apple Silicon tests on Qwen 3B across `llama.cpp` and `MLX` runtimes.

## Methodology and Limitations

> [!WARNING]
> Please note that all results documented in this repository are heavily **hardware-**, **runtime-**, **model-**, and **context-dependent**. 
> - **Hardware:** Variations in memory bandwidth, core counts, and thermal throttling (especially on mobile) significantly alter real-world metrics.
> - **Runtime:** Different inference engines (`llama.cpp`, `MLX`) utilize hardware accelerators (Metal, NEON) differently.
> - **Model & Context:** Quantization formats and context lengths dictate whether an operation is compute-bound or memory-bandwidth bound.

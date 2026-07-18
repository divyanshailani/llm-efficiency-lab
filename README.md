# LLM Efficiency Lab

Finding the best quality, memory, and speed trade-offs for quantized LLMs across Apple Silicon and Hosted GPUs, followed by selective mobile validation.

## Architecture & Tiers

We use a strict 3-tier testing pipeline to evaluate models (like Qwen 3B, 7B, 9B) before they ever touch an edge device.

### Tier 1: Local Apple Silicon
Our controlled research baseline, utilizing the massive memory bandwidth of Apple Silicon (Unified Memory) to measure raw prompt processing (TTFT) and decode speeds.
* **`llama.cpp` + Metal:** The GGUF standard baseline.
* **`mlx-lm`:** Clean MLX-native single-request benchmarks.
* **`oMLX`:** Serving overhead, persistent KV caching, concurrency, and SSD-cache experiments.

### Tier 2: Hosted GPU Validation
For workloads that exceed local memory constraints, we leverage hosted NVIDIA infrastructure to stress-test scaling limits.
* **Experiments:** TurboQuant implementations, aggressive KV-Cache quantization, large batch size stress tests, and long-context scaling (4K, 8K, 16K, 32K, and conditionally 128K).

### Tier 3: Selective Android Validation
Once the optimal configurations are mapped on Tiers 1 and 2, we selectively deploy the strongest candidates to constrained physical edge devices (e.g., 6GB RAM Android phones) to validate real-world feasibility.

## Repository Structure

* `telemetry/`: Core profiling engine for capturing hardware metrics.
* `experiments/`: Targeted hypothesis scripts spanning the three tiers.
* `results/`: Output telemetry, benchmarks, and JSON data.
* `docs/`: Theory and deployment documentation.

## Milestones
* **Phase 1 (Complete):** Established edge constraints (Context cliff and KV memory scaling) on Samsung Galaxy F23 5G via native `llama-bench`.
* **Phase 2 (Complete):** Local Apple Silicon baseline tests on Mac mini (M4, 16GB) for Qwythos-9B-v2 (`Q4_K_M` vs `Q8_0`) using `llama-bench`. See [docs/mac_mini_m4_benchmark_guide.md](docs/mac_mini_m4_benchmark_guide.md) and `experiments/06_mac_m4_baseline` for reproducible scripts.
* **Phase 3 (Active):** Local Apple Silicon tests on Qwen 3B across `llama.cpp` and `MLX` runtimes.

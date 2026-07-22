# Devices & Hardware Benchmarks Index

This directory organizes all empirical edge and serverless hardware evaluations into dedicated, self-contained device directories. Each device folder contains its own `README.md`, specialized `scripts/`, and official `results/`.

---

## 📱 Hardware Categories & Device Sitemap

### 1. 📱 [Samsung Galaxy F23 5G](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/samsung_f23_5g/README.md)
- **Target Device**: Samsung Galaxy F23 5G (Snapdragon 750G, 6GB Shared System RAM)
- **Target Model**: Qwen2.5 3B (`Q4_K_M`)
- **Runtime Environment**: Termux ARM64 Native `llama.cpp`
- **Key Finding**: Discovered the "Context Cliff" at **3,800 tokens** where KV-cache memory pressure triggers Android Low Memory Killer (LMK).
- **Scripts**: `scripts/run_context_sweep.py`, `scripts/01_quant_scaling_edge.py`, `scripts/02_termux_native_benchmark.py`

### 2. 💻 [Apple Mac Mini M4 (16GB)](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/mac_mini_m4/README.md)
- **Target Device**: Apple Mac Mini M4 (10-Core CPU, 16GB Unified Memory)
- **Target Models**: Qwythos 9B v2 (`Q4_K_M` & `Q8_0`), Gemma-2 9B
- **Runtime Environment**: Apple Metal native `llama-bench` & MLX engine
- **Key Finding**: Metal memory alignment gives `Q8_0` ~4% faster prompt evaluation than `Q4_K_M`.
- **Scripts**: `scripts/run_benchmark.sh`, `scripts/run_gemma_speed_benchmark.sh`, `scripts/run_kv_benchmark.sh`, `scripts/run_mlx_benchmark.py`, `scripts/run_agent_survival.sh`

### 3. 🚀 [NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/RTX_PRO_6000_16GB_RAM/README.md)
- **Target Device**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, ~1,800 GB/s peak memory bandwidth)
- **Target Models**: 
  - **Qwen3.6 Deckard 40B NVFP4**: ~34 t/s decode, 4,375 t/s ingestion.
  - **Ornith 1.0 35B NVFP4 MoE**: **~200 t/s decode** (6x faster), 8,970 t/s ingestion, **100% 6/6 Survival Scorecard**.
- **Runtime Environment**: vLLM `0.22.0` + FlashInfer CUDA 13.1 JIT (`SM120` compute capability)
- **Scripts**: `scripts/footprint_audit.py`, `scripts/local_nvfp4_speed_sweep.py`, `scripts/local_nvfp4_survival_benchmark.py`, `scripts/run_local_server.sh`

### 4. ⚡ [NVIDIA A100 (16GB VRAM)](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/A100_16GB_RAM/README.md)
- **Target Device**: NVIDIA A100 Tensor Core GPU (16GB VRAM Slice)
- **Runtime Environment**: vLLM serverless container
- **Scripts**: `scripts/run_speed_benchmark.py`, `scripts/run_survival_benchmark.py`

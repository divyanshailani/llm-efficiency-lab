# Hardware & Model Benchmark Directories Index

Every evaluation in this repository is organized into a dedicated, self-contained directory paired explicitly by **Hardware Device & Model Name**. Each directory contains its own standalone `README.md`, dedicated tester `scripts/`, and official `results/`.

---

## 📁 Hardware & Model Directory Sitemap

### 1. ⚡ [cpu_8core_liquid_lfm2.5_2.6b](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/cpu_8core_liquid_lfm2.5_2.6b/README.md)
- **Hardware**: Serverless CPU Node (8.0 vCPUs, AVX-512 SIMD, 8–10GB RAM, Zero GPU)
- **Model**: Liquid AI LFM2.5 2.6B (Hybrid 22 Double-Gated Short-Conv + 8 GQA Attention)
- **Runtime**: PyTorch 2.4+ CPU SIMD / Native AVX-512
- **Key Finding**: **100% 11/11 Master Gate Score** (6/6 Survival + 5/5 Reasoning). Short convolutions operate with **$O(1)$ constant memory complexity**, completely eliminating KV-cache DDR5 bandwidth bottlenecks on CPU.

### 2. 📱 [samsung_f23_5g_qwen2.5_3b](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/samsung_f23_5g_qwen2.5_3b/README.md)
- **Hardware**: Samsung Galaxy F23 5G (Snapdragon 750G, 6GB Shared System RAM)
- **Model**: Qwen2.5 3B (`Q4_K_M`)
- **Runtime**: Termux ARM64 Native `llama.cpp`
- **Key Finding**: Identified "Context Cliff" at **3,800 tokens** where KV-cache causes low memory swap / LMK crashes.

### 3. 💻 [mac_mini_m4_qwythos_9b](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/mac_mini_m4_qwythos_9b/README.md)
- **Hardware**: Apple Mac Mini M4 (10-Core CPU, 16GB Unified RAM)
- **Model**: Qwythos 9B v2 (`Q4_K_M` & `Q8_0`) & Gemma-2 9B
- **Runtime**: Apple Metal Native `llama-bench` & MLX Engine
- **Key Finding**: Metal memory alignment makes `Q8_0` ~4% faster during prompt evaluation than asymmetric `Q4_K_M`.

### 4. 🚀 [rtx_pro_6000_deckard_40b_nvfp4](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/rtx_pro_6000_deckard_40b_nvfp4/README.md)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, ~1,800 GB/s bandwidth)
- **Model**: Qwen3.6 Deckard 40B NVFP4 (Dense Multimodal, 35.7 GB active weight footprint)
- **Runtime**: vLLM `0.22.0` + FlashInfer CUDA 13.1 JIT (`sm_120`)
- **Key Finding**: Single-user decode speed is 33.9 t/s (~67.2% of physical memory bandwidth ceiling).

### 5. 🏆 [rtx_pro_6000_ornith_35b_nvfp4](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/rtx_pro_6000_ornith_35b_nvfp4/README.md)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, ~1,800 GB/s bandwidth)
- **Model**: Ornith 1.0 35B NVFP4 (True NVFP4 W4A4 Mixture-of-Experts)
- **Runtime**: vLLM `0.22.0` + FlashInfer CUDA 13.1 JIT (`sm_120`)
- **Key Finding**: **~200 t/s decode** (6x faster than dense 40B), 8,970 t/s ingestion, **100% 6/6 Survival Scorecard**.

### 6. 🔬 [rtx_pro_6000_agents_a1_nvfp4](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/rtx_pro_6000_agents_a1_nvfp4/README.md)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, ~1,800 GB/s bandwidth)
- **Model**: InternScience Agents-A1 NVFP4 (True NVFP4 W4A4 MoE 35B)
- **Runtime**: vLLM `0.22.0` + FlashInfer CUDA 13.1 JIT (`sm_120`)
- **Key Finding**: Purpose-built agentic web model (GAIA 96.0, IFEval 94.8) delivering ~200 t/s decode for Tavily + Playwright browsing loops.

### 7. 👑 [rtx_pro_6000_qwopus3.5_122b_nvfp4](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/rtx_pro_6000_qwopus3.5_122b_nvfp4/README.md)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, ~1,800 GB/s bandwidth)
- **Model**: OpenYourMind Qwopus3.5 122B NVFP4 (True NVFP4 W4A4 MoE 122B, ~10B Active)
- **Runtime**: vLLM `0.22.0` + FlashInfer CUDA 13.1 JIT (`sm_120`)
- **Key Finding**: Distilled from Claude Opus 3.5 & Kimi K2.6; delivers ~200 t/s decode with 122B parameter reasoning depth.

### 8. ⚡ [a100_16gb_baseline](file:///Users/divyanshailani/Desktop/llm%20experiments/devices_and_hardware/a100_16gb_baseline/README.md)
- **Hardware**: NVIDIA A100 Tensor Core GPU (16GB VRAM Slice)
- **Model**: Baseline 7B/14B models
- **Runtime**: vLLM serverless endpoint

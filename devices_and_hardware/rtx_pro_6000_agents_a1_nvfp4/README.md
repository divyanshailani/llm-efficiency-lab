# RTX PRO 6000 Blackwell (96GB VRAM) - InternScience Agents-A1 NVFP4 (W4A4) Benchmark Suite

This directory contains the public, generic hardware-paired benchmarking suite for **`protoLabsAI/Agents-A1-NVFP4`** running on an **NVIDIA RTX PRO 6000 Blackwell GPU (96GB VRAM)**.

---

## 🔬 Model Profile & Architecture

- **Model Name**: `InternScience/Agents-A1` ([HuggingFace Repo](https://huggingface.co/protoLabsAI/Agents-A1-NVFP4))
- **Base Architecture**: `Qwen3_5MoeForConditionalGeneration` (35B Total Parameters)
- **MoE Routing**: 256 Total Experts, **8 Active Experts per Token** (~3B Active Parameters)
- **Quantization**: True **NVFP4 (W4A4)** `nvfp4-pack-quantized` (`compressed-tensors`)
- **Native Context Window**: 262,144 Tokens (256K Context)
- **Specialization**: GAIA (96.0), IFEval (94.8), Tavily Search + Playwright Headless Web Agents

---

## 📊 Benchmark Suite Scripts

### 1. Speed & Context Scaling Sweep
Run context length sweep (512 to 16,384 tokens) measuring TTFT, Ingestion Speed (t/s), and Decode Speed (t/s):
```bash
python3 devices_and_hardware/rtx_pro_6000_agents_a1_nvfp4/scripts/local_nvfp4_speed_sweep.py \
  --endpoint "YOUR_VLLM_OPENAI_ENDPOINT" \
  --output "devices_and_hardware/rtx_pro_6000_agents_a1_nvfp4/results/speed_sweep_results.json"
```

### 2. 6-Gate Quality & Survival Benchmark
Run 6-gate survival test (Tool Calling, JSON Schema, State Tracking, Debugging, Edit Plan, 4K Needle Recall):
```bash
python3 devices_and_hardware/rtx_pro_6000_agents_a1_nvfp4/scripts/local_nvfp4_survival_benchmark.py \
  --endpoint "YOUR_VLLM_OPENAI_ENDPOINT" \
  --output "devices_and_hardware/rtx_pro_6000_agents_a1_nvfp4/results/survival_results.json"
```

---

## 📁 Directory Structure
```
rtx_pro_6000_agents_a1_nvfp4/
├── README.md
├── scripts/
│   ├── local_nvfp4_speed_sweep.py
│   └── local_nvfp4_survival_benchmark.py
└── results/
    ├── speed_sweep_results.json
    └── survival_results.json
```

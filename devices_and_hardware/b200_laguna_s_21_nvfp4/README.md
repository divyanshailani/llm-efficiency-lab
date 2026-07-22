# NVIDIA B200 (192GB VRAM) - Poolside Laguna-S-2.1 NVFP4 Benchmark Suite

This directory contains the public, generic hardware-paired benchmarking suite for **`poolside/Laguna-S-2.1-NVFP4`** running on an **NVIDIA B200 GPU (192GB VRAM)**.

---

## 🔬 Model Profile & Architecture

- **Model Name**: `poolside/Laguna-S-2.1-NVFP4` ([HuggingFace Repo](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4))
- **Base Architecture**: `LagunaForCausalLM` (~68B Total Parameters)
- **MoE Routing**: 256 Total Experts, **10 Active Experts per Token**
- **Quantization**: True **NVFP4 (W4A4)** `nvfp4-pack-quantized` (`compressed-tensors`)
- **Native Context Window**: 262,144 Tokens (256K Context)
- **vLLM Parsers**: `--tool-call-parser poolside_v1` & `--reasoning-parser poolside_v1`

---

## ⚡ Performance Summary (B200 NVFP4)

- **Peak Generation/Decode Speed**: **~208.5 tokens/sec**
- **Peak Ingestion Speed**: **7,915.69 tokens/sec** at 16,384 context
- **TTFT (Time to First Token)**: **1.08s** at 2K context, **2.07s** at 16K context
- **6-Gate Survival Score**: **4 / 6 PASSED (66.7%)**

---

## 📊 Benchmark Suite Scripts

### 1. Speed & Context Scaling Sweep
Run context length sweep (512 to 32,768 tokens) measuring TTFT, Ingestion Speed (t/s), and Decode Speed (t/s):
```bash
python3 devices_and_hardware/b200_laguna_s_21_nvfp4/scripts/local_nvfp4_speed_sweep.py \
  --endpoint "http://localhost:8000/v1" \
  --output "devices_and_hardware/b200_laguna_s_21_nvfp4/results/speed_sweep_results.json"
```

### 2. 6-Gate Quality & Survival Benchmark
Run 6-gate survival test (Tool Calling, JSON Schema, State Tracking, Debugging, Edit Plan, 4K Needle Recall):
```bash
python3 devices_and_hardware/b200_laguna_s_21_nvfp4/scripts/local_nvfp4_survival_benchmark.py \
  --endpoint "http://localhost:8000/v1" \
  --output "devices_and_hardware/b200_laguna_s_21_nvfp4/results/survival_results.json"
```

---

## 📁 Directory Structure
```
b200_laguna_s_21_nvfp4/
├── README.md
├── scripts/
│   ├── local_nvfp4_speed_sweep.py
│   └── local_nvfp4_survival_benchmark.py
└── results/
    ├── speed_sweep_results.json
    └── survival_results.json
```

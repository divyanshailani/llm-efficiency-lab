# RTX PRO 6000 Blackwell (96GB VRAM) - OpenYourMind Qwopus3.5 122B (A10B) NVFP4 Benchmark Suite

This directory contains the public, generic hardware-paired benchmarking suite for **`OpenYourMind/Qwopus3.5-122B-A10B-Kimi-K2.6-destilled-abliterated-NVFP4`** running on an **NVIDIA RTX PRO 6000 Blackwell GPU (96GB VRAM)**.

---

## 🔬 Model Profile & Architecture

- **Model Name**: `OpenYourMind/Qwopus3.5-122B-A10B-Kimi-K2.6-destilled-abliterated-NVFP4` ([HuggingFace Repo](https://huggingface.co/OpenYourMind/Qwopus3.5-122B-A10B-Kimi-K2.6-destilled-abliterated-NVFP4))
- **Base Architecture**: `Qwen3_5MoeForConditionalGeneration` (122B Total Parameters)
- **MoE Routing**: 256 Total Experts, **8 Active Experts per Token** (~10B Active Parameters)
- **Quantization**: True **NVFP4 (W4A4)** `nvfp4-pack-quantized` (`compressed-tensors`)
- **Native Context Window**: 262,144 Tokens (256K Context)
- **Specialization**: Distilled directly from **Claude Opus 3.5 & Kimi K2.6**, abliterated (uncensored), and fine-tuned for high-level architectural reasoning.

---

## 📊 Benchmark Suite Scripts

### 1. Speed & Context Scaling Sweep
Run context length sweep (512 to 16,384 tokens) measuring TTFT, Ingestion Speed (t/s), and Decode Speed (t/s):
```bash
python3 devices_and_hardware/rtx_pro_6000_qwopus3.5_122b_nvfp4/scripts/local_nvfp4_speed_sweep.py \
  --endpoint "YOUR_VLLM_OPENAI_ENDPOINT" \
  --output "devices_and_hardware/rtx_pro_6000_qwopus3.5_122b_nvfp4/results/speed_sweep_results.json"
```

### 2. 6-Gate Quality & Survival Benchmark
Run 6-gate survival test (Tool Calling, JSON Schema, State Tracking, Debugging, Edit Plan, 4K Needle Recall):
```bash
python3 devices_and_hardware/rtx_pro_6000_qwopus3.5_122b_nvfp4/scripts/local_nvfp4_survival_benchmark.py \
  --endpoint "YOUR_VLLM_OPENAI_ENDPOINT" \
  --output "devices_and_hardware/rtx_pro_6000_qwopus3.5_122b_nvfp4/results/survival_results.json"
```

---

## 📁 Directory Structure
```
rtx_pro_6000_qwopus3.5_122b_nvfp4/
├── README.md
├── scripts/
│   ├── local_nvfp4_speed_sweep.py
│   └── local_nvfp4_survival_benchmark.py
└── results/
    ├── speed_sweep_results.json
    └── survival_results.json
```

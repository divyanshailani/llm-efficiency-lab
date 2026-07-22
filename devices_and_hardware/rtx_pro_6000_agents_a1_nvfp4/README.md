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

## 📈 Official Benchmark Results & Performance Tables

### ⚡ 1. Context Speed & Latency Scaling Sweep
Evaluated on **NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)** using vLLM + FlashInfer (`sm_120`).

| Context Length | TTFT (s) | Ingestion Speed (t/s) | Generation / Decode Speed (t/s) | Total E2E Time (128 tokens) |
| :--- | :--- | :--- | :--- | :--- |
| **512 Tokens** | 735.09s *(cold-start)* | 0.70 t/s | **168.42 tokens/sec** | 735.85s |
| **2,048 Tokens** | 3.64s | 562.71 t/s | 🚀 **199.29 tokens/sec** | 4.28s |
| **4,096 Tokens** | **1.31s** | 3,107.03 t/s | **166.13 tokens/sec** | 2.08s |
| **8,192 Tokens** | 2.29s | 3,570.14 t/s | **164.29 tokens/sec** | 3.07s |
| **16,384 Tokens** | 2.31s | 🚀 **7,076.38 tokens/sec** | **160.31 tokens/sec** | 3.11s |

---

### 🛡️ 2. 6-Gate Quality & Survival Scorecard
**Scorecard**: **5/6 GATES PASSED (83.3%)**

| Gate # | Benchmark Test | Status | Inspection Notes & Output Snippet |
| :---: | :--- | :---: | :--- |
| **Gate 1** | **Tool Calling & Schema** | ✅ **PASSED** | Emitted clean XML tool call: `<tool_call><function=tavily_search><parameter=query>...` |
| **Gate 2** | **JSON Schema Validation** | ⚠️ **FAILED** | Truncated by `max_tokens=150` limit while outputting deep `<think>` reasoning block. |
| **Gate 3** | **Multi-Turn State Tracking** | ✅ **PASSED** | Correctly tracked state: `current_url='https://nimbz.ai/docs'`, `status='authenticated'`. |
| **Gate 4** | **Executable Debugging** | ✅ **PASSED** | Detected string multiplication bug in Python and output `int(val.strip('GB')) * 2`. |
| **Gate 5** | **Code Edit Plan** | ✅ **PASSED** | Produced clean multi-phase edit plan for adding exponential backoff retry logic. |
| **Gate 6** | **4K Needle Recall** | ✅ **PASSED** | Retrieved `SECRET CODE: AGENTS-A1-BLACKWELL-CHAMPION-9928` from 4,000 tokens of noise. |

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


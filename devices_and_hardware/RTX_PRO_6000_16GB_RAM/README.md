# NVIDIA RTX PRO 6000 (96GB VRAM)

This directory contains the deployment and benchmarking scripts for running the **Qwen3.6-40B-Deckard** model on the serverless Modal environment using a single NVIDIA RTX PRO 6000 Blackwell Generation GPU with 96GB of VRAM and 16GB of allocated system RAM.

## Hardware Capabilities & Limits (Q8 40B Model)

We conducted rigorous raw-hardware limits testing utilizing `llama-bench` natively compiled inside the container (bypassing HTTP overhead) and verified the exact thresholds for the `llama-server` engine.

### Flash Attention Scaling
Flash Attention is **mandatory** for this model architecture on this hardware to prevent severe ingestion latency degradation at high context lengths.

| Context Window | Flash Attention OFF | Flash Attention ON | Difference |
|----------------|---------------------|--------------------|------------|
| 512            | 2434 t/s            | 2481 t/s           | +2%        |
| 2K             | 2521 t/s            | 2592 t/s           | +3%        |
| 4K             | 2448 t/s            | 2572 t/s           | +5%        |
| 8K             | 2324 t/s            | 2537 t/s           | +9%        |
| **16K**        | **2113 t/s**        | **2469 t/s**       | **+16%**   |

**Max Baseline Speed:** `~2,500 tokens/sec` (Ingest), `30.85 tokens/sec` (Decode).

### KV Cache Limits
The 96GB of VRAM comfortably houses the 43GB Q8 model weights alongside a massive unquantized KV cache. We progressively scaled the `llama-server` context argument (`-c`) and verified that the hardware can successfully allocate and execute up to a **48,000 token context window** within the 53GB VRAM headroom without Out-Of-Memory (OOM) crashes.

## Reasoning Quirks (Survival Benchmark)
When running `deckard_survival_benchmark.py` (which explicitly decouples the `<think>` reasoning block from the main `content` string to avoid false positives), the model scored **5/6 on the Quality Gates**.

### Benchmark Scorecard: 5/6
- ✅ **1. Tool Calling (JSON Check)**
- ✅ **2. Schema Validity**
- ✅ **3. State Tracking (Normalized)**
- ✅ **4. Executable Debugging**
- ✅ **5. Edit-Plan Follow-Through**
- ❌ **6. Long-Context Recall (4K Needle)**

**The Repetition Glitch:**
Despite having 48,000 tokens of hardware headroom, the Qwen3.6 Deckard model **failed the 4K Long-Context Recall (Needle-in-a-Haystack) test**. 
- **Cause:** The model's internal reasoning heuristics suffered a severe failure mode when processing the highly synthetic, repetitive filler text (`"The quick brown fox..." * 350`). 
- **Effect:** The reasoning engine became confused and fell into an infinite repetition loop, echoing the filler text indefinitely inside its `<think>` block until it was forcefully cut off by the `max_tokens` API limit.

## Scripts (Local Testing & Diagnostics)
- `scripts/run_local_server.sh`: Example bash script to boot the model locally using `llama-server` on port 8000.
- `scripts/local_speed_sweep.py`: Rigorous API-layer speed test up to 16K context (points to localhost).
- `scripts/local_survival_benchmark.py`: The 6-gate quality test, including the isolated `<think>` parsing logic (points to localhost).
- `scripts/footprint_audit.py`: Zero-GPU cost diagnostic script auditing active weight footprint per token vs memory bandwidth limits.

---

## vLLM + FlashInfer NVFP4 (W4A4) Blackwell Deployment

We deployed the **Deckard 40B NVFP4** model on a single **NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)** serverless node on Modal using `vLLM` and `FlashInfer` (CUDA 13.1 JIT compiled for SM120).

### Rigorous Context Speed Sweep (Modal Endpoint API)

| Context Size | Ingestion Speed (TTFT) | Single-User Decode Speed | TTFT (s) |
|--------------|------------------------|--------------------------|----------|
| **512 tokens** | **541.11 t/s** | **33.92 t/s** | **0.983s** |
| **2,048 tokens** | **1,503.08 t/s** | **33.87 t/s** | **1.372s** |
| **4,096 tokens** | **2,117.82 t/s** | **33.68 t/s** | **1.942s** |
| **8,192 tokens** | **3,373.06 t/s** | **33.67 t/s** | **2.435s** |
| **16,384 tokens** | **4,374.75 t/s** | **32.88 t/s** | **3.749s** |

### Active Memory Footprint & Decode Ceiling Analysis

1. **Active Weight Footprint (35.7 GB)**:
   - Language Model Linears (NVFP4 W4A4): ~21.5 GB
   - Excluded Modules (BF16 16-bit): Visual Encoder & Projections (~6.2 GB), Embeddings & LM Head (~4.8 GB), Linear Attention / Hybrid Layers (~2.4 GB), RMSNorms & Routers (~0.8 GB).
   - Total Active Weight Read per Token: **35.7 GB** (Multimodal) / **29.5 GB** (Pure Text).

2. **Physical Decode Ceiling at Batch Size 1**:
   - RTX PRO 6000 Peak Bandwidth: **~1,800 GB/s**.
   - Theoretical Single-User Decode Ceiling: $\frac{1800 \text{ GB/s}}{35.7 \text{ GB}} = 50.42 \text{ tokens/sec}$.
   - Measured Single-User Decode Speed: **33.88 tokens/sec**.
   - Hardware Efficiency: **67.2% of theoretical memory bandwidth ceiling** (near-optimal under HTTP/vLLM framework overheads).

3. **Ingestion Parallelism**:
   - Ingestion (Prompt Processing) scales linearly up to **4,374.75 tokens/sec** at 16K context because pre-filling matrix multiplications heavily utilize Blackwell's tensor cores.

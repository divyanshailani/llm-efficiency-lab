# NVIDIA RTX PRO 6000 (96GB VRAM)

This directory contains the deployment and benchmarking scripts for running the **Qwen3.6-40B-Deckard** model on the serverless Modal environment using a single NVIDIA RTX PRO 6000 Ada Generation GPU with 96GB of VRAM and 16GB of allocated system RAM.

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

## Scripts (Local Testing)
- `scripts/run_local_server.sh`: Example bash script to boot the model locally using `llama-server` on port 8000.
- `scripts/local_speed_sweep.py`: Rigorous API-layer speed test up to 16K context (points to localhost).
- `scripts/local_survival_benchmark.py`: The 6-gate quality test, including the isolated `<think>` parsing logic (points to localhost).

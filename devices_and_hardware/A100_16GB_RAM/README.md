# Qwen 3.6 27B Cloud GPU Benchmark

This document provides reproducibility details and comprehensive architectural findings for testing the Qwen 3.6 27B model on a serverless Cloud GPU environment.

## Hardware Configuration
- **Device:** Serverless Cloud Instance
- **GPU:** 1x NVIDIA A100
- **VRAM:** 80 GB
- **System RAM:** 16 GB (Allocated baseline)

## Software Versions
- **Inference Engine:** `vLLM`
- **Attention Backend:** `FlashInfer` (Optimized C++ JIT Kernels)
- **Model:** `Qwen/Qwen3.6-27B` (bf16 precision, ~54 GB)

## Deployment Architecture

To host a model of this magnitude serverlessly, we deployed a containerized Python script using a cloud SDK. 

### Critical Infrastructure Lessons
During deployment, we uncovered several severe edge cases in serverless GPU hosting:

1. **The Web Server Blocking Bug:** Serverless HTTP proxies often demand that your setup function returns immediately. Using synchronous blocking calls (like `subprocess.run()`) to launch `vLLM` will trick the cloud proxy into thinking the container is permanently stuck initializing. This leads to the container being violently killed after its maximum startup timeout (e.g., 20 minutes), creating a devastating and expensive crash loop. **Fix:** Always use a non-blocking background spawn (like `subprocess.Popen()`) so the cloud proxy can connect to the port instantly.
2. **DNS Propagation Delays (The 404 Bug):** Serverless ingress routes take roughly 10-15 seconds to propagate to the global edge network after deployment. Automated testing scripts that ping the endpoint immediately will fail with a `404 Not Found`. **Fix:** Add a `sleep 15` buffer before pinging new endpoints.
3. **Ghost Containers:** Terminating scripts locally does not always gracefully kill the cloud container. Always explicitly run the cloud provider's stop command before deploying structural changes to avoid port conflicts and wasted credits.
4. **Volume Syncing Bugs:** Attempting to write telemetry to standard cloud volumes via disk I/O results in heavy caching delays (files won't appear to the client script until the container shuts down). **Fix:** We bypassed disk I/O entirely by using a synchronized Redis-backed cloud Dictionary for live, sub-second telemetry streaming.
5. **Multiline String Syntax Bugs:** When generating bash scripts programmatically using multiline strings (`"""`), unescaped double quotes inside tools like `awk` can prematurely terminate the string and trigger silent syntax errors. Always use raw strings (`r"""`) to safeguard shell escapes.

## Benchmark Results: Speed Sweep

Using our automated `qwen_speed_sweep.py` script, we benchmarked the A100's capacity across various context lengths.

### Raw Telemetry
During the sweep, we captured live hardware telemetry:
- **VRAM Usage:** `68,567 MB / 81,920 MB` (Flatlined at 0.85 utilization)
- **RAM Usage:** `~5.5 GB / 16 GB` (Extremely lightweight Python/Uvicorn overhead)

### Speed Metrics

| Context Size (Tokens) | Prompt Processing (Ingestion) | Decode Speed |
|-----------------------|-------------------------------|--------------|
| **512** | 361.12 t/s | 28.51 t/s |
| **2048** | 1177.82 t/s | 28.78 t/s |
| **4096** | 1871.54 t/s | 28.53 t/s |
| **8192** | 2515.93 t/s | 28.05 t/s |
| **16384** | **2828.91 t/s** | **28.00 t/s** |

> [!NOTE]
> **Why is Decode Speed "only" 28 t/s on an 80GB A100?**
> You might expect an A100 to generate text significantly faster. However, auto-regressive generation (Decode Speed) is completely bound by **Memory Bandwidth**, not raw compute (TFLOPS). To generate a single token, the GPU must read all 54 GB of model weights. 
> The A100 has a memory bandwidth of roughly 2,000 GB/s. 
> `2,000 GB/s / 54 GB = ~37 tokens/sec` absolute theoretical maximum. 
> Achieving a rock-solid **28 t/s** is an incredibly highly-optimized result for a 27B model at Batch Size 1. 

> [!TIP]
> Conversely, Prompt Processing (Ingestion) scales massively up to **2,828 t/s** because pre-filling a prompt is a highly parallelizable matrix multiplication task that utilizes the A100's massive compute cores, effectively bypassing the memory bandwidth bottleneck!

## Phase 2: Agent Survival Benchmark

After profiling the raw speed, we ran the model through our rigorous `qwen_survival_benchmark.py` to evaluate its agentic capabilities (Tool Use, State Tracking, Executable Code).

### Benchmark Score: 3/6

| Test Case | Result |
| :--- | :--- |
| **Tool JSON Check** | ❌ FAIL |
| **Schema Validity** | ❌ FAIL |
| **State Tracking** | ✅ PASS |
| **Executable Debugging** | ✅ PASS |
| **Edit-Plan Follow-Through** | ❌ FAIL |
| **Long-Context Recall (4K)** | ✅ PASS |

### Critical Findings: The "Reasoning" Trap
At first glance, a 3/6 score seems poor for a 27B model. However, analyzing the *raw* output reveals a critical architectural paradigm:

Qwen 3.6 27B is heavily fine-tuned as a **Reasoning Model**. It is essentially hardwired to "think out loud" (outputting its entire internal chain-of-thought scratchpad) before arriving at an answer. 

1. **Why it passed State Tracking, Debugging, and 4K Recall:** It is incredibly intelligent. It mathematically tracked state, successfully retrieved the `DELTA-9` needle buried in 4,000 tokens of gibberish, and properly identified a missing Python syntax colon.
2. **Why it failed Tool Calling & Edit-Plan Follow-Through:** Because it forcefully outputs its internal reasoning trace (e.g., `Here's a thinking process...`), it fundamentally breaks strict `JSON` generation schemas. Instead of silently triggering `vLLM`'s `hermes` tool parser to emit a raw JSON payload, the model leaked its conversational reasoning process to stdout. Furthermore, in the Edit-Plan test, it generated multiple conversational code blocks inside its reasoning trace, completely confusing our strict execution harnesses which expected exactly *one* committed code block.

### Final Conclusion
Qwen 3.6 27B on an A100 is an absolute powerhouse for **Data Extraction**, **Code Debugging**, and **Massive Context Summarization** (capable of chewing through 16K tokens at nearly 3,000 t/s). 

However, **it is explicitly NOT recommended for autonomous Agentic pipelines.** Its inability to strictly adhere to silent, raw JSON schemas without leaking chain-of-thought traces makes it highly volatile when interacting with external APIs or function-calling orchestrators. For rigid, tool-using multi-agent systems, models like Llama-3.1 are heavily preferred.

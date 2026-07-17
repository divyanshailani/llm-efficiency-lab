# Experiment 1: Quantization vs. Quality & Hardware Constraints

**Date**: July 2026
**Target Platform**: Android via Termux (Simulated on Modal cloud with 4 CPU Cores, 4096 MB RAM)
**Model Used**: `Qwen2.5-3B-Instruct` (GGUF format)

## 1. Goal
To evaluate how different levels of GGUF quantization affect RAM usage, inference speed (Tokens per Second), and baseline intelligence (Math logic) when heavily constrained to 4GB of RAM and 4 CPU cores, simulating an Android edge device.

## 2. Telemetry Results

| Quantization | RAM Used (MB) | Inference Speed (TPS) | GSM8K Math Logic |
| :--- | :--- | :--- | :--- |
| **Q8_0 (8-bit)** | 3,518 MB | 5.56 TPS | Passed |
| **Q4_K_M (4-bit)** | 2,006 MB | 1.84 TPS | Passed |
| **IQ3_M (3-bit)** | N/A | N/A | 404 Error (File not found in repo) |

## 3. Key Findings & Anomalies

### The Q8_0 "Barely Survived" Discovery
The `Q8_0` baseline model actually managed to load into memory on the 4GB limit, consuming ~3.5GB. This left only ~500MB of RAM for the OS overhead and context caching. While extremely dangerous for a real Android deployment (likely to OOM crash during prolonged usage or multitasking), it proves that an 8-bit 3B model is the absolute hard ceiling for 4GB RAM devices.

### The Quantization Compute Penalty (The "Q4 is Slower" Anomaly)
The most significant finding of this experiment was the inversion of expected speeds:
* **Q8_0**: 5.56 Tokens/sec
* **Q4_K_M**: 1.84 Tokens/sec (Almost 3x slower)

**Why did a smaller file run slower?**
Decoding complex `K_M` (mixed precision k-quants) matrices dynamically on a CPU requires highly specialized, low-level hardware kernel support (like AVX instructions on Intel/AMD or Neon on ARM). 
If the underlying CPU processor does not have optimized instructions for decoding these specific 4-bit blocks, `llama.cpp` falls back to scalar math. Scalar un-quantization is incredibly computationally expensive.
Conversely, `Q8_0` is just a standard 8-bit integer array. The CPU requires virtually zero computational overhead to process it, meaning it runs purely at the speed of memory bandwidth rather than compute limits.

**Deployment Lesson for Android (Termux):**
When deploying to mobile, you must balance RAM limits against the exact processor architecture (e.g., Snapdragon instruction sets). We might compress a model to 2GB to fit on the phone's RAM, only to find out the ARM processor doesn't have the right instruction set to decode it quickly, crushing our real-time generation speed. 

## 4. Next Steps
* Update the script to target `Q3_K_M` instead of `IQ3_M` to test the logic degradation at 3-bit.
* Investigate `llama.cpp` ARM optimizations for Android deployment to solve the `Q4` speed penalty.

## 5. Advanced LLMOps Telemetry (The Codex Upgrades)
To accurately benchmark an LLM on constrained hardware, simple "end-to-end" timers are insufficient. The following metrics must be isolated:

### A. TTFT (Time To First Token) vs Decode TPS
*   **Prefill (Prompt Processing):** The model reads the prompt and computes the KV Cache. This happens all at once.
*   **Decode (Generation):** The model predicts the next word, one by one.
**Why separate them?** If you measure end-to-end time, a fast generation speed might be masked by a slow prompt reading speed. In production, **TTFT** dictates user experience (users will wait for a stream, but not for a delayed start). You must use `stream=True` to capture the exact millisecond the first token fires.

### B. The GGUF `mmap` Warmup Trap
GGUF files use Memory Mapped Files (`mmap`). When `llama.cpp` loads the model, the OS doesn't immediately copy the 2GB file into physical RAM; it just creates a pointer. It only moves weights into physical RAM when the CPU actually asks for them during matrix multiplication (triggering page faults).
**Why it matters:** If you measure RAM immediately after loading the model, the OS will underreport it. You *must* run a "warmup" inference to force the OS to pull the weights into physical RAM, and *then* measure the RSS (Resident Set Size).

### C. The Long Prefill Probe
A model might generate words quickly but choke when reading a long document. Android chips often lack the memory bandwidth required for massive parallel matrix multiplication (the operation required for processing 1,000+ token prompts). You must run a separate benchmark specifically testing prompt ingestion speeds.

## 6. The Final Results Matrix (Codex Profiler)
Using the upgraded profiler with `mmap` warmup and statistical medians, we tested the absolute floor of the Qwen 3B model:

| Quantization | RAM (MB) | TTFT | Decode TPS | Long Prefill Speed | Math Logic Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q8_0** | 3,566 MB | 163 ms | **6.14 TPS** | 40.5 tok/s | Passed |
| **Q4_K_M** | 2,071 MB | 495 ms | 2.02 TPS | 11.7 tok/s | Passed |
| **Q3_K_M** | 1,706 MB | 494 ms | 2.03 TPS | 11.8 tok/s | Passed |
| **Q2_K** | **1,374 MB** | 444 ms | 2.25 TPS | 13.1 tok/s | **FAILED** (Logic Collapse) |

### The Q2_K Intelligence Floor
The 2-bit quantization dropped RAM usage down to an incredible 1.3 GB, making it trivially easy to fit on a cheap smartphone. However, the reasoning capabilities completely collapsed. 
When asked the 5 apples minus 2 plus 4 problem, the 2-bit model responded: 
> *"You start with 5 apples and then you eat 2, which is equivalent to having -2 apples. Then you buy 4 more, which is equivalent to having +4 apples. The result is -2 + 4 = 2."*

This proves that pushing quantization past 3-bit fundamentally destroys the dense mathematical weights of the network. The LLM loses its ability to track basic state logic, demonstrating the absolute baseline floor for LLMOps deployment.

## 7. Cloud Simulation vs Physical Edge Hardware
We tested the "Golden Standard" `Q4_K_M` quantization on both a simulated Cloud edge environment (Modal CPU) and a physical Android smartphone (Snapdragon 8-core via Termux).

| Metric | Modal Cloud CPU (4-Core, 4GB) | Physical Android Phone (8-Core, 5.4GB) |
| :--- | :--- | :--- |
| **TTFT (Time To First Token)** | 495 ms | **208.4 ms** (2.3x Faster) |
| **Decode Speed** | 2.02 TPS | **4.71 TPS** (2.3x Faster) |
| **Long Prefill (1k Tokens)** | 11.7 tok/s | **15.36 tok/s** (1.3x Faster) |

**Conclusion:** On a Samsung Galaxy F23 5G with Snapdragon 750G and 6GB RAM (LPDDR4X), the Qwen2.5 3B Instruct `Q4_K_M` ran locally in Termux at about 4.71 decode tokens/sec with ~208 ms TTFT using 4 worker threads. 

Physical mobile hardware (even mid-range chips with LPDDR4X memory) heavily outperforms heavily-partitioned standard cloud CPU containers for LLM inference. While the RSS drop from 3130MB to 3026MB during warmup suggests page reclamation or `mmap` allocator behavior (potentially involving zRAM/RAM Plus), the undeniable result is a massive latency and throughput victory for edge devices over simulated cloud CPUs.

### The ARM big.LITTLE Core Scaling Paradox
We re-ran the native Android benchmark using all 8 physical cores (`--threads 8`) to see if the model would scale. The results were worse:
*   **TTFT:** Slowed down to 234 ms (from 208 ms).
*   **Decode TPS:** Dropped to 4.52 TPS (from 4.71 TPS), with a massive minimum stutter of **2.01 TPS**.
*   **Long Prefill:** Marginally improved to 16.3 tok/s (from 15.3 tok/s).

**Why this happens:** The Snapdragon 750G uses an asymmetric "big.LITTLE" architecture (2 Fast Cores, 6 Slow Efficiency Cores). When forcing `llama.cpp` to use 8 threads for sequential token generation (Decode), the fast cores spend clock cycles waiting for the slow cores to finish their matrix math. Furthermore, firing all 8 cores causes immediate thermal throttling (as evidenced by the 2.01 TPS minimum drop). For Edge AI on mobile, capping threads to match the exact number of "Performance" cores (or just using 4) yields the highest and most stable TPS.

## 8. Context Scaling, KV Cache, and Lazy Mmap Memory
To find the limits of the 6GB RAM on the Samsung F23, we ran a context scaling sweep (`n_ctx` = 512, 1024, 2048, 3072, 4096) on the `Q4_K_M` model. 

Against initial expectations, the Android device easily survived `n_ctx=4096` without crashing.

| Context Window | RAM (Peak Warm) | TTFT | Decode Speed |
| :--- | :--- | :--- | :--- |
| **512** | 2872.11 MB | 217.8 ms | 4.84 TPS |
| **1024** | 3083.14 MB | 202.8 ms | 4.86 TPS |
| **2048** | 2724.42 MB | 237.0 ms | 4.16 TPS |
| **3072** | 2726.71 MB | 205.7 ms | 4.83 TPS |
| **4096** | 3077.18 MB | 207.3 ms | 4.88 TPS |

### Finding 1: Capacity vs Allocated Cache (The GQA Factor)
Passing `--ctx 4096` to `llama-cpp-python` does not instantly allocate the entire memory payload; it only sets the context *capacity* ceiling. The actual RAM footprint is proportional to the tokens processed and stored in the KV Cache. Because the Qwen2.5 3B model utilizes Grouped Query Attention (GQA), the KV tensors are highly compressed:
`36 layers * 2 KV tensors * 2 KV heads * 128 head_dim * 2 bytes ~= 36 KB per token.`

Even if the cache was fully saturated at 4096 tokens, it would only cost `~144 MB` of RAM, making it incredibly survivable for mobile Edge AI.

### Finding 2: Lazy Loading vs zRAM Panic
The chaotic fluctuations in Resident Set Size (RSS) during the sweep (e.g., memory dropping from 3083 MB to 2724 MB as context increased) demonstrate the mechanics of `mmap` (memory-mapped) file loading. The OS only keeps memory pages resident as they are touched, actively paging them in and out. While Android's LMKD or zRAM compression may play a role, RSS drops alone are insufficient to prove zRAM activation without deeper kernel profiling (e.g., checking `/proc/swaps`).

## 9. Native Context Saturation (Llama-Bench)
To scientifically find the true context limits without Python's overhead, we compiled `llama.cpp` natively onto the Android Bionic kernel and executed the official C++ `llama-bench` tool. 

By bypassing Python's `n_ctx` capacity abstraction and commanding `llama-bench` to actively prefill up to 3,800 tokens (`-p 3800 -n 128`), we forced the OS to physically ingest and store the maximum possible KV-cache payload.

| Test Profile | Prompt Processing Speed | Generation Speed |
| :--- | :--- | :--- |
| **512 prefill + 128 decode** | 10.33 tok/s | 5.08 tok/s |
| **2048 prefill + 128 decode** | 9.39 tok/s | 5.11 tok/s |
| **3800 prefill + 128 decode** | 8.37 tok/s | 5.05 tok/s |

### Finding 1: The Stability of GQA
The generation speed remained rock solid (`~5.05 TPS`) even when the context window was stuffed with 3,800 tokens. This proves that traversing a Grouped Query Attention (GQA) KV cache during generation costs almost nothing in memory bandwidth on edge devices, since the physical size of the cache never exceeded ~144MB. 

### Finding 2: The True Edge Bottleneck
The actual bottleneck for edge context scaling is not memory capacity, but rather the memory bandwidth required for initial prompt ingestion. Pumping 3,800 tokens into the model dropped the prefill speed down to 8.37 tok/s, meaning a user would have to wait over **7.5 minutes** just for the model to "read" the prompt before generating the first word.

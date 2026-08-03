# Samsung Galaxy F23 5G Edge Benchmarks

This directory contains the edge inference evaluation for the Samsung Galaxy F23 5G. These benchmarks validate the real-world performance, memory limits, and context scaling behavior of quantized models on a constrained mobile device.

## Hardware Configuration
- **Device:** Samsung Galaxy F23 5G
- **SoC:** Qualcomm Snapdragon 750G (`lito`) — 6x Kryo @ 1804 MHz + 2x Kryo @ 2208 MHz
- **GPU:** Adreno 619 (integrated, no dedicated VRAM)
- **RAM:** 6 GB (Shared System Memory)
- **OS:** Android 14 (SDK 34), kernel 4.19.152, unrooted
- **Environment:** Termux (Native Android Linux environment)

## Tested Scenarios & Models
The primary focus on this device was identifying the "Context Cliff"—the exact point where KV-cache size causes the 6GB system memory to swap and the application to crash (Out-Of-Memory).
- **Target Model:** Qwen2.5 3B (Quantized)
- **Key Metric:** Context Scaling limit (up to 3800 tokens) vs Memory Usage.

## Software & Tools Used
- **Termux:** Provides the sandboxed Linux shell to compile and execute the binaries on Android natively without rooting.
- **llama.cpp (ARM64 Native):** The core inference engine built with `make` inside Termux targeting the native CPU.
- **Python 3:** Used for the automation scripts inside the `scripts/` directory to orchestrate the context sweeps.

## GPU Offload: Attempted and Ruled Out

> [!IMPORTANT]
> GPU offload to the Adreno 619 was attempted via the llama.cpp OpenCL backend and
> **fails at the hardware level**. llama.cpp was successfully built and does detect the
> GPU (`GPUOpenCL: QUALCOMM Adreno(TM)`), but any GPU buffer allocation panics the
> device into a full reboot.
>
> **Cause:** the Adreno 619 has no dedicated VRAM and allocates from a fixed 252 MiB CMA
> (Contiguous Memory Allocator) pool, which sits at **0–40 kB free at idle** — already
> consumed by the display stack. Allocation failure inside the KGSL driver escalates to a
> kernel panic because that driver owns the display pipeline. Observed across two forced
> reboots, with the kernel GPU `reset_count` climbing 178 → 209.
>
> Critically, `-ngl 4` fails identically to `-ngl 99`: the failure occurs at the *first*
> allocation, so no layer count is safe. This needs root (`cma=512M` on kernel cmdline)
> or different hardware.
>
> **Before attempting GPU offload on any Android device, check `grep CmaFree /proc/meminfo`.**
> If it is near zero, offload will reboot the device.
>
> Full analysis, the three Termux/Android linker problems solved along the way, and a
> reproducible build recipe: [`GPU_OFFLOAD_EXPERIMENT.md`](GPU_OFFLOAD_EXPERIMENT.md)

## Reproduction Steps

1. **Setup Termux:**
   Install Termux from F-Droid, run `pkg update`, and install the necessary build tools (`clang`, `make`, `python`).
2. **Build llama.cpp:**
   Clone the `llama.cpp` repository inside Termux and run `make`.
3. **Execute Scripts:**
   Navigate to the `scripts/` directory on the device and run the Python wrappers. These scripts will automatically invoke the `llama-bench` binary at varying context lengths and record the failure points.

```bash
cd scripts/
# Example: Run the context sweep script
python3 run_context_sweep.py
```

## Repository Layout
- `scripts/`: Python orchestration scripts used to perform the sweep tests on-device:
  - `run_context_sweep.py`, `02_termux_native_benchmark.py` — original context-cliff sweeps.
  - `04_cpu_thread_fa_sweep.py` — thread/flash-attention optimization sweep with RAM +
    thermal telemetry and low-memory abort guards (CPU backend, safe).
  - `03_adreno_gpu_benchmark.py` — GPU offload harness with kgsl `gpubusy` proof-of-use
    sampling. **Retained for reference only; do not run on this device** (see the GPU
    offload note above — it will reboot the phone).
- `results/`: The JSON outputs from `llama-bench`, the markdown summary of the context
  cliff (`llama_bench_3800.md`), and `qwen25_3b_cpu_thread_fa_sweep.json` (thread/FA sweep).
- `GPU_OFFLOAD_EXPERIMENT.md`: Full write-up of the failed Adreno 619 GPU offload attempt.

## CPU Thread & Flash-Attention Optimization

Sweep over thread count and flash-attention on the stable CPU backend.
Qwen2.5-3B Q4_K_M, `pp=128`, `tg=32`, 3 reps per case, 45 s cooldown, plugged in.

| Threads | Flash-Attn | Prefill (t/s) | Decode (t/s) | TTFT @128tok (ms) | Peak RAM (MB) | Peak CPU (°C) |
|---:|:---:|---:|---:|---:|---:|---:|
| 2 | off | 11.11 | **5.50** | 11521 | 4152 | 74.8 |
| 4 | off | 13.78 | 4.92 | 9289 | 4005 | 80.3 |
| 6 | off | 15.73 | 5.08 | 8137 | 4064 | 82.2 |
| 8 | off | **17.48** | 4.71 | 7323 | 3851 | 82.5 |
| 2 | on | 10.15 | 5.36 | 12611 | 3817 | 70.9 |
| 4 | on | 11.78 | 5.05 | 10866 | 3843 | 79.6 |
| 6 | on | 14.65 | 5.22 | 8737 | 4075 | 82.5 |
| 8 | on | 15.48 | 4.72 | 8269 | 3934 | 82.5 |

### Key finding: prefill and decode want opposite thread counts

- **Prefill is compute-bound** — scales cleanly with threads, 11.11 → 17.48 t/s (+57%)
  going from 2 to 8 threads.
- **Decode is memory-bandwidth-bound** — peaks at **2 threads (5.50 t/s)** and *degrades*
  to 4.71 t/s at 8 threads. Extra cores add memory contention, not throughput.
- The previous `-t 4` default was the worst of both: neither good prefill nor good decode.
- **Flash-attention was a consistent slight loss** on this SoC (e.g. 17.48 → 15.48 t/s
  prefill at 8 threads) and is not recommended here.

**Tuning guidance:** use `-t 8` for prompt-heavy/batch ingestion, `-t 2` for interactive
chat where decode latency dominates. Best decode of 5.50 t/s is **+13% over the 4.86 t/s
baseline**, at a lower peak temperature (74.8 °C vs 80.3 °C).

> [!NOTE]
> Thermals stayed within a safe envelope (peak 82.5 °C) with the device plugged in and a
> 45 s cooldown between cases. The sweep harness gates on temperature (`--max-temp`) and
> aborts if available memory drops below a floor (`--min-avail-mb`).

## Benchmark Results

Below are the aggregated results from the native Termux `llama-bench` tests on the 6GB RAM device.

### Context Scaling Limits
The following table shows the memory load and generation speeds for the **Qwen2.5-3B (`Q4_K_M`)** model at various context window sizes:

| Context (Tokens) | TTFT (ms) | Decode (t/s) | Active RAM (MB) | Status |
|------------------|-----------|--------------|-----------------|--------|
| **512**          | 217.8 ms  | 4.84 t/s     | ~2,872 MB       | ✅ Success |
| **1024**         | 202.8 ms  | 4.86 t/s     | ~3,083 MB       | ✅ Success |
| **2048**         | 237.0 ms  | 4.16 t/s     | ~2,724 MB       | ✅ Success |
| **3072**         | 205.7 ms  | 4.83 t/s     | ~2,726 MB       | ✅ Success |
| **3800**         | -         | 5.05 t/s     | -               | ✅ Max Safe |
| **4096**         | 207.3 ms  | 4.88 t/s     | ~3,077 MB       | ⚠️ Near Swap |

> [!WARNING]
> While the device successfully processed up to 4096 tokens in our automated script, manual tests revealed that sustained operations around **3800 tokens** form a "Context Cliff". Going beyond this significantly risks Android's Low Memory Killer (LMK) terminating the Termux process due to KV-cache memory pressure.

### Peak Ingestion Speed
At the maximum safe context window of **3800 tokens**, the device achieves a prompt processing speed of **8.37 tokens/sec**, proving that local 3B models are practically feasible on entry-level Android devices for moderate-length conversations.

# Samsung Galaxy F23 5G Edge Benchmarks

This directory contains the edge inference evaluation for the Samsung Galaxy F23 5G. These benchmarks validate the real-world performance, memory limits, and context scaling behavior of quantized models on a constrained mobile device.

## Hardware Configuration
- **Device:** Samsung Galaxy F23 5G
- **RAM:** 6 GB (Shared System Memory)
- **Environment:** Termux (Native Android Linux environment)

## Tested Scenarios & Models
The primary focus on this device was identifying the "Context Cliff"—the exact point where KV-cache size causes the 6GB system memory to swap and the application to crash (Out-Of-Memory).
- **Target Model:** Qwen2.5 3B (Quantized)
- **Key Metric:** Context Scaling limit (up to 3800 tokens) vs Memory Usage.

## Software & Tools Used
- **Termux:** Provides the sandboxed Linux shell to compile and execute the binaries on Android natively without rooting.
- **llama.cpp (ARM64 Native):** The core inference engine built with `make` inside Termux targeting the native CPU.
- **Python 3:** Used for the automation scripts inside the `scripts/` directory to orchestrate the context sweeps.

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
- `scripts/`: Python orchestration scripts (`run_context_sweep.py`, `02_termux_native_benchmark.py`, etc.) used to perform the sweep tests on-device.
- `results/`: The JSON outputs from `llama-bench` and the markdown summary of the context cliff (`llama_bench_3800.md`) demonstrating the exact memory failure points on the 6GB device.

# Edge LLM Efficiency Lab

A reproducible benchmark framework mapping the mathematical breaking points of quantized small LLMs (1B - 9B) on highly constrained edge devices.

## Purpose
This repository is strictly designed to measure the absolute intelligence floor of edge deployments. It generates real-world Pareto maps for GGUF-quantized small LLMs on physical mobile/edge hardware, with honest measurements for:
*   **Memory Saturation:** Physical RSS footprint.
*   **Latency:** Exact Time-To-First-Token (TTFT).
*   **Throughput:** Decode Tokens-Per-Second (TPS) and Long Prefill speeds.
*   **Degradation:** Mapping the logic collapse of models below 3-bit quantization (e.g. `Q2_K`).

## Architecture

*   `telemetry/`: The core profiling engine for capturing honest hardware metrics.
*   `experiments/`: Custom hypothesis scripts (e.g., GGUF quantization scaling).
*   `pareto_maps/`: Output data charting Quality vs. Speed vs. RAM.
*   `docs/`: Theory, architecture notes, and deployment playbook.

## Setup & Requirements

**Environment:**
*   Python 3.11+
*   `llama-cpp-python` (Built with native compiler constraints, see docs for Android Bionic `spawn.h` workaround).

**Run an Experiment:**
```bash
python experiments/02_termux_native/02_termux_native_benchmark.py --threads 4
```

*Note on ARM Threading:* Sweep thread counts because big.LITTLE CPUs may perform best below total core count.

## Milestones
*   **Milestone 1:** Qwen2.5 3B quantization and thread-scaling on Samsung Galaxy F23 5G.

# Qwen3.6 Deckard 40B NVFP4 — NVIDIA RTX PRO 6000 Blackwell Benchmark

This directory contains the empirical evaluation, footprint audit, and speed benchmark for the **Qwen3.6 Deckard 40B NVFP4 (W4A4)** model deployed on an **NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)** hardware node.

## Hardware Configuration
- **GPU:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Architecture:** `sm_120` (Blackwell Compute Capability)
- **Peak Memory Bandwidth:** ~1,800 GB/s

## Software & Deployment Stack
- **Inference Engine:** `vLLM 0.22.0`
- **Attention Kernel Backend:** `FlashInfer` (CUDA 13.1 JIT compiled for Blackwell `sm_120`)
- **Quantization Format:** `compressed-tensors` (`nvfp4-pack-quantized` W4A4 with BF16 exclusions)

## Benchmark Results

### Rigorous Context Speed Sweep

| Context Size | Ingestion Speed (TTFT) | Single-User Decode Speed | TTFT (s) |
|--------------|------------------------|--------------------------|----------|
| **512 tokens** | **541.11 t/s** | **33.92 t/s** | **0.983s** |
| **2,048 tokens** | **1,503.08 t/s** | **33.87 t/s** | **1.372s** |
| **4,096 tokens** | **2,117.82 t/s** | **33.68 t/s** | **1.942s** |
| **8,192 tokens** | **3,373.06 t/s** | **33.67 t/s** | **2.435s** |
| **16,384 tokens** | **4,374.75 t/s** | **32.88 t/s** | **3.749s** |

### Active Weight Footprint Audit (35.7 GB)
- **NVFP4 W4A4 Linear Layers**: ~21.5 GB
- **BF16 16-Bit Exclusions**: Visual Encoder (~6.2 GB), Embeddings & LM Head (~4.8 GB), Linear Attention (~2.4 GB), RMSNorms & Routers (~0.8 GB).
- **Total Active Weight Read per Token**: **35.7 GB**.

### Hardware Bandwidth Efficiency
$$\text{Theoretical Single-User Ceiling} = \frac{1800 \text{ GB/s}}{35.7 \text{ GB}} = 50.42 \text{ t/s}$$
$$\text{Measured Single-User Decode} = 33.88 \text{ t/s} \implies \mathbf{67.2\% \text{ Memory Bandwidth Efficiency}}$$

## Directory Contents
- `scripts/footprint_audit.py`: Zero-GPU CPU diagnostic script to calculate exact active weight read per token and memory bandwidth ceilings.
- `results/`: Speed sweep JSON outputs and config proof metadata.

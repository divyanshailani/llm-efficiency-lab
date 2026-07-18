# Llama Bench 3800 Prompt Test

Device: Samsung Galaxy F23 5G
SoC: Snapdragon 750G
RAM: 6GB
Model: Qwen2.5 3B Instruct Q4_K_M GGUF
Backend: llama.cpp native Termux build
Threads: 4
KV type: F16/F16
mmap: true

## Results

| Test | Tokens | Speed |
|---|---:|---:|
| Prompt processing | 3800 | 8.37 tok/s |
| Standalone generation | 128 | 5.05 tok/s |

## Interpretation

This proves the phone can successfully process a 3800-token prompt with this 3B Q4 model without running out of memory. Note that the generation row is a separate standalone generation benchmark, not decode-after-3800-context.

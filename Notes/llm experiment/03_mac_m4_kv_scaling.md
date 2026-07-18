# Apple Silicon M4: KV Cache Quantization & Flash Attention

## Flash Attention Requirement
On Apple Silicon (`MTL` backend), quantized KV caches (like `q8_0` and `q4_0`) strictly require Flash Attention (`-fa 1`) to be active. Running KV cache quantization without Flash Attention will result in an immediate `failed to create context` error.

## The Asymmetric Precision Trap
While `llama.cpp` supports mixing key/value cache precisions (e.g., `K=q8_0, V=q4_0`), testing revealed a severe optimization gap in the Metal kernels for asymmetric block sizes.

When evaluating `K=q8_0, V=q4_0`:
- Prompt processing at 8192 context plummeted from ~193 t/s down to **53.6 t/s**.
- At 16384 context, the Metal compute threadgroup completely locked up and hung indefinitely.

**Conclusion:** Never mix KV cache block precisions on Apple Silicon until the compute kernels explicitly support asymmetric unified memory alignments.

## Symmetric `q8_0:q8_0` vs `f16:f16`
Symmetric 8-bit quantization is a massive win on the M4.

| Metric | `f16:f16` | `q8_0:q8_0` |
| :--- | :--- | :--- |
| **Prompt Processing (512 ctx)** | 208.97 t/s | 207.65 t/s |
| **Prompt Processing (16k ctx)** | 186.78 t/s | 180.90 t/s |
| **Decode Speed (TPS)** | 15.75 t/s | **16.25 t/s** |

> [!TIP]
> `q8_0` KV cache provides a free 0.5 t/s decode speedup by halving the memory bandwidth required to read the context cache during the auto-regressive generation phase, with less than a 3% penalty to prompt ingestion speed.

## Recommendation
Always use `-fa 1 --cache-type-k q8_0 --cache-type-v q8_0`. Avoid `q4_0` KV mixtures as they trigger severe unoptimized fallback kernels on Metal.

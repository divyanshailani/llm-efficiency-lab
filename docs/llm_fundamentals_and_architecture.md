# LLM Fundamentals & Architecture

## 1. The Physics of GPU Compute & Math
At its core, a neural network is just massive lists of numbers (Vectors) and grids of numbers (Matrices). 
- **The CPU:** A CPU has a few very smart cores (e.g., 8 or 16). It is great at doing complex logic sequentially (if/else statements) and switching tasks rapidly.
- **The GPU:** A GPU has thousands of "dumb" cores (e.g., an Nvidia RTX 4090 has over 16,000 CUDA cores). 

When an LLM runs, it is performing **Matrix Multiplication** (multiplying the input tokens by the weights of the model). Because matrix multiplication consists of thousands of independent addition and multiplication operations, a GPU can calculate them all simultaneously. This is why GPUs dominate AI—pure parallel throughput.

## 2. Quantization (Shrinking the Weights)
A model trained in FP16 (16-bit floating point) takes 2 bytes per parameter. A 3 Billion parameter model takes 6GB of VRAM just to sit idle.
- **PTQ (Post-Training Quantization):** The model is trained in FP16, and a post-processing script mathematically rounds the weights down to 8-bit, 4-bit, or 2-bit integers. 
- **QAT (Quantization-Aware Training):** The model is trained from scratch (or fine-tuned) knowing it will be compressed, allowing the network to learn to compensate for the lost precision during gradient descent.

### The Formats
- **AWQ / GPTQ / EXL2:** Optimized for GPUs. The weights are stored compressed in VRAM, and the GPU's tensor cores dequantize them back to FP16 on the fly in nanoseconds to do the matrix math.
- **GGUF (llama.cpp):** Optimized for CPUs and Apple Silicon (Unified Memory). Mathematically structured to run inference across standard system RAM and CPU threads.

## 3. The KV Cache (Key-Value Cache)
During Autoregressive Inference (predicting the next token one by one), the model needs to look back at all previous tokens to understand context. Recalculating the Attention matrices (the Keys and Values) for every past token on every single step is a massive waste of compute.
Instead, the model calculates the Key and Value matrices for a token *once*, and saves them into the RAM/VRAM. This is the KV Cache.

### The KV Cache Formula
The RAM required to store the KV cache for a single token:
`Memory = 2 (for Key and Value) × Number of Layers × Number of KV Heads × Head Dimension × Bytes_per_parameter (e.g., 2 for FP16)`

## 4. Architectural Fixes: MQA vs GQA
To shrink the KV Cache, researchers changed the architecture of the Attention mechanism.
- **MHA (Multi-Head Attention):** The original architecture. Every "Query" head has its own unique "Key" and "Value" head. (Massive KV Cache).
- **MQA (Multi-Query Attention):** ALL Query heads share exactly ONE single Key and Value head. (Extremely small KV Cache, but loses reasoning quality).
- **GQA (Grouped-Query Attention):** The perfect middle ground. Used in Llama 3 and Qwen. Groups of Query heads share a Key and Value head (e.g., 32 Query heads share 8 Key/Value heads). It shrinks the KV Cache by 4x without losing much quality.

## 5. KV Cache Quantization vs TurboQuant
Even with GQA, a 32,000 token context window will consume gigabytes of RAM.

### Standard KV Cache Quantization
Just like we quantized the model weights, inference engines (like vLLM) quantize the *cache* in real-time. Before saving a Key/Value matrix to RAM, they convert it from FP16 to 8-bit (INT8 or FP8) or 4-bit (INT4). This linearly shrinks the cache size by 2x to 4x. 

### TurboQuant (1-bit / 2-bit Residuals)
TurboQuant (and similar cutting-edge methods) goes much further. Instead of just rounding numbers to INT4, it uses **Vector Quantization** with residual steps. 
1. It groups vectors of numbers and maps them to a highly compressed "codebook".
2. It stores the "error" (the residual difference between the real number and the compressed number) in extremely tight 1-bit or 2-bit formats.
This allows the KV Cache to be compressed by 8x or 16x. The tradeoff is that if the context is extremely long, finding a specific "Needle in a Haystack" (exact factual recall) degrades because the tiny mathematical errors in compression start to compound over thousands of tokens.

## 6. GGUF Nomenclature (The HuggingFace Alphabet Soup)
When you look at a GGUF repository on HuggingFace, you will see a massive list of files like `Q4_K_M`, `Q4_0`, or `IQ3_S`. Here is how to decode them:

### The Prefix (The Method)
- **Q (Legacy & Standard):** Stands for Quantization. The baseline rounding methods.
- **IQ (Importance Matrix):** The absolute bleeding edge. Before quantizing, the algorithm feeds real data through the model to track *which specific weights actually light up the most*. It preserves those critical "smart" weights at higher precision, and aggressively crushes the unused ones.

### The Suffix (The Precision)
- **_0 or _1 (e.g., `Q4_0`):** The oldest, dumbest quantization. Smashes everything uniformly. `_0` is symmetric, `_1` is asymmetric. Fast, but worst quality.
- **_K (k-quants):** Mixed precision. Instead of forcing the entire model to 4-bit, it analyzes the layers. It keeps highly sensitive layers (like Attention) at 6-bit, and crushes boring feed-forward layers down to 3-bit or 4-bit. This is much smarter.

### The Size Identifier
- **S (Small):** Very aggressive compression. Smaller file, worse reasoning.
- **M (Medium):** The golden standard. The perfect Pareto frontier of size vs. intelligence.
- **L (Large):** Least aggressive. Largest file, highest intelligence.
- **XS / XXS:** Extreme small versions (usually used with IQ to squeeze models onto tiny RAM footprints).

**Rule of Thumb:** If you are deploying, always grab the **`Q4_K_M`** (if you just want a standard solid model) or an **`IQ`** variant if you are forced to go down to 3-bit or 2-bit.

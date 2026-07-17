import modal
import os
import statistics
import time

app = modal.App("termux-sim-benchmark")

N_CTX = 1024
N_THREADS = 4
MAX_TOKENS = 128
WARMUP_TOKENS = 16
BENCH_RUNS = 3

SHORT_PROMPT = (
    "Q: If I have 5 apples and eat 2, then buy 4 more, how many apples do I have? "
    "A: Let's think step by step."
)

LONG_PREFILL_PROMPT = (
    "You are benchmarking CPU prompt processing on a constrained mobile-like system. "
    "Read the following notes and then summarize the key deployment tradeoffs. "
    + "Quantized local language models trade file size, RAM use, prompt processing speed, "
    "decode speed, and quality retention. Smaller quantization can reduce memory pressure, "
    "but may add unpacking or dequantization overhead on CPUs without ideal kernels. "
) * 8

# Build the container image with required libraries
image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "llama-cpp-python", "psutil")
)

# Simulate Termux-ish constraints: 4 CPU cores and 4096 MB (4GB) of RAM.
@app.function(image=image, cpu=4.0, memory=4096)
def benchmark_quantization(repo_id: str, filename: str, model_name: str = "unknown"):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    import psutil

    def rss_mb(process):
        return process.memory_info().rss / (1024 * 1024)

    def tokenize_count(llm, text, add_bos=False):
        return len(llm.tokenize(text.encode("utf-8"), add_bos=add_bos))

    def percentile(values, pct):
        if not values:
            return None
        values = sorted(values)
        index = round((len(values) - 1) * pct)
        return values[index]

    def run_streamed_generation(llm, prompt, max_tokens):
        prompt_tokens = tokenize_count(llm, prompt, add_bos=True)
        start = time.perf_counter()
        first_token_at = None
        chunks = []

        stream = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=42,
            echo=False,
            stream=True,
        )

        for chunk in stream:
            text = chunk["choices"][0].get("text", "")
            if text and first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(text)

        end = time.perf_counter()
        response = "".join(chunks).strip()
        completion_tokens = tokenize_count(llm, response, add_bos=False)

        ttft_s = (first_token_at or end) - start
        total_s = end - start
        decode_s = max(total_s - ttft_s, 0.0)
        decode_tokens = max(completion_tokens - 1, 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_s": ttft_s,
            "total_s": total_s,
            "prefill_tps_approx": prompt_tokens / ttft_s if ttft_s > 0 else 0,
            "decode_tps": decode_tokens / decode_s if decode_s > 0 and decode_tokens > 0 else 0,
            "response": response,
        }
    
    print(f"--- Starting Benchmark for {filename} ---")
    start_dl = time.time()
    
    try:
        # 1. Download the GGUF model
        print(f"Downloading {filename}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        dl_time = time.time() - start_dl
        print(f"Downloaded in {dl_time:.2f}s")
    except Exception as e:
        return {"filename": filename, "error": f"Download Failed: {e}"}
        
    try:
        # 2. Measure RAM before loading
        process = psutil.Process(os.getpid())
        ram_before = rss_mb(process)
        
        # 3. Load model via llama.cpp
        # n_ctx controls KV-cache size; keep this modest for mobile-style testing.
        # n_threads binds inference to the 4 allocated CPU cores.
        print("Loading model into RAM...")
        llm = Llama(model_path=model_path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)
        
        # 4. Measure RAM after loading
        ram_after_load = rss_mb(process)
        ram_load_used = ram_after_load - ram_before
        print(f"RAM after load: {ram_load_used:.2f} MB")

        # 5. Warm up kernels/pages so the measured runs are less dominated by first-call costs.
        print("Warming up...")
        _ = llm(SHORT_PROMPT, max_tokens=WARMUP_TOKENS, temperature=0.0, seed=42, echo=False)

        ram_after_warmup = rss_mb(process)
        ram_used = ram_after_warmup - ram_before
        print(f"RAM after warmup: {ram_used:.2f} MB")

        # 6. Benchmark TTFT, approximate prompt processing, and decode speed.
        runs = []
        for i in range(BENCH_RUNS):
            print(f"Benchmark run {i + 1}/{BENCH_RUNS}...")
            runs.append(run_streamed_generation(llm, SHORT_PROMPT, MAX_TOKENS))

        # 7. Run a longer prefill-heavy prompt separately.
        print("Running long prefill probe...")
        prefill_probe = run_streamed_generation(llm, LONG_PREFILL_PROMPT, 16)

        ttft_ms_values = [r["ttft_s"] * 1000 for r in runs]
        decode_tps_values = [r["decode_tps"] for r in runs]
        prefill_tps_values = [r["prefill_tps_approx"] for r in runs]
        completion_token_values = [r["completion_tokens"] for r in runs]
        total_time_values = [r["total_s"] for r in runs]
        best_response = runs[0]["response"]

        print(f"Response sample: {best_response}")
        print(f"TTFT median: {statistics.median(ttft_ms_values):.1f} ms")
        print(f"Decode median: {statistics.median(decode_tps_values):.2f} tok/s")
        
        return {
            "filename": filename,
            "model_name": model_name,
            "ram_mb": round(ram_used, 2),
            "ram_load_mb": round(ram_load_used, 2),
            "ttft_ms_median": round(statistics.median(ttft_ms_values), 1),
            "ttft_ms_p95": round(percentile(ttft_ms_values, 0.95), 1),
            "prefill_tps_approx_median": round(statistics.median(prefill_tps_values), 2),
            "decode_tps_median": round(statistics.median(decode_tps_values), 2),
            "decode_tps_min": round(min(decode_tps_values), 2),
            "decode_tps_max": round(max(decode_tps_values), 2),
            "completion_tokens_median": round(statistics.median(completion_token_values), 1),
            "total_s_median": round(statistics.median(total_time_values), 2),
            "long_prefill_tokens": prefill_probe["prompt_tokens"],
            "long_prefill_tps_approx": round(prefill_probe["prefill_tps_approx"], 2),
            "response": best_response,
        }
    except Exception as e:
        # If the model requires more than 4GB RAM, the OS will kill the process (OOM)
        # or Python will catch a MemoryError.
        return {"filename": filename, "error": f"Execution Failed (Likely OOM): {e}"}

@app.local_entrypoint()
def main():
    models = [
        {
            "name": "Qwen2.5-3B-Instruct",
            "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "quants": [
                "qwen2.5-3b-instruct-q8_0.gguf",   # 8-bit baseline
                "qwen2.5-3b-instruct-q4_k_m.gguf", # 4-bit mixed quant
                "qwen2.5-3b-instruct-q3_k_m.gguf", # 3-bit mixed quant
                "qwen2.5-3b-instruct-q2_k.gguf",   # 2-bit stress test
            ],
        },
    ]
    
    print("Starting Modal Termux Simulation (4 Cores, 4GB RAM)...\n")
    
    results = []
    for model in models:
        for quant in model["quants"]:
            print(f"Dispatching {model['name']} / {quant} to Modal...")
            result = benchmark_quantization.remote(model["repo_id"], quant, model["name"])
            results.append(result)
            print("-" * 40)
        
    print("\n--- FINAL BENCHMARK RESULTS ---")
    for r in results:
        if "error" in r:
            print(f"[{r['filename']}] FAILED: {r['error']}")
        else:
            print(f"[{r['model_name']} / {r['filename']}]")
            print(
                f"    RAM warm: {r['ram_mb']} MB | RAM load: {r['ram_load_mb']} MB | "
                f"TTFT median: {r['ttft_ms_median']} ms | "
                f"PP approx: {r['prefill_tps_approx_median']} tok/s | "
                f"Decode median: {r['decode_tps_median']} tok/s "
                f"(min {r['decode_tps_min']}, max {r['decode_tps_max']})"
            )
            print(
                f"    Long prefill: {r['long_prefill_tokens']} prompt tok @ "
                f"{r['long_prefill_tps_approx']} tok/s approx"
            )
            print(f"    Answer: {r['response'][:100]}...\n")

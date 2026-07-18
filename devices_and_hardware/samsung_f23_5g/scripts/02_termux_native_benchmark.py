import os
import sys
if sys.platform == "android":
    sys.platform = "linux"
import time
import statistics
import argparse
import json
import platform
import urllib.request
from llama_cpp import Llama
import psutil

DEFAULT_N_CTX = 1024
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

def print_system_info():
    vm = psutil.virtual_memory()
    print("[SYSTEM]")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPUs: {psutil.cpu_count(logical=False)}")
    print(f"RAM total: {vm.total / (1024 * 1024):.0f} MB")
    print(f"RAM available: {vm.available / (1024 * 1024):.0f} MB")

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

def resolve_model_path(repo_id: str, filename: str, model_path: str | None):
    if model_path:
        expanded_path = os.path.expanduser(model_path)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Local model path does not exist: {expanded_path}")
        return expanded_path

    print(f"Downloading {filename}...")
    return hf_hub_download(repo_id=repo_id, filename=filename)

def run_benchmark(
    repo_id: str,
    filename: str,
    threads: int,
    n_ctx: int,
    max_tokens: int,
    measured_runs: int,
    model_path: str | None,
    result_json: str | None,
):
    print(f"\n--- Starting Benchmark for {filename} ---")
    start_model_resolve = time.time()
    
    try:
        resolved_model_path = model_path or filename
        if not os.path.exists(resolved_model_path):
            print(f"Downloading {filename}...")
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            urllib.request.urlretrieve(url, resolved_model_path)
            print(f"Downloaded in {time.time() - start_model_resolve:.2f}s")
        else:
            print(f"File {resolved_model_path} ready.")

        model_resolve_time = time.time() - start_model_resolve
        model_size_mb = os.path.getsize(resolved_model_path) / (1024 * 1024)
        print(f"Model path: {resolved_model_path}")
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Model ready in {model_resolve_time:.2f}s")
    except Exception as e:
        print(f"Model setup failed: {e}")
        return None
        
    try:
        process = psutil.Process(os.getpid())
        ram_before = rss_mb(process)
        
        print("Loading model into RAM...")
        llm = Llama(
            model_path=resolved_model_path,
            n_ctx=n_ctx,
            n_threads=threads,
            verbose=False,
        )
        
        ram_after_load = rss_mb(process)
        ram_load_used = ram_after_load - ram_before
        print(f"RAM after load: {ram_load_used:.2f} MB")

        print("Warming up (forcing page faults)...")
        _ = llm(SHORT_PROMPT, max_tokens=WARMUP_TOKENS, temperature=0.0, seed=42, echo=False)

        ram_after_warmup = rss_mb(process)
        ram_used = ram_after_warmup - ram_before
        print(f"RAM after warmup: {ram_used:.2f} MB")

        generation_runs = []
        for i in range(measured_runs):
            print(f"Benchmark run {i + 1}/{measured_runs}...")
            generation_runs.append(run_streamed_generation(llm, SHORT_PROMPT, max_tokens))

        print("Running long prefill probe...")
        prefill_probe = run_streamed_generation(llm, LONG_PREFILL_PROMPT, 16)

        ttft_ms_values = [r["ttft_s"] * 1000 for r in generation_runs]
        decode_tps_values = [r["decode_tps"] for r in generation_runs]
        prefill_tps_values = [r["prefill_tps_approx"] for r in generation_runs]
        completion_token_values = [r["completion_tokens"] for r in generation_runs]
        total_time_values = [r["total_s"] for r in generation_runs]
        
        print(f"\n[RESULTS: {filename}]")
        print(f"RAM warm: {ram_used:.2f} MB | RAM load: {ram_load_used:.2f} MB")
        print(f"TTFT median: {statistics.median(ttft_ms_values):.1f} ms | p95: {percentile(ttft_ms_values, 0.95):.1f} ms")
        print(f"PP approx median: {statistics.median(prefill_tps_values):.2f} tok/s")
        print(
            f"Decode median: {statistics.median(decode_tps_values):.2f} tok/s "
            f"(min {min(decode_tps_values):.2f}, max {max(decode_tps_values):.2f})"
        )
        print(
            f"Long prefill: {prefill_probe['prompt_tokens']} prompt tok @ "
            f"{prefill_probe['prefill_tps_approx']:.2f} tok/s approx"
        )
        print(f"Answer: {generation_runs[0]['response'][:150]}...")

        result = {
            "filename": filename,
            "model_path": resolved_model_path,
            "model_size_mb": round(model_size_mb, 2),
            "threads": threads,
            "n_ctx": n_ctx,
            "max_tokens": max_tokens,
            "runs": measured_runs,
            "ram_load_mb": round(ram_load_used, 2),
            "ram_warm_mb": round(ram_used, 2),
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
            "response": generation_runs[0]["response"],
        }

        if result_json:
            with open(os.path.expanduser(result_json), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"Saved JSON results to {result_json}")

        return result
        
    except Exception as e:
        print(f"Execution Failed (Likely OOM): {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Termux LLM Benchmark")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--ctx", type=int, default=DEFAULT_N_CTX, help="Context length")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Completion tokens per run")
    parser.add_argument("--runs", type=int, default=BENCH_RUNS, help="Measured runs after warmup")
    parser.add_argument("--model-path", help="Use an already-downloaded local GGUF instead of downloading")
    parser.add_argument("--result-json", default="termux_qwen25_3b_q4_k_m_result.json")
    args = parser.parse_args()

    repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    quants = [
        "qwen2.5-3b-instruct-q4_k_m.gguf",
    ]
    
    print(f"Starting Native Termux Benchmark ({args.threads} Threads)...\n")
    print_system_info()
    for quant in quants:
        run_benchmark(
            repo_id=repo_id,
            filename=quant,
            threads=args.threads,
            n_ctx=args.ctx,
            max_tokens=args.max_tokens,
            measured_runs=args.runs,
            model_path=args.model_path,
            result_json=args.result_json,
        )
        print("-" * 40)

"""
LFM2.5-2.6B Q8 llama.cpp sweep — thread count, batch, and Flash Attention.

One knob changes per run. Each run produces a JSON dict:
  { "variant": "...", "args": {...}, "llama_bench": {...} }

Usage:
  modal run deploy_lfm_q8_sweep.py
"""
import os, subprocess, json, modal

app = modal.App()

# ── Image: same native build as baseline ──────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "git", "curl")
    .run_commands(
        "git clone https://github.com/ggerganov/llama.cpp.git /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j 16",
    )
    .pip_install("huggingface_hub>=0.28.0")
)

REPO_ID    = "LiquidAI/LFM2.5-2.6B-GGUF"
GGUF_FILE  = "LFM2.5-2.6B-Q8_0.gguf"
MODEL_PATH = None   # resolved at runtime

# ── Prompt sweep template ──────────────────────────────────────────────────────
PROMPTS = "128,512"
GEN_TOKS = 8


def run_bench(variant: str, extra_args: list[str]) -> dict:
    from huggingface_hub import hf_hub_download
    global MODEL_PATH
    if MODEL_PATH is None:
        MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=GGUF_FILE, token=os.environ.get("HF_TOKEN"))

    cmd = [
        "/root/llama.cpp/build/bin/llama-bench",
        "-m", MODEL_PATH,
        "-p", PROMPTS,
        "-n", str(GEN_TOKS),
        "-r", "1",
        "-o", "json",
        "-t", "16",          # keep threads fixed for all runs
        *extra_args,
    ]
    print(f"[{variant}] {' '.join(cmd)}")
    pc = subprocess.run(cmd, capture_output=True, text=True, timeout=540)
    result = {"variant": variant, "args": extra_args}
    try:
        result["llama_bench"] = json.loads(pc.stdout)
    except json.JSONDecodeError:
        result["llama_bench_error"] = pc.stdout[:2000]
        result["llama_bench_stderr"] = pc.stderr[-1000:]
    return result


# ── Variants: one changed knob at a time ──────────────────────────────────────
VARIANTS = [
    # Baseline (reference run — same as deploy_lfm_q8_llama_bench.py)
    ("t16_b512_fa0",  []),

    # Batch sweep
    ("t16_b128_fa0",  ["-b", "128"]),
    ("t16_b256_fa0",  ["-b", "256"]),
    ("t16_b1024_fa0", ["-b", "1024"]),

    # Flash Attention modes
    ("t16_b512_fa1",  ["-fa", "on"]),
    ("t16_b512_fa0_explicit", ["-fa", "off"]),

    # No memory-mapping (forces full load into RAM)
    ("t16_b512_fa0_nommap", ["-mmp", "0"]),

    # KV cache type cell variant
    ("t16_b512_q8kv",  ["-ctk", "q8_0", "-ctv", "q8_0"]),

    # Fewer threads (8) — test thread scaling
    ("t8_b512_fa0",   ["-t", "8"]),
]


@app.function(
    image=image, cpu=16.0, timeout=600, retries=0,
    memory=16384,
)
def run_sweep():
    results = []
    for variant, args in VARIANTS:
        r = run_bench(variant, args)
        results.append(r)
        print(f"  → {variant} done")

    out = json.dumps(results, indent=2)
    print("\n" + "=" * 60)
    print("ALL RESULTS:")
    print(out)
    print("=" * 60)
    return results


@app.local_entrypoint()
def main():
    run_sweep.remote()
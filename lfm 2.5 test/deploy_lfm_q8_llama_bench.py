import os
import subprocess
import json
import modal

app = modal.App()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake", "git", "curl")
    .run_commands(
        "git clone https://github.com/ggerganov/llama.cpp.git /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j 16",
    )
    .pip_install("huggingface_hub>=0.28.0")
)

REPO_ID   = "LiquidAI/LFM2.5-2.6B-GGUF"
GGUF_FILE = "LFM2.5-2.6B-Q8_0.gguf"


@app.function(
    image=image,
    cpu=16.0,
    timeout=1200,
    retries=0,
    memory=16384,
)
def run_bench():
    from huggingface_hub import HfFileSystem
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Fetching {GGUF_FILE} from cache...")
    fs = HfFileSystem(token=hf_token)
    try:
        files = fs.ls(f"{REPO_ID}", recursive=False)
        gguf_files = [f for f in files if GGUF_FILE in f]
        if gguf_files:
            full_path = gguf_files[0]
            model_path = "/" + full_path
            print(f"Found in HfFileSystem: {model_path}")
        else:
            raise FileNotFoundError("not in HfFileSystem")
    except Exception as e:
        print(f"HfFileSystem error: {e}, downloading via hf_hub_download...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=GGUF_FILE,
            token=hf_token
        )
        print(f"Downloaded to {model_path}")

    # Verify the file exists and its size
    size = os.path.getsize(model_path)
    print(f"GGUF file size: {size / 1024**3:.2f} GB")

    # Check the binary
    bench_bin = "/root/llama.cpp/build/bin/llama-bench"
    print(f"Binary exists: {os.path.exists(bench_bin)}")
    print(f"Binary is executable: {os.access(bench_bin, os.X_OK)}")

    # Quick smoke test — just print help, capture both streams
    print("=== llama-bench smoke test ===")
    help_proc = subprocess.run(
        [bench_bin, "--help"],
        capture_output=True, text=True, timeout=30
    )
    print(f"help RC: {help_proc.returncode}")
    print(f"help STDOUT (first 200): {help_proc.stdout[:200]}")
    print(f"help STDERR (first 200): {help_proc.stderr[:200]}")

    # Run the actual benchmark
    cmd = [
        bench_bin,
        "-m", model_path,
        "-p", "128,512",
        "-n", "8",
        "-r", "1",
        "-t", "16",
        "-o", "json",
    ]
    print(f"Running: {' '.join(cmd)}")

    pc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(f"Bench RC: {pc.returncode}")
    print(f"Bench STDOUT (first 3000): {pc.stdout[:3000]}")
    print(f"Bench STDERR (first 1000): {pc.stderr[:1000]}")

    return pc.stdout


@app.local_entrypoint()
def main():
    res = run_bench.remote()
    print("\n" + "=" * 60)
    print("LLAMA-BENCH JSON OUTPUT:")
    print(res)
    print("=" * 60)
import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
import platform

# Apply Termux Android platform patch
if sys.platform == "android":
    sys.platform = "linux"

def get_device_metadata():
    try:
        model = subprocess.check_output(["getprop", "ro.product.model"], text=True).strip()
        soc = subprocess.check_output(["getprop", "ro.board.platform"], text=True).strip()
        android_ver = subprocess.check_output(["getprop", "ro.build.version.release"], text=True).strip()
    except Exception:
        model = "Unknown"
        soc = "Unknown"
        android_ver = "Unknown"
    
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        ram_gb = "Unknown"
    
    return {
        "phone_model": model,
        "SoC": soc,
        "RAM_GB": ram_gb,
        "Android_version": android_ver,
        "Python": sys.version.split(" ")[0],
    }

def worker_routine(model_path, ctx_size, threads):
    import psutil
    from llama_cpp import Llama
    
    SHORT_PROMPT = (
        "Q: If I have 5 apples and eat 2, then buy 4 more, how many apples do I have? "
        "A: Let's think step by step."
    )
    
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    
    # Load model with strict context limit
    llm = Llama(model_path=model_path, n_ctx=ctx_size, n_threads=threads, verbose=False)
    
    ram_load = (process.memory_info().rss / (1024 * 1024)) - ram_before
    
    # Warmup memory
    _ = llm(SHORT_PROMPT, max_tokens=16, temperature=0.0, echo=False)
    ram_warm = (process.memory_info().rss / (1024 * 1024)) - ram_before
    
    # Timed Benchmark
    prompt_tokens = len(llm.tokenize(SHORT_PROMPT.encode("utf-8"), add_bos=True))
    start = time.perf_counter()
    first_token_at = None
    chunks = []
    
    stream = llm(SHORT_PROMPT, max_tokens=128, temperature=0.0, echo=False, stream=True)
    for chunk in stream:
        text = chunk["choices"][0].get("text", "")
        if text and first_token_at is None:
            first_token_at = time.perf_counter()
        chunks.append(text)
        
    end = time.perf_counter()
    response = "".join(chunks).strip()
    completion_tokens = len(llm.tokenize(response.encode("utf-8"), add_bos=False))
    
    ttft_s = (first_token_at or end) - start
    decode_s = max(end - (first_token_at or end), 0.0)
    decode_tokens = max(completion_tokens - 1, 0)
    
    result = {
        "ctx": ctx_size,
        "status": "success",
        "ram_load_mb": round(ram_load, 2),
        "ram_warm_mb": round(ram_warm, 2),
        "ttft_ms": round(ttft_s * 1000, 1),
        "decode_tps": round(decode_tokens / decode_s if decode_s > 0 else 0, 2),
        "prefill_tps_approx": round(prompt_tokens / ttft_s if ttft_s > 0 else 0, 2)
    }
    
    # Strictly output only the JSON on the final line for the master process
    print(json.dumps(result))

def master_routine(model_path, threads, out_file):
    contexts = [512, 1024, 2048, 3072, 4096]
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        sys.exit(1)
        
    file_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)
    
    final_output = {
        "metadata": get_device_metadata(),
        "run_info": {
            "timestamp": datetime.utcnow().isoformat(),
            "model_filename": os.path.basename(model_path),
            "model_size_mb": file_size_mb,
            "threads": threads
        },
        "results": []
    }
    
    print(f"Starting Context Sweep on {os.path.basename(model_path)}")
    print(f"Device: {final_output['metadata']['phone_model']} ({final_output['metadata']['SoC']})")
    
    for ctx in contexts:
        print(f"\n--- Testing n_ctx = {ctx} ---")
        
        # Spawn isolated subprocess to prevent OOM pollution
        cmd = [sys.executable, __file__, "--worker", "--model", model_path, "--ctx", str(ctx), "--threads", str(threads)]
        try:
            # 15 minutes strict timeout per context
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            if process.returncode == 0:
                lines = process.stdout.strip().split('\n')
                worker_result = json.loads(lines[-1])
                print(f"Success: RAM {worker_result['ram_warm_mb']} MB | TTFT {worker_result['ttft_ms']} ms | Decode {worker_result['decode_tps']} TPS")
                final_output["results"].append(worker_result)
            else:
                print(f"Failed with exit code {process.returncode}")
                print(f"Stderr: {process.stderr}")
                final_output["results"].append({
                    "ctx": ctx,
                    "status": "failed",
                    "error": f"process killed or timeout (code {process.returncode})"
                })
        except subprocess.TimeoutExpired:
            print("Timeout expired (15 mins)")
            final_output["results"].append({
                "ctx": ctx,
                "status": "failed",
                "error": "process timeout (900s)"
            })
        except Exception as e:
            print(f"Fatal error: {e}")
            final_output["results"].append({
                "ctx": ctx,
                "status": "failed",
                "error": str(e)
            })
            
        # Iteratively save the JSON after EVERY context run. 
        # If Android OOM-kills the entire Termux process during the 4096 run, 
        # we will still have the 512, 1024, 2048, and 3072 results safely stored on disk.
        with open(out_file, "w") as f:
            json.dump(final_output, f, indent=2)
            
    print(f"\nSweep completed. Final results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Internal flag to run isolated memory process")
    parser.add_argument("--model", type=str, default="qwen2.5-3b-instruct-q4_k_m.gguf")
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=str, default="qwen25_3b_context_sweep.json")
    args = parser.parse_args()
    
    if args.worker:
        worker_routine(args.model, args.ctx, args.threads)
    else:
        master_routine(args.model, args.threads, args.out)

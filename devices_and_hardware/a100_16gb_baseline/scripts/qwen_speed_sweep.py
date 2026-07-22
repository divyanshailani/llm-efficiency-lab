import requests
import time
import sys
import json
import subprocess

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-27B"
CONTEXT_SIZES = [512, 2048, 4096, 8192, 16384]

def get_local_telemetry():
    telemetry = {}
    try:
        telemetry["ram"] = subprocess.getoutput("free -m | grep Mem | awk '{print $3 \"MB / \" $2 \"MB\"}'")
        telemetry["vram"] = subprocess.getoutput("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1 \"MB / \" $2 \"MB\"}'")
    except Exception as e:
        telemetry["error"] = str(e)
    return telemetry

def generate_context(target_tokens):
    # 'The quick brown fox jumps over the lazy dog. ' is roughly 10 tokens
    approx_tokens_per_repeat = 10
    repeats = target_tokens // approx_tokens_per_repeat
    base_phrase = "The quick brown fox jumps over the lazy dog. "
    return (base_phrase * repeats) + "\n\nSummarize the text above in exactly one sentence."

def run_sweep():
    print("=== Rigorous Context Size Sweep (Qwen 27B on A100) ===")
    
    print("\n[Waking up vLLM engine and loading weights into VRAM...]")
    try:
        requests.post(BASE_URL, json={"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}, timeout=600)
        print("[Warmup complete! Commencing benchmark sweep...]")
    except Exception as e:
        print(f"[ERROR] Warmup failed: {e}")
        
    results = []
    
    for size in CONTEXT_SIZES:
        print(f"\nTesting Context Size: ~{size} tokens")
        prompt = generate_context(size)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "stream": True,
            "temperature": 0.0
        }
        
        try:
            start_time = time.time()
            resp = requests.post(BASE_URL, json=payload, stream=True, timeout=120)
            resp.raise_for_status()
            
            ttft = None
            token_count = 0
            
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded == "data: [DONE]":
                        break
                    if decoded.startswith("data: "):
                        if ttft is None:
                            ttft = time.time() - start_time
                        
                        data = json.loads(decoded[6:])
                        if data["choices"][0].get("delta", {}).get("content"):
                            token_count += 1
                            
            end_time = time.time()
            total_time = end_time - start_time
            decode_time = total_time - ttft
            
            ingest_speed = size / ttft if ttft and ttft > 0 else 0
            decode_speed = token_count / decode_time if decode_time and decode_time > 0 else 0
            
            # Fetch live hardware telemetry locally
            telemetry = get_local_telemetry()
            
            print(f"  TTFT: {ttft:.3f}s | Ingest Speed: {ingest_speed:.2f} tok/s | Decode Speed: {decode_speed:.2f} tok/s")
            if "error" not in telemetry:
                print(f"  Hardware -> RAM: {telemetry.get('ram', 'N/A')} | VRAM: {telemetry.get('vram', 'N/A')}")
            
            results.append({
                "context": size,
                "ingest_speed": ingest_speed,
                "decode_speed": decode_speed,
                "telemetry": telemetry
            })
            
        except Exception as e:
            print(f"  [ERROR] Benchmark failed for size {size}: {e}")
            
    print("\n--- Final Results Markdown Table ---")
    print("| Context Size (Tokens) | Prompt Processing (Ingestion) | Decode Speed | VRAM Usage | RAM Usage |")
    print("|-----------------------|-------------------------------|--------------|------------|-----------|")
    for r in results:
        vram = r.get("telemetry", {}).get("vram", "N/A") if "error" not in r.get("telemetry", {}) else "N/A"
        ram = r.get("telemetry", {}).get("ram", "N/A") if "error" not in r.get("telemetry", {}) else "N/A"
        print(f"| {r['context']} | {r['ingest_speed']:.2f} t/s | {r['decode_speed']:.2f} t/s | {vram} | {ram} |")
    
    # Save raw json for documentation
    with open("speed_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_sweep()

# ornith_speed_sweep.py
import requests
import time
import statistics
import json

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "ornith-1.0-35b-nvfp4"
CONTEXT_SIZES = [512, 2048, 4096, 8192, 16384]

def generate_context(target_tokens):
    approx_tokens_per_repeat = 10
    repeats = target_tokens // approx_tokens_per_repeat
    base_phrase = "The quick brown fox jumps over the lazy dog. "
    return (base_phrase * repeats) + "\n\nSummarize the text above in exactly one sentence."

def run_sweep():
    print("=== Rigorous Context Size Sweep (Ornith 35B NVFP4 MoE on Blackwell 96GB) ===")
    
    print("\n[Waking up container and loading weights to VRAM...]")
    try:
        resp = requests.post(
            BASE_URL,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1},
            timeout=600,
        )
        resp.raise_for_status()
        print("[Warmup complete! Commencing benchmark sweep...]")
    except Exception as e:
        print(f"[ERROR] Warmup failed: {e}")
        
    results = []
    
    for size in CONTEXT_SIZES:
        print(f"\nTesting Context Size: ~{size} tokens (3 Iterations)")
        prompt = generate_context(size)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
        }
        
        ttfts = []
        ingest_speeds = []
        decode_speeds = []
        
        for iteration in range(3):
            try:
                start_time = time.time()
                resp = requests.post(BASE_URL, json=payload, stream=True, timeout=600)
                resp.raise_for_status()
                
                ttft = None
                prompt_tokens = 0
                completion_tokens = 0
                
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded == "data: [DONE]":
                            break
                        if decoded.startswith("data: "):
                            try:
                                data = json.loads(decoded[6:])
                            except json.JSONDecodeError:
                                continue
                                
                            delta = data["choices"][0].get("delta", {}) if data.get("choices") else {}
                            content = delta.get("content", "")
                            reasoning = delta.get("reasoning_content", "")
                            
                            if (content or reasoning) and ttft is None:
                                ttft = time.time() - start_time
                                
                            if "usage" in data and data["usage"]:
                                prompt_tokens = data["usage"].get("prompt_tokens", 0)
                                completion_tokens = data["usage"].get("completion_tokens", 0)
                                
                end_time = time.time()
                total_time = end_time - start_time
                decode_time = total_time - ttft if ttft else 0
                
                approx_warning = ""
                if prompt_tokens == 0:
                    prompt_tokens = int(len(prompt) / 4)
                    approx_warning = " (Approx Tokens)"
                if completion_tokens == 0:
                    completion_tokens = 128 
                
                ingest_speed = prompt_tokens / ttft if ttft and ttft > 0 else 0
                decode_speed = completion_tokens / decode_time if decode_time and decode_time > 0 else 0
                
                ttfts.append(ttft or 0.0)
                ingest_speeds.append(ingest_speed)
                decode_speeds.append(decode_speed)
                
                print(f"  Iter {iteration+1} - TTFT: {ttft or 0.0:.3f}s | Ingest: {ingest_speed:.2f} t/s{approx_warning} | Decode: {decode_speed:.2f} t/s")
                
            except Exception as e:
                print(f"  [ERROR] Iteration {iteration+1} failed: {e}")
                
        if ttfts and ingest_speeds and decode_speeds:
            med_ttft = statistics.median(ttfts)
            med_ingest = statistics.median(ingest_speeds)
            med_decode = statistics.median(decode_speeds)
            
            results.append({
                "context_target": size,
                "ttft_median": med_ttft,
                "ingest_median": med_ingest,
                "decode_median": med_decode,
            })
            print(f"  -> MEDIANS: TTFT: {med_ttft:.3f}s | Ingest: {med_ingest:.2f} t/s | Decode: {med_decode:.2f} t/s")
            
    print("\n--- Final Results Markdown Table ---")
    print("| Context Size (Target) | Median Ingestion | Median Decode Speed | Median TTFT |")
    print("|-----------------------|------------------|---------------------|-------------|")
    for r in results:
        print(f"| {r['context_target']} | {r['ingest_median']:.2f} t/s | {r['decode_median']:.2f} t/s | {r['ttft_median']:.3f}s |")
    
    with open("ornith_speed_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_sweep()

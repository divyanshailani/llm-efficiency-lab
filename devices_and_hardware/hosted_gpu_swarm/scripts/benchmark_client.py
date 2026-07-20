import time
import requests
import json
import sys

# We will populate this URL after Modal deploy completes
MODAL_ENDPOINT_URL = "https://kartikchijwani-maker--qwen-chat.modal.run"

def run_benchmark(prompt, max_tokens=512):
    print(f"\n--- Running Benchmark ---")
    print(f"Prompt: {prompt[:50]}...")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            MODAL_ENDPOINT_URL,
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
            timeout=600 # Initial HF download to Modal Volume can take up to 5 minutes
        )
        response.raise_for_status()
    except Exception as e:
        print(f"FAILED: {e}")
        return
        
    end_time = time.time()
    latency = end_time - start_time
    
    data = response.json()
    text = data.get("text", "")
    completion_tokens = data.get("completion_tokens", 0)
    
    print(f"Success! Latency: {latency:.2f} seconds")
    if completion_tokens > 0:
        tps = completion_tokens / latency
        print(f"Tokens/sec: {tps:.2f} (Note: includes network/TTFT overhead)")
        
    print(f"Output preview: {text[:100]}...\n")

if __name__ == "__main__":
    if "<YOUR_WORKSPACE_NAME>" in MODAL_ENDPOINT_URL:
        print("Please update MODAL_ENDPOINT_URL with your actual deployed URL.")
        sys.exit(1)
        
    print("Testing Cold Start (or Warm if recently pinged)...")
    run_benchmark("Write a python function to compute the Fibonacci sequence.")
    
    print("Testing Warm Start...")
    run_benchmark("Explain the theory of relativity in one sentence.")

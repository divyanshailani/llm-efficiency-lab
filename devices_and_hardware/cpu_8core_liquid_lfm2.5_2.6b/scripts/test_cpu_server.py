#!/usr/bin/env python3
"""
Liquid AI LFM2.5-2.6B 8-Core CPU Node Generic Tester Script
Evaluates hardware throughput, latency, and reasoning on any 8-core CPU hardware node.

Usage:
    python3 test_cpu_server.py --endpoint http://localhost:8000/v1
"""

import argparse
import time
import json
import requests

def test_inference(endpoint_url: str, prompt: str, max_tokens: int = 150):
    url = f"{endpoint_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": "LiquidAI/LFM2.5-2.6B",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    print(f"🚀 Sending prompt to {url}...")
    t0 = time.perf_counter()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        elapsed = time.perf_counter() - t0
        
        if response.status_code == 200:
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_generated = usage.get("completion_tokens", max_tokens)
            speed = tokens_generated / elapsed if elapsed > 0 else 0
            
            print("=" * 60)
            print("💬 Response:")
            print(reply)
            print("=" * 60)
            print(f"⚡ Tokens:     {tokens_generated}")
            print(f"🕒 Latency:    {elapsed:.2f}s")
            print(f"🚀 Throughput: {speed:.2f} tok/s on CPU")
            print("=" * 60)
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Liquid AI LFM2.5-2.6B on CPU infrastructure")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1", help="Base URL of OpenAI-compatible API")
    parser.add_argument("--prompt", default="Explain why short convolutions have O(1) constant memory complexity during LLM token decoding.", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max new tokens to generate")
    args = parser.parse_args()
    
    test_inference(args.endpoint, args.prompt, args.max_tokens)

if __name__ == "__main__":
    main()

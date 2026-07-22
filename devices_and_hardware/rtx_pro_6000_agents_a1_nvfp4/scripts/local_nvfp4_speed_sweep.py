#!/usr/bin/env python3
"""
local_nvfp4_speed_sweep.py

Context Speed Sweep Benchmark for InternScience Agents-A1 NVFP4 (W4A4)
Evaluates TTFT, Ingestion Speed (t/s), and Generation/Decode Speed (t/s)
across context windows (512 to 16,384 tokens).
"""

import os
import sys
import time
import json
import argparse
import requests

DEFAULT_ENDPOINT = "https://kartikchijwani-maker--agents-a1-nvfp4-node-serve.modal.run/v1"
MODEL_NAME = "agents-a1-nvfp4"

CONTEXT_TEST_SIZES = [512, 2048, 4096, 8192, 16384]
GEN_TOKENS = 128

PAD_TEXT = (
    "The NVIDIA RTX PRO 6000 Blackwell architecture with 96GB VRAM provides hardware acceleration "
    "for NVFP4 (4-bit floating point) weights and activations, leveraging tensor cores and FlashInfer attention. "
    "InternScience Agents-A1 is a 35B Mixture-of-Experts agentic model with 8 active experts per token. "
)

def generate_context_prompt(target_tokens: int) -> str:
    words = PAD_TEXT.split()
    target_words = int(target_tokens * 0.75)
    repeated = (words * ((target_words // len(words)) + 1))[:target_words]
    body = " ".join(repeated)
    return f"Context Payload:\n{body}\n\nTask: Summarize the key architectural benefits of NVFP4 and MoE routing above."

def run_speed_sweep(base_url: str, api_key: str, output_json: str):
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results = {
        "model": MODEL_NAME,
        "endpoint": base_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweeps": []
    }

    print("=" * 70)
    print(f"AGENTS-A1 NVFP4 SPEED SWEEP BENCHMARK")
    print(f"Endpoint: {endpoint}")
    print("=" * 70)

    for ctx_len in CONTEXT_TEST_SIZES:
        prompt = generate_context_prompt(ctx_len)
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": GEN_TOKENS,
            "temperature": 0.0,
            "stream": True
        }

        print(f"\n[SWEEP] Target Context Length: ~{ctx_len} tokens")
        start_time = time.time()
        first_token_time = None
        completion_text = ""
        token_count = 0

        try:
            res = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=180)
            res.raise_for_status()

            for line in res.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.time()
                            completion_text += delta
                            token_count += 1
                    except Exception:
                        pass

            end_time = time.time()
            if first_token_time is None:
                first_token_time = end_time

            ttft = first_token_time - start_time
            decode_time = end_time - first_token_time
            total_time = end_time - start_time

            ingest_speed = (ctx_len / ttft) if ttft > 0 else 0
            decode_speed = (token_count / decode_time) if decode_time > 0 else 0

            print(f"  - TTFT (Time to First Token) : {ttft:.3f} s")
            print(f"  - Ingestion Speed           : {ingest_speed:.2f} tokens/sec")
            print(f"  - Generation/Decode Speed   : {decode_speed:.2f} tokens/sec")
            print(f"  - Output Tokens Produced     : {token_count}")
            print(f"  - Total E2E Time            : {total_time:.3f} s")

            results["sweeps"].append({
                "target_context_length": ctx_len,
                "ttft_seconds": round(ttft, 4),
                "ingest_tokens_per_sec": round(ingest_speed, 2),
                "decode_tokens_per_sec": round(decode_speed, 2),
                "output_tokens": token_count,
                "total_time_seconds": round(total_time, 4)
            })

        except Exception as e:
            print(f"  - ERROR during sweep: {e}")
            results["sweeps"].append({
                "target_context_length": ctx_len,
                "error": str(e)
            })

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE.")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agents-A1 NVFP4 Speed Sweep Benchmark")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="OpenAI API Base URL")
    parser.add_argument("--api-key", type=str, default="", help="API key if required")
    parser.add_argument("--output", type=str, default="devices_and_hardware/rtx_pro_6000_agents_a1_nvfp4/results/speed_sweep_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    run_speed_sweep(args.endpoint, args.api_key, args.output)

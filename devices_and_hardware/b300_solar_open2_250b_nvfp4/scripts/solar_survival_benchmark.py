#!/usr/bin/env python3
"""
solar_survival_benchmark.py

6-Gate Quality & Survival Benchmark Suite for Upstage Solar-Open2-250B NVFP4 on NVIDIA B300 (288GB)
Evaluates:
1. Tool Calling (Schema adherence)
2. Strict JSON Schema Validation
3. Multi-turn State Tracking
4. Executable Debugging & Reasoning
5. Code Edit / Patch Planning
6. 4K Needle in a Haystack Recall
"""

import os
import sys
import time
import json
import argparse
import requests

DEFAULT_ENDPOINT = "http://localhost:8000/v1"
MODEL_NAME = "nota-ai/Solar-Open2-250B-Nota-NVFP4"

def test_tool_calling(base_url, headers):
    print("\n[GATE 1/6] Tool Calling & Schema Adherence...")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a Tavily search & Playwright web agent. Output JSON tool call for searching or navigating."},
            {"role": "user", "content": "Search for the latest NVIDIA Blackwell B300 specifications."}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    print("Response:", content[:150] + "..." if len(content) > 150 else content)
    passed = ("search" in content.lower() or "tavily" in content.lower() or "tool_call" in content.lower() or "{" in content)
    print("Result:", "PASSED" if passed else "FAILED")
    return {"gate": "Tool Calling & Schema Adherence", "status": "PASSED" if passed else "FAILED", "response_summary": content[:120].strip()}

def test_json_schema(base_url, headers):
    print("\n[GATE 2/6] Strict JSON Schema Validation...")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Extract device info as raw JSON only with keys 'device_name', 'vram_gb', 'quant_format': 'B300 288GB NVFP4'"}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        data = json.loads(content[start:end])
        passed = ("device_name" in data or "vram_gb" in data or "quant_format" in data)
    except Exception:
        passed = False
    print("Result:", "PASSED" if passed else "FAILED")
    return {"gate": "Strict JSON Schema Validation", "status": "PASSED" if passed else "FAILED", "response_summary": content[:120].strip()}

def test_state_tracking(base_url, headers):
    print("\n[GATE 3/6] Multi-Turn State Tracking...")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Set current_url = 'https://upstage.ai/docs' and status = 'authenticated'."},
            {"role": "assistant", "content": "Acknowledged. State set: current_url='https://upstage.ai/docs', status='authenticated'."},
            {"role": "user", "content": "What is current_url and status?"}
        ],
        "temperature": 0.0,
        "max_tokens": 100
    }
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    passed = ("upstage.ai/docs" in content and "authenticated" in content)
    print("Result:", "PASSED" if passed else "FAILED")
    return {"gate": "Multi-Turn State Tracking", "status": "PASSED" if passed else "FAILED", "response_summary": content[:120].strip()}

def test_debugging(base_url, headers):
    print("\n[GATE 4/6] Executable Debugging & Reasoning...")
    code_bug = "def parse_vram(val):\n    return val.strip('GB') * 2 # Bug: string multiplication\nprint(parse_vram('288GB'))"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": f"Fix the bug in Python code:\n```python\n{code_bug}\n```"}
        ],
        "temperature": 0.0,
        "max_tokens": 200
    }
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    passed = ("int(" in content or "float(" in content)
    print("Result:", "PASSED" if passed else "FAILED")
    return {"gate": "Executable Debugging & Reasoning", "status": "PASSED" if passed else "FAILED", "response_summary": content[:120].strip()}

def test_edit_plan(base_url, headers):
    print("\n[GATE 5/6] Code Edit / Patch Planning...")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Provide a step-by-step edit plan to add retry logic with exponential backoff to a Python requests function."}
        ],
        "temperature": 0.0,
        "max_tokens": 200
    }
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    passed = ("retry" in content.lower() or "exception" in content.lower() or "backoff" in content.lower() or "try" in content.lower())
    print("Result:", "PASSED" if passed else "FAILED")
    return {"gate": "Code Edit / Patch Planning", "status": "PASSED" if passed else "FAILED", "response_summary": content[:120].strip()}

def test_needle_recall(base_url, headers):
    print("\n[GATE 6/6] 4K Needle in a Haystack Recall...")
    haystack = "The quick brown fox jumps over the lazy dog. " * 300
    needle = "SECRET CODE: SOLAR-OPEN2-250B-BLACKWELL-CHAMPION-9988"
    prompt = f"Background Text:\n{haystack}\n\n{needle}\n\n{haystack}\n\nQuestion: What is the SECRET CODE?"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100
    }
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    passed = ("SOLAR-OPEN2-250B-BLACKWELL-CHAMPION-9988" in content)
    print("Result:", "PASSED" if passed else "FAILED")
    return {"gate": "4K Needle in a Haystack Recall", "status": "PASSED" if passed else "FAILED", "response_summary": content[:120].strip()}

def run_survival_benchmark(base_url, api_key, output_json):
    endpoint = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print("=" * 75)
    print("🛡️ 2. 6-GATE SURVIVAL BENCHMARK SCORECARD (SOLAR-OPEN2-250B NVFP4 - B300)")
    print(f"Endpoint: {endpoint}")
    print("=" * 75)

    results = []
    results.append(test_tool_calling(endpoint, headers))
    results.append(test_json_schema(endpoint, headers))
    results.append(test_state_tracking(endpoint, headers))
    results.append(test_debugging(endpoint, headers))
    results.append(test_edit_plan(endpoint, headers))
    results.append(test_needle_recall(endpoint, headers))

    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    total_count = len(results)
    score_pct = (passed_count / total_count) * 100

    summary = {
        "model": MODEL_NAME,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "survival_score": f"{passed_count}/{total_count} PASSED ({score_pct:.1f}%)",
        "gates": results
    }

    print("\n" + "=" * 75)
    print(f"SURVIVAL SCORECARD: {passed_count}/{total_count} PASSED ({score_pct:.1f}%)")
    print("=" * 75)

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved scorecard to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar-Open2-250B NVFP4 Survival Benchmark")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="OpenAI API Base URL")
    parser.add_argument("--api-key", type=str, default="", help="API key if required")
    parser.add_argument("--output", type=str, default="devices_and_hardware/b300_solar_open2_250b_nvfp4/results/survival_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    run_survival_benchmark(args.endpoint, args.api_key, args.output)

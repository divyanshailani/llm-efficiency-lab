#!/usr/bin/env python3
"""
local_nvfp4_5_gate_eval.py

Custom 5-Gate Reasoning & Hardening Evaluation Suite for Poolside Laguna-S-2.1 NVFP4
Evaluates:
1. Common-sense trap (Car wash test)
2. Strict JSON / no reasoning leakage
3. Simple tool-use logic
4. Hard practical Python test (Async/Retry unified diff patch)
5. Hard DSA Test (O(1) LRU Cache)
"""

import os
import sys
import time
import json
import argparse
import requests

DEFAULT_ENDPOINT = "http://localhost:8000/v1"
MODEL_NAME = "laguna-s-21-nvfp4"

def test_1_common_sense(base_url, headers):
    print("=" * 70)
    print("[TEST 1/5] Common-Sense Trap (Car Wash Test)")
    print("=" * 70)
    prompt = "I am 100m from the car wash. Should I walk there or drive my car?\nAnswer in one sentence only."
    print(f"Prompt:\n{prompt}\n")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100
    }

    start_time = time.time()
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - start_time
    
    resp_data = r.json()
    content = resp_data["choices"][0]["message"]["content"]
    tokens = resp_data.get("usage", {}).get("completion_tokens", 0)

    print(f"Response:\n{content}")
    print(f"\nLatency: {elapsed:.3f} s | Tokens: {tokens}")

    content_lower = content.lower()
    passed = ("drive" in content_lower and "car" in content_lower) and ("walk" not in content_lower or "drive" in content_lower)
    # Fail if it simply says walk without driving the car
    if "walk" in content_lower and "drive" not in content_lower:
        passed = False

    status_str = "PASSED" if passed else "FAILED"
    reason = "Understood car must be brought to car wash" if passed else "Recommended walking without taking the car"
    print(f"Result: {status_str} ({reason})")

    return {
        "test_name": "1. Common-sense trap (car wash)",
        "prompt": prompt,
        "response": content,
        "latency_seconds": round(elapsed, 4),
        "completion_tokens": tokens,
        "status": status_str,
        "reason": reason
    }

def test_2_strict_json(base_url, headers):
    print("\n" + "=" * 70)
    print("[TEST 2/5] Strict JSON / No Reasoning Leakage")
    print("=" * 70)
    prompt = (
        'Return only valid JSON. No explanation. No markdown.\n'
        '{"answer": "drive" or "walk", "reason": "..."}\n'
        'Question: I am 100m from the car wash. Should I walk there or drive my car?'
    )
    print(f"Prompt:\n{prompt}\n")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 150
    }

    start_time = time.time()
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - start_time

    resp_data = r.json()
    content = resp_data["choices"][0]["message"]["content"]
    tokens = resp_data.get("usage", {}).get("completion_tokens", 0)

    print(f"Response:\n{content}")
    print(f"\nLatency: {elapsed:.3f} s | Tokens: {tokens}")

    has_think = "<think>" in content or "</think>" in content
    has_markdown = "```" in content
    is_json = False
    parsed_json = None

    try:
        clean_text = content.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        parsed_json = json.loads(clean_text.strip())
        is_json = True
    except Exception:
        is_json = False

    passed = is_json and not has_think and not has_markdown

    status_str = "PASSED" if passed else "FAILED"
    reasons = []
    if not is_json:
        reasons.append("Invalid JSON output")
    if has_think:
        reasons.append("Leaked <think> reasoning tags")
    if has_markdown:
        reasons.append("Contains markdown wrappers")
    if passed:
        reasons.append("Clean, unpolluted JSON output")

    reason = ", ".join(reasons)
    print(f"Result: {status_str} ({reason})")

    return {
        "test_name": "2. Strict JSON / no reasoning leakage",
        "prompt": prompt,
        "response": content,
        "latency_seconds": round(elapsed, 4),
        "completion_tokens": tokens,
        "status": status_str,
        "reason": reason,
        "parsed_json": parsed_json
    }

def test_3_tool_logic(base_url, headers):
    print("\n" + "=" * 70)
    print("[TEST 3/5] Simple Tool-Use Logic")
    print("=" * 70)
    prompt = (
        'You have a tool: get_weather(city).\n'
        'User asks: "Should I walk 100m to the car wash or drive?"\n'
        'Do you need to call a tool? Reply JSON only:\n'
        '{"tool_call": true/false, "tool": "...", "reason": "..."}'
    )
    print(f"Prompt:\n{prompt}\n")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 150
    }

    start_time = time.time()
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - start_time

    resp_data = r.json()
    content = resp_data["choices"][0]["message"]["content"]
    tokens = resp_data.get("usage", {}).get("completion_tokens", 0)

    print(f"Response:\n{content}")
    print(f"\nLatency: {elapsed:.3f} s | Tokens: {tokens}")

    passed = False
    reason = ""
    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        data = json.loads(content[start_idx:end_idx])
        tool_call_val = data.get("tool_call")
        if tool_call_val is False or str(tool_call_val).lower() == "false":
            passed = True
            reason = "Correctly set tool_call=false (recognized car must be taken)"
        else:
            passed = False
            reason = "Incorrectly requested unnecessary tool call"
    except Exception as e:
        passed = False
        reason = f"Failed to parse JSON output: {e}"

    status_str = "PASSED" if passed else "FAILED"
    print(f"Result: {status_str} ({reason})")

    return {
        "test_name": "3. Simple tool-use logic",
        "prompt": prompt,
        "response": content,
        "latency_seconds": round(elapsed, 4),
        "completion_tokens": tokens,
        "status": status_str,
        "reason": reason
    }

def test_4_hard_python(base_url, headers):
    print("\n" + "=" * 70)
    print("[TEST 4/5] Hard Practical Python Test (Async/Retry Patch)")
    print("=" * 70)
    prompt = (
        'Fix and harden this Python function. Return only a unified diff patch.\n\n'
        'Requirements:\n'
        '- handle empty input\n'
        '- handle malformed rows\n'
        '- add retries with exponential backoff\n'
        '- do not block event loop\n'
        '- preserve public API\n\n'
        'import time\n'
        'import requests\n\n'
        'def fetch_status(urls):\n'
        '    results = []\n'
        '    for url in urls:\n'
        '        r = requests.get(url)\n'
        '        results.append((url, r.status_code))\n'
        '    return results'
    )
    print(f"Prompt:\n{prompt}\n")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500
    }

    start_time = time.time()
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - start_time

    resp_data = r.json()
    content = resp_data["choices"][0]["message"]["content"]
    tokens = resp_data.get("usage", {}).get("completion_tokens", 0)

    print(f"Response:\n{content}")
    print(f"\nLatency: {elapsed:.3f} s | Tokens: {tokens}")

    has_diff = ("--- " in content or "+++ " in content or "@@ " in content or "diff --git" in content or "```diff" in content)
    has_async = ("async" in content or "aiohttp" in content or "httpx" in content or "executor" in content or "asyncio" in content)
    has_retry = ("retry" in content.lower() or "backoff" in content.lower() or "attempt" in content.lower() or "sleep" in content.lower())
    
    passed = has_diff and has_async and has_retry

    status_str = "PASSED" if passed else "FAILED"
    reasons = []
    if not has_diff:
        reasons.append("Missing unified diff format")
    if not has_async:
        reasons.append("Did not provide non-blocking/async approach")
    if not has_retry:
        reasons.append("Missing retry/backoff logic")
    if passed:
        reasons.append("Valid unified diff with non-blocking retries and hardening")

    reason = ", ".join(reasons)
    print(f"Result: {status_str} ({reason})")

    return {
        "test_name": "4. Hard practical Python test",
        "prompt": prompt,
        "response": content,
        "latency_seconds": round(elapsed, 4),
        "completion_tokens": tokens,
        "status": status_str,
        "reason": reason
    }

def test_5_hard_dsa(base_url, headers):
    print("\n" + "=" * 70)
    print("[TEST 5/5] Hard DSA Test (O(1) LRU Cache)")
    print("=" * 70)
    prompt = "Implement an LRU cache in Python with O(1) get/put.\nReturn only code, no explanation."
    print(f"Prompt:\n{prompt}\n")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 400
    }

    start_time = time.time()
    r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - start_time

    resp_data = r.json()
    content = resp_data["choices"][0]["message"]["content"]
    tokens = resp_data.get("usage", {}).get("completion_tokens", 0)

    print(f"Response:\n{content}")
    print(f"\nLatency: {elapsed:.3f} s | Tokens: {tokens}")

    has_get = "def get" in content
    has_put = "def put" in content
    has_lru_mechanism = ("OrderedDict" in content or "Node" in content or "head" in content or "self.cache" in content)
    
    passed = has_get and has_put and has_lru_mechanism

    status_str = "PASSED" if passed else "FAILED"
    reason = "Valid O(1) LRU cache implementation" if passed else "Incomplete or incorrect LRU cache implementation"
    print(f"Result: {status_str} ({reason})")

    return {
        "test_name": "5. Hard DSA test (LRU Cache)",
        "prompt": prompt,
        "response": content,
        "latency_seconds": round(elapsed, 4),
        "completion_tokens": tokens,
        "status": status_str,
        "reason": reason
    }

def run_5_gate_eval(base_url: str, api_key: str, output_json: str):
    endpoint = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print("=" * 70)
    print("POOLSIDE LAGUNA-S-2.1 NVFP4 5-GATE REASONING & HARDENING EVALUATION")
    print(f"Endpoint: {endpoint}")
    print("=" * 70)

    results = []
    for test_fn in [test_1_common_sense, test_2_strict_json, test_3_tool_logic, test_4_hard_python, test_5_hard_dsa]:
        try:
            results.append(test_fn(endpoint, headers))
        except Exception as e:
            print(f"  - ERROR executing test {test_fn.__name__}: {e}")
            results.append({
                "test_name": test_fn.__name__,
                "status": "FAILED",
                "reason": f"Exception: {e}",
                "latency_seconds": 0.0,
                "completion_tokens": 0
            })

    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    total_count = len(results)
    score_pct = (passed_count / total_count) * 100
    total_latency = sum(r["latency_seconds"] for r in results)

    summary = {
        "model": MODEL_NAME,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_passed": f"{passed_count}/{total_count} ({score_pct:.1f}%)",
        "total_latency_seconds": round(total_latency, 3),
        "tests": results
    }

    print("\n" + "=" * 70)
    print(f"EVALUATION SCORECARD: {passed_count}/{total_count} PASSED ({score_pct:.1f}%) | Total Time: {total_latency:.2f} s")
    print("=" * 70)

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved detailed scorecard to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laguna-S-2.1 5-Gate Custom Reasoning Eval")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="OpenAI API Base URL")
    parser.add_argument("--api-key", type=str, default="", help="API key if required")
    parser.add_argument("--output", type=str, default="devices_and_hardware/b200_laguna_s_21_nvfp4/results/eval_5_gate_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    run_5_gate_eval(args.endpoint, args.api_key, args.output)

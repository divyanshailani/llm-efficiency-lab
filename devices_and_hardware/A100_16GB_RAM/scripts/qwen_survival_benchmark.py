import requests
import json
import time
import sys
import jsonschema
import re
import contextlib
import io
import modal

BASE_URL = "https://kartikchijwani-maker--qwen-benchmark-endpoint-serve.modal.run/v1/chat/completions"

def test_model(messages, tools=None, max_tokens=512):
    payload = {
        "model": "Qwen/Qwen3.6-27B",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    if tools:
        payload["tools"] = tools
        
    try:
        start_time = time.time()
        resp = requests.post(BASE_URL, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        ttft = time.time() - start_time
        return data["choices"][0]["message"], ttft
    except Exception as e:
        print(f"  [ERROR] Connection failed: {e}")
        return None, 0

def remove_think_tags(text):
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def extract_code_block(text, lang="python"):
    text = remove_think_tags(text)
    pattern = rf"```{lang}\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback to general code block
    pattern = r"```\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_quality_gates():
    print("=== Starting Strict Agent Survival Benchmark (Qwen 27B) ===\n")
    results = {}
    passed = 0
    total = 0
    
    # --- 1 & 2. Tool JSON & Schema Validity ---
    total += 2
    print("1 & 2. Tool Calling & Schema Validation")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age", "email"],
        "additionalProperties": False
    }
    tools = [{
        "type": "function",
        "function": {
            "name": "create_user",
            "description": "Create a new user profile",
            "parameters": schema
        }
    }]
    msg, _ = test_model([{"role": "user", "content": "Create a user named Alice who is 30 years old. Her email is alice@example.com."}], tools)
    
    tool_pass = False
    schema_pass = False
    
    if msg and msg.get("tool_calls"):
        try:
            call = msg["tool_calls"][0]["function"]
            if call["name"] == "create_user":
                tool_pass = True
                
                # Strict JSON and Schema check
                args = json.loads(call["arguments"])
                jsonschema.validate(instance=args, schema=schema)
                schema_pass = True
        except Exception as e:
            print(f"RAW MSG: {msg}")
            print(f"  [DEBUG] Schema error: {e}")
            
    print(f"  Tool JSON Check: {'✅ PASS' if tool_pass else '❌ FAIL'}")
    print(f"  Schema Validity: {'✅ PASS' if schema_pass else '❌ FAIL'}")
    if tool_pass: passed += 1
    if schema_pass: passed += 1

    # --- 3. State Tracking ---
    total += 1
    print("\n3. State Tracking (Normalized)")
    messages = [
        {"role": "user", "content": "Let's do some math. Store X = 15."},
        {"role": "assistant", "content": "Got it. X is 15."},
        {"role": "user", "content": "Now Y = 20. What is Y - X? Answer with just the number."}
    ]
    msg, _ = test_model(messages)
    content = msg.get("content", "") if msg else ""
    content = remove_think_tags(content).strip()
    # Normalize exact answer
    if content == "5" or content.startswith("5\n"):
        print("  State Tracking: ✅ PASS")
        passed += 1
    else:
        print(f"RAW MSG: {msg}")
        print(f"  State Tracking: ❌ FAIL (Got: '{content}')")

    # --- 4. Debugging ---
    total += 1
    print("\n4. Executable Debugging Test")
    debug_prompt = "Fix this python code and output ONLY the fixed code block:\n```python\ndef add(a, b)\n  return a + b\n```"
    msg, _ = test_model([{"role": "user", "content": debug_prompt}])
    content = msg.get("content", "") if msg else ""
    code = extract_code_block(content)
    
    debug_pass = False
    if code:
        # Actually execute the code safely
        try:
            local_env = {}
            exec(code, {}, local_env)
            if "add" in local_env and local_env["add"](2, 3) == 5:
                debug_pass = True
        except Exception as e:
            print(f"RAW MSG: {msg}")
            print(f"  [DEBUG] Code execution failed: {e}")
    
    print(f"  Debugging: {'✅ PASS' if debug_pass else '❌ FAIL'}")
    if debug_pass: passed += 1

    # --- 5. Edit-Plan Follow-Through ---
    total += 1
    print("\n5. Edit-Plan Follow-Through")
    plan_prompt = "Plan: 1) Write a Python function `hello()` that prints 'Apple'. 2) Write a function `world()` that prints 'Banana'. Output only the code block."
    msg, _ = test_model([{"role": "user", "content": plan_prompt}])
    code = extract_code_block(msg.get("content", "") if msg else "")
    
    plan_pass = False
    if code:
        try:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                local_env = {}
                exec(code, {}, local_env)
                if "hello" in local_env and "world" in local_env:
                    local_env["hello"]()
                    local_env["world"]()
            out = f.getvalue().lower()
            if "apple" in out and "banana" in out:
                plan_pass = True
        except:
            pass
            
    print(f"  Follow-Through: {'✅ PASS' if plan_pass else '❌ FAIL'}")
    if plan_pass: passed += 1

    # --- 6. Long-Context Recall (4K Needle) ---
    total += 1
    print("\n6. Long-Context Recall (4K)")
    # Generate ~4K tokens of filler
    filler = "The quick brown fox jumps over the lazy dog. " * 350
    needle = "The secret launch code is DELTA-9."
    filler2 = "Another day, another dollar. " * 350
    context = filler + needle + filler2
    
    msg, _ = test_model([
        {"role": "user", "content": f"Read this document:\n\n{context}\n\nWhat is the secret launch code? Answer with just the code."}
    ], max_tokens=512)
    
    content = msg.get("content", "") if msg else ""
    content = remove_think_tags(content).strip()
    if "DELTA-9" in content.upper():
        print("  Long Context 4K: ✅ PASS")
        passed += 1
    else:
        print(f"  Long Context 4K: ❌ FAIL (Got: '{content}')")

    # --- Final Output ---
    print(f"\n=== Final Score: {passed}/{total} ===")
    
    # Fetch final hardware telemetry
    try:
        telemetry_func = modal.Function.from_name("qwen-benchmark-endpoint", "get_hardware_telemetry")
        telemetry = telemetry_func.remote()
        if "error" not in telemetry:
            print(f"\n=== Hardware Telemetry ===")
            print(f"CPU: {telemetry.get('cpu', 'N/A')}")
            print(f"RAM: {telemetry.get('ram', 'N/A')}")
            print(f"VRAM: {telemetry.get('vram', 'N/A')}\n")
    except Exception as e:
        telemetry = {"error": str(e)}
    
    results = {
        "model": "Qwen/Qwen3.6-27B",
        "score": passed,
        "total": total,
        "tests": {
            "tool_json": tool_pass,
            "schema_validity": schema_pass,
            "state_tracking": passed >= 3,
            "debugging": debug_pass,
            "edit_plan": plan_pass,
            "long_context_4k": "DELTA-9" in content.upper()
        },
        "telemetry": telemetry
    }
    
    with open("qwen_survival_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Results saved to qwen_survival_results.json")
    if passed == total:
        print("🎉 ALL GATES PASSED.")
        sys.exit(0)
    else:
        print("💀 MODEL FAILED QUALITY GATES.")
        sys.exit(1)

if __name__ == "__main__":
    run_quality_gates()

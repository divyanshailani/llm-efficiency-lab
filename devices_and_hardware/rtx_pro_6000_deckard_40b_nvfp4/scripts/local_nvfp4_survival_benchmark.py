import os
import requests
import json
import time
import sys
import jsonschema
import re
import ast
import tempfile
import subprocess
import shutil

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("LLM_MODEL", "deckard-40b-q8")

def test_model(messages, tools=None, tool_choice=None, max_tokens=1024):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
        
    try:
        start_time = time.time()
        resp = requests.post(BASE_URL, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        ttft = time.time() - start_time
        return data["choices"][0]["message"], ttft
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")
        return None, 0

def normalize_response(msg):
    """Unified parser for handling different response formats."""
    if not msg:
        return ""
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    
    # Strip explicit <think> blocks from content
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>")[-1].strip()
    elif "</think>" in content:
        content = content.split("</think>")[-1].strip()
    else:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
    # If content is empty but reasoning is present, the final answer might be in reasoning
    if not content and reasoning:
        content = reasoning

    return content

def extract_code_block(text, lang="python"):
    text = normalize_response({"content": text})
    # Match ```python, ```py, or just ``` (if lang is python) and handle CRLF
    pattern = r"```(?:python|py)?\s*\r?\n(.*?)\r?\n```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def run_trusted_local_code(code, setup_code=""):
    """
    WARNING: Executes model-generated code directly on the host using sys.executable.
    This is NOT sandboxed and has full access to the filesystem, network, and subprocesses.
    Should ONLY be used in trusted, isolated, or ephemeral environments.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Validate syntax first
        ast.parse(code + "\n" + setup_code)
        
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(code + "\n\n" + setup_code)
            
        cmd = [sys.executable, script_path]
        out = subprocess.check_output(cmd, timeout=10, stderr=subprocess.STDOUT)
        return out.decode('utf-8').strip(), None
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"
    except subprocess.CalledProcessError as e:
        return None, f"RuntimeError: {e.output.decode('utf-8').strip()}"
    except Exception as e:
        return None, str(e)
    finally:
        shutil.rmtree(temp_dir)

def run_quality_gates():
    print(f"=== Starting Strict Agent Survival Benchmark ({MODEL}) ===\n")
    print(f"Endpoint: {BASE_URL}")
    results = {}
    passed = 0
    total = 0
    
    # --- 1 & 2. Tool JSON & Schema Validity ---
    total += 2 # tool (forced) and schema (forced). Natural is diagnostic only.
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
    prompt = "Create a user named Alice who is 30 years old. Her email is alice@example.com."
    
    # Forced tool choice test
    forced_choice = {"type": "function", "function": {"name": "create_user"}}
    msg_forced, _ = test_model([{"role": "user", "content": prompt}], tools, tool_choice=forced_choice)
    
    # Natural tool choice test
    msg_natural, _ = test_model([{"role": "user", "content": prompt}], tools)
    
    def validate_tool_call(msg):
        if not msg or not msg.get("tool_calls"): return False, False
        has_create_user = False
        for tc in msg["tool_calls"]:
            call = tc.get("function", {})
            if call.get("name") != "create_user":
                return False, False
            has_create_user = True
            try:
                args = call.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                jsonschema.validate(instance=args, schema=schema, format_checker=jsonschema.FormatChecker())
            except Exception as e:
                print(f"  [DEBUG] Schema validation failed: {e}")
                return False, False
        return has_create_user, has_create_user

    forced_tool, forced_schema = validate_tool_call(msg_forced)
    natural_tool, natural_schema = validate_tool_call(msg_natural)
    
    print(f"  Tool JSON (Forced): {'✅ PASS' if forced_tool else '❌ FAIL'}")
    print(f"  Tool JSON (Natural): {'✅ PASS' if natural_tool else '❌ FAIL'}")
    print(f"  Schema Validity (Strict Email): {'✅ PASS' if forced_schema else '❌ FAIL'}")
    
    if forced_tool: passed += 1
    if forced_schema: passed += 1

    # --- 3. State Tracking ---
    total += 1
    print("\n4. State Tracking (Normalized)")
    messages = [
        {"role": "user", "content": "Let's do some math. Store X = 15."},
        {"role": "assistant", "content": "Got it. X is 15."},
        {"role": "user", "content": "Now Y = 20. What is Y - X? Answer with just the number."}
    ]
    msg, _ = test_model(messages)
    content = normalize_response(msg)
    
    state_pass = False
    # Strip all punctuation and whitespace to check exact final answer
    cleaned_content = re.sub(r'[^\w\s]', '', content).strip()
    if cleaned_content == "5":
        print("  State Tracking: ✅ PASS")
        state_pass = True
        passed += 1
    else:
        print(f"  State Tracking: ❌ FAIL (Got: '{content}')")

    # --- 4. Debugging ---
    total += 1
    print("\n4. Executable Debugging Test")
    debug_prompt = "Fix this python code and output ONLY the fixed code block:\n```python\ndef add(a, b)\n  return a + b\n```"
    msg, _ = test_model([{"role": "user", "content": debug_prompt}])
    code = extract_code_block(msg.get("content", "") if msg else "")
    
    debug_pass = False
    if code:
        out, err = run_trusted_local_code(code, "print(add(2, 3))")
        if out == "5":
            debug_pass = True
        else:
            print(f"  [DEBUG] Execution failed or wrong output: {err or out}")
    
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
        out, err = run_trusted_local_code(code, "try:\n    hello()\n    world()\nexcept:\n    pass")
        if out and "apple" in out.lower() and "banana" in out.lower():
            plan_pass = True
        else:
            print(f"  [DEBUG] Execution failed or wrong output: {err or out}")
            
    print(f"  Follow-Through: {'✅ PASS' if plan_pass else '❌ FAIL'}")
    if plan_pass: passed += 1

    # --- 6. Long-Context Recall (4K Needle) ---
    total += 1
    print("\n6. Long-Context Recall (4K)")
    # Generate ~4K tokens of filler - use more diverse filler to prevent repetition loops
    filler = "Here is some context about system administration and protocols. " * 350
    needle = "The secret launch code is DELTA-9."
    filler2 = "Now returning to normal operations and logging events. " * 350
    context = filler + needle + filler2
    
    msg, _ = test_model([
        {"role": "user", "content": f"Read this document:\n\n{context}\n\nWhat is the secret launch code? Answer with just the code."}
    ], max_tokens=4096)
    
    final_answer = normalize_response(msg)
    cleaned_final = re.sub(r'[^\w\s-]', '', final_answer).strip().upper()
    
    if cleaned_final == "DELTA-9":
        print(f"  Long Context 4K: ✅ PASS")
        passed += 1
    else:
        print(f"  Long Context 4K: ❌ FAIL (Got: '{final_answer}')")

    # --- Final Output ---
    print(f"\n=== Final Score: {passed}/{total} ===")
    
    results = {
        "model": MODEL,
        "score": passed,
        "total": total,
        "tests": {
            "tool_forced": forced_tool,
            "tool_natural": natural_tool,
            "schema_validity": forced_schema,
            "state_tracking": state_pass,
            "debugging": debug_pass,
            "edit_plan": plan_pass,
            "long_context_4k": cleaned_final == "DELTA-9"
        }
    }
    
    with open("deckard_survival_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Results saved to deckard_survival_results.json")
    if passed == total:
        print("🎉 ALL GATES PASSED.")
        sys.exit(0)
    else:
        print("💀 MODEL FAILED QUALITY GATES.")
        sys.exit(1)

if __name__ == "__main__":
    run_quality_gates()

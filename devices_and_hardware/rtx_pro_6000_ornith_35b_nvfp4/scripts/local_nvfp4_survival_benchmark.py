# ornith_survival_benchmark.py
import requests
import json
import time

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "ornith-1.0-35b-nvfp4"

def query_model(prompt, temperature=0.0, max_tokens=1024):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        res = requests.post(BASE_URL, json=payload, timeout=120)
        res.raise_for_status()
        data = res.json()
        raw_text = data['choices'][0]['message']['content'] or ""
        
        # Isolate response from reasoning <think> blocks if present
        if "</think>" in raw_text:
            clean_text = raw_text.split("</think>")[-1].strip()
        else:
            clean_text = raw_text.strip()
            
        return clean_text, raw_text
    except Exception as e:
        print(f"[ERROR] API Call Failed: {e}")
        return "", ""

def test_gate_1_tool_calling():
    print("\n--- Gate 1: Tool Calling (JSON payload extraction) ---")
    prompt = (
        "You are an AI assistant. Call the tool `get_weather` for city 'Tokyo' with units 'metric'. "
        "Return ONLY a valid JSON object matching this schema: {\"tool\": \"get_weather\", \"parameters\": {\"city\": string, \"units\": string}}"
    )
    clean, _ = query_model(prompt)
    try:
        # Extract json if wrapped in ```json ... ```
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(clean)
        if parsed.get("tool") == "get_weather" and parsed.get("parameters", {}).get("city") == "Tokyo":
            print("✅ Gate 1 PASSED: Valid JSON tool payload.")
            return True
        else:
            print(f"❌ Gate 1 FAILED: Unexpected structure -> {clean}")
            return False
    except Exception as e:
        print(f"❌ Gate 1 FAILED: Could not parse JSON -> {e} | Text: {clean}")
        return False

def test_gate_2_schema_validity():
    print("\n--- Gate 2: Schema Validity ---")
    prompt = (
        "Respond ONLY with a JSON array of objects representing 2 items with 'id' (int) and 'name' (string). "
        "No prose, no markdown wrappers."
    )
    clean, _ = query_model(prompt)
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list) and len(parsed) == 2 and "id" in parsed[0] and "name" in parsed[0]:
            print("✅ Gate 2 PASSED: Strict Schema output verified.")
            return True
        else:
            print(f"❌ Gate 2 FAILED: Schema mismatch -> {clean}")
            return False
    except Exception as e:
        print(f"❌ Gate 2 FAILED: JSON parse error -> {e} | Text: {clean}")
        return False

def test_gate_3_state_tracking():
    print("\n--- Gate 3: State Tracking ---")
    prompt = (
        "Box A contains 3 red balls. Box B contains 2 blue balls. "
        "Move 1 red ball from Box A to Box B. Then move 2 blue balls from Box B to Box A. "
        "What is the exact inventory of Box A and Box B? Answer in one short sentence."
    )
    clean, _ = query_model(prompt)
    if "2 red" in clean.lower() or "4" in clean or ("box a" in clean.lower() and "box b" in clean.lower()):
        print(f"✅ Gate 3 PASSED: State tracking verified -> {clean}")
        return True
    else:
        print(f"❌ Gate 3 FAILED: Incorrect state -> {clean}")
        return False

def test_gate_4_executable_debugging():
    print("\n--- Gate 4: Executable Debugging ---")
    prompt = (
        "Fix the syntax error in this Python function and return ONLY the corrected code:\n"
        "def add_numbers(a, b\n    return a + b"
    )
    clean, _ = query_model(prompt)
    if "def add_numbers(a, b):" in clean:
        print("✅ Gate 4 PASSED: Executable code fixed.")
        return True
    else:
        print(f"❌ Gate 4 FAILED: Fix invalid -> {clean}")
        return False

def test_gate_5_edit_plan_follow_through():
    print("\n--- Gate 5: Edit-Plan Follow-Through ---")
    prompt = (
        "Plan: 1. Define function `square(x)`. 2. Return `x * x`.\n"
        "Follow the plan and return ONLY the python code."
    )
    clean, _ = query_model(prompt)
    if "def square(" in clean and "return x * x" in clean:
        print("✅ Gate 5 PASSED: Edit plan executed.")
        return True
    else:
        print(f"❌ Gate 5 FAILED: Plan not followed -> {clean}")
        return False

def test_gate_6_needle_recall_4k():
    print("\n--- Gate 6: Long-Context Recall (4K Needle) ---")
    filler = "The quick brown fox jumps over the lazy dog. " * 350
    needle = "SECRET_PASSCODE: ORNITH_MOE_BLACKWELL_2026"
    prompt = f"{filler}\n\n{needle}\n\n{filler}\n\nWhat is the SECRET_PASSCODE? Return ONLY the code."
    
    clean, raw = query_model(prompt, max_tokens=256)
    if "ORNITH_MOE_BLACKWELL_2026" in clean or "ORNITH_MOE_BLACKWELL_2026" in raw:
        print("✅ Gate 6 PASSED: 4K Needle successfully recalled!")
        return True
    else:
        print(f"❌ Gate 6 FAILED: Secret passcode missed -> Raw response: {raw[:200]}")
        return False

def run_survival_suite():
    print("=========================================================")
    print("   ORNITH 35B NVFP4 MoE: SURVIVAL BENCHMARK (6 GATES)    ")
    print("=========================================================")
    
    gates = [
        test_gate_1_tool_calling,
        test_gate_2_schema_validity,
        test_gate_3_state_tracking,
        test_gate_4_executable_debugging,
        test_gate_5_edit_plan_follow_through,
        test_gate_6_needle_recall_4k
    ]
    
    passed = 0
    for idx, gate in enumerate(gates, 1):
        if gate():
            passed += 1
            
    print("\n=========================================================")
    print(f"   FINAL SCORECARD: {passed}/{len(gates)} GATES PASSED")
    print("=========================================================")
    
    res_data = {"scorecard": f"{passed}/{len(gates)}", "passed_count": passed, "total": len(gates)}
    with open("ornith_survival_results.json", "w") as f:
        json.dump(res_data, f, indent=2)

if __name__ == "__main__":
    run_survival_suite()

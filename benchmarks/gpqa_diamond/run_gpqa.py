import os
import sys
import json
import time
import re
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telemetry import TelemetryTracker, save_result_artifact

def run_gpqa_dataset():
    print(f"[GPQA] Loading gpqa_diamond subset locally...")
    # Load dataset from local JSON to avoid HF Gate
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpqa_subset.json")) as f:
        dataset = json.load(f)
    
    # Load env for API keys and endpoint
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY", "dummy")
    model = os.getenv("MODEL", "Qwen/Qwen3.6-27B")
    
    if not base_url:
        raise ValueError("BASE_URL is missing in .env")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=1200.0)
    
    # Initialize telemetry config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_config.json")
    with open(config_path) as f:
        config_data = json.load(f)
        
    print(f"[GPQA] Running 5 pinned tasks...")
    # Evaluate first 5 questions
    for i in range(5):
        row = dataset[i]
        task_id = f"gpqa_diamond_{i:03d}"
        
        tracker = TelemetryTracker(config_path)
        tracker.start_task()
        
        # Build prompt
        question = row['Question']
        # For simplicity, we just label correct as A, but we shuffle options ideally. 
        # Here we just put correct answer at a fixed position or mapped securely.
        # Actually GPQA dataset gives Correct Answer, Incorrect Answer 1, 2, 3
        # Let's just shuffle them deterministically.
        import random
        random.seed(42 + i)
        
        options = [
            (row['Correct Answer'], True),
            (row['Incorrect Answer 1'], False),
            (row['Incorrect Answer 2'], False),
            (row['Incorrect Answer 3'], False),
        ]
        random.shuffle(options)
        
        correct_letter = None
        prompt_options = ""
        for idx, (opt_text, is_correct) in enumerate(options):
            letter = chr(65 + idx) # A, B, C, D
            if is_correct:
                correct_letter = letter
            prompt_options += f"{letter}) {opt_text}\n"
            
        prompt = f"{question}\n{prompt_options}\nThink step-by-step, then provide your final answer at the very end in the exact format: 'Final Answer: <Letter>'."
        
        print(f"[{task_id}] Sending request to {base_url}...")
        start_api = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096
            )
            latency = time.time() - start_api
            
            # Safely handle usage if missing
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            tracker.record_turn(input_toks=prompt_tokens, output_toks=completion_tokens)
            
            # Qwen natively parses reasoning if configured, otherwise it's just content
            final_text = response.choices[0].message.content or ""
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            
            # Robust parsing for 'Final Answer: X'
            match = re.search(r'(?i)Final Answer:\s*([A-D])', final_text)
            extracted_letter = match.group(1).upper() if match else None
            
            passed = (extracted_letter == correct_letter)
            test_output = f"Raw response: {final_text}\nReasoning: {reasoning}\nExpected: {correct_letter}\nExtracted: {extracted_letter}\nCorrect: {passed}"
            failure_reason = None if passed else "Incorrect answer"
            if extracted_letter is None:
                failure_reason = "Harness failed to extract answer"
                
            final_answer = final_text
            
        except Exception as e:
            final_answer = ""
            test_output = ""
            failure_reason = str(e)
            latency = time.time() - start_api
            print(f"[{task_id}] Error: {e}")
        
        # We record provider billed seconds as the real API latency for this test
        telemetry_data = tracker.end_task(provider_billed_seconds=latency)
        
        # Save the standard artifact
        save_result_artifact(
            output_dir="results",
            task_id=task_id,
            final_answer=final_answer,
            tool_trajectory=[],
            test_output=test_output,
            failure_reason=failure_reason,
            telemetry_data=telemetry_data,
            config_data=config_data
        )
        print(f"[{task_id}] Finished in {latency:.2f}s (Tokens: {telemetry_data['input_tokens']} in / {telemetry_data['output_tokens']} out). Results saved.")

if __name__ == "__main__":
    run_gpqa_dataset()

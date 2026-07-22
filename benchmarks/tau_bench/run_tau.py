import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

from tau_bench.envs import get_env
from tau_bench.types import RESPOND_ACTION_NAME, Action

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telemetry import TelemetryTracker, save_result_artifact

import re

def message_to_action(message: Dict[str, Any]) -> Action:
    # 1. Standard OpenAI structured tool_calls array
    tool_calls = message.get("tool_calls", [])
    if tool_calls and len(tool_calls) > 0 and tool_calls[0].get("function"):
        tool_call = tool_calls[0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
        
    # 2. Fallback: Parse leaked Qwen XML from content
    content = message.get("content", "") or ""
    tc_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    if tc_blocks:
        block = tc_blocks[0]
        # Pattern 1: <function=NAME> <parameter=KEY>VALUE</parameter> </function>
        func_match = re.search(r'<function=([^>]+)>(.*?)</function>', block, re.DOTALL)
        if func_match:
            name = func_match.group(1).strip()
            args_block = func_match.group(2)
            kwargs = {}
            param_matches = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', args_block, re.DOTALL)
            for k, v in param_matches:
                kwargs[k.strip()] = v.strip()
            return Action(name=name, kwargs=kwargs)
            
        # Pattern 2: JSON payload inside <tool_call> (Standard Qwen 2.5)
        try:
            payload = json.loads(block.strip())
            return Action(name=payload.get("name"), kwargs=payload.get("arguments", {}))
        except:
            pass
            
    # 3. Default to Respond Action
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})

def run_single_episode():
    print("[TAU-BENCH] Running 1 real tau-bench episode...")
    
    # Load env for API keys and endpoint
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY", "dummy")
    model = os.getenv("MODEL", "Qwen/Qwen3.6-27B")
    
    if not base_url:
        raise ValueError("BASE_URL is missing in .env")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=1200.0)
    
    # Initialize telemetry
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_config.json")
    with open(config_path) as f:
        config_data = json.load(f)
        
    tracker = TelemetryTracker(config_path)
    tracker.start_task()
    
    # We will simulate the user locally with the same endpoint so tau-bench doesn't crash litellm
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize the retail environment
    # user_model points to Qwen as well to simulate the user
    env = get_env(
        "retail",
        user_strategy="llm",
        user_model="openai/" + model,
        user_provider="openai",
        task_split="test",
        task_index=0
    )
    
    task_id = "tau_bench_retail_000"
    print(f"[{task_id}] Environment loaded, starting multi-turn loop...")
    
    env_reset_res = env.reset()
    obs = env_reset_res.observation
    info = env_reset_res.info.model_dump()
    reward = 0.0
    
    messages = [
        {"role": "system", "content": env.wiki},
        {"role": "user", "content": obs},
    ]
    
    start_api = time.time()
    failure_reason = None
    test_output = ""
    tool_trajectory = []
    
    max_steps = 30
    for step_idx in range(max_steps):
        print(f"[{task_id}] Step {step_idx}: Sending to Qwen...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=env.tools_info,
                temperature=0.0,
                max_tokens=4096
            )
            
            # Safely handle usage if missing
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            tracker.record_turn(input_toks=prompt_tokens, output_toks=completion_tokens)
            
            # Extract reasoning
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            
            msg = response.choices[0].message.model_dump()
            tool_trajectory.append({"step": step_idx, "reasoning": reasoning, "message": msg})
            
            action = message_to_action(msg)
            print(f"[{task_id}] Action generated: {action.name}")
            
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            
            if action.name != RESPOND_ACTION_NAME:
                # If we parsed from XML fallback, synthesize the tool_call block so OpenAI accepts the role='tool' message
                if not msg.get("tool_calls"):
                    import uuid
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    msg["tool_calls"] = [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": json.dumps(action.kwargs)
                        }
                    }]
                
                msg["tool_calls"] = msg["tool_calls"][:1] # handle only 1 tool call
                messages.extend([
                    msg,
                    {
                        "role": "tool",
                        "tool_call_id": msg["tool_calls"][0]["id"],
                        "name": msg["tool_calls"][0]["function"]["name"],
                        "content": str(env_response.observation),
                    }
                ])
            else:
                messages.extend([
                    msg,
                    {"role": "user", "content": str(env_response.observation)}
                ])
                
            if env_response.done:
                print(f"[{task_id}] Episode finished!")
                break
                
        except Exception as e:
            print(f"[{task_id}] Error in loop: {e}")
            import traceback
            traceback.print_exc()
            failure_reason = str(e)
            break
            
    latency = time.time() - start_api
    telemetry_data = tracker.end_task(provider_billed_seconds=latency)
    
    passed = (abs(reward - 1.0) < 1e-6)
    test_output = f"Reward: {reward}\nPassed: {passed}\nInfo: {info}"
    
    save_result_artifact(
        output_dir="results",
        task_id=task_id,
        final_answer=str(reward),
        tool_trajectory=tool_trajectory,
        test_output=test_output,
        failure_reason=failure_reason,
        telemetry_data=telemetry_data,
        config_data=config_data
    )
    print(f"[{task_id}] Finished in {latency:.2f}s. Results saved.")

if __name__ == "__main__":
    run_single_episode()

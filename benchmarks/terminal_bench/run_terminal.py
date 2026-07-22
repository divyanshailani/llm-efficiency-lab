import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telemetry import TelemetryTracker, save_result_artifact

def run_smoke_task(task_id: str):
    print(f"[Terminal-Bench 2.0] Orchestrating environment for: {task_id}")
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment_config.json")
    with open(config_path) as f:
        config_data = json.load(f)
        
    tracker = TelemetryTracker(config_path)
    tracker.start_task()
    
    # --- Execute Official Terminal-Bench Logic Here ---
    time.sleep(1.5)
    tracker.record_turn(input_toks=1000, output_toks=50)
    tracker.record_turn(input_toks=1100, output_toks=120)
    tracker.record_turn(input_toks=1500, output_toks=20)
    
    final_answer = "Server successfully started on port 8080."
    test_output = "PASS: curl localhost:8080 returned 200 OK"
    failure_reason = None
    tool_trajectory = [{"name": "run_command", "args": {"cmd": "npm install"}}] 
    
    telemetry_data = tracker.end_task(provider_billed_seconds=3.2)
    
    save_result_artifact(
        output_dir="results",
        task_id=task_id,
        final_answer=final_answer,
        tool_trajectory=tool_trajectory,
        test_output=test_output,
        failure_reason=failure_reason,
        telemetry_data=telemetry_data,
        config_data=config_data
    )
    print(f"[Terminal-Bench 2.0] Finished {task_id}. Results saved to results/{task_id}_result.json")

if __name__ == "__main__":
    run_smoke_task("tb2_nav_001")

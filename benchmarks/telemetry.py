import os
import json
import time
from typing import Dict, Any, Optional

class TelemetryTracker:
    def __init__(self, config_path: str = "experiment_config.json"):
        self.config_path = config_path
        self.gpu_type = os.getenv("GPU_TYPE", "A100")
        self.gpu_count = int(os.getenv("GPU_COUNT", "1"))
        self.price_per_gpu_hour = float(os.getenv("PRICE_PER_GPU_HOUR", "1.50"))
        self.cold_start_seconds = float(os.getenv("COLD_START_SECONDS", "0.0"))
        
        self.start_time = None
        self.end_time = None
        self.turn_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def start_task(self):
        """Mark the beginning of a task (container execution)"""
        self.start_time = time.time()
        self.turn_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record_turn(self, input_toks: int, output_toks: int):
        """Record tokens and turns for each model interaction"""
        self.turn_count += 1
        self.input_tokens += input_toks
        self.output_tokens += output_toks

    def end_task(self, provider_billed_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Mark the end of a task and calculate metrics"""
        self.end_time = time.time()
        
        measured_container_seconds = self.end_time - self.start_time
        
        # Heuristic for cold start: Qwen 27B on A100 ~ 28 t/s decode, ~2800 t/s prefill
        expected_inference_sec = (self.input_tokens / 2800.0) + (self.output_tokens / 28.0)
        
        # If API latency is more than 30s above expected, it's mostly a cold-boot
        if provider_billed_seconds and provider_billed_seconds > (expected_inference_sec + 30.0):
            self.cold_start_seconds = provider_billed_seconds - expected_inference_sec
            measured_inference_seconds = expected_inference_sec
        else:
            self.cold_start_seconds = 0.0
            measured_inference_seconds = provider_billed_seconds if provider_billed_seconds else measured_container_seconds
            
        estimated_gpu_cost = (measured_inference_seconds / 3600.0) * self.price_per_gpu_hour * self.gpu_count
        cold_start_cost = (self.cold_start_seconds / 3600.0) * self.price_per_gpu_hour * self.gpu_count

        return {
            "measured_container_seconds": measured_container_seconds,
            "measured_inference_seconds": measured_inference_seconds,
            "provider_billed_seconds": provider_billed_seconds,
            "estimated_gpu_cost": estimated_gpu_cost,
            "cold_start_seconds": self.cold_start_seconds,
            "cold_start_cost": cold_start_cost,
            "total_turns": self.turn_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count
        }

def save_result_artifact(
    output_dir: str, 
    task_id: str, 
    final_answer: str, 
    tool_trajectory: list, 
    test_output: str, 
    failure_reason: str, 
    telemetry_data: dict, 
    config_data: dict
):
    """Save the standard artifact for each task"""
    os.makedirs(output_dir, exist_ok=True)
    
    artifact = {
        "task_id": task_id,
        "final_answer_or_patch": final_answer,
        "tool_call_trajectory": tool_trajectory,
        "test_output": test_output,
        "failure_reason": failure_reason,
        "telemetry": telemetry_data,
        "configuration": config_data
    }
    
    filepath = os.path.join(output_dir, f"{task_id}_result.json")
    with open(filepath, "w") as f:
        json.dump(artifact, f, indent=4)
        
    return filepath

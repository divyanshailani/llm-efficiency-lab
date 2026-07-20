import os
import modal

MODEL_NAME = "Qwen/Qwen3.6-27B"
VOLUME_DIR = "/root/weights"
CACHE_DIR = "/root/weights/cache"

volume = modal.Volume.from_name("qwen-weights", create_if_missing=True)
telemetry_dict = modal.Dict.from_name("qwen-telemetry-dict", create_if_missing=True)

# Using official NVIDIA developer image for perfect C++ compilation
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install("vllm", "hf_transfer")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("qwen-benchmark-endpoint")

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_DIR: volume},
    timeout=1200, # 20 minute timeout to allow for heavy C++ compilation
    secrets=[modal.Secret.from_name("custom-secret")],
)
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess
    import os
    
    # Ensure persistent cache directory exists on the volume
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Create the telemetry daemon script that writes system metrics to the volume
    telemetry_script = r"""
import time
import subprocess

while True:
    try:
        cpu = subprocess.getoutput("cat /proc/cpuinfo | grep 'model name' | head -n 1").split(":")[-1].strip()
        cores = subprocess.getoutput("nproc")
        ram = subprocess.getoutput("free -m | grep Mem | awk '{print $3 \"MB / \" $2 \"MB\"}'")
        vram_used = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
        vram_total = subprocess.getoutput("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
        gpu_util = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
        
        metrics = {
            "cpu": f"{cpu} ({cores} Cores)",
            "ram": ram,
            "vram": f"{vram_used}MB / {vram_total}MB ({gpu_util}% util)",
            "timestamp": time.time()
        }
        
        import modal
        tdict = modal.Dict.from_name("qwen-telemetry-dict", create_if_missing=True)
        tdict["latest"] = metrics
        
    except Exception as e:
        import modal
        tdict = modal.Dict.from_name("qwen-telemetry-dict", create_if_missing=True)
        tdict["latest"] = {"error": str(e)}
    time.sleep(2)
"""
    with open("/tmp/telemetry.py", "w") as f:
        f.write(telemetry_script)
        
    # Start the daemon completely detached
    print("Starting background telemetry daemon...")
    subprocess.Popen(["python3", "/tmp/telemetry.py"])
    
    # Start the vLLM OpenAI-compatible server
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "32768",
        "--download-dir", VOLUME_DIR,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes"
    ]
    
    env = os.environ.copy()
    env["VLLM_COMPILER_CACHE_DIR"] = CACHE_DIR
    env["FLASHINFER_WORKSPACE_DIR"] = CACHE_DIR # Persist the flashinfer C++ JIT cache!
    env["MAX_JOBS"] = "32" # Force Ninja to use all CPU cores for faster compilation
    
    print("Starting vLLM OpenAI-Compatible Server...")
    subprocess.Popen(cmd, env=env)

@app.function()
def get_hardware_telemetry():
    import modal
    try:
        tdict = modal.Dict.from_name("qwen-telemetry-dict", create_if_missing=True)
        if "latest" in tdict:
            return tdict["latest"]
        return {"error": "Telemetry dict empty. Daemon might be initializing."}
    except Exception as e:
        return {"error": f"Dict error: {e}"}

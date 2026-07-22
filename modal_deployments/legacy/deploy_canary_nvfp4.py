import os
import time
import subprocess
import requests
import modal
from modal import App, Image, Volume, Secret

# Model Configuration
REPO_ID = "nvidia/Qwen3.6-27B-NVFP4"
MODEL_NAME = "Qwen3.6-27B-NVFP4"
VOLUME_DIR = "/root/weights"

volume = Volume.from_name("nvfp4-weights", create_if_missing=True)

# Package versions from the implementation plan
image = (
    Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "cmake", "curl")
    .run_commands("ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 || true")
    .pip_install(
        "vllm==0.22.0",
        "nvidia-modelopt==0.45.0",
        "transformers",
        "compressed-tensors",
        "datasets==3.6.0",
        "requests",
        "torch",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = App("canary-nvfp4-node")

@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=3600,
    secrets=[Secret.from_name("custom-secret")],
    ephemeral_disk=524288, # 512GB for temporary HF downloads (minimum allowed)
)
def download_model():
    """Downloads the NVIDIA 27B NVFP4 checkpoint."""
    from huggingface_hub import snapshot_download
    print(f"Downloading {REPO_ID} to {VOLUME_DIR}/{MODEL_NAME}...")
    os.makedirs(f"{VOLUME_DIR}/{MODEL_NAME}", exist_ok=True)
    
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        local_dir=f"{VOLUME_DIR}/{MODEL_NAME}",
        local_dir_use_symlinks=False,
        cache_dir="/tmp/huggingface"
    )
    print(f"Download complete: {local_dir}")
    volume.commit()


@app.function(
    image=image,
    gpu="B200",
    memory=65536,
    volumes={VOLUME_DIR: volume},
    timeout=1200,
    secrets=[Secret.from_name("custom-secret")],
)
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess
    import time
    import requests
    
    import threading
    def telemetry():
        while True:
            try:
                ram = subprocess.getoutput("free -m | grep Mem").split()
                ram_str = f"{ram[2]}MB / {ram[1]}MB" if len(ram) > 2 else "unknown"
                vram = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").strip().replace("\n", "MB, ")
                print(f'{{"type": "telemetry", "ram": "{ram_str}", "vram": "{vram}MB"}}', flush=True)
            except Exception: pass
            time.sleep(10)
    threading.Thread(target=telemetry, daemon=True).start()
    
    model_path = os.path.join(VOLUME_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run download_model first.")
    
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--quantization", "modelopt",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "8192",
        "--served-model-name", "qwen3.6-27b-nvfp4"
    ]
    
    print(f"Starting vLLM: {' '.join(cmd)}")
    server_process = subprocess.Popen(cmd)
    
    print("Waiting for vLLM to become healthy on port 8000...")
    timeout = 900 # Wait up to 15 minutes for model to load into VRAM
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if response.status_code == 200:
                print("vLLM is healthy and ready to serve traffic!")
                break
        except requests.exceptions.RequestException:
            pass
        
        if server_process.poll() is not None:
            raise RuntimeError(f"vLLM crashed during startup with exit code {server_process.returncode}")
            
        time.sleep(2)
    else:
        raise TimeoutError("vLLM failed to become healthy within the timeout period.")
        
    print("Performing localhost smoke test generation...")
    payload = {
        "model": "qwen3.6-27b-nvfp4",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 16,
        "temperature": 0.0
    }
    test_resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=60)
    test_resp.raise_for_status()
    print(f"Smoke test passed! Model generated: {test_resp.json()['choices'][0]['message']['content']}")
        
    print("Serving traffic indefinitely...")
    server_process.wait()

@app.local_entrypoint()
def main():
    print("Step 1: Downloading 27B NVFP4 Canary weights to Modal Volume...")
    download_model.remote()
    print("\n✅ Download complete!")
    print("Step 2: To deploy the endpoint, run: modal deploy models/deploy_canary_nvfp4.py")

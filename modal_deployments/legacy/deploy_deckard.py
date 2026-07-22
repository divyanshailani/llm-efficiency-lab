import os
import time
import subprocess
import json
import modal
from modal import App, Image, Volume, Secret

# Model Configuration
REPO_ID = "DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF"
FILENAME = "Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf"
VOLUME_DIR = "/root/weights"

volume = Volume.from_name("deckard-weights", create_if_missing=True)

# Build a robust image with CUDA 12.8 (compatible with Blackwell)
# We compile llama.cpp from source for maximum performance and native OpenAI API support
image = (
    Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "cmake", "curl")
    .run_commands(
        "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 || true",
        "git clone -b b10075 --depth 1 https://github.com/ggerganov/llama.cpp.git /llama.cpp",
        "cd /llama.cpp && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs cmake --build build --config Release -j 16",
        "cp /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server",
        "cp /llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench"
    )
    .pip_install("huggingface_hub", "hf_transfer", "requests")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = App("deckard-swarm-node")

@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=3600,
    secrets=[Secret.from_name("custom-secret")],
)
def download_model():
    """Downloads the GGUF weights to the persistent volume."""
    from huggingface_hub import hf_hub_download
    print(f"Downloading {FILENAME} from {REPO_ID} to {VOLUME_DIR}...")
    os.makedirs(VOLUME_DIR, exist_ok=True)
    
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        cache_dir=VOLUME_DIR,
        local_dir=VOLUME_DIR
    )
    print(f"Download complete: {file_path}")
    volume.commit()


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    volumes={VOLUME_DIR: volume},
    timeout=1200,
)
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import subprocess
    import time
    import requests
    
    # 1. Start plain JSON stdout telemetry in the background
    telemetry_script = r"""
import time
import subprocess
import json

while True:
    try:
        cpu = subprocess.getoutput("cat /proc/cpuinfo | grep 'model name' | head -n 1").split(":")[-1].strip()
        cores = subprocess.getoutput("nproc")
        ram_output = subprocess.getoutput("free -m | grep Mem").split()
        ram_used, ram_total = ram_output[2], ram_output[1]
        
        vram_used = subprocess.getoutput("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").strip()
        vram_total = subprocess.getoutput("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits").strip()
        gpu_util = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").strip()
        
        metrics = {
            "type": "telemetry",
            "cpu": f"{cpu} ({cores} Cores)",
            "ram": f"{ram_used}MB / {ram_total}MB",
            "vram": f"{vram_used}MB / {vram_total}MB",
            "gpu_util": f"{gpu_util}%",
            "timestamp": time.time()
        }
        print(json.dumps(metrics), flush=True)
    except Exception as e:
        print(json.dumps({"type": "telemetry_error", "error": str(e)}), flush=True)
    time.sleep(10)
"""
    with open("/tmp/telemetry.py", "w") as f:
        f.write(telemetry_script)
    subprocess.Popen(["python3", "/tmp/telemetry.py"])
    
    # 2. Launch llama-server asynchronously
    model_path = os.path.join(VOLUME_DIR, FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run download_model first.")
    
    # -ngl 999 offloads all layers to the GPU
    # -c 32768 sets the context window to 32K initially (to test the 50GB VRAM headroom hypothesis)
    cmd = [
        "llama-server",
        "-m", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "-ngl", "999",
        "-c", "48000", 
        "-fa", "on",
        "--parallel", "1", # Start with 1 sequence for the first 43GB Q8 baseline
        "--alias", "deckard-40b-q8"
    ]
    
    print(f"Starting llama-server: {' '.join(cmd)}")
    server_process = subprocess.Popen(cmd, stdout=None, stderr=None)
    
    # 3. Robust /health polling loop (Replacing the fixed 15s sleep)
    print("Waiting for llama-server to become healthy on port 8000...")
    timeout = 600 # Wait up to 10 minutes for model to load into VRAM
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if response.status_code == 200:
                print("llama-server is healthy and ready to serve traffic!")
                break
        except requests.exceptions.ConnectionError:
            pass
        
        # Check if the process crashed prematurely
        if server_process.poll() is not None:
            raise RuntimeError(f"llama-server crashed during startup with exit code {server_process.returncode}")
            
        time.sleep(2)
    else:
        raise TimeoutError("llama-server failed to become healthy within the timeout period.")

@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    volumes={VOLUME_DIR: volume},
    timeout=1200,
)
def run_bench(flash_attn: bool = False):
    """Runs the native llama-bench on the GPU to measure raw hardware limits."""
    import subprocess
    model_path = os.path.join(VOLUME_DIR, FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run download_model first.")
        
    cmd = [
        "llama-bench",
        "-m", model_path,
        "-n", "128",
        "-p", "512,2048,4096,8192,16384",
    ]
    if flash_attn:
        cmd.append("-fa")
        cmd.append("1")
    else:
        cmd.append("-fa")
        cmd.append("0")
        
    print(f"Running native llama-bench (Flash Attention: {flash_attn}): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main():
    print("Step 1: Downloading GGUF weights to Modal Volume...")
    download_model.remote()
    print("\n✅ Download complete! The weights are now cached in the cloud.")
    print("Step 2: To deploy the endpoint, please run:")
    print("        modal deploy models/deploy_deckard.py")

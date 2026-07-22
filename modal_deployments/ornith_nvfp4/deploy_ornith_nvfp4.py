# deploy_ornith_nvfp4.py
import os
import sys
import modal
from modal import App, Image, Volume, Secret

# ---------------------------------------------------------------------
# Named Modal resources
# ---------------------------------------------------------------------
APP_NAME = os.environ.get("ORNITH_DEPLOY_APP", "ornith-nvfp4-node")
VOLUME_NAME = os.environ.get("ORNITH_VOLUME_NAME", "ornith-nvfp4-weights")
SECRET_NAME = os.environ.get("ORNITH_MODAL_SECRET", "custom-secret")

MODEL_DIR_NAME = os.environ.get("ORNITH_MODEL_DIR_NAME", "ornith-1.0-35b-nvfp4")
SERVED_MODEL_NAME = os.environ.get("ORNITH_SERVED_NAME", "ornith-1.0-35b-nvfp4")
VOLUME_DIR = "/root/weights"

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [Secret.from_name(SECRET_NAME)]

# ---------------------------------------------------------------------
# Image Definition with CUDA 13.1 & PyTorch Spoofing Patch
# ---------------------------------------------------------------------
base_packages = [
    "vllm==0.22.0",
    "nvidia-modelopt==0.45.0",
    "transformers>=5.9.0",
    "compressed-tensors",
    "datasets>=3.6.0",
    "requests",
    "torch",
    "accelerate",
    "huggingface_hub",
    "hf_transfer",
    "safetensors",
]

image = (
    Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "curl",
        "wget",
        "ca-certificates",
    )
    .run_commands(
        "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 || true"
    )
    .pip_install(*base_packages)
    .env(
        {
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_FLASHINFER_SAMPLER": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
            "XDG_CACHE_HOME": "/root/weights/.cache",
            "FLASHINFER_CUDA_ARCH_LIST": "12.0",
            "TORCH_CUDA_ARCH_LIST": "12.0",
        }
    )
    .pip_install("flashinfer-python")
    .run_commands(
        "sed -i \"s/cuda = .*/cuda = '13.1'/g\" /usr/local/lib/python3.10/site-packages/torch/version.py"
    )
)

app = App(APP_NAME)

@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    memory=65536,
    volumes={VOLUME_DIR: volume},
    timeout=86400,
    secrets=secrets,
    max_containers=1,
)
@modal.web_server(port=8000, startup_timeout=1200)
def serve():
    import json
    import time
    import subprocess
    import threading
    import signal
    import shlex
    import requests

    model_path = os.path.join(VOLUME_DIR, MODEL_DIR_NAME)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}. Run download first.")

    os.makedirs("/root/weights/.cache", exist_ok=True)

    base_env = os.environ.copy()
    base_env.update(
        {
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_FLASHINFER_SAMPLER": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": "/root/weights/.cache",
        }
    )

    print("=== DEPLOYMENT PREFLIGHT (ORNITH 35B NVFP4) ===")
    print(f"Model path: {model_path}")
    print(f"App name: {APP_NAME}")

    try:
        print(subprocess.getoutput("nvidia-smi"))
    except Exception:
        pass

    stop_event = threading.Event()

    def terminate_process(proc):
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(20)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def wait_until_healthy(proc, timeout: int) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if stop_event.is_set():
                return False
            if proc.poll() is not None:
                print(f"vLLM exited during startup with code {proc.returncode}")
                return False
            try:
                r = requests.get("http://127.0.0.1:8000/health", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "::",
        "--port",
        "8000",
        "--quantization",
        "compressed-tensors",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--trust-remote-code",
    ]

    print("Starting vLLM Ornith 35B NVFP4 server...")
    print("CMD: " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        env=base_env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

    if wait_until_healthy(proc, 1200):
        print("Ornith 35B vLLM server is healthy and ready to serve traffic!")
        try:
            smoke = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
            smoke.raise_for_status()
            print("Smoke test passed! Models:", smoke.json())
        except Exception as e:
            print(f"WARNING: smoke test failed: {e}")

        print("Startup complete. Returning control to Modal web_server proxy.")
        return
    else:
        print("vLLM failed to start.")
        terminate_process(proc)
        raise RuntimeError("vLLM server startup failed.")

@app.local_entrypoint()
def main():
    print("Deploying Ornith 35B NVFP4 endpoint to Modal...")
    print(f"Run: modal deploy modal_deployments/ornith_nvfp4/deploy_ornith_nvfp4.py --name {APP_NAME}")

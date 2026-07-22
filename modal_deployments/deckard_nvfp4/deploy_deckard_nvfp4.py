# deploy_deckard_nvfp4.py
import os
import sys
import modal
from modal import App, Image, Volume, Secret

# ---------------------------------------------------------------------
# Named Modal resources
# ---------------------------------------------------------------------
APP_NAME = os.environ.get("DECKARD_DEPLOY_APP", "deckard-nvfp4-node")
VOLUME_NAME = os.environ.get("DECKARD_VOLUME_NAME", "nvfp4-weights")
SECRET_NAME = os.environ.get("DECKARD_MODAL_SECRET", "custom-secret")

MODEL_NAME = os.environ.get("DECKARD_QUANTIZED_NAME", "deckard-40b-nvfp4")
VOLUME_DIR = "/root/weights"

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [Secret.from_name(SECRET_NAME)]

# ---------------------------------------------------------------------
# Image
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
)

# Optional:
# If you need a Blackwell-specific FlashInfer wheel index, set:
#   FLASHINFER_EXTRA_INDEX_URL="https://..."
# before running `modal deploy`.
flashinfer_extra_index = os.environ.get("FLASHINFER_EXTRA_INDEX_URL", "").strip()

if flashinfer_extra_index:
    image = image.pip_install(
        "flashinfer-python",
        extra_index_url=flashinfer_extra_index,
    )
else:
    image = image.pip_install("flashinfer-python")

image = image.run_commands(
    "sed -i \"s/cuda = .*/cuda = '13.1'/g\" /usr/local/lib/python3.10/site-packages/torch/version.py"
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

    # -----------------------------------------------------------------
    # Basic paths and preflight
    # -----------------------------------------------------------------
    model_path = os.path.join(VOLUME_DIR, MODEL_NAME)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}.")

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}.")

    with open(config_path, "r") as f:
        config = json.load(f)

    qcfg = config.get("quantization_config")

    if qcfg is None:
        if os.environ.get("REQUIRE_TRUE_NVFP4", "1") == "1":
            raise RuntimeError(
                "config.json does not contain quantization_config. "
                "Refusing to deploy because true NVFP4 cannot be verified. "
                "Set REQUIRE_TRUE_NVFP4=0 to override."
            )
    else:
        qtext = json.dumps(qcfg)

        if "NVFP4A16" in qtext:
            raise RuntimeError(
                "Deploy blocked: model is quantized as NVFP4A16 (W4A16), "
                "not true NVFP4 W4A4. Re-quantize first."
            )

        if "NVFP4" not in qtext:
            print(
                "WARNING: quantization_config does not explicitly mention NVFP4. "
                "Proceeding, but performance may be incorrect."
            )

    os.makedirs("/root/weights/.cache", exist_ok=True)

    # -----------------------------------------------------------------
    # Runtime env
    # -----------------------------------------------------------------
    base_env = os.environ.copy()
    base_env.update(
        {
            "VLLM_ATTENTION_BACKEND": base_env.get("VLLM_ATTENTION_BACKEND", "FLASHINFER"),
            "VLLM_USE_FLASHINFER_SAMPLER": base_env.get("VLLM_USE_FLASHINFER_SAMPLER", "1"),
            "VLLM_FLASHINFER_SAMPLER": base_env.get("VLLM_FLASHINFER_SAMPLER", "1"),
            "VLLM_WORKER_MULTIPROC_METHOD": base_env.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
            "HF_HUB_OFFLINE": base_env.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": base_env.get("TRANSFORMERS_OFFLINE", "1"),
            "XDG_CACHE_HOME": base_env.get("XDG_CACHE_HOME", "/root/weights/.cache"),
        }
    )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    print("=== DEPLOYMENT PREFLIGHT ===")
    print(f"Model path: {model_path}")
    print(f"App name: {APP_NAME}")
    print(f"Model name: {MODEL_NAME}")

    try:
        print(subprocess.getoutput("nvidia-smi"))
    except Exception:
        pass

    try:
        import vllm

        print(f"vLLM version: {getattr(vllm, '__version__', 'unknown')}")
    except Exception as e:
        print(f"WARNING: could not import vLLM for version check: {e}")

    try:
        import flashinfer

        print(f"FlashInfer version: {getattr(flashinfer, '__version__', 'unknown')}")
    except Exception as e:
        print(f"WARNING: FlashInfer import check failed: {e}")

    print("Relevant environment:")
    for k in sorted(base_env.keys()):
        if k.startswith(("VLLM_", "FLASHINFER", "CUDA", "HF_", "TRANSFORMERS_")):
            print(f"{k}={base_env[k]}")

    # -----------------------------------------------------------------
    # Graceful shutdown + telemetry
    # -----------------------------------------------------------------
    stop_event = threading.Event()
    state = {"proc": None}

    def handle_signal(signum, frame):
        print(f"Received signal {signum}. Initiating graceful shutdown.")
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
    except Exception as e:
        print(f"WARNING: could not install signal handlers: {e}")

    def telemetry():
        while not stop_event.is_set():
            try:
                ram = subprocess.getoutput("free -m | grep Mem").split()
                ram_str = f"{ram[2]}MB / {ram[1]}MB" if len(ram) > 2 else "unknown"

                vram = (
                    subprocess.getoutput(
                        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
                    )
                    .strip()
                    .replace("\n", "MB, ")
                )

                print(
                    json.dumps(
                        {
                            "type": "telemetry",
                            "ram": ram_str,
                            "vram": f"{vram}MB",
                        }
                    ),
                    flush=True,
                )
            except Exception:
                pass

            stop_event.wait(30)

    threading.Thread(target=telemetry, daemon=True).start()

    # -----------------------------------------------------------------
    # Process helpers
    # -----------------------------------------------------------------
    def terminate_process(proc):
        if proc is None:
            return

        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

            try:
                proc.wait(20)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

                try:
                    proc.wait(10)
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

    # -----------------------------------------------------------------
    # vLLM command
    # -----------------------------------------------------------------
    max_model_len = os.environ.get("MAX_MODEL_LEN", "32768")
    max_num_seqs = os.environ.get("MAX_NUM_SEQS", "1")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")

    base_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        MODEL_NAME,
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
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--trust-remote-code",
    ]

    if os.environ.get("ENABLE_AUTO_TOOL_CHOICE", "1") == "1":
        base_cmd += [
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            os.environ.get("TOOL_CALL_PARSER", "hermes"),
        ]

    if os.environ.get("ENABLE_PREFIX_CACHING", "0") == "1":
        base_cmd.append("--enable-prefix-caching")

    extra_args = shlex.split(os.environ.get("EXTRA_VLLM_ARGS", ""))

    # -----------------------------------------------------------------
    # Backend profiles
    # -----------------------------------------------------------------
    forced_backend = os.environ.get("FORCE_ATTENTION_BACKEND", "").strip()

    if forced_backend:
        env = base_env.copy()
        env["VLLM_ATTENTION_BACKEND"] = forced_backend

        if "FLASHINFER" in forced_backend.upper():
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
            env["VLLM_FLASHINFER_SAMPLER"] = "1"
        else:
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            env["VLLM_FLASHINFER_SAMPLER"] = "0"

        profiles = [
            {
                "name": f"forced-{forced_backend}",
                "args": [],
                "env": env,
            }
        ]
    else:
        flashinfer_env = base_env.copy()
        flashinfer_env.update(
            {
                "VLLM_ATTENTION_BACKEND": "FLASHINFER",
                "VLLM_USE_FLASHINFER_SAMPLER": "1",
                "VLLM_FLASHINFER_SAMPLER": "1",
            }
        )

        flash_attn_env = base_env.copy()
        flash_attn_env.update(
            {
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_FLASHINFER_SAMPLER": "0",
            }
        )

        profiles = [
            {
                "name": "flashinfer-cudagraph",
                "args": [],
                "env": flashinfer_env,
            },
            {
                "name": "flashinfer-eager",
                "args": ["--enforce-eager"],
                "env": flashinfer_env,
            },
            {
                "name": "flash-attn-eager",
                "args": ["--enforce-eager"],
                "env": flash_attn_env,
            },
        ]

    startup_timeout = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "1100"))
    max_runtime_restarts = int(os.environ.get("MAX_RUNTIME_RESTARTS", "2"))

    # -----------------------------------------------------------------
    # Supervisor loop
    # -----------------------------------------------------------------
    for profile in profiles:
        restarts = 0

        while restarts <= max_runtime_restarts and not stop_event.is_set():
            cmd = base_cmd + profile["args"] + extra_args
            env = profile["env"]

            print(f"Starting vLLM profile: {profile['name']}")
            print("CMD: " + " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,
            )

            state["proc"] = proc

            if wait_until_healthy(proc, startup_timeout):
                print("vLLM is healthy and ready to serve traffic!")

                try:
                    smoke = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
                    smoke.raise_for_status()
                    print("Smoke test passed! Models:", smoke.json())
                except Exception as e:
                    print(f"WARNING: smoke test failed: {e}")

                print("Startup complete. Returning control to Modal web_server proxy.")
                return

                restarts += 1
                print(
                    f"vLLM profile '{profile['name']}' exited with code "
                    f"{proc.returncode}. Restart {restarts}/{max_runtime_restarts}."
                )

                time.sleep(10)
                continue

            else:
                print(f"Profile '{profile['name']}' failed startup.")
                terminate_process(proc)
                state["proc"] = None
                break

        if stop_event.is_set():
            return

    raise RuntimeError(
        "vLLM failed to start with all configured profiles. "
        "Check FlashInfer compatibility, model artifact, and GPU logs."
    )


@app.local_entrypoint()
def main():
    print("To deploy the named NVFP4 Deckard endpoint, run:")
    print(f"    modal deploy deploy_deckard_nvfp4.py --name {APP_NAME}")
    print()
    print("For local dev / serve mode, run:")
    print("    modal serve deploy_deckard_nvfp4.py")
    print()
    print("Function lookup example:")
    print(f'    modal.Function.lookup("{APP_NAME}", "serve")')
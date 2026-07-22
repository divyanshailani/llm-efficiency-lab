# download_ornith_nvfp4.py
import os
import modal
from modal import App, Image, Volume, Secret

APP_NAME = os.environ.get("ORNITH_DOWNLOAD_APP", "ornith-downloader")
VOLUME_NAME = os.environ.get("ORNITH_VOLUME_NAME", "ornith-nvfp4-weights")
SECRET_NAME = os.environ.get("ORNITH_MODAL_SECRET", "custom-secret")

REPO_ID = os.environ.get("ORNITH_REPO_ID", "sakamakismile/Ornith-1.0-35B-NVFP4")
MODEL_DIR_NAME = os.environ.get("ORNITH_MODEL_DIR_NAME", "ornith-1.0-35b-nvfp4")

VOLUME_DIR = "/root/weights"

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [Secret.from_name(SECRET_NAME)]

image = (
    Image.from_registry(
        "nvidia/cuda:13.1.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "curl", "wget", "ca-certificates")
    .pip_install(
        "huggingface_hub",
        "hf_transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
)

app = App(APP_NAME)

@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=7200,
    secrets=secrets,
    ephemeral_disk=524288,
    retries=0,
)
def download_model():
    """
    Downloads sakamakismile/Ornith-1.0-35B-NVFP4 into Modal Volume.
    """
    from huggingface_hub import snapshot_download

    dest = os.path.join(VOLUME_DIR, MODEL_DIR_NAME)
    os.makedirs(dest, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    print(f"Downloading {REPO_ID} to {dest}...")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=dest,
        cache_dir="/tmp/huggingface",
        token=token,
        max_workers=8,
        ignore_patterns=[
            "*.bin",
            "*.pth",
            "*.gguf",
            "original/*",
        ],
    )

    print("Download complete.")
    volume.commit()
    print("Volume committed to Modal successfully!")

@app.local_entrypoint()
def main():
    print(f"Starting download of {REPO_ID} to Modal Volume '{VOLUME_NAME}'...")
    download_model.remote()
    print("All done.")

# quantize_deckard_nvfp4.py
import os
import json
import shutil
import glob
import hashlib
from typing import Set, Tuple

import modal
from modal import App, Image, Volume, Secret

# ---------------------------------------------------------------------
# Named Modal resources
# ---------------------------------------------------------------------
APP_NAME = os.environ.get("DECKARD_QUANTIZER_APP", "deckard-quantizer")
VOLUME_NAME = os.environ.get("DECKARD_VOLUME_NAME", "nvfp4-weights")
SECRET_NAME = os.environ.get("DECKARD_MODAL_SECRET", "custom-secret")

REPO_ID = os.environ.get(
    "DECKARD_REPO_ID",
    "DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking",
)
MODEL_NAME = os.environ.get("DECKARD_BF16_NAME", "Qwen3.6-40B-Deckard-BF16")
QUANTIZED_NAME = os.environ.get("DECKARD_QUANTIZED_NAME", "deckard-40b-nvfp4")

VOLUME_DIR = "/root/weights"

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [Secret.from_name(SECRET_NAME)]

# ---------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------
image = (
    Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
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
    .pip_install(
        "nvidia-modelopt==0.45.0",
        "transformers>=5.9.0",
        "compressed-tensors",
        "llmcompressor",
        "datasets>=3.6.0",
        "torch",
        "torchvision",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
        "safetensors",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = App(APP_NAME)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _dir_gib(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024**3)


def _safetensors_keys(directory: str, open_shards: bool = False) -> Tuple[Set[str], Set[str]]:
    """
    Returns:
      keys, shards
    """
    from safetensors import safe_open

    keys: Set[str] = set()
    shards: Set[str] = set()

    index_path = os.path.join(directory, "model.safetensors.index.json")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)

        weight_map = idx.get("weight_map", {})
        shards = set(weight_map.values())
        keys = set(weight_map.keys())

        for shard in shards:
            shard_path = os.path.join(directory, shard)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard {shard} declared in index but missing from disk!")

            if open_shards:
                with safe_open(shard_path, framework="pt", device="cpu") as f_st:
                    keys.update(f_st.keys())
    else:
        for st in glob.glob(os.path.join(directory, "*.safetensors")):
            shards.add(os.path.basename(st))
            with safe_open(st, framework="pt", device="cpu") as f_st:
                keys.update(f_st.keys())

    # Sidecars not listed in the main index
    for st in glob.glob(os.path.join(directory, "*.safetensors")):
        bn = os.path.basename(st)
        if bn not in shards:
            with safe_open(st, framework="pt", device="cpu") as f_st:
                keys.update(f_st.keys())
            shards.add(bn)

    return keys, shards


def _assert_true_nvfp4(config: dict) -> None:
    qcfg = config.get("quantization_config")
    if not qcfg:
        raise ValueError("quantization_config not found in output config.json!")

    qmethod = qcfg.get("quant_method")
    if qmethod not in {"compressed-tensors", "compressed_tensors"}:
        raise ValueError(
            f"Unexpected quant_method: {qmethod}. Expected compressed-tensors."
        )

    qtext = json.dumps(qcfg)

    # Strictly reject W4A16 masquerading as NVFP4
    if "NVFP4A16" in qtext:
        raise ValueError(
            "Output config still contains NVFP4A16. This is W4A16, not true NVFP4 W4A4."
        )

    if "NVFP4" not in qtext:
        raise ValueError(
            "Output config does not contain NVFP4. True W4A4 quantization likely failed."
        )


def _load_calibration_dataset(tokenizer, samples: int):
    """
    Loads a text calibration dataset and returns a HF Dataset with a 'text' column.
    """
    from datasets import Dataset, load_dataset

    repo = os.environ.get("CALIBRATION_DATASET", "HuggingFaceH4/ultrachat_200k")
    split = os.environ.get(
        "CALIBRATION_SPLIT",
        f"train_sft[:{max(samples * 8, 64)}]",
    )
    token = os.environ.get("HF_TOKEN")

    load_kwargs = {
        "path": repo,
        "split": split,
        "token": token,
        "trust_remote_code": True,
    }

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"Primary calibration split failed ({e}). Falling back to streaming full split.")
        load_kwargs["split"] = "train_sft"
        load_kwargs["streaming"] = True
        ds = load_dataset(**load_kwargs)

    texts = []

    for example in ds:
        messages = example.get("messages")
        text = ""

        if messages:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback if chat template application fails
                parts = []
                for m in messages:
                    if isinstance(m, dict):
                        parts.append(str(m.get("content", "")))
                text = "\n".join(parts)

        if not text:
            text = (
                example.get("prompt")
                or example.get("text")
                or example.get("instruction")
                or ""
            )

        if isinstance(text, list):
            text = " ".join(str(x) for x in text)

        text = str(text).strip()

        if len(text) >= 32:
            texts.append({"text": text})

        if len(texts) >= samples:
            break

    if len(texts) < samples:
        raise RuntimeError(
            f"Only collected {len(texts)} calibration samples, but {samples} were required."
        )

    return Dataset.from_list(texts)


# ---------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------
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
    Downloads the BF16 Deckard model into the Modal volume.
    """
    from huggingface_hub import snapshot_download

    dest = os.path.join(VOLUME_DIR, MODEL_NAME)
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
    print("Volume committed to Modal.")


@app.function(
    image=image,
    gpu="B200",
    memory=131072,
    ephemeral_disk=524288,
    volumes={VOLUME_DIR: volume},
    timeout=14400,
    secrets=secrets,
    retries=0,
)
def quantize_deckard():
    """
    True NVFP4 W4A4 quantization for Blackwell.
    """
    import torch
    from transformers import AutoTokenizer, AutoProcessor

    try:
        from transformers import Qwen3_5ForConditionalGeneration as ModelCls
        model_loader_name = "Qwen3_5ForConditionalGeneration"
    except Exception:
        from transformers import AutoModelForCausalLM as ModelCls
        model_loader_name = "AutoModelForCausalLM"

    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor import oneshot

    model_path = os.path.join(VOLUME_DIR, MODEL_NAME)
    final_out_path = os.path.join(VOLUME_DIR, QUANTIZED_NAME)
    out_path = final_out_path + "_temp"

    if os.path.exists(final_out_path):
        print(f"Warning: Output path {final_out_path} already exists. Deleting to overwrite.")
        shutil.rmtree(final_out_path)

    if os.path.exists(out_path):
        print(f"Removing stale temp path {out_path}.")
        shutil.rmtree(out_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Source model not found at {model_path}. Run download_model first."
        )

    print(f"Loading model with {model_loader_name} (this may take a while)...")

    try:
        model = ModelCls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"Primary model loader failed ({e}). Falling back to AutoModelForCausalLM.")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model.config.use_cache = False

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        processor = None
        print(f"WARNING: Failed to load AutoProcessor: {e}")

    calibration_samples = int(os.environ.get("CALIBRATION_SAMPLES", "512"))
    max_seq_length = int(os.environ.get("CALIBRATION_MAX_SEQ_LENGTH", "4096"))

    print(
        f"Loading calibration dataset: samples={calibration_samples}, "
        f"max_seq_length={max_seq_length}"
    )
    ds = _load_calibration_dataset(tokenizer, calibration_samples)
    print(f"Calibration dataset ready with {len(ds)} samples.")

    ignore = [
        "lm_head",
        "re:.*embed.*",
        "re:.*norm.*",
        "re:.*linear_attn.*",
        "re:.*visual.*",
        "re:.*router.*",
    ]

    extra_ignore = os.environ.get("QUANT_IGNORE", "")
    if extra_ignore:
        ignore.extend([x.strip() for x in extra_ignore.split(",") if x.strip()])

    print("Defining true NVFP4 recipe...")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=ignore,
    )

    print("Applying oneshot quantization with calibration data...")
    try:
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=max_seq_length,
            num_calibration_samples=calibration_samples,
        )
    except TypeError as e:
        print(f"oneshot TypeError: {e}")
        print("Retrying with num_calibration_steps for compatibility...")
        try:
            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=max_seq_length,
                num_calibration_steps=calibration_samples,
            )
        except TypeError:
            print("Retrying minimal oneshot call...")
            oneshot(
                model=model,
                dataset=ds,
                recipe=recipe,
                max_seq_length=max_seq_length,
            )

    print("Inspecting quantized vs excluded module counts...")
    quantized_count = 0
    standard_count = 0

    for name, module in model.named_modules():
        mod_type = type(module).__name__

        is_linear = "Linear" in mod_type
        is_quantized = (
            "Compressed" in mod_type
            or "Quantized" in mod_type
            or "QuantLinear" in mod_type
            or hasattr(module, "weight_scale")
            or hasattr(module, "weight_packed")
        )

        if is_linear or is_quantized:
            if is_quantized:
                quantized_count += 1
            else:
                standard_count += 1

    print(
        f"Post-quantization module counts: "
        f"Quantized={quantized_count}, Standard Linear={standard_count}"
    )

    if quantized_count == 0:
        raise RuntimeError(
            "Zero modules were quantized! The quantization failed or ignored all layers."
        )

    print(f"Saving quantized model to {out_path}...")
    os.makedirs(out_path, exist_ok=True)

    model.save_pretrained(
        out_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print("Saving tokenizer...")
    tokenizer.save_pretrained(out_path)

    if processor is not None:
        print("Saving processor...")
        processor.save_pretrained(out_path)

    print("Copying remaining metadata files...")
    skip_suffix = (
        ".safetensors",
        ".bin",
        ".pth",
        ".ckpt",
        ".index.json",
    )

    for fn in os.listdir(model_path):
        src = os.path.join(model_path, fn)
        dst = os.path.join(out_path, fn)

        if (
            os.path.isfile(src)
            and not fn.endswith(skip_suffix)
            and not os.path.exists(dst)
        ):
            shutil.copy2(src, dst)

    print("Validating config.json for true NVFP4...")
    with open(os.path.join(out_path, "config.json"), "r") as f:
        config = json.load(f)

    _assert_true_nvfp4(config)
    print("config.json validation passed: true NVFP4 detected.")

    print("Validating output artifact size...")
    size_gib = _dir_gib(out_path)
    min_gib = float(os.environ.get("MIN_OUTPUT_GIB", "20"))
    max_gib = float(os.environ.get("MAX_OUTPUT_GIB", "26"))
    strict_size = os.environ.get("STRICT_OUTPUT_SIZE", "1") == "1"

    print(f"Output artifact size: {size_gib:.2f} GiB")

    if size_gib < min_gib or size_gib > max_gib:
        msg = (
            f"Output size {size_gib:.2f} GiB is outside expected range "
            f"[{min_gib:.2f}, {max_gib:.2f}] GiB."
        )
        if strict_size:
            raise ValueError(msg)
        else:
            print("WARNING: " + msg)

    print("Validating artifact integrity...")

    if not os.path.exists(os.path.join(out_path, "model.safetensors.index.json")):
        raise FileNotFoundError("model.safetensors.index.json is missing.")

    src_keys, _ = _safetensors_keys(model_path, open_shards=False)
    out_keys, out_shards = _safetensors_keys(out_path, open_shards=False)

    # Check for missing base tensor prefixes
    src_bases = {
        k[: -len(".weight")]
        for k in src_keys
        if k.endswith(".weight")
    }

    missing_bases = []
    for base in src_bases:
        if not any(k == base or k.startswith(base + ".") for k in out_keys):
            missing_bases.append(base)

    if missing_bases:
        raise RuntimeError(
            f"Quantization dropped {len(missing_bases)} base modules completely! "
            f"Missing: {missing_bases[:20]}"
        )

    # Strict MTP preservation check
    src_mtp = {k for k in src_keys if "mtp" in k.lower()}
    missing_mtp = src_mtp - out_keys

    if missing_mtp:
        raise RuntimeError(
            f"MTP preservation failed! Missing exact MTP tensors: {missing_mtp}"
        )

    print("Artifact integrity checks passed.")

    # Optional hashes
    if os.environ.get("SKIP_HASHES", "0") != "1":
        print("Computing checksums...")

        def sha256(filepath: str) -> str:
            h = hashlib.sha256()
            with open(filepath, "rb") as file:
                while chunk := file.read(8192 * 1024):
                    h.update(chunk)
            return h.hexdigest()

        hashes = {}
        files_to_hash = [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ] + sorted(out_shards)

        for critical_file in files_to_hash:
            p = os.path.join(out_path, critical_file)
            if os.path.exists(p):
                print(f"Hashing {critical_file}...")
                hashes[critical_file] = sha256(p)

        with open(os.path.join(out_path, "checksums.json"), "w") as f:
            json.dump(hashes, f, indent=2)

        print("Checksums saved.")

    # Environment / versions record
    import transformers
    import llmcompressor

    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llmcompressor": llmcompressor.__version__,
    }

    with open(os.path.join(out_path, "quant_environment.json"), "w") as f:
        json.dump(versions, f, indent=2)

    print("Finalizing artifact...")
    if os.path.exists(final_out_path):
        shutil.rmtree(final_out_path)

    os.replace(out_path, final_out_path)

    volume.commit()
    print("Volume committed to Modal!")


@app.local_entrypoint()
def main():
    if os.environ.get("SKIP_DOWNLOAD", "0") != "1":
        print("Step 1: Downloading BF16 model...")
        download_model.remote()
    else:
        print("Skipping download because SKIP_DOWNLOAD=1")

    print("Step 2: Quantizing Deckard to true NVFP4 W4A4...")
    quantize_deckard.remote()

    print("Done.")
import os
import time

# Route HuggingFace cache to the external SSD to save internal Mac storage
os.environ["HF_HOME"] = "/Volumes/ssd/hf_cache"

from mlx_lm import load, generate

print("Loading MLX OptiQ 4-bit model... (This will download ~8.2GB to your SSD if not cached)")

# We explicitly load the OptiQ model without MTP sidecars
model, tokenizer = load("mlx-community/Qwythos-9B-v2-OptiQ-4bit")

print("\nModel loaded successfully! Warming up Metal backend...")

prompt = "Analyze the philosophical themes in Dostoyevsky's Crime and Punishment."
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Warmup run
_ = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=10, verbose=False)

print("\nRunning benchmark...")

# mlx_lm.generate with verbose=True natively outputs PP and Decode tokens-per-sec!
output = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=256, verbose=True)

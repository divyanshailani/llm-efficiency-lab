#!/bin/bash
# Local testing script for Qwen3.6-40B-Deckard

# Assuming llama.cpp is compiled and available in your PATH
llama-server \
  -m models/Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-QT-HIGH-Q8_0.gguf \
  --port 8000 \
  --ctx-size 16384 \
  --n-gpu-layers 120 \
  --flash-attn

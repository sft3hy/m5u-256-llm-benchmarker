#!/usr/bin/env bash
# Downloads weights for the LLM leg in both GGUF (llama.cpp) and MLX
# (mlx-lm / oMLX) formats. This is a LOT of data -- the full matrix below
# is roughly 700-900GB on disk. Comment out what you don't need, or pass
# a model key as $1 to fetch just one.
#
# Usage:
#   ./download_models.sh                 # everything
#   ./download_models.sh qwen3.8-27b     # just one model, both formats
set -euo pipefail
source .venv/bin/activate 2>/dev/null || true

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS_DIR/gguf" "$MODELS_DIR/mlx"

fetch_gguf () {
  local repo="$1" dest="$2"
  echo ">> GGUF: $repo"
  huggingface-cli download "$repo" --local-dir "$MODELS_DIR/gguf/$dest" --local-dir-use-symlinks False
}

fetch_mlx () {
  local repo="$1" dest="$2"
  echo ">> MLX: $repo"
  huggingface-cli download "$repo" --local-dir "$MODELS_DIR/mlx/$dest" --local-dir-use-symlinks False
}

target="${1:-all}"

# --- Agentic coding ---
if [[ "$target" == "all" || "$target" == "qwen3.8-27b" ]]; then
  fetch_gguf unsloth/Qwen3.8-27B-GGUF qwen3.8-27b
  fetch_mlx  mlx-community/Qwen3.8-27B-4bit qwen3.8-27b-4bit
  fetch_mlx  mlx-community/Qwen3.8-27B-8bit qwen3.8-27b-8bit
fi

if [[ "$target" == "all" || "$target" == "qwen3.8-flash-next" ]]; then
  fetch_gguf unsloth/Qwen3.8-Flash-Next-GGUF qwen3.8-flash-next
  fetch_mlx  mlx-community/Qwen3.8-Flash-Next-4bit qwen3.8-flash-next-4bit
  fetch_mlx  mlx-community/Qwen3.8-Flash-Next-8bit qwen3.8-flash-next-8bit
fi

if [[ "$target" == "all" || "$target" == "glm-5.3" ]]; then
  echo "!! GLM-5.3 1-bit is ~145GB. Confirm free disk space before continuing."
  fetch_gguf unsloth/GLM-5.3-GGUF glm-5.3-1bit   # grab the smallest UD-IQ1_* file from the repo listing
  fetch_mlx  mlx-community/GLM-5.3-1bit glm-5.3-1bit
fi

if [[ "$target" == "all" || "$target" == "glm-5.3-flash" ]]; then
  fetch_gguf unsloth/GLM-5.3-Flash-GGUF glm-5.3-flash
  fetch_mlx  mlx-community/GLM-5.3-Flash-4bit glm-5.3-flash-4bit
fi

if [[ "$target" == "all" || "$target" == "deepseek-v4-flash" ]]; then
  echo "!! DeepSeek-V4-Flash 8-bit is ~290GB -- likely too big for a 256GB machine. Downloading 4-bit only by default."
  fetch_gguf unsloth/DeepSeek-V4-Flash-0731-GGUF deepseek-v4-flash-4bit
  fetch_mlx  mlx-community/DeepSeek-V4-Flash-0731-4bit deepseek-v4-flash-4bit
  # Uncomment only if you're on the 512GB config, or want to test swap/paging behavior:
  # fetch_gguf unsloth/DeepSeek-V4-Flash-0731-GGUF deepseek-v4-flash-8bit
  # fetch_mlx  mlx-community/DeepSeek-V4-Flash-0731-8bit deepseek-v4-flash-8bit
fi

# --- American ---
if [[ "$target" == "all" || "$target" == "muse-glimmer" ]]; then
  fetch_gguf unsloth/Muse-Glimmer-30B-GGUF muse-glimmer
  fetch_mlx  mlx-community/Muse-Glimmer-30B-4bit muse-glimmer-4bit
  fetch_mlx  mlx-community/Muse-Glimmer-30B-8bit muse-glimmer-8bit
fi

if [[ "$target" == "all" || "$target" == "gemma4-31b" ]]; then
  fetch_gguf unsloth/gemma-4-31b-GGUF gemma4-31b
  fetch_mlx  mlx-community/gemma-4-31b-4bit gemma4-31b-4bit
  fetch_mlx  mlx-community/gemma-4-31b-8bit gemma4-31b-8bit
fi

if [[ "$target" == "all" || "$target" == "gpt-oss-120b" ]]; then
  fetch_gguf unsloth/gpt-oss-120b-GGUF gpt-oss-120b
  fetch_mlx  mlx-community/gpt-oss-120b gpt-oss-120b   # native MXFP4, don't re-quantize
fi

# --- Image / video (single copy, format depends on runner) ---
if [[ "$target" == "all" || "$target" == "flux2-dev" ]]; then
  huggingface-cli download black-forest-labs/FLUX.2-dev --local-dir "$MODELS_DIR/flux2-dev" --local-dir-use-symlinks False
fi

if [[ "$target" == "all" || "$target" == "sd-3.5-large" ]]; then
  huggingface-cli download stabilityai/stable-diffusion-3.5-large --local-dir "$MODELS_DIR/sd-3.5-large" --local-dir-use-symlinks False
fi

if [[ "$target" == "all" || "$target" == "wan2.2" ]]; then
  huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir "$MODELS_DIR/wan2.2" --local-dir-use-symlinks False
fi

if [[ "$target" == "all" || "$target" == "hunyuanvideo" ]]; then
  huggingface-cli download tencent/HunyuanVideo --local-dir "$MODELS_DIR/hunyuanvideo" --local-dir-use-symlinks False
fi

echo "== Done. Weights are under $MODELS_DIR =="
df -h "$MODELS_DIR"

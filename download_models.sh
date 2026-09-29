#!/usr/bin/env bash
# Downloads weights for the LLM leg in both GGUF (llama.cpp) and MLX
# (mlx-lm / oMLX) formats. This downloads only the specific quantization
# variants needed for the benchmark matrix.
#
# Usage:
#   ./download_models.sh                 # everything
#   ./download_models.sh qwen3.8-27b     # just one model, both formats
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate" 2>/dev/null || true

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS_DIR/gguf" "$MODELS_DIR/mlx"

check_disk_space () {
  local used_kb
  used_kb=$(du -sk "$MODELS_DIR" 2>/dev/null | awk '{print $1}')
  local used_gb=$(( used_kb / 1024 / 1024 ))
  if (( used_gb >= 1450 )); then
    echo "!! SAFETY HALT: Models directory has reached ${used_gb} GB (limit 1.5 TB)." >&2
    echo "Aborting remaining downloads to preserve SSD space." >&2
    exit 0
  fi

  local avail_kb
  avail_kb=$(df -k "$MODELS_DIR" | awk 'NR==2 {print $4}')
  local avail_gb=$(( avail_kb / 1024 / 1024 ))
  if (( avail_gb <= 150 )); then
    echo "!! SAFETY HALT: Available disk space is down to ${avail_gb} GB (< 150 GB buffer)." >&2
    echo "Aborting remaining downloads to protect disk." >&2
    exit 0
  fi
}

download_hf () {
  check_disk_space
  local repo="$1" dest="$2"
  shift 2 || true

  local exit_code=0
  if command -v hf &>/dev/null; then
    if [[ "$DRY_RUN" == true ]]; then
      hf download "$repo" --dry-run --local-dir "$dest" "$@" || exit_code=$?
    else
      hf download "$repo" --local-dir "$dest" "$@" || exit_code=$?
    fi
  elif command -v huggingface-cli &>/dev/null; then
    if [[ "$DRY_RUN" == true ]]; then
      huggingface-cli download "$repo" --dry-run --local-dir "$dest" "$@" || exit_code=$?
    else
      huggingface-cli download "$repo" --local-dir "$dest" "$@" || exit_code=$?
    fi
  else
    echo "ERROR: Neither 'hf' nor 'huggingface-cli' found in PATH." >&2
    echo "Please activate your virtual environment: source .venv/bin/activate" >&2
    exit 1
  fi

  if [[ $exit_code -ne 0 ]]; then
    echo ">> [SKIP] Failed or gated repository: $repo (HTTP error or requires auth). Continuing..." >&2
  fi
}

fetch_gguf () {
  local repo="$1" dest="$2"
  shift 2 || true
  echo ">> GGUF: $repo -> $dest"
  download_hf "$repo" "$MODELS_DIR/gguf/$dest" "$@"
}

fetch_mlx () {
  local repo="$1" dest="$2"
  shift 2 || true
  echo ">> MLX: $repo -> $dest"
  download_hf "$repo" "$MODELS_DIR/mlx/$dest" "$@"
}

DRY_RUN=false
target="all"

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=true
  else
    target="$arg"
  fi
done

# --- Agentic coding ---
if [[ "$target" == "all" || "$target" == "qwen3.8-27b" ]]; then
  fetch_gguf unsloth/Qwen3.8-27B-GGUF qwen3.8-27b-4bit --include "*Q4_K_M*.gguf"
  fetch_gguf unsloth/Qwen3.8-27B-GGUF qwen3.8-27b-8bit --include "*Q8_0*.gguf"
  fetch_mlx  mlx-community/Qwen3.8-27B-4bit qwen3.8-27b-4bit
  fetch_mlx  mlx-community/Qwen3.8-27B-8bit qwen3.8-27b-8bit
fi

if [[ "$target" == "all" || "$target" == "qwen3.8-flash-next" ]]; then
  fetch_gguf unsloth/Qwen3.8-Flash-Next-GGUF qwen3.8-flash-next-4bit --include "UD-Q4_K_XL/*"
  fetch_mlx  mlx-community/Qwen3.8-Flash-Next-4bit qwen3.8-flash-next-4bit
  # 8-bit is ~270GB combined -- uncomment only if you have extra storage headroom:
  # fetch_gguf unsloth/Qwen3.8-Flash-Next-GGUF qwen3.8-flash-next-8bit --include "Q8_0/*"
  # fetch_mlx  mlx-community/Qwen3.8-Flash-Next-8bit qwen3.8-flash-next-8bit
fi

if [[ "$target" == "all" || "$target" == "glm-5.3" ]]; then
  echo "!! GLM-5.3 1-bit is ~145GB. Confirm free disk space before continuing."
  fetch_gguf unsloth/GLM-5.3-GGUF glm-5.3-1bit --include "UD-IQ1_S/*"
  # Note: MLX has no 1-bit quantization kernel (supports 2,3,4,8 bit only).
  # 1-bit is exclusively a llama.cpp/GGML feature. For MLX, see glm-5.3-flash below.
fi

if [[ "$target" == "all" || "$target" == "glm-5.3-flash" ]]; then
  # MLX format is faster on Apple Silicon and saves 200GB vs pulling both:
  fetch_mlx  mlx-community/GLM-5.3-Flash-4bit glm-5.3-flash-4bit
  # fetch_gguf unsloth/GLM-5.3-Flash-GGUF glm-5.3-flash-4bit --include "UD-Q4_K_XL/*"
fi

if [[ "$target" == "all" || "$target" == "deepseek-v4-flash" ]]; then
  echo "!! DeepSeek-V4-Flash 8-bit is ~290GB -- likely too big for a 256GB machine. Downloading 4-bit only by default."
  fetch_gguf unsloth/DeepSeek-V4-Flash-0731-GGUF deepseek-v4-flash-4bit --include "UD-Q4_K_XL/*"
  fetch_mlx  mlx-community/DeepSeek-V4-Flash-4bit deepseek-v4-flash-4bit
  # Uncomment only if you're on the 512GB config, or want to test swap/paging behavior:
  # fetch_gguf unsloth/DeepSeek-V4-Flash-0731-GGUF deepseek-v4-flash-8bit --include "UD-Q8_K_XL/*"
  # fetch_mlx  mlx-community/DeepSeek-V4-Flash-8bit deepseek-v4-flash-8bit
fi

# --- American ---
if [[ "$target" == "all" || "$target" == "muse-glimmer" ]]; then
  fetch_gguf unsloth/Muse-Glimmer-30B-GGUF muse-glimmer-4bit --include "*UD-Q4_K_XL.gguf"
  fetch_gguf unsloth/Muse-Glimmer-30B-GGUF muse-glimmer-8bit --include "*Q8_0.gguf"
  fetch_mlx  mlx-community/Muse-Glimmer-30B-4bit muse-glimmer-4bit
  fetch_mlx  mlx-community/Muse-Glimmer-30B-8bit muse-glimmer-8bit
fi

if [[ "$target" == "all" || "$target" == "gemma4-31b" ]]; then
  fetch_gguf unsloth/gemma-4-31B-it-GGUF gemma4-31b-4bit --include "*Q4_K_M*.gguf"
  fetch_gguf unsloth/gemma-4-31B-it-GGUF gemma4-31b-8bit --include "*Q8_0*.gguf"
  fetch_mlx  mlx-community/gemma-4-31b-4bit gemma4-31b-4bit
  fetch_mlx  mlx-community/gemma-4-31b-8bit gemma4-31b-8bit
fi

if [[ "$target" == "all" || "$target" == "gpt-oss-120b" ]]; then
  fetch_gguf unsloth/gpt-oss-120b-GGUF gpt-oss-120b-native-mxfp4 --include "gpt-oss-120b-F16.gguf"
  fetch_mlx  lmstudio-community/gpt-oss-120b-MLX-8bit gpt-oss-120b-native-mxfp4
fi

# --- Image / video (single copy, format depends on runner) ---
if [[ "$target" == "all" || "$target" == "flux2-dev" ]]; then
  echo ">> Image: FLUX.2-dev"
  download_hf black-forest-labs/FLUX.2-dev "$MODELS_DIR/flux2-dev"
fi

if [[ "$target" == "all" || "$target" == "sd-3.5-large" ]]; then
  echo ">> Image: SD-3.5-large"
  download_hf stabilityai/stable-diffusion-3.5-large "$MODELS_DIR/sd-3.5-large"
fi

if [[ "$target" == "all" || "$target" == "wan2.2" ]]; then
  echo ">> Video: Wan2.1 (T2V-1.3B)"
  download_hf Wan-AI/Wan2.1-T2V-1.3B "$MODELS_DIR/wan2.2"
fi

if [[ "$target" == "all" || "$target" == "hunyuanvideo" ]]; then
  echo ">> Video: HunyuanVideo"
  download_hf tencent/HunyuanVideo "$MODELS_DIR/hunyuanvideo"
fi

echo "== Done. Weights are under $MODELS_DIR =="
df -h "$MODELS_DIR"

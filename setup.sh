#!/usr/bin/env bash
# Setup for the M5 Ultra 256GB benchmark kit.
# Run this once on the Mac Studio itself (not in any sandbox).
set -euo pipefail

echo "== Checking for Homebrew =="
if ! command -v brew &>/dev/null; then
  echo "Homebrew not found. Install it from https://brew.sh first."
  exit 1
fi

echo "== Installing llama.cpp (Metal build) =="
brew install llama.cpp

echo "== Installing oMLX =="
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx

echo "== Setting up Python env for mlx-lm + hf downloads + timing/plotting =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install mlx mlx-lm "huggingface_hub[cli]" psutil pandas matplotlib pyyaml

echo "== Diffusion / video deps (PyTorch with MPS backend + diffusers) =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # MPS build ships in the default wheel on macOS
pip install diffusers transformers accelerate sentencepiece imageio imageio-ffmpeg

echo "== mflux for MLX-native Flux inference (optional, faster than diffusers on Apple Silicon) =="
pip install mflux

echo "== Geekbench CLI note =="
cat <<'EOF'

Geekbench 6 and Geekbench AI ship a CLI binary inside the .app bundle:
  /Applications/Geekbench 6.app/Contents/Resources/geekbench6
  /Applications/Geekbench AI.app/Contents/Resources/geekbench_ai
Both support --export-json for scriptable results. Download the apps from
https://www.geekbench.com first (this script does not install them).

AmorphousDiskMark and 3DMark Wild Life Extreme are GUI-only -- there is no
supported CLI for either. Run those manually; see README.md for the
recording checklist.
EOF

echo "== Done. Activate the venv in new shells with: source .venv/bin/activate =="

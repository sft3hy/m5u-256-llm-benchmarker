"""
Model registry for the M5 Ultra 256GB benchmark kit.

Every entry is a real, released model as of Sept 2026 (verified by web search
while this kit was built). Repo names are the *pattern* the community uses
(Unsloth for GGUF, mlx-community for MLX) -- double-check the exact repo id
on Hugging Face before downloading, since exact quant repo names shift as
new quants get uploaded.

Sizes are approximate on-disk / resident-memory figures for the *weights*
at that quantization. Actual peak RAM during inference is higher (KV cache,
activations, context) -- see README budget notes.
"""

LLMS = {
    # ---- Agentic coding ----
    "qwen3.8-27b": {
        "family": "Qwen3.8-27B (dense, 27B)",
        "vendor": "Alibaba",
        "released": "2026-08-14",
        "gguf_repo": "unsloth/Qwen3.8-27B-GGUF",
        "mlx_repo": "mlx-community/Qwen3.8-27B-{quant}",
        "quants": ["4bit", "8bit"],
        "approx_gb": {"4bit": 15, "8bit": 29},
        "notes": "Dense model, fits comfortably at both quants.",
    },
    "qwen3.8-flash-next": {
        "family": "Qwen3.8-Flash-Next (MoE, 125B total / 6B active, +51B n-gram table)",
        "vendor": "Alibaba",
        "released": "2026-08-26",
        "gguf_repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "mlx_repo": "mlx-community/Qwen3.8-Flash-Next-{quant}",
        "quants": ["4bit", "8bit"],
        "approx_gb": {"4bit": 65, "8bit": 130},
        "notes": "Qwen4 architecture preview. MoE -- good tok/s despite size.",
    },
    "glm-5.3": {
        "family": "GLM-5.3 (MoE, 744B total / 40B active)",
        "vendor": "Zhipu / Z.ai",
        "released": "2026-08-29",
        "gguf_repo": "unsloth/GLM-5.3-GGUF",
        "mlx_repo": None,  # MLX has no 1-bit quantization kernel; see glm-5.3-flash
        "quants": ["1bit"],
        "approx_gb": {"1bit": 145},
        "notes": (
            "744B total params -- 1-bit (Unsloth UD-IQ1 style) is the only "
            "quant that fits a 256GB pool with headroom for KV cache/OS. "
            "Benchmarked resident in unified memory via llama.cpp."
        ),
    },
    "glm-5.3-flash": {
        "family": "GLM-5.3-Flash (multimodal, MIT license)",
        "vendor": "Zhipu / Z.ai",
        "released": "2026-08-26",
        "gguf_repo": "unsloth/GLM-5.3-Flash-GGUF",
        "mlx_repo": "mlx-community/GLM-5.3-Flash-{quant}",
        "quants": ["4bit"],
        "approx_gb": {"4bit": 205},
        "notes": (
            "320B-class architecture, native 4-bit MLX release."
        ),
    },
    "deepseek-v4-flash": {
        "family": "DeepSeek-V4-Flash-0731 (MoE, 284B total / 13B active)",
        "vendor": "DeepSeek",
        "released": "2026-07-31",
        "gguf_repo": "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        "mlx_repo": "mlx-community/DeepSeek-V4-Flash-{quant}",
        "quants": ["4bit", "8bit"],
        "approx_gb": {"4bit": 150, "8bit": 290},
        "notes": (
            "WARNING: 8-bit (~290GB) will NOT fit a 256GB unified memory "
            "pool once you subtract OS + KV cache. Use q4/q5/q6 for the "
            "8-bit-class run, or skip that leg on this machine -- it needs "
            "the 512GB M5 Ultra config."
        ),
    },
    # ---- American ----
    "muse-glimmer": {
        "family": "Muse Glimmer (dense, 30B)",
        "vendor": "Meta Superintelligence Labs",
        "released": "2026-08-10",
        "gguf_repo": "unsloth/Muse-Glimmer-30B-GGUF",
        "mlx_repo": "mlx-community/Muse-Glimmer-30B-{quant}",
        "quants": ["4bit", "8bit"],
        "approx_gb": {"4bit": 17, "8bit": 32},
        "notes": "Distilled from Muse Spark. Ships with its own speculative decoder (DFlash).",
    },
    "gemma4-31b": {
        "family": "Gemma 4 (dense, 31B)",
        "vendor": "Google DeepMind",
        "released": "2026-03-31",
        "gguf_repo": "unsloth/gemma-4-31B-it-GGUF",
        "mlx_repo": "mlx-community/gemma-4-31b-{quant}",
        "quants": ["4bit", "8bit"],
        "approx_gb": {"4bit": 17, "8bit": 33},
        "notes": "Apache 2.0. Also available as 26B-A4B MoE if you want a MoE/dense contrast at similar size.",
    },
    "gpt-oss-120b": {
        "family": "gpt-oss-120b (MoE, 117B total / ~5.1B active)",
        "vendor": "OpenAI",
        "released": "2025-08",
        "gguf_repo": "unsloth/gpt-oss-120b-GGUF",
        "mlx_repo": "lmstudio-community/gpt-oss-120b-MLX-8bit",
        "quants": ["native-mxfp4"],
        "approx_gb": {"native-mxfp4": 65},
        "notes": "Ships natively in MXFP4 -- don't re-quantize, use the native release for both engines.",
    },
}

IMAGE_MODELS = {
    "flux2-dev": {
        "family": "FLUX.2 [dev] (32B, hybrid Mistral-3-24B VLM + rectified flow transformer)",
        "vendor": "Black Forest Labs",
        "released": "2025-11-25",
        "hf_repo": "black-forest-labs/FLUX.2-dev",
        "mlx_repo": "mlx-community/FLUX.2-dev-4bit",  # via mflux
        "runner": "mflux (MLX) or diffusers (PyTorch/MPS) or ComfyUI",
        "notes": "License is non-commercial for [dev]. Use Klein (Apache 2.0) instead if you need a commercially-clean open weight.",
    },
    "sd-3.5-large": {
        "family": "Stable Diffusion 3.5 Large (SUBSTITUTED -- see note)",
        "vendor": "Stability AI",
        "released": "2024-10-22",
        "hf_repo": "stabilityai/stable-diffusion-3.5-large",
        "mlx_repo": "mlx-community/stable-diffusion-3.5-large-4bit",
        "runner": "diffusers (PyTorch/MPS) or mflux-style MLX port",
        "notes": (
            "'Stable Diffusion 2.5' does not appear to exist in Stability's "
            "lineup (checked their release history: 1.5 -> 2.0 -> 2.1 -> "
            "SDXL -> SD3 -> SD3.5 -> Stable LM 2.5 [a *language* model, not "
            "image]). Swapped in SD 3.5 Large as the closest same-vendor "
            "comparison point to Flux.2. If you meant something else "
            "(SDXL, SD 2.1, a specific fine-tune), edit this entry."
        ),
    },
}

VIDEO_MODELS = {
    "wan2.2": {
        "family": "Wan 2.2 (MoE video, A14B: 27B total / 14B active)",
        "vendor": "Alibaba Tongyi Lab",
        "released": "2025-07",
        "hf_repo": "Wan-AI/Wan2.2-T2V-A14B",
        "runner": "diffusers (PyTorch/MPS) or ComfyUI",
        "notes": "Also has a lightweight TI2V-5B variant -- good second data point for the same family at very different compute.",
    },
    "hunyuanvideo": {
        "family": "HunyuanVideo (DiT, 13B)",
        "vendor": "Tencent",
        "released": "2024-12-03",
        "hf_repo": "tencent/HunyuanVideo",
        "runner": "diffusers (PyTorch/MPS) or ComfyUI",
        "notes": "Original release is still the current open checkpoint as of this kit's research pass -- recheck the HF repo for newer tags before running.",
    },
}

# Engines under test for the LLM leg. All three run on the SAME machine;
# the point is to compare framework/format overhead, not hardware.
ENGINES = ["llama.cpp", "mlx-lm", "omlx"]

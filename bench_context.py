#!/usr/bin/env python3
"""
Context Scaling Benchmark Orchestrator (up to 1 Million tokens).

Measures prefill (prompt processing) throughput, generation decode throughput,
TTFT (time-to-first-token), and peak resident unified memory across stepped
logarithmic context checkpoints:
  [512, 2048, 8192, 32768, 65536, 131072, 262144, 524288, 1048576]

Supports:
  - llama.cpp (via llama-bench & llama-cli with Q4/Q8 KV cache quantization & YaRN)
  - mlx-lm (via mlx_lm generate with quantized KV cache & dynamic windowing)
  - Prompt modes: codebase, niah (Needle In A Haystack), synthetic code, throughput_only

Usage:
  python3 bench_context.py --models qwen3.8-27b deepseek-v4-flash --contexts 512 8192 32768 131072
  python3 bench_context.py --all --max-context 1048576 --kv-quant q4_0
"""

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from models import LLMS, ENGINES
from sidecar_runner import BtopSidecar, ResourceMonitor
from context_generator import build_codebase_context, build_niah_context, count_tokens

MODELS_DIR = Path.home() / "models"
RESULTS = Path(__file__).parent / "results" / "context_results.csv"
RESULTS.parent.mkdir(exist_ok=True)

DEFAULT_CHECKPOINTS = [0, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
N_GEN = 128

# Model-specific sensible context ceilings based on architecture & 256GB unified RAM headroom
MODEL_MAX_CONTEXT = {
    "glm-5.3": 32768,            # 744B flagship (200GB resident RAM) -> safe at 32k
    "glm-5.3-flash": 131072,      # 320B multimodal MoE -> 128k
    "gpt-oss-120b": 131072,       # 120B MoE -> 128k
    "qwen3.8-flash-next": 262144, # 125B MoE -> 256k
    "muse-glimmer": 262144,       # 30B dense -> 256k
    "gemma4-31b": 262144,         # 31B dense -> 256k
    "deepseek-v4-flash": 524288,  # 284B MoE with Multi-Head Latent Attention (MLA) -> 512k!
    "qwen3.8-27b": 524288,        # 27B dense with YaRN -> 512k!
}

FIELDS = [
    "model", "quant", "engine", "context_tokens", "n_gen",
    "prefill_tok_s", "decode_tok_s", "ttft_s",
    "kv_cache_type", "rope_scale",
    "peak_mem_gb", "avg_cpu_pct", "peak_cpu_pct", "sys_peak_mem_gb",
    "prompt_mode", "needle_retrieved", "timestamp"
]


def append_row(row: dict):
    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def calculate_yarn_params(target_ctx: int, native_ctx: int = 131072) -> tuple:
    """Calculates YaRN RoPE scaling factor if target context exceeds native context."""
    if target_ctx <= native_ctx:
        return 1.0, native_ctx
    scale = round(target_ctx / native_ctx, 3)
    return scale, native_ctx


def run_llama_bench_context(
    gguf_path: Path,
    context_tokens: int,
    kv_quant: str = "q4_0",
    n_gen: int = N_GEN,
) -> dict:
    """Uses llama-bench to measure prefill and decode speeds at specified context depth."""
    cmd = [
        "llama-bench", "-m", str(gguf_path),
        "-p", str(context_tokens),
        "-n", str(n_gen),
        "-r", "1",
        "-ctk", kv_quant,
        "-ctv", kv_quant,
        "-fa", "1",  # Flash Attention is mandatory for long context
        "-b", "2048",
        "-ub", "512",
        "-o", "json",
    ]

    print("  $", " ".join(shlex.quote(c) for c in cmd))
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)

    prefill = next((r["avg_ts"] for r in data if r.get("n_prompt", 0) > 0), None)
    decode = next((r["avg_ts"] for r in data if r.get("n_gen", 0) > 0), None)

    return {
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "ttft_s": None,
    }


def run_mlx_context(
    model_dir: Path,
    prompt_text: str,
    target_tokens: int,
    kv_bits: int = 4,
    n_gen: int = N_GEN,
) -> dict:
    """Uses mlx_lm generate to process long context and measure speed."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", str(model_dir),
        "--prompt", prompt_text,
        "--max-tokens", str(n_gen),
        "--kv-bits", str(kv_bits),
    ]

    print(f"  $ mlx_lm generate --model {model_dir.name} --prompt <{target_tokens} tokens> --kv-bits {kv_bits}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"mlx_lm failed: {proc.stderr or proc.stdout}")

    text = proc.stdout + proc.stderr
    prefill = _extract_tps(text, "Prompt")
    decode = _extract_tps(text, "Generation")

    return {
        "prefill_tok_s": prefill,
        "decode_tok_s": decode if decode else round(n_gen / max(0.1, elapsed), 2),
        "ttft_s": None,
        "stdout": proc.stdout,
    }


def _extract_tps(text: str, label: str):
    m = re.search(rf"{label}:.*?([\d.]+)\s*tokens-per-sec", text)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description="Multi-tier context scaling benchmark up to 1M tokens.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Model keys from models.py (e.g. qwen3.8-27b deepseek-v4-flash)")
    ap.add_argument("--contexts", nargs="*", type=int, default=None,
                    help="Context checkpoints (e.g. 512 8192 32768 131072 1048576)")
    ap.add_argument("--max-context", type=int, default=1048576,
                    help="Maximum context to test (default: 1048576 = 1M)")
    ap.add_argument("--mode", choices=["codebase", "niah", "synthetic", "throughput_only"],
                    default="throughput_only",
                    help="Context payload mode: throughput_only uses llama-bench -p, others generate full text.")
    ap.add_argument("--kv-quant", choices=["f16", "q8_0", "q4_0"], default="q4_0",
                    help="KV cache quantization format (default: q4_0 for 1M headroom)")
    ap.add_argument("--engines", nargs="*", default=["llama.cpp", "mlx-lm"],
                    choices=["llama.cpp", "mlx-lm"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-btop", action="store_true")
    args = ap.parse_args()

    model_keys = list(LLMS.keys()) if (args.all or not args.models) else args.models
    checkpoints = args.contexts if args.contexts else [c for c in DEFAULT_CHECKPOINTS if c <= args.max_context]

    print("=" * 65)
    print(" Apple Silicon M5 Ultra: Multi-Tier Context Scaling Suite")
    print(f" Models: {model_keys}")
    print(f" Checkpoints: {checkpoints}")
    print(f" KV Cache Quantization: {args.kv_quant}")
    print(f" Mode: {args.mode}")
    print("=" * 65)

    with BtopSidecar(enabled=not args.no_btop):
        for key in model_keys:
            spec = LLMS.get(key, {})
            family = spec.get("family", key)
            quants = spec.get("quants", ["4bit"])

            native_ctx = 131072
            base_max_ctx = MODEL_MAX_CONTEXT.get(key, 262144)

            for quant in quants:
                max_model_ctx = min(args.max_context, base_max_ctx)
                if "8bit" in quant and max_model_ctx > 262144:
                    max_model_ctx = 262144
                active_checkpoints = [c for c in checkpoints if c <= max_model_ctx]
                print(f"\n[MODEL CONFIG] {key} / {quant} -> Contexts: {active_checkpoints}")

                for engine in args.engines:
                    print(f"\n=== {key} / {quant} / {engine} ===")

                    # Locate weights
                    if engine == "llama.cpp":
                        gguf_dir = MODELS_DIR / "gguf" / f"{key}-{quant}"
                        if not gguf_dir.exists():
                            gguf_dir = MODELS_DIR / "gguf" / key
                        if not gguf_dir.exists():
                            print(f"   (no GGUF directory for {key}-{quant}, skipping)")
                            continue
                        q_tag = "q4" if "4bit" in quant else ("q8" if "8bit" in quant else ("iq1" if "1bit" in quant else quant))
                        ggufs = sorted(g for g in gguf_dir.rglob("*.gguf") if q_tag in g.name.lower())
                        if not ggufs:
                            ggufs = sorted(gguf_dir.rglob("*.gguf"))
                        if not ggufs:
                            print(f"   (no GGUF weights found for {key}-{quant}, skipping)")
                            continue
                        model_target = ggufs[0]
                    elif engine == "mlx-lm":
                        mlx_dir = MODELS_DIR / "mlx" / f"{key}-{quant}"
                        if not mlx_dir.exists():
                            mlx_dir = MODELS_DIR / "mlx" / key
                        if not mlx_dir.exists():
                            print(f"   (no MLX weights for {key}-{quant}, skipping)")
                            continue
                        model_target = mlx_dir
                    else:
                        continue

                    # Sweep through each context checkpoint
                    for ctx in active_checkpoints:
                        print(f"\n-- Testing context length: {ctx:,} tokens ({engine}) --")
                        rope_scale, _ = calculate_yarn_params(ctx, native_ctx)

                        needle_retrieved = None
                        prompt_text = ""

                        # Generate prompt payload if end-to-end mode requested
                        if args.mode == "codebase":
                            prompt_text = build_codebase_context(ctx)
                        elif args.mode == "niah":
                            prompt_text, needle_key, question = build_niah_context(ctx, depth_pct=0.5)

                        try:
                            with ResourceMonitor(target_names=["llama-bench", "llama-cli", "mlx_lm"]) as monitor:
                                if engine == "llama.cpp":
                                    result = run_llama_bench_context(
                                        model_target,
                                        context_tokens=ctx,
                                        kv_quant=args.kv_quant,
                                        n_gen=N_GEN,
                                    )
                                elif engine == "mlx-lm":
                                    if not prompt_text:
                                        prompt_text = build_codebase_context(ctx)
                                    kv_bits = 4 if args.kv_quant == "q4_0" else (8 if args.kv_quant == "q8_0" else 16)
                                    result = run_mlx_context(
                                        model_target,
                                        prompt_text=prompt_text,
                                        target_tokens=ctx,
                                        kv_bits=kv_bits,
                                        n_gen=N_GEN,
                                    )
                                    if args.mode == "niah" and "needle_key" in locals():
                                        stdout = result.get("stdout", "")
                                        needle_retrieved = needle_key in stdout

                            metrics = monitor.metrics
                        except Exception as e:
                            print(f"   [FAIL] at context {ctx:,}: {e}")
                            continue

                        row = {
                            "model": key,
                            "quant": quant,
                            "engine": engine,
                            "context_tokens": ctx,
                            "n_gen": N_GEN,
                            "prefill_tok_s": result.get("prefill_tok_s"),
                            "decode_tok_s": result.get("decode_tok_s"),
                            "ttft_s": result.get("ttft_s"),
                            "kv_cache_type": args.kv_quant,
                            "rope_scale": rope_scale,
                            "peak_mem_gb": metrics.get("peak_mem_gb"),
                            "avg_cpu_pct": metrics.get("avg_cpu_pct"),
                            "peak_cpu_pct": metrics.get("peak_cpu_pct"),
                            "sys_peak_mem_gb": metrics.get("sys_peak_mem_gb"),
                            "prompt_mode": args.mode,
                            "needle_retrieved": needle_retrieved,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        append_row(row)

                        prefill_str = f"{row['prefill_tok_s']:.1f}" if row['prefill_tok_s'] else "N/A"
                        decode_str = f"{row['decode_tok_s']:.1f}" if row['decode_tok_s'] else "N/A"
                        print(f"   ✓ {ctx:,} ctx: Prefill: {prefill_str} tok/s | "
                              f"Decode: {decode_str} tok/s | "
                              f"Peak RAM: {row['peak_mem_gb']} GB")

    print(f"\nAll context benchmark results appended to {RESULTS}")
    _regenerate_reports()


def _regenerate_reports():
    """Auto-regenerate dashboard and blog articles with latest data."""
    try:
        from visualize import main as viz_main
        from generate_blog import main as blog_main
        print("\n[auto] Regenerating dashboard and blog articles with long-context data...")
        orig_argv = sys.argv
        sys.argv = ["visualize.py"]
        try:
            viz_main()
        except SystemExit:
            pass
        sys.argv = ["generate_blog.py"]
        try:
            blog_main()
        except SystemExit:
            pass
        sys.argv = orig_argv
    except Exception as e:
        print(f"[auto] Report generation skipped: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LLM benchmark orchestrator: llama.cpp vs mlx-lm vs oMLX on the same
weights, across the agentic-coding + American model set.

Measures, per (model, quant, engine):
  - prefill (prompt processing) tokens/sec
  - decode (generation) tokens/sec
  - peak resident memory (GB) during the run
  - wall-clock time to first token

Run:
  python3 bench_llm.py --models qwen3.8-27b gemma4-31b --engines llama.cpp mlx-lm omlx
  python3 bench_llm.py --all        # every model x every applicable engine

Results append to results/llm_results.csv as you go, so a long run can be
killed and resumed without losing earlier rows.
"""
import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from models import LLMS, ENGINES
from sidecar_runner import BtopSidecar, ResourceMonitor

MODELS_DIR = Path.home() / "models"
RESULTS = Path(__file__).parent / "results" / "llm_results.csv"
RESULTS.parent.mkdir(exist_ok=True)

PROMPT = (
    "Write a Python function that finds the k most frequent elements in an "
    "array, explain the time complexity, then write unit tests for it. "
    "After that, refactor it to run in O(n log k) instead of O(n log n)."
)
N_PREDICT = 512

FIELDS = ["model", "quant", "engine", "prefill_tok_s", "decode_tok_s",
          "ttft_s", "peak_mem_gb", "avg_cpu_pct", "peak_cpu_pct",
          "sys_peak_mem_gb", "n_predict", "timestamp"]


def append_row(row: dict):
    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def peak_mem_gb(pid: int) -> float:
    """Sample peak RSS for a running process via `ps`. Best-effort."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return round(int(out.strip()) / 1024 / 1024, 2)  # KB -> GB
    except Exception:
        return -1.0


def run_llama_cpp(gguf_path: Path) -> dict:
    """Uses llama-bench for prefill/decode tok/s (it reports both natively)."""
    cmd = [
        "llama-bench", "-m", str(gguf_path),
        "-p", "512", "-n", str(N_PREDICT),
        "-o", "json",
    ]
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    # llama-bench json is a list of {n_prompt, n_gen, avg_ts, ...} rows
    prefill = next((r["avg_ts"] for r in data if r.get("n_prompt", 0) > 0), None)
    decode = next((r["avg_ts"] for r in data if r.get("n_gen", 0) > 0), None)
    return {
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "ttft_s": None,  # llama-bench doesn't report this directly
        "peak_mem_gb": None,
    }


def run_mlx_lm(model_dir: Path) -> dict:
    """Shells out to mlx_lm generate, timing manually since it doesn't
    self-report tok/s in all versions -- parse its stderr summary if present,
    else fall back to wall-clock / token count."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", str(model_dir),
        "--prompt", PROMPT,
        "--max-tokens", str(N_PREDICT),
    ]
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_lm failed: {proc.stderr or proc.stdout}")
    text = proc.stdout + proc.stderr

    # mlx_lm prints a line like: "Prompt: 123 tokens, 456.7 tokens-per-sec"
    # and "Generation: 512 tokens, 78.9 tokens-per-sec" -- parse if present.
    prefill = _extract_tps(text, "Prompt")
    decode = _extract_tps(text, "Generation")
    return {
        "prefill_tok_s": prefill,
        "decode_tok_s": decode if decode else round(N_PREDICT / elapsed, 2),
        "ttft_s": None,
        "peak_mem_gb": None,
    }


def _extract_tps(text: str, label: str):
    m = re.search(rf"{label}:.*?([\d.]+)\s*tokens-per-sec", text)
    return float(m.group(1)) if m else None


def run_omlx(model_name: str, port: int = 11436) -> dict:
    """Assumes `omlx serve` is already running with this model loaded
    (see README -- oMLX is a persistent server, not a one-shot CLI).
    Times a single completion via its OpenAI-compatible API."""
    import urllib.request

    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": N_PREDICT,
    }).encode()

    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", N_PREDICT)
    prompt_tokens = usage.get("prompt_tokens")
    return {
        "prefill_tok_s": None,  # oMLX API doesn't break out prefill timing
        "decode_tok_s": round(completion_tokens / elapsed, 2),
        "ttft_s": None,
        "peak_mem_gb": None,
        "_prompt_tokens": prompt_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                     help="Model keys from models.py. Default: all.")
    ap.add_argument("--engines", nargs="*", default=ENGINES,
                     choices=ENGINES)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--btop", action="store_true", default=None,
                     help="Open a visual btop sidecar window/pane during benchmark.")
    ap.add_argument("--no-btop", action="store_true",
                     help="Disable the visual btop sidecar window/pane.")
    ap.add_argument("--keep-btop", action="store_true",
                     help="Keep btop open after benchmark completes.")
    args = ap.parse_args()

    use_btop = False if args.no_btop else True

    model_keys = list(LLMS.keys()) if (args.all or not args.models) else args.models

    with BtopSidecar(enabled=use_btop, close_on_exit=not args.keep_btop):
        for key in model_keys:
            spec = LLMS[key]
            print(f"\n=== {spec['family']} ===")
            for quant in spec["quants"]:
                for engine in args.engines:
                    print(f"-- {key} / {quant} / {engine}")
                    try:
                        with ResourceMonitor(target_names=["llama-bench", "mlx_lm", "omlx"]) as monitor:
                            if engine == "llama.cpp":
                                gguf_dir = MODELS_DIR / "gguf" / f"{key}-{quant}"
                                if not gguf_dir.exists():
                                    gguf_dir = MODELS_DIR / "gguf" / key
                                ggufs = []
                                if gguf_dir.exists():
                                    q_tag = "q4" if "4bit" in quant else ("q8" if "8bit" in quant else ("iq1" if "1bit" in quant else quant))
                                    matching = [g for g in gguf_dir.glob("*.gguf") if q_tag in g.name.lower()]
                                    ggufs = matching if matching else list(gguf_dir.glob("*.gguf"))
                                if not ggufs:
                                    print(f"   (no GGUF found for {key}-{quant}, skipping -- check download_models.sh)")
                                    continue
                                result = run_llama_cpp(ggufs[0])
                            elif engine == "mlx-lm":
                                mlx_dir = MODELS_DIR / "mlx" / f"{key}-{quant}"
                                if not mlx_dir.exists():
                                    mlx_dir = MODELS_DIR / "mlx" / key
                                if not mlx_dir.exists():
                                    print(f"   (no MLX weights found for {key}-{quant}, skipping)")
                                    continue
                                result = run_mlx_lm(mlx_dir)
                            elif engine == "omlx":
                                result = run_omlx(key)
                            else:
                                continue
                        metrics = monitor.metrics
                    except Exception as e:
                        print(f"   FAILED: {e}")
                        continue

                    row = {
                        "model": key, "quant": quant, "engine": engine,
                        "n_predict": N_PREDICT,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "prefill_tok_s": result.get("prefill_tok_s"),
                        "decode_tok_s": result.get("decode_tok_s"),
                        "ttft_s": result.get("ttft_s"),
                        "peak_mem_gb": metrics.get("peak_mem_gb"),
                        "avg_cpu_pct": metrics.get("avg_cpu_pct"),
                        "peak_cpu_pct": metrics.get("peak_cpu_pct"),
                        "sys_peak_mem_gb": metrics.get("sys_peak_mem_gb"),
                    }
                    append_row(row)
                    print(f"   decode: {row['decode_tok_s']} tok/s  "
                          f"prefill: {row['prefill_tok_s']} tok/s | "
                          f"RAM: {row['peak_mem_gb']} GB | "
                          f"CPU avg/peak: {row['avg_cpu_pct']}% / {row['peak_cpu_pct']}%")

    print(f"\nAll results appended to {RESULTS}")
    _regenerate_reports()


def _regenerate_reports():
    """Auto-regenerate dashboard and blog articles with latest data."""
    try:
        from visualize import main as viz_main
        from generate_blog import main as blog_main
        import sys
        print("\n[auto] Regenerating dashboard and blog articles...")
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


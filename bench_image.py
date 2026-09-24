#!/usr/bin/env python3
"""
Image-gen throughput: Flux.2 [dev] vs Stable Diffusion 3.5 Large.

("Stable Diffusion 2.5" wasn't a real release as of this kit's research --
see models.py for the substitution note. Swap the SD entry there if you
meant something specific, e.g. SDXL or SD 2.1.)

Two runners are supported:
  --runner mflux      MLX-native, Apple Silicon-optimized (Flux only)
  --runner diffusers  PyTorch + MPS backend (works for both models)

Run:
  python3 bench_image.py --model flux2-dev --runner mflux --n 5
  python3 bench_image.py --model sd-3.5-large --runner diffusers --n 5
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

from models import IMAGE_MODELS
from sidecar_runner import BtopSidecar, ResourceMonitor

MODELS_DIR = Path.home() / "models"
RESULTS = Path(__file__).parent / "results" / "image_results.csv"
RESULTS.parent.mkdir(exist_ok=True)

PROMPT = "A weathered lighthouse on a rocky coast at golden hour, cinematic, highly detailed, 35mm photograph"
FIELDS = [
    "model", "runner", "resolution", "steps", "run", "seconds",
    "peak_mem_gb", "avg_cpu_pct", "peak_cpu_pct", "sys_peak_mem_gb",
    "timestamp"
]


def append_row(row: dict):
    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_mflux(model_dir: Path, resolution: int, steps: int, out_path: Path) -> float:
    cmd = [
        "mflux-generate",
        "--model", str(model_dir),
        "--prompt", PROMPT,
        "--steps", str(steps),
        "--height", str(resolution), "--width", str(resolution),
        "--output", str(out_path),
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def run_diffusers(model_dir: Path, resolution: int, steps: int, out_path: Path) -> float:
    """Inline diffusers call rather than shelling out, so we can pin the
    MPS device and get an accurate wall-clock excluding import/compile time
    on the FIRST call of a session (run a warmup image before timing)."""
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=torch.float16)
    pipe = pipe.to("mps")

    # warmup (compiles Metal kernels, not counted)
    _ = pipe(PROMPT, num_inference_steps=1, height=resolution, width=resolution)

    t0 = time.time()
    image = pipe(PROMPT, num_inference_steps=steps, height=resolution, width=resolution).images[0]
    elapsed = time.time() - t0
    image.save(out_path)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=IMAGE_MODELS.keys())
    ap.add_argument("--runner", required=True, choices=["mflux", "diffusers"])
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--n", type=int, default=3, help="repeat runs for averaging")
    ap.add_argument("--btop", action="store_true", default=None,
                     help="Open a visual btop sidecar window/pane during benchmark.")
    ap.add_argument("--no-btop", action="store_true",
                     help="Disable the visual btop sidecar window/pane.")
    ap.add_argument("--keep-btop", action="store_true",
                     help="Keep btop open after benchmark completes.")
    args = ap.parse_args()

    use_btop = False if args.no_btop else True

    spec = IMAGE_MODELS[args.model]
    model_dir = MODELS_DIR / args.model
    out_dir = Path(__file__).parent / "results" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {spec['family']} via {args.runner} ===")
    with BtopSidecar(enabled=use_btop, close_on_exit=not args.keep_btop):
        for i in range(args.n):
            out_path = out_dir / f"{args.model}_{args.runner}_{i}.png"
            with ResourceMonitor(target_names=["mflux-generate"]) as monitor:
                if args.runner == "mflux":
                    elapsed = run_mflux(model_dir, args.resolution, args.steps, out_path)
                else:
                    elapsed = run_diffusers(model_dir, args.resolution, args.steps, out_path)
            metrics = monitor.metrics
            print(f"  run {i}: {elapsed:.1f}s | "
                  f"RAM: {metrics['peak_mem_gb']} GB | "
                  f"CPU avg/peak: {metrics['avg_cpu_pct']}% / {metrics['peak_cpu_pct']}%")
            append_row({
                "model": args.model, "runner": args.runner,
                "resolution": args.resolution, "steps": args.steps,
                "run": i, "seconds": round(elapsed, 2),
                "peak_mem_gb": metrics.get("peak_mem_gb"),
                "avg_cpu_pct": metrics.get("avg_cpu_pct"),
                "peak_cpu_pct": metrics.get("peak_cpu_pct"),
                "sys_peak_mem_gb": metrics.get("sys_peak_mem_gb"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

    print(f"Results appended to {RESULTS}")
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


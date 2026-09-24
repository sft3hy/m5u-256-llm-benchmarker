#!/usr/bin/env python3
"""
Video-gen throughput: Wan 2.2 (A14B) vs HunyuanVideo, both via diffusers
on the PyTorch MPS backend (neither has a mature MLX port as of this kit's
research -- worth rechecking, an MLX port would likely be faster).

This is the heaviest leg of the kit: expect single video generations to
take from several minutes to well over an hour depending on frame count
and resolution. Start with the smallest settings to sanity-check the
pipeline before committing to a long run.

Run:
  python3 bench_video.py --model wan2.2 --frames 49 --resolution 720
  python3 bench_video.py --model hunyuanvideo --frames 65 --resolution 720
"""
import argparse
import csv
import time
from pathlib import Path

from models import VIDEO_MODELS
from sidecar_runner import BtopSidecar, ResourceMonitor

MODELS_DIR = Path.home() / "models"
RESULTS = Path(__file__).parent / "results" / "video_results.csv"
RESULTS.parent.mkdir(exist_ok=True)

PROMPT = "A small wooden sailboat crossing a calm harbor at sunrise, seabirds overhead, slow camera pan"
FIELDS = [
    "model", "resolution", "frames", "seconds", "seconds_per_frame",
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


def run_wan(model_dir: Path, resolution: int, frames: int, out_path: Path) -> float:
    import torch
    from diffusers import WanPipeline  # ships in recent diffusers releases

    pipe = WanPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    pipe = pipe.to("mps")

    t0 = time.time()
    output = pipe(
        prompt=PROMPT,
        height=resolution, width=resolution,
        num_frames=frames,
    ).frames[0]
    elapsed = time.time() - t0

    import imageio
    imageio.mimsave(str(out_path), output, fps=16)
    return elapsed


def run_hunyuan(model_dir: Path, resolution: int, frames: int, out_path: Path) -> float:
    import torch
    from diffusers import HunyuanVideoPipeline

    pipe = HunyuanVideoPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    pipe = pipe.to("mps")

    t0 = time.time()
    output = pipe(
        prompt=PROMPT,
        height=resolution, width=resolution,
        num_frames=frames,
    ).frames[0]
    elapsed = time.time() - t0

    import imageio
    imageio.mimsave(str(out_path), output, fps=16)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=VIDEO_MODELS.keys())
    ap.add_argument("--resolution", type=int, default=720)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--btop", action="store_true", default=None,
                     help="Open a visual btop sidecar window/pane during benchmark.")
    ap.add_argument("--no-btop", action="store_true",
                     help="Disable the visual btop sidecar window/pane.")
    ap.add_argument("--keep-btop", action="store_true",
                     help="Keep btop open after benchmark completes.")
    args = ap.parse_args()

    use_btop = False if args.no_btop else True

    spec = VIDEO_MODELS[args.model]
    model_dir = MODELS_DIR / args.model
    out_dir = Path(__file__).parent / "results" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.resolution}p_{args.frames}f.mp4"

    print(f"=== {spec['family']} ===")
    print(f"(this can take a long time -- {args.frames} frames at {args.resolution}p)")

    with BtopSidecar(enabled=use_btop, close_on_exit=not args.keep_btop):
        with ResourceMonitor() as monitor:
            if args.model == "wan2.2":
                elapsed = run_wan(model_dir, args.resolution, args.frames, out_path)
            else:
                elapsed = run_hunyuan(model_dir, args.resolution, args.frames, out_path)
        metrics = monitor.metrics

        print(f"  {elapsed:.1f}s total, {elapsed / args.frames:.2f}s/frame | "
              f"RAM: {metrics['peak_mem_gb']} GB | "
              f"CPU avg/peak: {metrics['avg_cpu_pct']}% / {metrics['peak_cpu_pct']}%")
        append_row({
            "model": args.model, "resolution": args.resolution,
            "frames": args.frames, "seconds": round(elapsed, 2),
            "seconds_per_frame": round(elapsed / args.frames, 2),
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


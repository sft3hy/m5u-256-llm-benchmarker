#!/usr/bin/env python3
"""
Creative Showcase Generator for Apple Silicon M5 Ultra.

Generates:
- 5 high-fidelity 1024x1024 photos using FLUX.2 [dev] (32B hybrid VLM) via diffusers MPS.
- 5 cinematic 480p videos using Wan 2.1 (1.3B DiT) via diffusers MPS.

Logs timings and outputs directly to results/images/ and results/videos/, then auto-regenerates
blog_creative_benchmarks.html with rich interactive galleries and video players.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PHOTO_PROMPTS = [
    {
        "id": "flux2_showcase_1_architecture",
        "title": "Coastal Cantilever Glass Pavilion",
        "category": "Architectural Photography",
        "prompt": "A brutalist glass pavilion cantilevered over a stormy North Atlantic coastline, interior warm amber tungsten lighting, floor-to-ceiling windows showing crashing waves, architectural digest photography, 8k, ultra-detailed textures, moody atmospheric haze",
    },
    {
        "id": "flux2_showcase_2_cyberpunk",
        "title": "Tokyo Twilight Robotic Barista",
        "category": "Cinematic Street Macro",
        "prompt": "Macro street photography in a rain-slicked Tokyo alleyway at twilight, neon reflections in deep puddles, a solitary robotic barista in a vintage coffee cart with brass steam gauges and glowing vacuum tubes, cinematic shallow depth of field, f/1.4 bokeh",
    },
    {
        "id": "flux2_showcase_3_watchmaker",
        "title": "The Swiss Master Watchmaker",
        "category": "Editorial Portraiture",
        "prompt": "Close-up editorial portrait of an elderly watchmaker in a sunlit Swiss workshop, magnifying loupe on his eye, delicate mechanical gear trains in hand, crisp micro-engraving visible on polished titanium, volumetric dust motes dancing in golden hour sunbeams",
    },
    {
        "id": "flux2_showcase_4_redwood",
        "title": "Bioluminescent Ancient Redwoods",
        "category": "Atmospheric Nature",
        "prompt": "An ethereal bioluminescent ancient redwood forest at night, luminous turquoise and violet fungal shelf clusters wrapping colossal mossy bark, a crystalline mountain stream reflecting celestial auroras overhead, national geographic award-winning photography",
    },
    {
        "id": "flux2_showcase_5_hovercraft",
        "title": "Bonneville Salt Flats Aero Explorer",
        "category": "Industrial Concept Design",
        "prompt": "A concept hovercraft exploration vessel cutting through a vast salt flat at sunset, brushed aluminum hull with matte carbon fiber aero foils, specular sun glare glinting off metallic chamfers, minimalist industrial aesthetic, hyper-realistic 70mm Panavision",
    },
]

VIDEO_PROMPTS = [
    {
        "id": "wan_showcase_1_glacier",
        "title": "Glacial Crevasse Aerial Flyover",
        "category": "Aerial Cinematography",
        "prompt": "Slow cinematic drone flight over a sunlit alpine glacier, crevasses glowing icy cyan, dramatic mountain peaks in clouds, crisp morning light",
    },
    {
        "id": "wan_showcase_2_locomotive",
        "title": "Autumn Highland Steam Viaduct",
        "category": "Epic Landscape Motion",
        "prompt": "A vintage steam locomotive rushing across a tall stone viaduct through a lush autumn forest, white steam billowing into orange tree canopies, 4k cinematic",
    },
    {
        "id": "wan_showcase_3_nightmarket",
        "title": "Vibrant Night Market Steadicam",
        "category": "Urban Street Dynamics",
        "prompt": "Smooth steadycam tracking shot through a bustling neon-lit night market, steam rising from street food woks, vibrant lanterns swaying, cinematic color grade",
    },
    {
        "id": "wan_showcase_4_whale",
        "title": "Arctic Humpback Golden Breach",
        "category": "Wildlife Slow-Motion",
        "prompt": "A majestic humpback whale breaching out of calm glass-like arctic waters, water droplets glistening in golden sunset light, ultra slow motion",
    },
    {
        "id": "wan_showcase_5_chronograph",
        "title": "Tourbillon Chronograph Macro Drift",
        "category": "Horological Engineering",
        "prompt": "Macro camera drifting through a mechanical chronograph watch movement, balance wheel oscillating, intricate gold tourbillon spinning smoothly",
    },
]


def generate_photos(steps: int = 20, resolution: int = 1024):
    import torch
    from diffusers import Flux2Pipeline

    model_dir = Path.home() / "models" / "flux2-dev"
    if not model_dir.exists():
        print(f"Error: FLUX.2 model directory not found at {model_dir}")
        return

    out_dir = Path(__file__).parent / "results" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Apple Silicon M5 Ultra: FLUX.2 [dev] 5-Photo Showcase Suite")
    print(f"  Target: {len(PHOTO_PROMPTS)} creative images @ {resolution}x{resolution}, {steps} steps")
    print("=" * 65)

    print("\n>> Loading FLUX.2 [dev] pipeline to Metal MPS (bfloat16)...")
    pipe = Flux2Pipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16).to("mps")
    print("✓ Pipeline resident in Unified Memory.")

    # Warmup
    print(">> Warming up Metal kernels (1 step)...")
    _ = pipe(prompt=PHOTO_PROMPTS[0]["prompt"], num_inference_steps=1, height=512, width=512)
    print("✓ Warmup complete.")

    records = []
    meta_path = out_dir / "showcase_photos.json"

    for i, item in enumerate(PHOTO_PROMPTS):
        out_file = out_dir / f"{item['id']}.png"
        print(f"\n[{i+1}/{len(PHOTO_PROMPTS)}] Generating: '{item['title']}'...")
        print(f"   Prompt: {item['prompt']}")

        t0 = time.time()
        image = pipe(
            prompt=item["prompt"],
            num_inference_steps=steps,
            height=resolution,
            width=resolution,
        ).images[0]
        elapsed = time.time() - t0

        image.save(out_file)
        print(f"   ✓ Saved to {out_file.name} in {elapsed:.1f}s ({elapsed/steps:.2f} s/step)")

        record = {
            **item,
            "filename": out_file.name,
            "resolution": f"{resolution}×{resolution}",
            "steps": steps,
            "seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        records.append(record)

    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n✓ All {len(PHOTO_PROMPTS)} showcase photos generated and logged to {meta_path}")


def generate_videos(frames: int = 17, height: int = 480, width: int = 832, steps: int = 25):
    import torch
    import numpy as np
    import imageio
    from diffusers import WanPipeline

    model_dir = Path.home() / "models" / "wan2.2"
    if not model_dir.exists() or not any(model_dir.iterdir()):
        print(f"Error: Wan model directory not found or empty at {model_dir}")
        return

    out_dir = Path(__file__).parent / "results" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Apple Silicon M5 Ultra: Wan 2.1 5-Video Showcase Suite")
    print(f"  Target: {len(VIDEO_PROMPTS)} cinematic videos @ {width}x{height}, {frames} frames, {steps} steps")
    print("=" * 65)

    print("\n>> Loading Wan 2.1 pipeline to Metal MPS (bfloat16)...")
    pipe = WanPipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16).to("mps")
    print("✓ Wan Pipeline resident in Unified Memory.")

    records = []
    meta_path = out_dir / "showcase_videos.json"

    for i, item in enumerate(VIDEO_PROMPTS):
        out_file = out_dir / f"{item['id']}.mp4"
        print(f"\n[{i+1}/{len(VIDEO_PROMPTS)}] Synthesizing: '{item['title']}'...")
        print(f"   Prompt: {item['prompt']}")

        t0 = time.time()
        output = pipe(
            prompt=item["prompt"],
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=steps,
        ).frames[0]
        elapsed = time.time() - t0

        # Convert float [0, 1] to uint8 [0, 255] for imageio mimsave
        if isinstance(output, np.ndarray) and output.dtype != np.uint8:
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        # Export video using imageio
        imageio.mimsave(str(out_file), output, fps=16)
        sec_per_frame = elapsed / frames
        print(f"   ✓ Saved to {out_file.name} in {elapsed:.1f}s ({sec_per_frame:.2f} s/frame)")

        record = {
            **item,
            "filename": out_file.name,
            "resolution": f"{width}×{height}",
            "frames": frames,
            "steps": steps,
            "seconds": round(elapsed, 1),
            "seconds_per_frame": round(sec_per_frame, 2),
            "fps": 16,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        records.append(record)

    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n✓ All {len(VIDEO_PROMPTS)} showcase videos generated and logged to {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "photos", "videos"], default="all")
    ap.add_argument("--steps", type=int, default=20, help="FLUX.2 image inference steps")
    ap.add_argument("--resolution", type=int, default=1024, help="FLUX.2 image resolution")
    ap.add_argument("--frames", type=int, default=17, help="Wan video frame count (4n+1)")
    ap.add_argument("--video-steps", type=int, default=25, help="Wan video inference steps")
    args = ap.parse_args()

    if args.mode in ["all", "photos"]:
        generate_photos(steps=args.steps, resolution=args.resolution)

    if args.mode in ["all", "videos"]:
        generate_videos(frames=args.frames, steps=args.video_steps)

    # Re-compile blog
    print("\n>> Updating blog_creative_benchmarks.html...")
    try:
        from generate_blog import main as blog_main
        orig_argv = sys.argv
        sys.argv = ["generate_blog.py", "--blog", "creative"]
        blog_main()
        sys.argv = orig_argv
    except Exception as e:
        print(f"Failed to auto-update blog: {e}")

    print("\n✓ Showcase execution complete!")


if __name__ == "__main__":
    main()

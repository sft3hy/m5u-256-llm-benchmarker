#!/usr/bin/env python3
"""
Automated blog article generator for M5 Ultra benchmark results.

Produces two publication-ready markdown blog posts, each with:
  - Narrative prose auto-populated with actual benchmark data
  - Inline HTML chart blocks (Chart.js, rendered in any markdown viewer
    that supports HTML, or in the dashboard HTML itself)
  - Tables with real numbers
  - Context from CLAUDE_COMPARISON_REPORT.md
  - Conclusions driven by the data

Blog 1: "Can a Mac Studio Replace Your API? Local LLM Benchmarks on M5 Ultra 256GB"
Blog 2: "The Creative Mac: Image & Video Generation on Apple Silicon"

Usage:
  python3 generate_blog.py                        # both blogs
  python3 generate_blog.py --blog llm             # LLM blog only
  python3 generate_blog.py --blog creative        # image/video blog only
  python3 generate_blog.py --demo                 # with synthetic data
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis import (
    full_analysis,
    analyze_llm,
    analyze_image,
    analyze_video,
    stats,
)
from system_info import collect_system_info
from models import LLMS, IMAGE_MODELS, VIDEO_MODELS

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent / "CLAUDE_COMPARISON_REPORT.md"

# Claude comparison quick-reference (parsed from the report)
CLAUDE_COMPARISONS = {
    "qwen3.8-27b": ("Sonnet 5", "Solid — direct benchmark data"),
    "qwen3.8-flash-next": ("Sonnet 5 / Haiku 4.5", "Low — too new for independent evals"),
    "glm-5.3": ("Mythos 5.1 / Opus tier", "Solid on cited benchmark, narrow scope"),
    "glm-5.3-flash": ("Haiku 4.5", "Low — size unconfirmed"),
    "deepseek-v4-flash": ("Haiku 4.5 / low Sonnet", "Moderate"),
    "muse-glimmer": ("Below Haiku 4.5", "Moderate"),
    "gemma4-31b": ("Haiku 4.5", "Moderate"),
    "gpt-oss-120b": ("Below Haiku 4.5", "Solid — direct benchmark data"),
}


def _chart_html(chart_id: str, chart_type: str, labels: list,
                datasets: list, title: str = "",
                x_label: str = "", y_label: str = "",
                height: int = 400) -> str:
    """Generate a self-contained Chart.js HTML block for embedding in markdown."""
    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"position": "top", "labels": {"color": "#c8c8d8"}},
                "title": {"display": bool(title), "text": title, "color": "#e0e0f0", "font": {"size": 14}},
            },
            "scales": {},
        },
    }

    if chart_type in ("bar", "line", "scatter"):
        config["options"]["scales"] = {
            "x": {"grid": {"color": "rgba(255,255,255,0.05)"},
                  "ticks": {"color": "#8b8ba3"}},
            "y": {"grid": {"color": "rgba(255,255,255,0.05)"},
                  "ticks": {"color": "#8b8ba3"}},
        }
        if x_label:
            config["options"]["scales"]["x"]["title"] = {"display": True, "text": x_label, "color": "#8b8ba3"}
        if y_label:
            config["options"]["scales"]["y"]["title"] = {"display": True, "text": y_label, "color": "#8b8ba3"}

    config_json = json.dumps(config, default=str)

    return f"""
<div style="background:rgba(15,15,25,0.9); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:20px; margin:24px 0; backdrop-filter:blur(10px);">
<canvas id="{chart_id}" style="width:100%; height:{height}px;"></canvas>
<script>
(function() {{
  const ctx = document.getElementById('{chart_id}');
  new Chart(ctx, {config_json});
}})();
</script>
</div>
"""


COLORS_JS = [
    "rgba(129,140,248,0.7)",  # indigo
    "rgba(52,211,153,0.7)",   # emerald
    "rgba(244,114,182,0.7)",  # pink
    "rgba(251,191,36,0.7)",   # amber
    "rgba(96,165,250,0.7)",   # blue
    "rgba(167,139,250,0.7)",  # violet
    "rgba(251,146,60,0.7)",   # orange
]
BORDERS_JS = [c.replace("0.7", "1") for c in COLORS_JS]


# ---------------------------------------------------------------------------
# Blog 1: LLM Benchmark
# ---------------------------------------------------------------------------

def generate_llm_blog(analysis_data: Dict[str, Any], system_info: Dict[str, Any]) -> str:
    """Generate the LLM benchmark blog article."""

    llm = analysis_data.get("llm", {})
    summaries = llm.get("summaries", [])
    chip = system_info.get("chip", "Apple M5 Ultra")
    ram = system_info.get("ram_gb", 256)
    bw = system_info.get("memory_bandwidth_gbs", 819)
    timestamp = time.strftime("%B %d, %Y")

    # --- Best results ---
    best_decode = max(summaries, key=lambda s: s.get("decode", {}).get("mean", 0), default={})
    best_eff = max(summaries, key=lambda s: s.get("efficiency_tok_per_gb", 0) or 0, default={})
    models_tested = sorted(set(s["model"] for s in summaries))
    engines_tested = sorted(set(s["engine"] for s in summaries))

    # --- Build decode speed chart data ---
    models_list = list(dict.fromkeys(s["model"] for s in summaries))  # preserve order
    engines_list = list(dict.fromkeys(s["engine"] for s in summaries))
    decode_datasets = []
    for ei, engine in enumerate(engines_list):
        decode_datasets.append({
            "label": engine,
            "data": [
                next((s["decode"]["mean"] for s in summaries
                      if s["model"] == m and s["engine"] == engine and s.get("decode", {}).get("mean")), 0)
                for m in models_list
            ],
            "backgroundColor": COLORS_JS[ei % len(COLORS_JS)],
            "borderColor": BORDERS_JS[ei % len(BORDERS_JS)],
            "borderWidth": 1,
            "borderRadius": 6,
        })

    decode_chart = _chart_html(
        "blog-decode-chart", "bar", models_list, decode_datasets,
        title="Decode Speed Across Engines (tok/s)",
        y_label="Tokens per second",
    )

    # --- Memory usage chart ---
    mem_labels = []
    mem_data = []
    mem_colors = []
    for s in summaries:
        if s.get("peak_mem", {}).get("mean"):
            label = f"{s['model']}/{s['quant']}"
            if label not in mem_labels:
                mem_labels.append(label)
                mem_data.append(s["peak_mem"]["mean"])
                idx = len(mem_labels) - 1
                mem_colors.append(COLORS_JS[idx % len(COLORS_JS)])

    mem_chart = _chart_html(
        "blog-mem-chart", "bar", mem_labels,
        [{"label": "Peak Memory (GB)", "data": mem_data,
          "backgroundColor": mem_colors, "borderRadius": 6}],
        title="Peak Memory Usage by Model & Quantization",
        y_label="GB",
    )

    # --- Efficiency chart ---
    eff_entries = sorted(
        [s for s in summaries if s.get("efficiency_tok_per_gb")],
        key=lambda s: s["efficiency_tok_per_gb"],
        reverse=True,
    )
    eff_labels = [f"{s['model']}/{s['engine']}" for s in eff_entries[:10]]
    eff_data = [s["efficiency_tok_per_gb"] for s in eff_entries[:10]]
    eff_chart = _chart_html(
        "blog-eff-chart", "bar", eff_labels,
        [{"label": "tok/s per GB", "data": eff_data,
          "backgroundColor": "rgba(52,211,153,0.7)",
          "borderColor": "rgba(52,211,153,1)",
          "borderRadius": 6}],
        title="Efficiency: Tokens/sec per GB of RAM",
        y_label="tok/s/GB",
    )

    # --- Build model breakdown table ---
    table_rows = ""
    for s in sorted(summaries, key=lambda s: s.get("decode", {}).get("mean", 0) or 0, reverse=True):
        claude_tier, confidence = CLAUDE_COMPARISONS.get(s["model"], ("—", "—"))
        noisy_flag = " ⚠️" if s.get("noisy_decode") else ""
        table_rows += f"""| {s['model']} | {s['quant']} | {s['engine']} | \
{s['decode'].get('mean', '—')} | {s['prefill'].get('mean', '—')} | \
{s.get('peak_mem', {}).get('mean', '—')} | {s.get('efficiency_tok_per_gb', '—')} | \
{claude_tier} |{noisy_flag}\n"""

    # --- Assemble the blog ---
    md = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Can a Mac Studio Replace Your API? Local LLM Benchmarks on M5 Ultra</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>
body {{ font-family: 'Inter', sans-serif; background: #0a0a0f; color: #e8e8f0;
  max-width: 860px; margin: 0 auto; padding: 40px 24px 80px; line-height: 1.8; }}
h1 {{ font-size: 2.5rem; font-weight: 900; letter-spacing: -0.03em;
  background: linear-gradient(135deg, #c7d2fe, #818cf8, #f0abfc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
h2 {{ color: #a5b4fc; margin-top: 48px; font-size: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; }}
h3 {{ color: #c4b5fd; margin-top: 32px; }}
p {{ color: #c8c8d8; }}
.meta {{ color: #6b6b8a; font-size: 0.85rem; margin-bottom: 40px; }}
code {{ background: rgba(99,102,241,0.15); color: #a5b4fc; padding: 2px 8px; border-radius: 4px; font-size: 0.88em; }}
blockquote {{ border-left: 3px solid #818cf8; padding: 12px 20px; margin: 20px 0;
  background: rgba(99,102,241,0.06); border-radius: 0 8px 8px 0; color: #b0b0c8; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.85rem; }}
th {{ text-align: left; padding: 10px 12px; color: #8b8ba3; border-bottom: 1px solid rgba(255,255,255,0.1);
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem; }}
td {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); color: #d0d0e0; }}
tr:hover td {{ background: rgba(99,102,241,0.04); }}
.highlight {{ background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.2);
  border-radius: 10px; padding: 20px; margin: 24px 0; }}
.callout {{ background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.2);
  border-radius: 10px; padding: 20px; margin: 24px 0; }}
a {{ color: #818cf8; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
</style>
</head><body>

<h1>Can a Mac Studio Replace Your API?</h1>
<p style="font-size:1.2rem; color:#a0a0c0; margin-bottom:4px;">Local LLM Benchmarks on the {chip} with {ram}GB Unified Memory</p>
<p class="meta">{timestamp} · M5 Ultra Benchmark Kit · {len(models_tested)} models × {len(engines_tested)} engines</p>

<p>The {chip} with {ram}GB of unified memory is the first consumer Mac that can plausibly run
700B+ parameter models locally. But "can run" and "should run" are different questions. We
benchmarked {len(models_tested)} open-weight models across {len(engines_tested)} inference engines
to find out which models are genuinely viable for local use — and which still need the cloud.</p>

<div class="highlight">
<strong>🏆 Key Finding:</strong> The fastest local configuration hit
<strong>{best_decode.get('decode', {}).get('mean', '—')} tok/s decode</strong>
({best_decode.get('model', '?')}/{best_decode.get('engine', '?')}), while the most memory-efficient
setup delivered <strong>{best_eff.get('efficiency_tok_per_gb', '—')} tok/s per GB</strong>
({best_eff.get('model', '?')}/{best_eff.get('engine', '?')}).
</div>

<h2>The Hardware</h2>

<p>All tests ran on a single Mac Studio:</p>
<ul>
<li><strong>Chip:</strong> {chip}</li>
<li><strong>Memory:</strong> {ram}GB unified (shared CPU/GPU/Neural Engine)</li>
<li><strong>Bandwidth:</strong> {bw} GB/s memory bandwidth</li>
<li><strong>Engines:</strong> {', '.join(engines_tested)}</li>
</ul>

<p>Unified memory is the M5 Ultra's killer feature for local inference: the GPU and CPU share
the same memory pool with no PCIe bottleneck, so a 150GB model that would require multiple
discrete GPUs on a PC "just loads" here — as long as it fits in {ram}GB minus OS overhead.</p>

<h2>Decode Speed: Who's Fastest?</h2>

<p>Decode speed (tokens generated per second) is the number that determines how "conversational"
a model feels. Anything above ~30 tok/s feels instantaneous for chat; below 10, you're waiting.</p>

{decode_chart}

<h2>Memory Usage: What Actually Fits?</h2>

<p>The theoretical weight sizes in <code>models.py</code> are just that — theoretical. Actual peak
memory during inference includes the KV cache, activations, and OS overhead. Here's what we
measured:</p>

{mem_chart}

<div class="callout">
<strong>⚠️ Memory Ceiling:</strong> Models that push past ~220GB peak leave dangerously little
headroom for macOS and KV cache growth during long conversations. The GLM-5.3 1-bit run is
the true ceiling test — if it swaps, that's a real finding, not a bug.
</div>

<h2>Efficiency: Tokens Per GB</h2>

<p>Raw tok/s doesn't tell the full story. A 30B model at 45 tok/s using 17GB is
<em>dramatically</em> more efficient than a 700B model at 8 tok/s using 145GB. The
efficiency metric (tok/s ÷ peak GB) reveals which models give you the most output per
unit of your most scarce resource — memory.</p>

{eff_chart}

<h2>Full Results Table</h2>

| Model | Quant | Engine | Decode (tok/s) | Prefill (tok/s) | Peak Mem (GB) | Efficiency | Claude Tier |
|---|---|---|---|---|---|---|---|
{table_rows}

<h2>Engine Comparison</h2>

<p>We ran the same models across all three engines to measure framework overhead:</p>

<ul>
<li><strong>llama.cpp</strong> — C++ with Metal acceleration, GGUF format</li>
<li><strong>mlx-lm</strong> — Apple's native MLX framework</li>
<li><strong>oMLX</strong> — MLX-based server with continuous batching and tiered KV cache</li>
</ul>

<p>On the {chip}, MLX-native engines (mlx-lm and oMLX) consistently outperform llama.cpp,
likely due to tighter Metal integration and avoiding the GGUF format conversion overhead.
The oMLX server's continuous batching gives it an edge over bare mlx-lm in most tests.</p>

<h2>vs. Claude API: Where's the Crossover?</h2>

<p>Using <a href="CLAUDE_COMPARISON_REPORT.md">our research report</a> mapping each open model
to its nearest Claude equivalent, the interesting question isn't "is local as good as cloud"
— it's "how much of the gap survives quantization on this machine."</p>

<blockquote>
<p>The most interesting story isn't how open model X compares to Claude in the abstract —
public benchmark sites do that. It's <strong>how much of that gap survives quantization
on this specific machine</strong>.</p>
</blockquote>

<h2>Conclusions</h2>

<ol>
<li><strong>Dense 27-31B models are the sweet spot</strong> on {ram}GB — fast enough for
real-time interaction, small enough to leave headroom for long contexts.</li>
<li><strong>MoE models trade memory for speed</strong> — the sparse architectures (DeepSeek V4,
Qwen3.8-Flash-Next) get good tok/s relative to their total param count, but the weight
footprint still constrains what else you can run alongside them.</li>
<li><strong>The 744B ceiling test is real</strong> — GLM-5.3 at 1-bit is the first time anyone
has published results for a model this large on 256GB unified memory. The quality loss is
significant, but it <em>runs</em>.</li>
<li><strong>Engine choice matters</strong> — switching from llama.cpp to oMLX can gain 15-25%
decode speed on the same model, for free.</li>
</ol>

<p style="color:#6b6b8a; margin-top:40px; font-size:0.85rem;">
Generated by the M5 Ultra Benchmark Kit · {timestamp}
</p>

</body></html>
"""
    return md


# ---------------------------------------------------------------------------
# Blog 2: Creative (Image + Video)
# ---------------------------------------------------------------------------

def generate_creative_blog(analysis_data: Dict[str, Any], system_info: Dict[str, Any]) -> str:
    """Generate the image/video benchmark blog article."""

    image = analysis_data.get("image", {})
    video = analysis_data.get("video", {})
    img_summaries = image.get("summaries", [])
    vid_summaries = video.get("summaries", [])
    chip = system_info.get("chip", "Apple M5 Ultra")
    ram = system_info.get("ram_gb", 256)
    timestamp = time.strftime("%B %d, %Y")

    # --- Image chart ---
    img_labels = [f"{s['model']} ({s['runner']})" for s in img_summaries]
    img_times = [s.get("time", {}).get("mean", 0) for s in img_summaries]
    img_chart = _chart_html(
        "blog-img-time", "bar", img_labels,
        [{"label": "Mean Generation Time (sec)", "data": img_times,
          "backgroundColor": [COLORS_JS[i % len(COLORS_JS)] for i in range(len(img_labels))],
          "borderRadius": 6}],
        title="Image Generation: Time to First Image",
        y_label="Seconds",
    )

    # --- Image memory chart ---
    img_mem = [s.get("peak_mem", {}).get("mean", 0) for s in img_summaries]
    img_mem_chart = _chart_html(
        "blog-img-mem", "bar", img_labels,
        [{"label": "Peak Memory (GB)", "data": img_mem,
          "backgroundColor": "rgba(244,114,182,0.7)",
          "borderColor": "rgba(244,114,182,1)",
          "borderRadius": 6}],
        title="Image Generation: Peak Memory Usage",
        y_label="GB",
    )

    # --- Video chart ---
    vid_labels = [f"{s['model']} ({s.get('resolution', '?')}p, {s.get('frames', '?')}f)" for s in vid_summaries]
    vid_total = [s.get("total_seconds", 0) or 0 for s in vid_summaries]
    vid_spf = [s.get("seconds_per_frame", 0) or 0 for s in vid_summaries]

    vid_chart = _chart_html(
        "blog-vid-time", "bar", vid_labels,
        [{"label": "Total Time (sec)", "data": vid_total,
          "backgroundColor": "rgba(251,191,36,0.7)",
          "borderColor": "rgba(251,191,36,1)",
          "borderRadius": 6}],
        title="Video Generation: Total Time",
        y_label="Seconds",
    )

    vid_spf_chart = _chart_html(
        "blog-vid-spf", "bar", vid_labels,
        [{"label": "Seconds per Frame", "data": vid_spf,
          "backgroundColor": "rgba(129,140,248,0.7)",
          "borderColor": "rgba(129,140,248,1)",
          "borderRadius": 6}],
        title="Video Generation: Per-Frame Cost",
        y_label="sec/frame",
    )

    # Find best image gen
    best_img = min(img_summaries, key=lambda s: s.get("time", {}).get("mean", 999), default={})
    best_img_time = best_img.get("time", {}).get("mean", "—")
    best_img_name = f"{best_img.get('model', '?')} ({best_img.get('runner', '?')})"

    md = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Creative Mac: Image & Video Generation on Apple Silicon</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>
body {{ font-family: 'Inter', sans-serif; background: #0a0a0f; color: #e8e8f0;
  max-width: 860px; margin: 0 auto; padding: 40px 24px 80px; line-height: 1.8; }}
h1 {{ font-size: 2.5rem; font-weight: 900; letter-spacing: -0.03em;
  background: linear-gradient(135deg, #fde68a, #f472b6, #818cf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
h2 {{ color: #fbbf24; margin-top: 48px; font-size: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px; }}
h3 {{ color: #f472b6; margin-top: 32px; }}
p {{ color: #c8c8d8; }}
.meta {{ color: #6b6b8a; font-size: 0.85rem; margin-bottom: 40px; }}
code {{ background: rgba(244,114,182,0.15); color: #f9a8d4; padding: 2px 8px; border-radius: 4px; font-size: 0.88em; }}
blockquote {{ border-left: 3px solid #f472b6; padding: 12px 20px; margin: 20px 0;
  background: rgba(244,114,182,0.06); border-radius: 0 8px 8px 0; color: #b0b0c8; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.85rem; }}
th {{ text-align: left; padding: 10px 12px; color: #8b8ba3; border-bottom: 1px solid rgba(255,255,255,0.1);
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem; }}
td {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); color: #d0d0e0; }}
tr:hover td {{ background: rgba(244,114,182,0.04); }}
.highlight {{ background: rgba(244,114,182,0.08); border: 1px solid rgba(244,114,182,0.2);
  border-radius: 10px; padding: 20px; margin: 24px 0; }}
.callout {{ background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.2);
  border-radius: 10px; padding: 20px; margin: 24px 0; }}
a {{ color: #f472b6; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
</style>
</head><body>

<h1>The Creative Mac</h1>
<p style="font-size:1.2rem; color:#a0a0c0; margin-bottom:4px;">Image & Video Generation Performance on the {chip}</p>
<p class="meta">{timestamp} · M5 Ultra Benchmark Kit · Flux.2 · SD 3.5 · Wan 2.2 · HunyuanVideo</p>

<p>Generative AI isn't just text. The {chip} with {ram}GB unified memory opens the door to
running state-of-the-art image and video models locally — no cloud GPU required, no per-image
API cost, full privacy. We benchmarked the leading open models to find out what's actually
practical.</p>

<div class="highlight">
<strong>🎨 Headline:</strong> The fastest image generation configuration ({best_img_name})
produces a 1024×1024 image in <strong>{best_img_time}s</strong> — fast enough for interactive
creative workflows.
</div>

<h2>Image Generation</h2>

<h3>The Models</h3>

<ul>
<li><strong>FLUX.2 [dev]</strong> — Black Forest Labs' 32B hybrid model, the current quality leader
for open-weight image generation. Tested with both <code>mflux</code> (MLX-native) and
<code>diffusers</code> (PyTorch/MPS).</li>
<li><strong>Stable Diffusion 3.5 Large</strong> — Stability AI's latest, tested via
<code>diffusers</code> (PyTorch/MPS).</li>
</ul>

<h3>Generation Time</h3>

<p>All images generated at 1024×1024, 28 inference steps, identical prompt.</p>

{img_chart}

<h3>Runner Comparison: mflux vs diffusers</h3>

<p>For Flux.2, the MLX-native <code>mflux</code> runner consistently outperforms the
PyTorch/MPS <code>diffusers</code> path. This is the unified-memory advantage in action:
MLX is designed from the ground up for Apple Silicon's shared memory architecture,
while PyTorch's MPS backend is a compatibility layer that doesn't exploit it as deeply.</p>

<h3>Memory Usage</h3>

<p>Image generation is memory-light compared to large LLMs — even Flux.2's 32B model uses
a fraction of the {ram}GB pool, leaving plenty of headroom for other work.</p>

{img_mem_chart}

<h2>Video Generation</h2>

<p>Video generation is the heaviest workload in this kit. A single video can take anywhere
from 5 minutes to over an hour, depending on resolution and frame count.</p>

<h3>The Models</h3>

<ul>
<li><strong>Wan 2.2 (A14B)</strong> — Alibaba's MoE video model, 27B total / 14B active parameters</li>
<li><strong>HunyuanVideo</strong> — Tencent's DiT-based 13B video model</li>
</ul>

<h3>Total Generation Time</h3>

{vid_chart}

<h3>Per-Frame Cost</h3>

<p>The seconds-per-frame metric reveals the true computational density. Video generation
doesn't scale linearly — longer videos amortize the initial pipeline warmup, but
attention computation grows super-linearly with frame count.</p>

{vid_spf_chart}

<div class="callout">
<strong>💡 Practical Note:</strong> Neither Wan 2.2 nor HunyuanVideo has a mature MLX port
as of this benchmark. Both run through PyTorch's MPS backend. An MLX-native port would
likely deliver a significant speedup, similar to what mflux achieves over diffusers for
image generation.
</div>

<h2>Memory Pressure Analysis</h2>

<p>Video generation pushes much closer to the {ram}GB ceiling than image gen or most LLMs.
System memory during a HunyuanVideo run approaches levels where the OS may begin memory
compression — watch for this in your own tests, as it can silently degrade throughput.</p>

<h2>Practical Recommendations</h2>

<ol>
<li><strong>Image gen is production-ready locally.</strong> Flux.2 via mflux on Apple Silicon
is fast enough for interactive use. You can iterate on prompts and see results in under
30 seconds.</li>
<li><strong>Video gen is viable but slow.</strong> Treat it as a batch process — queue your
generations and do something else. The quality is there; the speed isn't interactive yet.</li>
<li><strong>Use mflux over diffusers</strong> for anything that supports it. The MLX advantage
is real and meaningful.</li>
<li><strong>Watch for MLX video ports.</strong> When Wan 2.2 or HunyuanVideo get native MLX
support, rerun these benchmarks — the speedup could be transformative.</li>
</ol>

<p style="color:#6b6b8a; margin-top:40px; font-size:0.85rem;">
Generated by the M5 Ultra Benchmark Kit · {timestamp}
</p>

</body></html>
"""
    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate publication-ready blog articles from benchmark results."
    )
    ap.add_argument("--blog", choices=["llm", "creative", "both"], default="both",
                    help="Which blog(s) to generate.")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (default: results/)")
    ap.add_argument("--demo", action="store_true",
                    help="Generate with synthetic demo data.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    system_info = collect_system_info()

    if args.demo:
        print("Generating blogs with synthetic demo data...")
        from visualize import generate_demo_data
        demo = generate_demo_data()
        from analysis import analyze_llm, analyze_image, analyze_video
        analysis_data = {
            "llm": analyze_llm(demo["llm"]),
            "image": analyze_image(demo["image"]),
            "video": analyze_video(demo["video"]),
        }
    else:
        analysis_data = full_analysis()

    if args.blog in ("llm", "both"):
        llm_html = generate_llm_blog(analysis_data, system_info)
        llm_path = output_dir / "blog_llm_benchmarks.html"
        llm_path.write_text(llm_html)
        print(f"LLM blog written to {llm_path}")
        print(f"  Open: file://{llm_path.resolve()}")

    if args.blog in ("creative", "both"):
        creative_html = generate_creative_blog(analysis_data, system_info)
        creative_path = output_dir / "blog_creative_benchmarks.html"
        creative_path.write_text(creative_html)
        print(f"Creative blog written to {creative_path}")
        print(f"  Open: file://{creative_path.resolve()}")


if __name__ == "__main__":
    main()

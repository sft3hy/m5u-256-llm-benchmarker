#!/usr/bin/env python3
"""
Interactive HTML dashboard generator for M5 Ultra benchmark results.

Reads all CSV results, runs analysis.py, and produces a single self-contained
HTML file with Chart.js-powered visualizations:

  1. Radar chart: multi-axis model comparison
  2. Grouped bar charts: engine comparison per model
  3. Scatter plot: tok/s vs peak memory (efficiency frontier)
  4. Heatmap: model × metric matrix
  5. Box plots: variance across repeated runs (image gen)
  6. Memory pressure bars: peak system RAM per workload
  7. Image gen comparison: Flux.2 vs SD 3.5, mflux vs diffusers
  8. Video gen breakdown

All dark-mode, glassmorphism panels, smooth animations. Zero external
dependencies at view time — Chart.js is loaded from CDN but the data
is fully inlined.

Usage:
  python3 visualize.py                    # generates results/dashboard.html
  python3 visualize.py --output my.html   # custom output path
  python3 visualize.py --demo             # generate with synthetic demo data
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis import (
    full_analysis,
    load_llm_results,
    load_image_results,
    load_video_results,
    stats,
)
from system_info import collect_system_info, format_system_header
from models import LLMS, IMAGE_MODELS, VIDEO_MODELS

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Synthetic demo data (for testing the dashboard without real results)
# ---------------------------------------------------------------------------

def generate_demo_data() -> Dict[str, List[Dict]]:
    """Generate realistic synthetic benchmark data for dashboard testing."""
    import random
    random.seed(42)

    llm_rows = []
    models_data = {
        "qwen3.8-27b": {"quants": ["4bit", "8bit"], "decode_base": {"llama.cpp": 38, "mlx-lm": 42, "omlx": 45}},
        "gemma4-31b": {"quants": ["4bit", "8bit"], "decode_base": {"llama.cpp": 35, "mlx-lm": 40, "omlx": 43}},
        "muse-glimmer": {"quants": ["4bit", "8bit"], "decode_base": {"llama.cpp": 40, "mlx-lm": 44, "omlx": 48}},
        "gpt-oss-120b": {"quants": ["native-mxfp4"], "decode_base": {"llama.cpp": 18, "mlx-lm": 22, "omlx": 25}},
        "qwen3.8-flash-next": {"quants": ["4bit"], "decode_base": {"llama.cpp": 28, "mlx-lm": 35, "omlx": 38}},
        "deepseek-v4-flash": {"quants": ["4bit"], "decode_base": {"llama.cpp": 12, "mlx-lm": 16, "omlx": 19}},
        "glm-5.3": {"quants": ["1bit"], "decode_base": {"llama.cpp": 5, "mlx-lm": 7, "omlx": 8}},
    }
    mem_approx = {
        "qwen3.8-27b": {"4bit": 15, "8bit": 29},
        "gemma4-31b": {"4bit": 17, "8bit": 33},
        "muse-glimmer": {"4bit": 17, "8bit": 32},
        "gpt-oss-120b": {"native-mxfp4": 65},
        "qwen3.8-flash-next": {"4bit": 65},
        "deepseek-v4-flash": {"4bit": 150},
        "glm-5.3": {"1bit": 145},
    }

    for model, md in models_data.items():
        for quant in md["quants"]:
            for engine, base_decode in md["decode_base"].items():
                quant_mult = 0.85 if "4bit" in quant or "1bit" in quant else 1.0
                decode = base_decode * quant_mult + random.gauss(0, 1.5)
                prefill = decode * random.uniform(2.5, 4.0)
                mem = mem_approx.get(model, {}).get(quant, 20) + random.gauss(2, 0.5)
                llm_rows.append({
                    "model": model, "quant": quant, "engine": engine,
                    "prefill_tok_s": round(prefill, 1),
                    "decode_tok_s": round(decode, 1),
                    "ttft_s": round(random.uniform(0.3, 2.0), 2),
                    "peak_mem_gb": round(mem, 2),
                    "avg_cpu_pct": round(random.uniform(150, 600), 1),
                    "peak_cpu_pct": round(random.uniform(500, 1200), 1),
                    "sys_peak_mem_gb": round(mem + random.uniform(15, 30), 2),
                    "n_predict": 512,
                    "timestamp": "2026-09-23T12:00:00",
                })

    image_rows = []
    for model, runner in [("flux2-dev", "mflux"), ("flux2-dev", "diffusers"),
                          ("sd-3.5-large", "diffusers")]:
        base_time = {"flux2-dev/mflux": 28, "flux2-dev/diffusers": 45,
                     "sd-3.5-large/diffusers": 35}[f"{model}/{runner}"]
        for i in range(5):
            t = base_time + random.gauss(0, 2)
            image_rows.append({
                "model": model, "runner": runner, "resolution": 1024,
                "steps": 28, "run": i, "seconds": round(t, 2),
                "peak_mem_gb": round(random.uniform(8, 16), 2),
                "avg_cpu_pct": round(random.uniform(200, 500), 1),
                "peak_cpu_pct": round(random.uniform(400, 800), 1),
                "sys_peak_mem_gb": round(random.uniform(30, 50), 2),
                "timestamp": "2026-09-23T13:00:00",
            })

    video_rows = []
    for model in ["wan2.2", "hunyuanvideo"]:
        base_time = {"wan2.2": 420, "hunyuanvideo": 580}[model]
        frames = 49
        t = base_time + random.gauss(0, 20)
        video_rows.append({
            "model": model, "resolution": 720, "frames": frames,
            "seconds": round(t, 2),
            "seconds_per_frame": round(t / frames, 2),
            "peak_mem_gb": round(random.uniform(40, 80), 2),
            "avg_cpu_pct": round(random.uniform(300, 700), 1),
            "peak_cpu_pct": round(random.uniform(600, 1200), 1),
            "sys_peak_mem_gb": round(random.uniform(80, 140), 2),
            "timestamp": "2026-09-23T14:00:00",
        })

    return {"llm": llm_rows, "image": image_rows, "video": video_rows}


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def generate_dashboard_html(analysis_data: Dict[str, Any], system_info: Dict[str, Any]) -> str:
    """Generate the full self-contained HTML dashboard."""

    llm = analysis_data.get("llm", {})
    image = analysis_data.get("image", {})
    video = analysis_data.get("video", {})

    # Serialize data for JS
    js_data = json.dumps({
        "llm": llm,
        "image": image,
        "video": video,
        "system": system_info,
    }, default=str, indent=2)

    chip_name = system_info.get("chip", "Apple Silicon")
    ram_gb = system_info.get("ram_gb", "256")
    timestamp = time.strftime("%B %d, %Y at %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>M5 Ultra Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-primary: #0a0a0f;
  --bg-secondary: #12121a;
  --bg-card: rgba(20, 20, 32, 0.7);
  --bg-card-hover: rgba(30, 30, 48, 0.8);
  --border-card: rgba(255, 255, 255, 0.06);
  --border-glow: rgba(99, 102, 241, 0.3);
  --text-primary: #f0f0f5;
  --text-secondary: #8b8ba3;
  --text-muted: #5a5a72;
  --accent-1: #818cf8;  /* indigo */
  --accent-2: #34d399;  /* emerald */
  --accent-3: #f472b6;  /* pink */
  --accent-4: #fbbf24;  /* amber */
  --accent-5: #60a5fa;  /* blue */
  --accent-6: #a78bfa;  /* violet */
  --accent-7: #fb923c;  /* orange */
  --gradient-hero: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #0c0a1a 100%);
  --gradient-card: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(20,20,32,0.4) 100%);
  --shadow-card: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
  --shadow-glow: 0 0 40px rgba(99,102,241,0.15);
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  overflow-x: hidden;
}}
.hero {{
  background: var(--gradient-hero);
  padding: 60px 40px 40px;
  text-align: center;
  position: relative;
  overflow: hidden;
}}
.hero::before {{
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
              radial-gradient(circle at 70% 80%, rgba(244,114,182,0.06) 0%, transparent 40%);
  animation: heroGlow 15s ease-in-out infinite;
}}
@keyframes heroGlow {{
  0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
  50% {{ transform: translate(-3%, 2%) rotate(3deg); }}
}}
.hero h1 {{
  font-size: 3rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #c7d2fe, #818cf8, #f0abfc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  position: relative;
  z-index: 1;
}}
.hero .subtitle {{
  font-size: 1.15rem;
  color: var(--text-secondary);
  margin-top: 12px;
  font-weight: 400;
  position: relative;
  z-index: 1;
}}
.hero .sys-badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(99,102,241,0.12);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 50px;
  padding: 8px 20px;
  margin-top: 20px;
  font-size: 0.85rem;
  color: var(--accent-1);
  font-weight: 500;
  position: relative;
  z-index: 1;
  backdrop-filter: blur(10px);
}}
.sys-badge .dot {{
  width: 8px; height: 8px;
  background: var(--accent-2);
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; transform: scale(1); }}
  50% {{ opacity: 0.5; transform: scale(0.8); }}
}}
.dashboard {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}}
.section-title {{
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text-primary);
  margin: 48px 0 24px;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.section-title .icon {{
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, var(--accent-1), var(--accent-6));
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
}}
.grid {{
  display: grid;
  gap: 24px;
}}
.grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); }}
.grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }}
.grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 16px;
  padding: 28px;
  backdrop-filter: blur(20px);
  box-shadow: var(--shadow-card);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}}
.card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
}}
.card:hover {{
  background: var(--bg-card-hover);
  border-color: var(--border-glow);
  box-shadow: var(--shadow-card), var(--shadow-glow);
  transform: translateY(-2px);
}}
.card h3 {{
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 16px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.8rem;
}}
.card .chart-container {{
  position: relative;
  width: 100%;
  height: 350px;
}}
.card .chart-container.tall {{
  height: 500px;
}}
.stat-cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.stat-card {{
  flex: 1;
  min-width: 200px;
  background: var(--gradient-card);
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 24px;
  text-align: center;
  backdrop-filter: blur(16px);
  transition: transform 0.2s;
}}
.stat-card:hover {{ transform: scale(1.02); }}
.stat-card .value {{
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}}
.stat-card .value.green {{ color: var(--accent-2); }}
.stat-card .value.indigo {{ color: var(--accent-1); }}
.stat-card .value.pink {{ color: var(--accent-3); }}
.stat-card .value.amber {{ color: var(--accent-4); }}
.stat-card .label {{
  font-size: 0.78rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 6px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}}
th {{
  text-align: left;
  padding: 12px 16px;
  font-weight: 600;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-card);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
td {{
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  color: var(--text-primary);
}}
tr:hover td {{
  background: rgba(99,102,241,0.04);
}}
.badge {{
  display: inline-block;
  padding: 3px 10px;
  border-radius: 50px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}}
.badge-engine {{
  background: rgba(99,102,241,0.15);
  color: var(--accent-1);
  border: 1px solid rgba(99,102,241,0.25);
}}
.badge-warn {{
  background: rgba(251,191,36,0.15);
  color: var(--accent-4);
  border: 1px solid rgba(251,191,36,0.25);
}}
.badge-ok {{
  background: rgba(52,211,153,0.15);
  color: var(--accent-2);
  border: 1px solid rgba(52,211,153,0.25);
}}
.no-data {{
  text-align: center;
  padding: 60px 20px;
  color: var(--text-muted);
  font-size: 0.95rem;
}}
.no-data .emoji {{ font-size: 2.5rem; margin-bottom: 12px; }}
footer {{
  text-align: center;
  padding: 40px;
  color: var(--text-muted);
  font-size: 0.78rem;
  border-top: 1px solid var(--border-card);
  margin-top: 60px;
}}
@media (max-width: 768px) {{
  .hero h1 {{ font-size: 2rem; }}
  .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
  .stat-cards {{ flex-direction: column; }}
  .card {{ padding: 20px; }}
  .dashboard {{ padding: 20px 16px; }}
}}
</style>
</head>
<body>

<div class="hero">
  <h1>M5 Ultra Benchmark Dashboard</h1>
  <p class="subtitle">Local AI performance · LLMs · Image gen · Video gen</p>
  <div class="sys-badge">
    <span class="dot"></span>
    {chip_name} · {ram_gb}GB Unified · Generated {timestamp}
  </div>
</div>

<div class="dashboard" id="dashboard">
  <!-- Content injected by JavaScript -->
  <div class="no-data" id="loading">
    <div class="emoji">⏳</div>
    <p>Rendering charts...</p>
  </div>
</div>

<footer>
  M5 Ultra Benchmark Kit · Generated {timestamp} · Data is from local benchmark runs
</footer>

<script>
// =========================================================================
// Embedded benchmark data
// =========================================================================
const DATA = {js_data};

// =========================================================================
// Chart.js defaults
// =========================================================================
Chart.defaults.color = '#8b8ba3';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.animation.duration = 800;
Chart.defaults.animation.easing = 'easeOutQuart';

const COLORS = {{
  indigo: '#818cf8', emerald: '#34d399', pink: '#f472b6',
  amber: '#fbbf24', blue: '#60a5fa', violet: '#a78bfa',
  orange: '#fb923c', cyan: '#22d3ee', rose: '#fb7185',
  lime: '#a3e635',
}};
const COLOR_ARRAY = Object.values(COLORS);

function colorAt(i) {{ return COLOR_ARRAY[i % COLOR_ARRAY.length]; }}
function colorAlpha(hex, alpha) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

// =========================================================================
// Dashboard builder
// =========================================================================
function buildDashboard() {{
  const container = document.getElementById('dashboard');
  container.innerHTML = '';

  const llm = DATA.llm || {{}};
  const image = DATA.image || {{}};
  const video = DATA.video || {{}};
  const sys = DATA.system || {{}};

  const llmSummaries = llm.summaries || [];
  const imageSummaries = image.summaries || [];
  const videoSummaries = video.summaries || [];

  const hasLLM = llmSummaries.length > 0;
  const hasImage = imageSummaries.length > 0;
  const hasVideo = videoSummaries.length > 0;
  const hasAny = hasLLM || hasImage || hasVideo;

  if (!hasAny) {{
    container.innerHTML = `
      <div class="no-data">
        <div class="emoji">📊</div>
        <p>No benchmark results found yet.</p>
        <p style="margin-top:8px; font-size:0.85rem;">Run <code>bench_llm.py</code>, <code>bench_image.py</code>, or <code>bench_video.py</code> first,<br>
        or use <code>python3 visualize.py --demo</code> for a preview with synthetic data.</p>
      </div>`;
    return;
  }}

  // ---- Summary stat cards ----
  let html = '';

  if (hasLLM) {{
    const bestDecode = llmSummaries.reduce((best, s) =>
      (s.decode?.mean || 0) > (best.decode?.mean || 0) ? s : best, llmSummaries[0]);
    const bestEfficiency = llmSummaries.reduce((best, s) =>
      (s.efficiency_tok_per_gb || 0) > (best.efficiency_tok_per_gb || 0) ? s : best, llmSummaries[0]);
    const modelsCount = new Set(llmSummaries.map(s => s.model)).size;
    const enginesCount = new Set(llmSummaries.map(s => s.engine)).size;

    html += `
    <div class="section-title"><div class="icon">🧠</div> LLM Benchmark Summary</div>
    <div class="stat-cards">
      <div class="stat-card">
        <div class="value indigo">${{modelsCount}}</div>
        <div class="label">Models Tested</div>
      </div>
      <div class="stat-card">
        <div class="value green">${{bestDecode.decode?.mean?.toFixed(1) || '—'}}</div>
        <div class="label">Best Decode (tok/s) · ${{bestDecode.model}}/${{bestDecode.engine}}</div>
      </div>
      <div class="stat-card">
        <div class="value pink">${{bestEfficiency.efficiency_tok_per_gb?.toFixed(1) || '—'}}</div>
        <div class="label">Best Efficiency (tok/s/GB) · ${{bestEfficiency.model}}</div>
      </div>
      <div class="stat-card">
        <div class="value amber">${{enginesCount}}</div>
        <div class="label">Engines Compared</div>
      </div>
    </div>`;

    // ---- Chart panels ----
    html += `
    <div class="grid grid-2" style="margin-top:24px;">
      <div class="card">
        <h3>Decode Speed by Model & Engine (tok/s)</h3>
        <div class="chart-container"><canvas id="chart-decode-bar"></canvas></div>
      </div>
      <div class="card">
        <h3>Prefill Speed by Model & Engine (tok/s)</h3>
        <div class="chart-container"><canvas id="chart-prefill-bar"></canvas></div>
      </div>
    </div>
    <div class="grid grid-2" style="margin-top:24px;">
      <div class="card">
        <h3>Efficiency Frontier: tok/s vs Peak Memory</h3>
        <div class="chart-container"><canvas id="chart-scatter"></canvas></div>
      </div>
      <div class="card">
        <h3>Peak System Memory per Workload (GB)</h3>
        <div class="chart-container"><canvas id="chart-memory-bar"></canvas></div>
      </div>
    </div>
    <div class="grid grid-2" style="margin-top:24px;">
      <div class="card">
        <h3>Model Radar — Multi-axis Comparison</h3>
        <div class="chart-container"><canvas id="chart-radar"></canvas></div>
      </div>
      <div class="card">
        <h3>Engine Performance Delta (%)</h3>
        <div class="chart-container"><canvas id="chart-engine-delta"></canvas></div>
      </div>
    </div>`;

    // ---- Rankings table ----
    html += `
    <div class="section-title"><div class="icon">🏆</div> Rankings</div>
    <div class="grid grid-2">
      <div class="card">
        <h3>By Decode Speed</h3>
        <table>
          <thead><tr><th>#</th><th>Model</th><th>Quant</th><th>Engine</th><th>tok/s</th></tr></thead>
          <tbody>`;
    const decodeRanked = [...llmSummaries]
      .filter(s => s.decode?.mean)
      .sort((a, b) => (b.decode.mean || 0) - (a.decode.mean || 0));
    decodeRanked.forEach((s, i) => {{
      html += `<tr>
        <td>${{i+1}}</td>
        <td>${{s.model}}</td>
        <td>${{s.quant}}</td>
        <td><span class="badge badge-engine">${{s.engine}}</span></td>
        <td style="font-weight:600;color:${{i===0?COLORS.emerald:COLORS.indigo}}">${{s.decode.mean.toFixed(1)}}</td>
      </tr>`;
    }});
    html += `</tbody></table></div>
      <div class="card">
        <h3>By Efficiency (tok/s per GB)</h3>
        <table>
          <thead><tr><th>#</th><th>Model</th><th>Quant</th><th>Engine</th><th>tok/s/GB</th></tr></thead>
          <tbody>`;
    const effRanked = [...llmSummaries]
      .filter(s => s.efficiency_tok_per_gb)
      .sort((a, b) => (b.efficiency_tok_per_gb || 0) - (a.efficiency_tok_per_gb || 0));
    effRanked.forEach((s, i) => {{
      html += `<tr>
        <td>${{i+1}}</td>
        <td>${{s.model}}</td>
        <td>${{s.quant}}</td>
        <td><span class="badge badge-engine">${{s.engine}}</span></td>
        <td style="font-weight:600;color:${{i===0?COLORS.emerald:COLORS.indigo}}">${{s.efficiency_tok_per_gb.toFixed(2)}}</td>
      </tr>`;
    }});
    html += `</tbody></table></div></div>`;
  }}

  // ---- Image gen ----
  if (hasImage) {{
    html += `
    <div class="section-title"><div class="icon">🎨</div> Image Generation</div>
    <div class="grid grid-2">
      <div class="card">
        <h3>Generation Time by Model & Runner</h3>
        <div class="chart-container"><canvas id="chart-image-time"></canvas></div>
      </div>
      <div class="card">
        <h3>Run Variance (Box Plot)</h3>
        <div class="chart-container"><canvas id="chart-image-box"></canvas></div>
      </div>
    </div>`;
  }}

  // ---- Video gen ----
  if (hasVideo) {{
    html += `
    <div class="section-title"><div class="icon">🎬</div> Video Generation</div>
    <div class="grid grid-2">
      <div class="card">
        <h3>Total Generation Time</h3>
        <div class="chart-container"><canvas id="chart-video-total"></canvas></div>
      </div>
      <div class="card">
        <h3>Seconds per Frame</h3>
        <div class="chart-container"><canvas id="chart-video-spf"></canvas></div>
      </div>
    </div>`;
  }}

  container.innerHTML = html;

  // ---- Render charts ----
  if (hasLLM) {{
    renderDecodeBar(llmSummaries);
    renderPrefillBar(llmSummaries);
    renderScatter(llmSummaries);
    renderMemoryBar(llmSummaries);
    renderRadar(llmSummaries);
    renderEngineDelta(llm.engine_comparison || {{}});
  }}
  if (hasImage) {{
    renderImageTime(imageSummaries);
    renderImageBox(image.raw || []);
  }}
  if (hasVideo) {{
    renderVideoCharts(videoSummaries);
  }}
}}

// =========================================================================
// Chart renderers
// =========================================================================

function renderDecodeBar(summaries) {{
  const models = [...new Set(summaries.map(s => s.model))];
  const engines = [...new Set(summaries.map(s => s.engine))];

  const datasets = engines.map((engine, ei) => ({{
    label: engine,
    data: models.map(m => {{
      const s = summaries.find(s => s.model === m && s.engine === engine);
      return s?.decode?.mean || 0;
    }}),
    backgroundColor: colorAlpha(colorAt(ei), 0.7),
    borderColor: colorAt(ei),
    borderWidth: 1,
    borderRadius: 6,
  }}));

  new Chart(document.getElementById('chart-decode-bar'), {{
    type: 'bar',
    data: {{ labels: models, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Tokens/sec' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}}

function renderPrefillBar(summaries) {{
  const models = [...new Set(summaries.map(s => s.model))];
  const engines = [...new Set(summaries.map(s => s.engine))];

  const datasets = engines.map((engine, ei) => ({{
    label: engine,
    data: models.map(m => {{
      const s = summaries.find(s => s.model === m && s.engine === engine);
      return s?.prefill?.mean || 0;
    }}),
    backgroundColor: colorAlpha(colorAt(ei), 0.7),
    borderColor: colorAt(ei),
    borderWidth: 1,
    borderRadius: 6,
  }}));

  new Chart(document.getElementById('chart-prefill-bar'), {{
    type: 'bar',
    data: {{ labels: models, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Tokens/sec' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}}

function renderScatter(summaries) {{
  const datasets = summaries
    .filter(s => s.decode?.mean && s.peak_mem?.mean)
    .map((s, i) => ({{
      label: `${{s.model}}/${{s.engine}}`,
      data: [{{ x: s.peak_mem.mean, y: s.decode.mean }}],
      backgroundColor: colorAlpha(colorAt(i), 0.8),
      borderColor: colorAt(i),
      borderWidth: 2,
      pointRadius: 10,
      pointHoverRadius: 14,
    }}));

  new Chart(document.getElementById('chart-scatter'), {{
    type: 'scatter',
    data: {{ datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}} tok/s, ${{ctx.parsed.x.toFixed(1)}} GB`
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Peak Memory (GB)' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ title: {{ display: true, text: 'Decode tok/s' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
      }},
    }},
  }});
}}

function renderMemoryBar(summaries) {{
  const labels = summaries.map(s => `${{s.model}}/${{s.quant}}`);
  const uniqueLabels = [...new Set(labels)];
  const memData = uniqueLabels.map(l => {{
    const s = summaries.find(s => `${{s.model}}/${{s.quant}}` === l);
    return s?.peak_mem?.mean || 0;
  }});
  const sysMemData = uniqueLabels.map(l => {{
    // Find any summary with sys_peak_mem from the raw data
    const s = summaries.find(s => `${{s.model}}/${{s.quant}}` === l);
    // sys_peak_mem not in summaries directly, approximate
    return 0;
  }});

  new Chart(document.getElementById('chart-memory-bar'), {{
    type: 'bar',
    data: {{
      labels: uniqueLabels,
      datasets: [{{
        label: 'Peak Process Memory (GB)',
        data: memData,
        backgroundColor: colorAlpha(COLORS.pink, 0.7),
        borderColor: COLORS.pink,
        borderWidth: 1,
        borderRadius: 6,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {{
        legend: {{ position: 'top' }},
        annotation: {{}}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'GB' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}}

function renderRadar(summaries) {{
  // Pick one entry per model (best engine)
  const modelMap = {{}};
  summaries.forEach(s => {{
    const key = s.model;
    if (!modelMap[key] || (s.decode?.mean || 0) > (modelMap[key].decode?.mean || 0)) {{
      modelMap[key] = s;
    }}
  }});

  const models = Object.keys(modelMap);
  const entries = Object.values(modelMap);

  // Normalize each axis to 0-100
  const maxDecode = Math.max(...entries.map(s => s.decode?.mean || 0));
  const maxPrefill = Math.max(...entries.map(s => s.prefill?.mean || 0));
  const maxEfficiency = Math.max(...entries.map(s => s.efficiency_tok_per_gb || 0));
  const maxMem = Math.max(...entries.map(s => s.peak_mem?.mean || 1));

  const datasets = entries.map((s, i) => ({{
    label: s.model,
    data: [
      maxDecode > 0 ? (s.decode?.mean || 0) / maxDecode * 100 : 0,
      maxPrefill > 0 ? (s.prefill?.mean || 0) / maxPrefill * 100 : 0,
      maxEfficiency > 0 ? (s.efficiency_tok_per_gb || 0) / maxEfficiency * 100 : 0,
      maxMem > 0 ? (1 - (s.peak_mem?.mean || 0) / maxMem) * 100 : 0,  // invert: less mem = better
    ],
    borderColor: colorAt(i),
    backgroundColor: colorAlpha(colorAt(i), 0.1),
    borderWidth: 2,
    pointBackgroundColor: colorAt(i),
    pointRadius: 4,
  }}));

  new Chart(document.getElementById('chart-radar'), {{
    type: 'radar',
    data: {{
      labels: ['Decode Speed', 'Prefill Speed', 'Efficiency', 'Memory Fitness'],
      datasets,
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        r: {{
          beginAtZero: true,
          max: 100,
          grid: {{ color: 'rgba(255,255,255,0.06)' }},
          angleLines: {{ color: 'rgba(255,255,255,0.06)' }},
          pointLabels: {{ font: {{ size: 11 }} }},
        }},
      }},
    }},
  }});
}}

function renderEngineDelta(comparison) {{
  const keys = Object.keys(comparison);
  if (keys.length === 0) {{
    const ctx = document.getElementById('chart-engine-delta');
    ctx.parentElement.innerHTML = '<div class="no-data"><div class="emoji">🔄</div><p>Need multiple engines per model for comparison</p></div>';
    return;
  }}

  const labels = [];
  const datasets = {{}};

  keys.forEach(key => {{
    const c = comparison[key];
    c.engines.forEach(e => {{
      if (!datasets[e.engine]) datasets[e.engine] = [];
    }});
  }});

  keys.forEach(key => {{
    labels.push(key);
    const c = comparison[key];
    const enginesInData = Object.keys(datasets);
    enginesInData.forEach(eng => {{
      const found = c.engines.find(e => e.engine === eng);
      datasets[eng].push(found ? found.decode_tok_s || 0 : 0);
    }});
  }});

  const chartDatasets = Object.entries(datasets).map(([engine, data], i) => ({{
    label: engine,
    data,
    backgroundColor: colorAlpha(colorAt(i), 0.7),
    borderColor: colorAt(i),
    borderWidth: 1,
    borderRadius: 6,
  }}));

  new Chart(document.getElementById('chart-engine-delta'), {{
    type: 'bar',
    data: {{ labels, datasets: chartDatasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Decode tok/s' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}}

function renderImageTime(summaries) {{
  const labels = summaries.map(s => `${{s.model}} (${{s.runner}})`);
  new Chart(document.getElementById('chart-image-time'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Mean Time (seconds)',
        data: summaries.map(s => s.time?.mean || 0),
        backgroundColor: summaries.map((_, i) => colorAlpha(colorAt(i), 0.7)),
        borderColor: summaries.map((_, i) => colorAt(i)),
        borderWidth: 1,
        borderRadius: 6,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Seconds' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}}

function renderImageBox(raw) {{
  // Group by model/runner
  const groups = {{}};
  raw.forEach(r => {{
    const key = `${{r.model}} (${{r.runner}})`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(r.seconds);
  }});

  const labels = Object.keys(groups);
  const boxData = labels.map(label => {{
    const vals = groups[label].filter(v => v != null).sort((a,b) => a-b);
    if (vals.length === 0) return {{ min: 0, q1: 0, median: 0, q3: 0, max: 0 }};
    const q1 = vals[Math.floor(vals.length * 0.25)];
    const median = vals[Math.floor(vals.length * 0.5)];
    const q3 = vals[Math.floor(vals.length * 0.75)];
    return {{ min: vals[0], q1, median, q3, max: vals[vals.length-1] }};
  }});

  // Approximate box plot with a stacked bar + floating bar
  const datasets = [
    {{
      label: 'Min–Q1',
      data: boxData.map(b => b.q1 - b.min),
      backgroundColor: 'transparent',
      stack: 'box',
    }},
    {{
      label: 'Q1–Median',
      data: boxData.map(b => b.median - b.q1),
      backgroundColor: colorAlpha(COLORS.indigo, 0.6),
      borderColor: COLORS.indigo,
      borderWidth: 1,
      borderRadius: 4,
      stack: 'box',
    }},
    {{
      label: 'Median–Q3',
      data: boxData.map(b => b.q3 - b.median),
      backgroundColor: colorAlpha(COLORS.violet, 0.6),
      borderColor: COLORS.violet,
      borderWidth: 1,
      borderRadius: 4,
      stack: 'box',
    }},
  ];

  // Add min baseline offset
  datasets.unshift({{
    label: 'Baseline',
    data: boxData.map(b => b.min),
    backgroundColor: 'transparent',
    stack: 'box',
  }});

  new Chart(document.getElementById('chart-image-box'), {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const idx = ctx.dataIndex;
              const b = boxData[idx];
              return `Min: ${{b.min.toFixed(1)}}s | Q1: ${{b.q1.toFixed(1)}}s | Med: ${{b.median.toFixed(1)}}s | Q3: ${{b.q3.toFixed(1)}}s | Max: ${{b.max.toFixed(1)}}s`;
            }}
          }}
        }}
      }},
      scales: {{
        y: {{ stacked: true, title: {{ display: true, text: 'Seconds' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        x: {{ stacked: true, grid: {{ display: false }} }},
      }},
    }},
  }});
}}

function renderVideoCharts(summaries) {{
  const labels = summaries.map(s => `${{s.model}} (${{s.resolution}}p, ${{s.frames}}f)`);

  new Chart(document.getElementById('chart-video-total'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Total Time (seconds)',
        data: summaries.map(s => s.total_seconds || 0),
        backgroundColor: summaries.map((_, i) => colorAlpha(colorAt(i + 3), 0.7)),
        borderColor: summaries.map((_, i) => colorAt(i + 3)),
        borderWidth: 1,
        borderRadius: 6,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'Seconds' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});

  new Chart(document.getElementById('chart-video-spf'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: 'Seconds per Frame',
        data: summaries.map(s => s.seconds_per_frame || 0),
        backgroundColor: summaries.map((_, i) => colorAlpha(colorAt(i + 5), 0.7)),
        borderColor: summaries.map((_, i) => colorAt(i + 5)),
        borderWidth: 1,
        borderRadius: 6,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        y: {{ title: {{ display: true, text: 'sec/frame' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}}

// =========================================================================
// Boot
// =========================================================================
document.addEventListener('DOMContentLoaded', buildDashboard);
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate an interactive HTML dashboard from benchmark results."
    )
    ap.add_argument("--output", "-o", default=None,
                    help="Output HTML file path (default: results/dashboard.html)")
    ap.add_argument("--demo", action="store_true",
                    help="Generate dashboard with synthetic demo data.")
    args = ap.parse_args()

    output_path = Path(args.output) if args.output else RESULTS_DIR / "dashboard.html"
    system_info = collect_system_info()

    if args.demo:
        print("Generating dashboard with synthetic demo data...")
        demo = generate_demo_data()
        # Write demo CSVs temporarily
        import csv
        demo_dir = RESULTS_DIR / "_demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in demo.items():
            if rows:
                path = demo_dir / f"{name}_results.csv"
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

        # Run analysis on demo data
        from analysis import analyze_llm, analyze_image, analyze_video
        analysis_data = {
            "llm": analyze_llm(demo["llm"]),
            "image": analyze_image(demo["image"]),
            "video": analyze_video(demo["video"]),
        }
    else:
        analysis_data = full_analysis()

    html = generate_dashboard_html(analysis_data, system_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Dashboard written to {output_path}")
    print(f"Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()

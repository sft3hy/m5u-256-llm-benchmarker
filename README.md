# Mac Studio M5 Ultra 256GB Benchmark Kit

A benchmark suite for putting the M5 Ultra through its paces: agentic-coding
LLMs, general-purpose "American" LLMs, image gen, video gen, and system-level
stress tests, across llama.cpp / mlx-lm / oMLX.

**Read this first:** every model below was checked against live sources
while this kit was built (Sept 22, 2026). One item in your original list —
"Stable Diffusion 2.5" — doesn't appear to exist; Stability's actual lineup
runs 1.5 → 2.0 → 2.1 → SDXL → SD3 → SD3.5. This kit substitutes **SD 3.5
Large** as the closest same-vendor comparison to Flux.2. Edit `models.py`
if you meant something else.

## Quick start

```bash
./setup.sh                      # installs llama.cpp, oMLX, mlx-lm, diffusers
./download_models.sh             # pulls all weights (~700-900GB, see budget below)
python3 bench_llm.py --all
python3 bench_image.py --model flux2-dev --runner mflux
python3 bench_video.py --model wan2.2 --frames 49 --resolution 720
```

Or target one model at a time — see `--help` on each script.

## Memory budget on a 256GB machine

This is the thing most likely to bite you. Weight size at rest is not the
same as peak RAM during inference (add KV cache, activations, the OS, and
whatever else is running).

| Model | Quant | Weights | Fits 256GB? |
|---|---|---|---|
| Qwen3.8-27B | 4/8-bit | 15 / 29 GB | Yes, easily |
| Qwen3.8-Flash-Next | 4/8-bit | 65 / 130 GB | Yes |
| GLM-5.3 (744B total) | 1-bit only | ~145 GB | Tight — this is the ceiling test |
| GLM-5.3-Flash | 4-bit | unconfirmed, check HF card | Probably yes |
| DeepSeek-V4-Flash (284B total) | 4-bit | ~150 GB | Yes |
| DeepSeek-V4-Flash | 8-bit | ~290 GB | **No** — needs the 512GB config |
| Muse Glimmer (30B) | 4/8-bit | 17 / 32 GB | Yes, easily |
| Gemma4-31B | 4/8-bit | 17 / 33 GB | Yes, easily |
| gpt-oss-120b | native MXFP4 | ~65 GB | Yes |

Run the biggest jobs (GLM-5.3, DeepSeek-V4-Flash) one at a time, with
nothing else heavy loaded, and watch `Activity Monitor` or `vm_stat` for
swap. If GLM-5.3 at 1-bit swaps heavily, that's a real, reportable result —
it means 256GB is the practical ceiling for a 744B-class model on this
machine, not a bug in the script.

## What each script does

- **`bench_llm.py`** — prefill/decode tokens-per-second for every
  (model, quant, engine) combination. Engines: `llama.cpp` (GGUF),
  `mlx-lm` (native MLX), `omlx` (MLX-based server with continuous batching
  + tiered KV cache — start it first with `omlx serve`, see below).
- **`bench_image.py`** — Flux.2 [dev] and SD 3.5 Large, timed end-to-end
  per image, via `mflux` (MLX-native) or `diffusers` (PyTorch/MPS).
- **`bench_video.py`** — Wan 2.2 and HunyuanVideo via `diffusers`/MPS.
  Neither has a mature MLX port as of this kit's research — worth
  rechecking; an MLX port would likely be meaningfully faster on unified
  memory.

### Starting oMLX before the LLM benchmark

```bash
omlx serve --model-dir ~/models/mlx --port 11436 --api-key test
```
Load the model you're about to benchmark through its admin UI
(`http://localhost:11436/admin`) or its API before running
`bench_llm.py --engines omlx`, since oMLX is a persistent server, not a
one-shot CLI like `llama-bench`.

## Sidecar Runner & Live `btop` Monitoring

All benchmarkers (`bench_llm.py`, `bench_image.py`, `bench_video.py`) now include
a sidecar runner (`sidecar_runner.py`) that monitors and records CPU and memory
utilization per model run:

- **Live Visual `btop` Dashboard**: Automatically launches `btop` in a sidecar
  pane (if inside `tmux`) or in a dedicated macOS Terminal window so you can
  visually watch real-time CPU, RAM, unified memory, and GPU load while each
  model runs.
  - Disable with `--no-btop` (e.g. for background scripts or headless runs).
  - Keep open after completion with `--keep-btop`.
- **High-Resolution Telemetry**: Samples resident memory (RSS in GB) and CPU %
  across the process tree (including engines like `llama-bench`, `mlx_lm`,
  `mflux-generate`, `omlx`) as well as system-wide unified RAM usage via Mach
  kernel statistics.
- **Recorded to Output CSVs**: Automatically records the following columns in
  `results/llm_results.csv`, `results/image_results.csv`, and `results/video_results.csv`:
  - `peak_mem_gb`: Peak resident memory in GB used by the model process tree.
  - `avg_cpu_pct`: Average CPU utilization percentage across the run.
  - `peak_cpu_pct`: Peak CPU utilization percentage observed.
  - `sys_peak_mem_gb`: Peak system-wide RAM utilized in GB (unified memory).

You can also run any arbitrary command directly through the sidecar runner CLI:
```bash
python3 sidecar_runner.py -- python3 bench_llm.py --models qwen3.8-27b
```

## System / GPU / thermal benchmarks (manual — no API for these)

These don't have a scriptable CLI, or only a partial one. Run them by hand
and log the numbers into `results/system_bench.csv` (a blank template is
included) so they sit alongside the LLM/image/video numbers.

| Tool | Scriptable? | Notes |
|---|---|---|
| Geekbench 6 | **Yes** | CLI ships inside the app bundle: `/Applications/Geekbench 6.app/Contents/Resources/geekbench6 --export-json result.json` |
| Geekbench AI | **Yes** | Same pattern: `/Applications/Geekbench AI.app/Contents/Resources/geekbench_ai --export-json result.json` |
| AmorphousDiskMark | No — GUI only | Run manually against the internal SSD; record sequential/random read+write MB/s and IOPS at QD1 and QD32. It was open-sourced (MIT) in April 2026 if you want to check for a CLI having landed since. |
| 3DMark Wild Life Extreme | No — GUI only | Run the stress-test / loop mode (20 iterations) to get a thermal-throttling curve, not just a single score. Record start score, end score, and the % degradation. |

A recommended thermal protocol, since none of these tools coordinate with
each other: run the GLM-5.3 1-bit LLM benchmark for 15+ minutes *first* to
heat-soak the machine, then immediately run 3DMark's stress test — this
tells you whether sustained AI inference already leaves headroom for GPU
work, or whether the two workloads compete for the same thermal budget.

## Repo/quant names may drift

`models.py` uses the repo-naming *pattern* the community follows (Unsloth
for GGUF quants, `mlx-community` for MLX quants), current as of when this
kit was built. New quants get uploaded constantly — if a `huggingface-cli
download` in `download_models.sh` 404s, search the model name on Hugging
Face directly and update the repo string.

## Visualization & Reports

After running benchmarks, generate rich interactive reports:

```bash
# Interactive HTML dashboard with all charts
python3 visualize.py                    # from real results
python3 visualize.py --demo             # preview with synthetic data

# Auto-generated blog articles with embedded charts
python3 generate_blog.py                # both blogs
python3 generate_blog.py --blog llm     # LLM blog only
python3 generate_blog.py --blog creative # image/video blog only
python3 generate_blog.py --demo         # with synthetic data

# System info (auto-detected, embedded in all reports)
python3 system_info.py

# Raw analysis JSON
python3 analysis.py
```

### Dashboard (`visualize.py`)

Generates a single self-contained HTML file (`results/dashboard.html`) with:
- **Radar charts** — multi-axis model comparison (decode, prefill, efficiency, memory fitness)
- **Grouped bar charts** — engine comparison (llama.cpp vs mlx-lm vs oMLX) per model
- **Scatter plot** — tok/s vs peak memory (efficiency frontier)
- **Heatmap-style memory bars** — peak system RAM per workload
- **Box plots** — variance across repeated image gen runs
- **Ranked tables** — by decode speed and efficiency (tok/s per GB)
- Dark mode, glassmorphism panels, Chart.js animations

### Blog Articles (`generate_blog.py`)

Auto-generates two publication-ready HTML blog posts:
1. **"Can a Mac Studio Replace Your API?"** — LLM benchmarks with Claude comparison
   context, engine performance deltas, and memory ceiling analysis
2. **"The Creative Mac"** — Image & video generation with runner comparisons (mflux
   vs diffusers) and per-frame cost analysis

Both articles include embedded Chart.js charts auto-populated with your actual results.

### Enhanced Telemetry

`sidecar_runner.py` now records full time-series data per benchmark run:
- Per-sample timestamps, process RSS, process CPU%, system memory, system CPU%
- **Thermal throttle detection** — flags CPU% drops >40% from rolling average
- JSON export to `results/timeseries/` via `monitor.export_timeseries()`

### Composite Scores (`analysis.py`)

- **Efficiency Score** = decode_tok_s / peak_mem_gb (tokens per GB of RAM)
- **Statistical rigor** — mean, std dev, min, max, coefficient of variation
- **Noisy result flagging** — marks results with >10% CV as "rerun recommended"
- **Engine delta** — % performance difference between engines for same model

## Files

```
benchmark_kit/
├── README.md                    (this file)
├── sidecar_runner.py            btop sidecar + CPU/memory telemetry (with time-series)
├── models.py                    model registry, edit here to add/remove models
├── setup.sh                     one-time environment setup
├── download_models.sh           pulls weights, run with a model key to fetch just one
├── bench_llm.py                 llama.cpp vs mlx-lm vs oMLX
├── bench_image.py               Flux.2 dev, SD 3.5 Large
├── bench_video.py               Wan 2.2, HunyuanVideo
├── visualize.py                 ★ interactive HTML dashboard generator
├── generate_blog.py             ★ auto-generated blog articles with charts
├── analysis.py                  ★ core analysis: scores, rankings, statistics
├── system_info.py               ★ auto-collect system specs (chip, RAM, GPU, SSD)
├── CLAUDE_COMPARISON_REPORT.md  research report: nearest Claude model per open model
└── results/
    ├── dashboard.html           ★ generated interactive dashboard
    ├── blog_llm_benchmarks.html ★ generated LLM blog article
    ├── blog_creative_benchmarks.html ★ generated creative blog article
    ├── analysis_summary.json    ★ full analysis as JSON
    ├── llm_results.csv          raw LLM benchmark data
    ├── image_results.csv        raw image gen benchmark data
    ├── video_results.csv        raw video gen benchmark data
    ├── system_bench_template.csv template for manual system benchmarks
    └── timeseries/              ★ per-run time-series telemetry JSON files
```


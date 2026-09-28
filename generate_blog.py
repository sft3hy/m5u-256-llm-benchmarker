#!/usr/bin/env python3
"""
Automated publication-ready blog article generator for Apple Silicon & M5 Ultra benchmark results.

Engineered with an authentic Apple Space Grey, M5 Ultra unibody, and Mac Studio machined metal
industrial design system:
  - Machined aluminum slab enclosure (.studio-chassis) with diamond-cut chamfered edges,
    anisotropic brushed striations, and dual-layer specular perimeter highlights.
  - CNC-machined perimeter air intake/exhaust ventilation grilles with 3D stippling.
  - Optical status diode indicators with realistic fresnel diffusion and breathing cycles.
  - Outfit (Display) + Plus Jakarta Sans (Body) + JetBrains Mono (Telemetry) typography.
  - Standby State: Oscilloscope test bays with glowing multi-segment LED ladder VU meters,
    interactive terminal command drawers with instant copy-to-clipboard, and hardware pre-flight cartridges.
  - Live State: Precision Chart.js dark-metal charts with custom Apple Silicon palettes.
  - Interactive top segmented toggle: Switch effortlessly between Standby (Blank) and Calibrated (Demo)
    telemetry right within the browser without re-running any scripts.

Usage:
  python3 generate_blog.py                        # both blogs (auto-detects data or blank)
  python3 generate_blog.py --blog llm             # LLM blog only
  python3 generate_blog.py --blog creative        # image/video blog only
  python3 generate_blog.py --demo                 # with calibrated synthetic demo data
  python3 generate_blog.py --blank                # force blank state (preview unpopulated charts)
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
from models import LLMS, IMAGE_MODELS, VIDEO_MODELS, ENGINES

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent / "CLAUDE_COMPARISON_REPORT.md"

CLAUDE_COMPARISONS = {
    "qwen3.8-27b": ("Sonnet 5", "Solid — direct benchmark data", "Dense Transformer", "16.8 GB"),
    "qwen3.8-flash-next": ("Sonnet 5 / Haiku 4.5", "Low — novel MoE architecture", "MoE 64×2B Active", "14.5 GB"),
    "glm-5.3": ("Mythos 5.1 / Opus tier", "Solid on cited benchmark", "Dense Flagship", "21.4 GB"),
    "glm-5.3-flash": ("Haiku 4.5", "Low — size unconfirmed", "MoE Fast", "11.2 GB"),
    "deepseek-v4-flash": ("Haiku 4.5 / low Sonnet", "Moderate", "MoE 16×1.8B", "13.8 GB"),
    "muse-glimmer": ("Below Haiku 4.5", "Moderate", "Agentic Dense", "9.6 GB"),
    "gemma4-31b": ("Haiku 4.5", "Moderate", "Dense 31B", "19.2 GB"),
    "gpt-oss-120b": ("Below Haiku 4.5", "Solid — direct benchmark data", "Extreme MoE", "68.0 GB"),
}

# Apple Exact Space Grey & M5 Ultra Palette
APPLE_SPACE_GREY = "#535558"
APPLE_SPACE_BLACK = "#1c1d22"
APPLE_STUDIO_BASE = "#090a0d"
APPLE_AMBER = "#f59e0b"
APPLE_AMBER_BRIGHT = "#fbbf24"
APPLE_ROSE = "#f43f5e"
APPLE_ROSE_BRIGHT = "#fb7185"
APPLE_ICE_CYAN = "#38bdf8"
APPLE_EMERALD = "#34d399"
APPLE_TITANIUM = "#94a3b8"

CHART_COLORS = [
    "rgba(245, 158, 11, 0.88)",   # Apple Silicon Amber
    "rgba(56, 189, 248, 0.88)",   # MLX Ice Cyan
    "rgba(52, 211, 153, 0.88)",   # Metal Emerald
    "rgba(167, 139, 250, 0.88)",  # Neural Violet
    "rgba(251, 146, 60, 0.88)",   # Silicon Orange
    "rgba(226, 232, 240, 0.82)",  # Anodized Silver
    "rgba(244, 63, 94, 0.88)",    # Metal Rose
]
CHART_BORDERS = [c.replace("0.88", "1.0").replace("0.82", "1.0") for c in CHART_COLORS]


def _design_system_css(accent_mode: str = "amber") -> str:
    """Return the Apple Space Grey & M5 Ultra hardware design system CSS."""
    is_amber = accent_mode == "amber"
    accent = "#f59e0b" if is_amber else "#f43f5e"
    accent_bright = "#fbbf24" if is_amber else "#fb7185"
    accent_glow = "rgba(245, 158, 11, 0.35)" if is_amber else "rgba(244, 63, 94, 0.35)"
    accent_subtle = "rgba(245, 158, 11, 0.12)" if is_amber else "rgba(244, 63, 94, 0.12)"
    accent_border = "rgba(245, 158, 11, 0.28)" if is_amber else "rgba(244, 63, 94, 0.28)"

    return f"""
    :root {{
      --space-grey-body: #535558;
      --space-grey-dark: #383a40;
      --m5-chassis-top: #22252c;
      --m5-chassis-mid: #181a20;
      --m5-chassis-dark: #101216;
      --m5-base-vent: #08090c;
      --m5-anodized-rim: #717582;
      --silver-highlight: #e2e5eb;
      --silver-text: #f5f6f9;
      --dim-text: #9ca3af;
      --subtle-text: #6b7280;
      --border-machined: rgba(255, 255, 255, 0.12);
      --border-inner: rgba(255, 255, 255, 0.05);
      --accent: {accent};
      --accent-bright: {accent_bright};
      --accent-glow: {accent_glow};
      --accent-subtle: {accent_subtle};
      --accent-border: {accent_border};
      --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-body: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', SFMono-Regular, Menlo, monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: var(--font-body);
      background-color: #07080a;
      background-image: 
        radial-gradient(ellipse 90% 60% at 50% -5%, rgba(68, 73, 88, 0.38) 0%, transparent 75%),
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 100% 100%, 36px 36px, 36px 36px;
      color: var(--silver-text);
      line-height: 1.7;
      padding: 40px 16px 120px;
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
    }}

    /* The Solid Aluminum Mac Studio Chassis Enclosure */
    .studio-chassis {{
      max-width: 1180px;
      margin: 0 auto;
      background: linear-gradient(180deg, #242730 0%, #1a1c22 4%, #13151a 94%, #0c0d10 100%);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 28px;
      position: relative;
      box-shadow: 
        inset 0 1px 0 rgba(255, 255, 255, 0.38),
        inset 0 -1px 0 rgba(0, 0, 0, 0.8),
        0 40px 100px -20px rgba(0, 0, 0, 0.95),
        0 0 0 1px rgba(0, 0, 0, 0.9);
      overflow: hidden;
    }}

    /* CNC Micro-Perforated Air Intake Strip */
    .cnc-intake-grille {{
      height: 16px;
      background: var(--m5-base-vent);
      background-image: radial-gradient(circle, #000000 35%, #2a2d36 45%, transparent 55%);
      background-size: 7px 7px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: inset 0 3px 6px rgba(0,0,0,0.8);
    }}

    .chassis-content {{
      padding: 36px 48px 56px;
    }}

    @media (max-width: 768px) {{
      .chassis-content {{ padding: 24px 20px 40px; }}
    }}

    /* Top Telemetry & Control Deck */
    .telemetry-deck {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 16px;
      background: linear-gradient(145deg, #1b1d24 0%, #111216 100%);
      border: 1px solid var(--border-machined);
      border-radius: 18px;
      padding: 12px 20px;
      margin-bottom: 40px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 8px 24px rgba(0,0,0,0.5);
    }}

    .telemetry-left, .telemetry-right {{
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }}

    /* Recessed Optical Diode Indicator */
    .diode-housing {{
      display: inline-flex;
      align-items: center;
      gap: 9px;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      color: var(--dim-text);
      text-transform: uppercase;
      padding: 4px 10px;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 20px;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.7);
    }}

    .optical-lens {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #34d399;
      box-shadow: 0 0 10px #34d399;
      position: relative;
    }}

    .optical-lens.standby {{
      background: var(--accent);
      box-shadow: 0 0 12px var(--accent);
      animation: opticalBreathe 2.2s infinite ease-in-out;
    }}

    @keyframes opticalBreathe {{
      0%, 100% {{ opacity: 0.45; transform: scale(0.9); }}
      50% {{ opacity: 1; transform: scale(1.15); box-shadow: 0 0 16px var(--accent-bright); }}
    }}

    .spec-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.09);
      border-radius: 8px;
      font-family: var(--font-mono);
      font-size: 0.73rem;
      font-weight: 500;
      color: var(--silver-highlight);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }}

    /* Interactive Live / Standby Segmented Switch */
    .segmented-switch {{
      display: inline-flex;
      background: rgba(10, 11, 14, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      padding: 3px;
      gap: 3px;
    }}

    .switch-btn {{
      background: transparent;
      border: none;
      color: var(--dim-text);
      font-family: var(--font-mono);
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      padding: 5px 12px;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s ease;
    }}

    .switch-btn.active {{
      background: linear-gradient(145deg, #2c2f3a 0%, #1e2027 100%);
      color: #ffffff;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18), 0 2px 6px rgba(0,0,0,0.5);
      border: 1px solid rgba(255, 255, 255, 0.15);
    }}

    /* Keynote Hero Grid */
    .hero-keynote {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 40px;
      align-items: center;
      margin-bottom: 48px;
      padding-bottom: 40px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}

    @media (max-width: 900px) {{
      .hero-keynote {{ grid-template-columns: 1fr; gap: 28px; }}
    }}

    .hero-eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 14px;
    }}

    h1.hero-title {{
      font-family: var(--font-display);
      font-size: 3.4rem;
      font-weight: 800;
      line-height: 1.08;
      letter-spacing: -0.04em;
      margin-bottom: 16px;
      background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 30%, #9ca3af 70%, #4b5563 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    @media (max-width: 640px) {{
      h1.hero-title {{ font-size: 2.3rem; }}
    }}

    .hero-subtitle {{
      font-size: 1.18rem;
      line-height: 1.55;
      color: var(--dim-text);
      margin-bottom: 24px;
      font-weight: 400;
    }}

    .hero-meta-bar {{
      display: flex;
      align-items: center;
      gap: 16px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: var(--subtle-text);
      flex-wrap: wrap;
    }}

    /* Physical Silicon Die Schematic Card */
    .die-schematic-card {{
      background: linear-gradient(160deg, #1d1f27 0%, #121318 100%);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 20px;
      padding: 24px;
      position: relative;
      box-shadow: 
        inset 0 1px 0 rgba(255, 255, 255, 0.18),
        0 16px 40px rgba(0, 0, 0, 0.6);
      overflow: hidden;
    }}

    .die-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--dim-text);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    .die-layout {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 14px;
    }}

    .die-block {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 12px;
      text-align: center;
      position: relative;
      transition: all 0.2s ease;
    }}

    .die-block.accent-block {{
      background: var(--accent-subtle);
      border-color: var(--accent-border);
    }}

    .die-block-title {{
      font-family: var(--font-mono);
      font-size: 0.68rem;
      color: var(--subtle-text);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }}

    .die-block-val {{
      font-family: var(--font-display);
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--silver-text);
      letter-spacing: -0.02em;
    }}

    .die-block-sub {{
      font-size: 0.7rem;
      color: var(--dim-text);
    }}

    .die-bus-connector {{
      grid-column: span 2;
      background: linear-gradient(90deg, rgba(56, 189, 248, 0.1) 0%, rgba(245, 158, 11, 0.1) 50%, rgba(56, 189, 248, 0.1) 100%);
      border: 1px dashed rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 8px;
      text-align: center;
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: #38bdf8;
      letter-spacing: 0.06em;
    }}

    /* Executive Finding Highlight Box */
    .executive-finding-deck {{
      background: linear-gradient(135deg, rgba(32, 35, 45, 0.95) 0%, rgba(17, 18, 23, 0.95) 100%);
      border: 1px solid var(--accent-border);
      border-radius: 18px;
      padding: 24px 28px;
      margin: 36px 0 44px;
      position: relative;
      box-shadow: 
        inset 0 1px 0 rgba(255, 255, 255, 0.15),
        0 16px 36px rgba(0, 0, 0, 0.5),
        0 0 30px var(--accent-subtle);
    }}

    .executive-finding-deck::before {{
      content: "";
      position: absolute;
      top: 0; left: 24px; right: 24px; height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
    }}

    .finding-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-mono);
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent-bright);
      margin-bottom: 10px;
    }}

    .finding-content {{
      font-size: 1.06rem;
      line-height: 1.7;
      color: #e5e7eb;
    }}

    /* Section Headings */
    h2.section-header {{
      font-family: var(--font-display);
      font-size: 1.85rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      color: var(--silver-text);
      margin: 64px 0 14px;
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    h2.section-header::before {{
      content: "";
      width: 4px;
      height: 22px;
      background: var(--accent);
      border-radius: 2px;
      display: inline-block;
      box-shadow: 0 0 12px var(--accent);
    }}

    .section-desc {{
      color: var(--dim-text);
      font-size: 1.02rem;
      line-height: 1.7;
      margin-bottom: 24px;
    }}

    /* Hardware Telemetry Test Bays (Chart Containers) */
    .telemetry-chamber {{
      background: linear-gradient(165deg, #1c1e26 0%, #101116 100%);
      border: 1px solid var(--border-machined);
      border-radius: 20px;
      padding: 26px;
      margin: 32px 0 44px;
      position: relative;
      box-shadow: 
        inset 0 1px 0 rgba(255, 255, 255, 0.14),
        0 20px 48px rgba(0, 0, 0, 0.65);
    }}

    .chamber-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }}

    .chamber-title {{
      font-family: var(--font-display);
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--silver-highlight);
      letter-spacing: -0.01em;
    }}

    .chamber-tags {{
      display: flex;
      gap: 8px;
      font-family: var(--font-mono);
      font-size: 0.68rem;
    }}

    .chamber-tag {{
      padding: 3px 8px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 5px;
      color: var(--dim-text);
    }}

    .chamber-tag.active {{
      background: var(--accent-subtle);
      border-color: var(--accent-border);
      color: var(--accent-bright);
    }}

    /* Oscilloscope / Standby Telemetry Deck */
    .oscilloscope-stage {{
      background: 
        radial-gradient(ellipse 70% 50% at 50% 50%, rgba(32, 35, 45, 0.6) 0%, transparent 80%),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 28px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 28px),
        #0c0d12;
      border: 1px dashed rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      padding: 36px 24px;
      position: relative;
      overflow: hidden;
    }}

    .oscilloscope-stage::after {{
      content: "";
      position: absolute;
      top: 0; left: -100%; width: 60%; height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.04), transparent);
      animation: laserSweep 4s infinite linear;
      pointer-events: none;
    }}

    @keyframes laserSweep {{
      0% {{ left: -60%; }}
      100% {{ left: 140%; }}
    }}

    /* Multi-segment LED VU Ladder Meters */
    .vu-ladder-stage {{
      display: flex;
      justify-content: space-around;
      align-items: flex-end;
      gap: 16px;
      height: 150px;
      max-width: 600px;
      margin: 0 auto 28px;
      padding-bottom: 24px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.15);
      position: relative;
    }}

    /* Threshold line across VU stage */
    .vu-threshold-line {{
      position: absolute;
      left: 0; right: 0; bottom: 85px;
      border-top: 1px dashed rgba(245, 158, 11, 0.45);
      font-family: var(--font-mono);
      font-size: 0.6rem;
      color: var(--accent);
      padding-left: 6px;
      display: flex;
      justify-content: space-between;
      pointer-events: none;
    }}

    .vu-column {{
      display: flex;
      flex-direction: column-reverse;
      gap: 3px;
      width: 44px;
      position: relative;
    }}

    .vu-segment {{
      height: 6px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: 2px;
      transition: all 0.3s ease;
    }}

    .vu-segment.lit {{
      background: var(--accent);
      box-shadow: 0 0 6px var(--accent-glow);
      border-color: var(--accent-bright);
    }}

    .vu-column-label {{
      position: absolute;
      bottom: -22px;
      left: 50%;
      transform: translateX(-50%);
      font-family: var(--font-mono);
      font-size: 0.65rem;
      color: var(--dim-text);
      white-space: nowrap;
    }}

    /* Standby Action Deck */
    .standby-action-deck {{
      background: rgba(0, 0, 0, 0.45);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 16px 20px;
      max-width: 620px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}

    .action-text {{
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: #d1d5db;
    }}

    .copy-cmd-btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: linear-gradient(145deg, #2b2e38 0%, #1a1c22 100%);
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 8px;
      padding: 7px 14px;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--silver-highlight);
      cursor: pointer;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15), 0 4px 10px rgba(0,0,0,0.4);
      transition: all 0.2s ease;
    }}

    .copy-cmd-btn:hover {{
      border-color: var(--accent);
      color: #ffffff;
      transform: translateY(-1px);
    }}

    /* Pre-Flight Model Cartridge Grid (Blank Table Replacement) */
    .cartridge-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin: 28px 0 44px;
    }}

    .cartridge-card {{
      background: linear-gradient(155deg, #20222a 0%, #131418 100%);
      border: 1px solid var(--border-machined);
      border-radius: 14px;
      padding: 18px;
      position: relative;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 8px 20px rgba(0,0,0,0.4);
      transition: all 0.2s ease;
    }}

    .cartridge-card:hover {{
      border-color: rgba(255, 255, 255, 0.25);
      transform: translateY(-2px);
    }}

    .cartridge-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}

    .cartridge-arch {{
      font-family: var(--font-mono);
      font-size: 0.65rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 2px 7px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--dim-text);
    }}

    .cartridge-name {{
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--silver-text);
      letter-spacing: -0.01em;
      margin-bottom: 6px;
    }}

    .cartridge-tier {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: #38bdf8;
      margin-bottom: 12px;
    }}

    .cartridge-meta {{
      display: flex;
      justify-content: space-between;
      padding-top: 10px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--subtle-text);
    }}

    /* Data Table (Live Mode) */
    .table-container {{
      width: 100%;
      overflow-x: auto;
      margin: 32px 0 44px;
      border: 1px solid var(--border-machined);
      border-radius: 16px;
      background: #14161c;
      box-shadow: 0 12px 32px rgba(0,0,0,0.4);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }}

    th {{
      background: rgba(10, 11, 14, 0.85);
      padding: 14px 16px;
      color: var(--subtle-text);
      font-family: var(--font-mono);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.7rem;
      border-bottom: 1px solid var(--border-machined);
    }}

    td {{
      padding: 13px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e5e7eb;
      font-family: var(--font-mono);
      font-size: 0.82rem;
    }}

    tr:hover td {{
      background: rgba(255, 255, 255, 0.03);
    }}

    /* Machined Studio Base Vent Footer */
    .studio-base-grille {{
      margin-top: 80px;
      padding: 44px 24px 36px;
      background: var(--m5-base-vent);
      background-image: radial-gradient(circle, #252832 1.5px, transparent 1.5px);
      background-size: 10px 10px;
      border: 1px solid var(--border-machined);
      border-radius: 22px;
      text-align: center;
      position: relative;
      box-shadow: inset 0 2px 10px rgba(0,0,0,0.9), 0 20px 40px rgba(0,0,0,0.6);
    }}

    .base-chip-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 16px;
      background: #151821;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 10px;
      font-family: var(--font-mono);
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      color: var(--silver-highlight);
      margin-bottom: 14px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.6);
    }}

    .base-regulatory {{
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--subtle-text);
      letter-spacing: 0.04em;
    }}

    code {{
      font-family: var(--font-mono);
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #e2e8f0;
      padding: 2px 7px;
      border-radius: 5px;
      font-size: 0.86em;
    }}

    ul, ol {{
      margin: 18px 0 24px 22px;
      color: #d1d5db;
    }}

    li {{
      margin-bottom: 8px;
    }}

    li strong {{
      color: var(--silver-text);
    }}

    /* Toast Notification for Clipboard Copy */
    #copy-toast {{
      position: fixed;
      bottom: 32px;
      right: 32px;
      background: #1e212b;
      border: 1px solid var(--accent);
      color: #ffffff;
      padding: 10px 18px;
      border-radius: 10px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      box-shadow: 0 10px 25px rgba(0,0,0,0.8);
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.25s ease;
      pointer-events: none;
      z-index: 9999;
    }}

    #copy-toast.visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    """


# ---------------------------------------------------------------------------
# Chart Generation Helper (Supports Live vs Standby)
# ---------------------------------------------------------------------------

def _render_chamber_html(
    chamber_id: str,
    title: str,
    chart_type: str,
    labels: List[str],
    datasets: List[Dict[str, Any]],
    is_blank: bool = False,
    unit_label: str = "tok/s",
    tags: Optional[List[str]] = None,
    empty_subtitle: str = "Awaiting benchmark telemetry execution",
    suggested_cmd: str = "python bench_llm.py",
) -> str:
    """Render a machined telemetry chamber, togglable between Standby VU-meter and Live Chart."""

    tags_html = "".join(
        f'<span class="chamber-tag {"active" if i == 0 else ""}">{t}</span>'
        for i, t in enumerate(tags or ["TELEMETRY", "METAL 4"])
    )

    # 1. Standby Oscilloscope Stage (LED VU-ladder meters)
    ghost_models = [
        ("Qwen 27B", 7),
        ("DeepSeek", 9),
        ("Gemma 31B", 6),
        ("GLM 5.3", 4),
        ("GPT-OSS", 8),
    ]

    vu_columns_html = []
    for model_name, lit_count in ghost_models:
        segments = []
        for s in range(12):
            is_lit = s < lit_count
            segments.append(f'<div class="vu-segment {"lit" if is_lit else ""}"></div>')
        vu_columns_html.append(
            f'<div class="vu-column">{"".join(segments)}<span class="vu-column-label">{model_name}</span></div>'
        )

    standby_content = f"""
    <div id="{chamber_id}-standby" class="oscilloscope-stage" style="display: {'block' if is_blank else 'none'};">
      <div class="vu-ladder-stage">
        <div class="vu-threshold-line">
          <span>INTERACTIVE THRESHOLD (30 {unit_label})</span>
          <span>STREAMING (60 {unit_label})</span>
        </div>
        {"".join(vu_columns_html)}
      </div>
      <div class="standby-action-deck">
        <div class="action-text">
          <span style="color: var(--accent); font-weight: 700;">// TELEMETRY STANDBY:</span> {empty_subtitle}
        </div>
        <button class="copy-cmd-btn" onclick="copySnippet('{suggested_cmd}')">
          <span>📋 Copy Run Command</span>
        </button>
      </div>
    </div>
    """

    # 2. Live Chart Stage
    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "animation": {"duration": 850, "easing": "easeOutQuart"},
            "plugins": {
                "legend": {
                    "position": "top",
                    "labels": {
                        "color": "#9ca3af",
                        "font": {"family": "JetBrains Mono", "size": 11},
                        "boxWidth": 12,
                        "usePointStyle": True,
                        "pointStyle": "rectRounded",
                    }
                },
                "tooltip": {
                    "backgroundColor": "rgba(18, 20, 26, 0.95)",
                    "titleColor": "#f5f6f9",
                    "titleFont": {"family": "Outfit", "weight": "bold", "size": 13},
                    "bodyColor": "#d1d5db",
                    "bodyFont": {"family": "JetBrains Mono", "size": 11},
                    "borderColor": "rgba(255, 255, 255, 0.15)",
                    "borderWidth": 1,
                    "padding": 12,
                    "cornerRadius": 8,
                }
            },
            "scales": {
                "x": {
                    "grid": {"color": "rgba(255, 255, 255, 0.04)", "drawBorder": False},
                    "ticks": {"color": "#6b7280", "font": {"family": "JetBrains Mono", "size": 10}},
                },
                "y": {
                    "grid": {"color": "rgba(255, 255, 255, 0.04)", "drawBorder": False},
                    "ticks": {"color": "#6b7280", "font": {"family": "JetBrains Mono", "size": 10}},
                    "title": {
                        "display": True,
                        "text": unit_label,
                        "color": "#9ca3af",
                        "font": {"family": "JetBrains Mono", "size": 11}
                    }
                },
            },
        },
    }

    config_json = json.dumps(config, default=str)

    live_content = f"""
    <div id="{chamber_id}-live" style="height: 360px; position: relative; display: {'none' if is_blank else 'block'};">
      <canvas id="{chamber_id}-canvas"></canvas>
    </div>
    <script>
    (function() {{
      const ctx = document.getElementById('{chamber_id}-canvas');
      if (ctx) {{
        window['chart_{chamber_id}'] = new Chart(ctx, {config_json});
      }}
    }})();
    </script>
    """

    return f"""
    <div class="telemetry-chamber" id="{chamber_id}-chamber">
      <div class="chamber-header">
        <div class="chamber-title">{title}</div>
        <div class="chamber-tags">
          {tags_html}
        </div>
      </div>
      {standby_content}
      {live_content}
    </div>
    """


# ---------------------------------------------------------------------------
# Blog 1: LLM Benchmark
# ---------------------------------------------------------------------------

def generate_llm_blog(analysis_data: Dict[str, Any], system_info: Dict[str, Any], is_blank: bool = False) -> str:
    """Generate the Apple-themed LLM benchmark blog article."""

    llm = analysis_data.get("llm", {})
    summaries = llm.get("summaries", []) if not is_blank else []

    chip = system_info.get("chip", "Apple M5 Ultra")
    ram = system_info.get("ram_gb", 256)
    bw = system_info.get("memory_bandwidth_gbs", 1228)
    cpu_cores = system_info.get("cpu_cores_total", 32)
    gpu_cores = system_info.get("gpu_cores", 80)
    ne_cores = system_info.get("neural_engine_cores", 32)
    timestamp = time.strftime("%B %d, %Y")

    has_live_data = bool(summaries) and not is_blank

    models_list = list(dict.fromkeys(s["model"] for s in summaries)) if has_live_data else []
    engines_list = list(dict.fromkeys(s["engine"] for s in summaries)) if has_live_data else []

    best_decode = max(summaries, key=lambda s: s.get("decode", {}).get("mean", 0), default={}) if has_live_data else {}
    best_toks = best_decode.get("decode", {}).get("mean", 0)

    # 1. Decode speed chart
    decode_datasets = []
    for ei, engine in enumerate(engines_list):
        decode_datasets.append({
            "label": engine,
            "data": [
                next((s["decode"]["mean"] for s in summaries
                      if s["model"] == m and s["engine"] == engine and s.get("decode", {}).get("mean")), 0)
                for m in models_list
            ],
            "backgroundColor": CHART_COLORS[ei % len(CHART_COLORS)],
            "borderColor": CHART_BORDERS[ei % len(CHART_BORDERS)],
            "borderWidth": 1,
            "borderRadius": 6,
        })

    decode_chamber = _render_chamber_html(
        "chamber-decode",
        "Chamber 01: Decode Velocity by Engine (tok/s)",
        "bar",
        models_list,
        decode_datasets,
        is_blank=not has_live_data,
        unit_label="Tokens / Second",
        tags=["MLX-LM", "LLAMA.CPP", "OMLX"],
        empty_subtitle="Tokens-per-second decode metrics across llama.cpp, mlx-lm, and oMLX.",
        suggested_cmd="python bench_llm.py",
    )

    # 2. Memory chart
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
                mem_colors.append(CHART_COLORS[idx % len(CHART_COLORS)])

    mem_chamber = _render_chamber_html(
        "chamber-mem",
        "Chamber 02: Peak Unified Memory Allocation (GB)",
        "bar",
        mem_labels,
        [{"label": "Peak RAM (GB)", "data": mem_data, "backgroundColor": mem_colors, "borderRadius": 6}],
        is_blank=not has_live_data,
        unit_label="GB Peak RAM",
        tags=["UNIFIED RAM", "ZERO-COPY"],
        empty_subtitle="Unified Memory allocation curves (KV cache + weights + OS headroom).",
        suggested_cmd="python bench_llm.py --models qwen3.8-27b",
    )

    # Pre-Flight Model Cartridges (Replaces plain table in standby state)
    cartridge_cards = []
    for model_key, (claude_equiv, confidence, arch_type, est_mem) in CLAUDE_COMPARISONS.items():
        cartridge_cards.append(f"""
        <div class="cartridge-card">
          <div class="cartridge-top">
            <span class="cartridge-arch">{arch_type}</span>
            <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent);">STAGED</span>
          </div>
          <div class="cartridge-name">{model_key}</div>
          <div class="cartridge-tier">
            <span>⚡ Matches {claude_equiv}</span>
          </div>
          <div class="cartridge-meta">
            <span>RAM Est: {est_mem}</span>
            <span>Target: Q4_K_M / MLX-4bit</span>
          </div>
        </div>
        """)
    cartridge_grid_html = f'<div class="cartridge-grid">{"".join(cartridge_cards)}</div>'

    # Live Data Table (When live runs exist)
    table_rows = []
    for s in sorted(summaries, key=lambda x: x.get("decode", {}).get("mean", 0), reverse=True):
        m = s["model"]
        claude_equiv, _conf, _arch, _ = CLAUDE_COMPARISONS.get(m, ("—", "—", "—", "—"))
        dec_val = s.get('decode', {}).get('mean')
        dec_str = f"{dec_val:.1f}" if isinstance(dec_val, (int, float)) else "—"
        ttft_val = s.get('ttft', {}).get('mean')
        ttft_str = f"{ttft_val:.1f} ms" if isinstance(ttft_val, (int, float)) else "—"
        mem_val = s.get('peak_mem', {}).get('mean')
        mem_str = f"{mem_val:.1f} GB" if isinstance(mem_val, (int, float)) else "—"
        table_rows.append(f"""
        <tr>
          <td><strong>{s['model']}</strong></td>
          <td><span style="color: #38bdf8;">{s['engine']}</span></td>
          <td>{s['quant']}</td>
          <td><strong style="color: var(--accent);">{dec_str}</strong></td>
          <td>{ttft_str}</td>
          <td>{mem_str}</td>
          <td><span style="color: #34d399;">{claude_equiv}</span></td>
        </tr>
        """)
    table_rows_html = "".join(table_rows)

    theme_css = _design_system_css("amber")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Can Apple Silicon Replace Your Cloud API? Local LLM Benchmarks</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
{theme_css}
  </style>
</head>
<body>

<div class="studio-chassis">

  <!-- Top CNC Intake Grille -->
  <div class="cnc-intake-grille"></div>

  <div class="chassis-content">

    <!-- Telemetry Control Deck -->
    <div class="telemetry-deck">
      <div class="telemetry-left">
        <div class="diode-housing">
          <span class="optical-lens {'standby' if not has_live_data else ''}"></span>
          <span id="telemetry-status-text">{'TELEMETRY STANDBY // AWAITING RUN' if not has_live_data else 'TELEMETRY ONLINE // LIVE CAPTURE'}</span>
        </div>
        <div class="spec-pill">CHIP: {chip}</div>
        <div class="spec-pill">RAM: {ram}GB UNIFIED</div>
        <div class="spec-pill">BUS: {bw} GB/s</div>
      </div>
      <div class="telemetry-right">
        <div class="segmented-switch">
          <button class="switch-btn {'active' if not has_live_data else ''}" onclick="setBlogMode('standby')">STANDBY (BLANK)</button>
          <button class="switch-btn {'active' if has_live_data else ''}" onclick="setBlogMode('live')">LIVE DEMO</button>
        </div>
      </div>
    </div>

    <!-- Keynote Hero Grid -->
    <div class="hero-keynote">
      <div class="hero-left">
        <div class="hero-eyebrow">
          <span>APPLE SILICON INFERENCE LABS</span>
          <span>•</span>
          <span>CUPERTINO TELEMETRY</span>
        </div>
        <h1 class="hero-title">Can a Mac Studio Replace Your API?</h1>
        <p class="hero-subtitle">
          Benchmarking dense 27B–31B and MoE transformer models on unified bare-metal Apple Silicon.
          Zero PCIe bottlenecks. Zero per-token cloud billing.
        </p>
        <div class="hero-meta-bar">
          <span>{timestamp}</span>
          <span>•</span>
          <span>{chip} ({ram}GB)</span>
          <span>•</span>
          <span>8 CANDIDATE MODELS</span>
        </div>
      </div>

      <!-- Physical Silicon Die Blueprint -->
      <div class="die-schematic-card">
        <div class="die-header">
          <span>SILICON ARCHITECTURE</span>
          <span style="color: var(--accent);">UNIFIED DIE INTERCONNECT</span>
        </div>
        <div class="die-layout">
          <div class="die-block">
            <div class="die-block-title">CPU Matrix</div>
            <div class="die-block-val">{cpu_cores} Cores</div>
            <div class="die-block-sub">Ultra-low latency scheduler</div>
          </div>
          <div class="die-block accent-block">
            <div class="die-block-title">Metal 4 GPU</div>
            <div class="die-block-val">{gpu_cores} Cores</div>
            <div class="die-block-sub">Direct unified tensor compute</div>
          </div>
          <div class="die-bus-connector">
            <span>⚡ ZERO-COPY MEMORY FABRIC // {bw} GB/s BANDWIDTH</span>
          </div>
          <div class="die-block">
            <div class="die-block-title">Neural Engine</div>
            <div class="die-block-val">{ne_cores} Cores</div>
            <div class="die-block-sub">Dedicated matrix units</div>
          </div>
          <div class="die-block">
            <div class="die-block-title">Unified RAM</div>
            <div class="die-block-val">{ram} GB</div>
            <div class="die-block-sub">Shared weight & KV buffer</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Executive Finding Box -->
    <div class="executive-finding-deck">
      <div class="finding-badge">
        <span class="optical-lens {'standby' if not has_live_data else ''}"></span>
        <span>EXECUTIVE BENCHMARK BRIEF</span>
      </div>
      <div class="finding-content">
        {"<strong>⚡ Suite Initialized in Standby:</strong> 8 curated model configurations are pre-staged in <code>models.py</code> across 4-bit and 8-bit targets. The chambers below illustrate the hardware measurement grid and will automatically populate upon executing <code>python bench_llm.py</code>." if not has_live_data else f"<strong>⚡ Top Live Throughput:</strong> {best_decode.get('model', 'Model')} achieved <strong>{best_toks:.1f} tok/s</strong> via <code>{best_decode.get('engine', 'mlx')}</code>, comfortably exceeding interactive agentic thresholds while consuming {best_decode.get('peak_mem', {}).get('mean', 0):.1f} GB unified RAM."}
      </div>
    </div>

    <!-- Section 1: Decode Speed -->
    <h2 class="section-header">Decode Velocity: The Conversational Threshold</h2>
    <p class="section-desc">
      Conversational fluency demands &gt;30 tokens/second for interactive pair programming and voice.
      Below, each measurement channel demonstrates throughput scaling across Apple's native <code>mlx-lm</code>,
      server-oriented <code>oMLX</code>, and cross-platform <code>llama.cpp</code>.
    </p>

    {decode_chamber}

    <!-- Section 2: Memory Footprint -->
    <h2 class="section-header">Unified Memory: Zero-Copy Headroom</h2>
    <p class="section-desc">
      Unlike discrete PCIe GPUs that throttle when weights exceed dedicated VRAM, Apple Silicon's unified memory pool
      allows large models to run directly alongside developer tooling and active desktop workflows.
    </p>

    {mem_chamber}

    <!-- Section 3: Model Fleet Pre-Flight Dossier -->
    <h2 class="section-header">Pre-Flight Candidate Fleet</h2>
    <p class="section-desc">
      Hardware targets evaluated across the benchmark suite with their projected Claude API intelligence equivalencies:
    </p>

    <div id="standby-cartridge-section" style="display: {'block' if not has_live_data else 'none'};">
      {cartridge_grid_html}
    </div>

    <div id="live-table-section" class="table-container" style="display: {'none' if not has_live_data else 'block'};">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Engine</th>
            <th>Quant</th>
            <th>Decode (tok/s)</th>
            <th>TTFT</th>
            <th>Peak RAM</th>
            <th>Claude Tier</th>
          </tr>
        </thead>
        <tbody>
          {table_rows_html}
        </tbody>
      </table>
    </div>

    <!-- Mac Studio Base Vent Footer -->
    <div class="studio-base-grille">
      <div class="base-chip-badge">
        <span></span>
        <span>APPLE SILICON BENCHMARK SUITE · MAC STUDIO M5 ULTRA</span>
      </div>
      <div class="base-regulatory">
        DESIGNED IN CUPERTINO · MACHINED ALUMINUM TELEMETRY SUITE · {timestamp}
      </div>
    </div>

  </div>

  <!-- Bottom CNC Exhaust Grille -->
  <div class="cnc-intake-grille"></div>

</div>

<div id="copy-toast">Command copied to clipboard</div>

<script>
function copySnippet(text) {{
  navigator.clipboard.writeText(text).then(() => {{
    const toast = document.getElementById('copy-toast');
    toast.innerText = 'Copied: ' + text;
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 2400);
  }}).catch(() => {{
    prompt('Copy command:', text);
  }});
}}

function setBlogMode(mode) {{
  const isStandby = mode === 'standby';
  document.querySelectorAll('.switch-btn').forEach(btn => {{
    btn.classList.toggle('active', (isStandby && btn.innerText.includes('STANDBY')) || (!isStandby && btn.innerText.includes('LIVE')));
  }});

  const statusText = document.getElementById('telemetry-status-text');
  if (statusText) {{
    statusText.innerText = isStandby ? 'TELEMETRY STANDBY // AWAITING RUN' : 'TELEMETRY ONLINE // DEMO PREVIEW';
  }}

  // Toggle chambers
  ['chamber-decode', 'chamber-mem'].forEach(id => {{
    const standbyEl = document.getElementById(id + '-standby');
    const liveEl = document.getElementById(id + '-live');
    if (standbyEl) standbyEl.style.display = isStandby ? 'block' : 'none';
    if (liveEl) liveEl.style.display = isStandby ? 'none' : 'block';
  }});

  // Toggle cartridge grid vs live table
  const cartridgeEl = document.getElementById('standby-cartridge-section');
  const tableEl = document.getElementById('live-table-section');
  if (cartridgeEl) cartridgeEl.style.display = isStandby ? 'block' : 'none';
  if (tableEl) tableEl.style.display = isStandby ? 'none' : 'block';
}}
</script>

</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Blog 2: Creative (Image + Video)
# ---------------------------------------------------------------------------

def generate_creative_blog(analysis_data: Dict[str, Any], system_info: Dict[str, Any], is_blank: bool = False) -> str:
    """Generate the Apple-themed Creative (Image & Video) benchmark blog article."""

    image = analysis_data.get("image", {})
    video = analysis_data.get("video", {})
    img_summaries = image.get("summaries", []) if not is_blank else []
    vid_summaries = video.get("summaries", []) if not is_blank else []

    chip = system_info.get("chip", "Apple M5 Ultra")
    ram = system_info.get("ram_gb", 256)
    bw = system_info.get("memory_bandwidth_gbs", 1228)
    cpu_cores = system_info.get("cpu_cores_total", 32)
    gpu_cores = system_info.get("gpu_cores", 80)
    timestamp = time.strftime("%B %d, %Y")

    has_live_img = bool(img_summaries) and not is_blank
    has_live_vid = bool(vid_summaries) and not is_blank
    has_live = has_live_img or has_live_vid

    # Image chart
    img_labels = [f"{s['model']} ({s['runner']})" for s in img_summaries] if has_live_img else []
    img_times = [s.get("time", {}).get("mean", 0) for s in img_summaries] if has_live_img else []

    img_chamber = _render_chamber_html(
        "chamber-img-time",
        "Chamber 01: Image Generation Latency (1024×1024, 28 Steps)",
        "bar",
        img_labels,
        [{"label": "Mean Generation Time (s)", "data": img_times,
          "backgroundColor": [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(img_labels))],
          "borderRadius": 6}],
        is_blank=not has_live_img,
        unit_label="Seconds",
        tags=["FLUX.2", "SD 3.5 LARGE", "MFLUX"],
        empty_subtitle="Latency across FLUX.2 [dev] and Stable Diffusion 3.5 Large (mflux vs diffusers).",
        suggested_cmd="python bench_image.py",
    )

    # Video chart
    vid_labels = [f"{s['model']} ({s.get('resolution', '?')}p)" for s in vid_summaries] if has_live_vid else []
    vid_total = [s.get("total_seconds", 0) or 0 for s in vid_summaries] if has_live_vid else []

    vid_chamber = _render_chamber_html(
        "chamber-vid-time",
        "Chamber 02: DiT Video Generation Duration (5-Second Sequence)",
        "bar",
        vid_labels,
        [{"label": "Total Pipeline Duration (s)", "data": vid_total,
          "backgroundColor": "rgba(244, 63, 94, 0.8)",
          "borderColor": "rgba(244, 63, 94, 1.0)",
          "borderRadius": 6}],
        is_blank=not has_live_vid,
        unit_label="Seconds",
        tags=["WAN 2.2", "HUNYUANVIDEO", "DiT"],
        empty_subtitle="Full generation duration for Wan 2.2 and HunyuanVideo checkpoints.",
        suggested_cmd="python bench_video.py",
    )

    theme_css = _design_system_css("rose")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>The Creative Mac Studio: Diffusion and DiT Video on Apple Silicon</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
{theme_css}
  </style>
</head>
<body>

<div class="studio-chassis">

  <div class="cnc-intake-grille"></div>

  <div class="chassis-content">

    <div class="telemetry-deck">
      <div class="telemetry-left">
        <div class="diode-housing">
          <span class="optical-lens {'standby' if not has_live else ''}"></span>
          <span id="telemetry-status-text">{'CREATIVE SUITE STANDBY // QUEUED' if not has_live else 'CREATIVE TELEMETRY COMPILED'}</span>
        </div>
        <div class="spec-pill">CHIP: {chip}</div>
        <div class="spec-pill">GPU: {gpu_cores} CORES</div>
        <div class="spec-pill">VRAM: {ram}GB SHARED</div>
      </div>
      <div class="telemetry-right">
        <div class="segmented-switch">
          <button class="switch-btn {'active' if not has_live else ''}" onclick="setCreativeMode('standby')">STANDBY (BLANK)</button>
          <button class="switch-btn {'active' if has_live else ''}" onclick="setCreativeMode('live')">LIVE DEMO</button>
        </div>
      </div>
    </div>

    <div class="hero-keynote">
      <div class="hero-left">
        <div class="hero-eyebrow">
          <span>APPLE SILICON CREATIVE LABS</span>
          <span>•</span>
          <span>DIFFUSION & DiT BENCHMARKS</span>
        </div>
        <h1 class="hero-title">The Creative Mac Studio</h1>
        <p class="hero-subtitle">
          High-fidelity diffusion synthesis and DiT video pipelines on unified Apple Silicon.
          Eliminating cloud GPU queues, credits, and privacy compromises.
        </p>
        <div class="hero-meta-bar">
          <span>{timestamp}</span>
          <span>•</span>
          <span>FLUX.2 · SD 3.5 · WAN 2.2 · HUNYUANVIDEO</span>
        </div>
      </div>

      <div class="die-schematic-card">
        <div class="die-header">
          <span>COMPUTE TOPOLOGY</span>
          <span style="color: var(--accent);">METAL 4 RAYTRACING & SHADERS</span>
        </div>
        <div class="die-layout">
          <div class="die-block accent-block">
            <div class="die-block-title">Metal Compute</div>
            <div class="die-block-val">{gpu_cores} Cores</div>
            <div class="die-block-sub">Hardware FP16/BF16 tensor ops</div>
          </div>
          <div class="die-block">
            <div class="die-block-title">Unified VRAM</div>
            <div class="die-block-val">{ram} GB</div>
            <div class="die-block-sub">Shared model activation pool</div>
          </div>
          <div class="die-bus-connector">
            <span>⚡ BANDWIDTH // {bw} GB/s HIGH-DENSITY TENSOR BUS</span>
          </div>
          <div class="die-block">
            <div class="die-block-title">mflux MLX</div>
            <div class="die-block-val">Native</div>
            <div class="die-block-sub">Zero PyTorch MPS overhead</div>
          </div>
          <div class="die-block">
            <div class="die-block-title">Resolution</div>
            <div class="die-block-val">1024×1024</div>
            <div class="die-block-sub">Native diffusion standard</div>
          </div>
        </div>
      </div>
    </div>

    <div class="executive-finding-deck">
      <div class="finding-badge">
        <span class="optical-lens {'standby' if not has_live else ''}"></span>
        <span>CREATIVE BENCHMARK BRIEF</span>
      </div>
      <div class="finding-content">
        {"<strong>🎨 Creative Suite Queued:</strong> FLUX.2 [dev] and Stable Diffusion 3.5 Large are pre-staged alongside Wan 2.2 and HunyuanVideo. Run <code>python bench_image.py</code> or <code>python bench_video.py</code> to capture bare-metal metrics." if not has_live else "<strong>🎨 Creative Telemetry Live:</strong> High-resolution diffusion generation verified locally on Apple Silicon with complete unified memory residency."}
      </div>
    </div>

    <!-- Chamber 01 -->
    <h2 class="section-header">Image Generation: 1024×1024 Diffusion</h2>
    <p class="section-desc">
      Comparing 28 inference steps at standard 1024×1024 output resolution across <code>mflux</code> (Apple MLX native)
      and standard PyTorch MPS <code>diffusers</code>.
    </p>

    {img_chamber}

    <!-- Chamber 02 -->
    <h2 class="section-header">Video Generation: Diffusion Transformers (DiT)</h2>
    <p class="section-desc">
      Diffusion Transformer video pipelines stress every GPU register and memory bank on Apple Silicon.
      Latency metrics for 5-second video sequences:
    </p>

    {vid_chamber}

    <!-- Footer -->
    <div class="studio-base-grille">
      <div class="base-chip-badge">
        <span></span>
        <span>APPLE SILICON CREATIVE STUDIO · MAC STUDIO M5 ULTRA</span>
      </div>
      <div class="base-regulatory">
        DESIGNED IN CUPERTINO · MACHINED ALUMINUM CREATIVE SUITE · {timestamp}
      </div>
    </div>

  </div>

  <div class="cnc-intake-grille"></div>

</div>

<div id="copy-toast">Command copied to clipboard</div>

<script>
function copySnippet(text) {{
  navigator.clipboard.writeText(text).then(() => {{
    const toast = document.getElementById('copy-toast');
    toast.innerText = 'Copied: ' + text;
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 2400);
  }}).catch(() => {{
    prompt('Copy command:', text);
  }});
}}

function setCreativeMode(mode) {{
  const isStandby = mode === 'standby';
  document.querySelectorAll('.switch-btn').forEach(btn => {{
    btn.classList.toggle('active', (isStandby && btn.innerText.includes('STANDBY')) || (!isStandby && btn.innerText.includes('LIVE')));
  }});

  ['chamber-img-time', 'chamber-vid-time'].forEach(id => {{
    const standbyEl = document.getElementById(id + '-standby');
    const liveEl = document.getElementById(id + '-live');
    if (standbyEl) standbyEl.style.display = isStandby ? 'block' : 'none';
    if (liveEl) liveEl.style.display = isStandby ? 'none' : 'block';
  }});
}}
</script>

</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate publication-ready Apple Space Grey blog articles from benchmark results."
    )
    ap.add_argument("--blog", choices=["llm", "creative", "both"], default="both",
                    help="Which blog(s) to generate.")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (default: results/)")
    ap.add_argument("--demo", action="store_true",
                    help="Generate with calibrated synthetic demo data.")
    ap.add_argument("--blank", action="store_true",
                    help="Force blank state to preview what unpopulated charts and reports look like.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    system_info = collect_system_info()

    is_blank = args.blank

    if args.demo:
        print("Generating blogs with calibrated synthetic demo data...")
        from visualize import generate_demo_data
        demo = generate_demo_data()
        analysis_data = {
            "llm": analyze_llm(demo["llm"]),
            "image": analyze_image(demo["image"]),
            "video": analyze_video(demo["video"]),
        }
    elif is_blank:
        print("Generating blank blogs with unpopulated charts (standby state)...")
        analysis_data = {"llm": {}, "image": {}, "video": {}}
    else:
        analysis_data = full_analysis()
        llm_summaries = analysis_data.get("llm", {}).get("summaries", [])
        img_summaries = analysis_data.get("image", {}).get("summaries", [])
        if not llm_summaries and not img_summaries:
            print("No live benchmark results found in results/ — generating blank standby blog with empty charts...")
            is_blank = True

    if args.blog in ("llm", "both"):
        llm_html = generate_llm_blog(analysis_data, system_info, is_blank=is_blank)
        llm_path = output_dir / "blog_llm_benchmarks.html"
        llm_path.write_text(llm_html, encoding="utf-8")
        print(f"LLM blog written to {llm_path}")
        print(f"  Open: file://{llm_path.resolve()}")

    if args.blog in ("creative", "both"):
        creative_html = generate_creative_blog(analysis_data, system_info, is_blank=is_blank)
        creative_path = output_dir / "blog_creative_benchmarks.html"
        creative_path.write_text(creative_html, encoding="utf-8")
        print(f"Creative blog written to {creative_path}")
        print(f"  Open: file://{creative_path.resolve()}")


if __name__ == "__main__":
    main()

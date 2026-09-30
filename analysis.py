#!/usr/bin/env python3
"""
Core analysis library for benchmark results.

Reads CSV results from all three benchmark legs (LLM, image, video) plus
the system_bench template, computes composite scores, statistical summaries,
and rankings. Used by both visualize.py and generate_blog.py.
"""

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models import LLMS, IMAGE_MODELS, VIDEO_MODELS, ENGINES

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load a CSV file, converting numeric fields automatically."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cleaned = {}
            for k, v in row.items():
                if v is None or v.strip() == "":
                    cleaned[k] = None
                    continue
                try:
                    num = float(v)
                except ValueError:
                    cleaned[k] = v
                    continue
                # Keep the int/float distinction the rest of the kit relies
                # on, but drop non-finite values: 'inf' used to blow up the
                # int() cast below, and NaN survived to poison every mean,
                # CV and ranking downstream.
                if not math.isfinite(num):
                    cleaned[k] = None
                elif num.is_integer() and abs(num) < 1e15:
                    cleaned[k] = int(num)
                else:
                    cleaned[k] = num
            rows.append(cleaned)
    return rows


def load_llm_results() -> List[Dict[str, Any]]:
    return _load_csv(RESULTS_DIR / "llm_results.csv")


def load_image_results() -> List[Dict[str, Any]]:
    return _load_csv(RESULTS_DIR / "image_results.csv")


def load_video_results() -> List[Dict[str, Any]]:
    return _load_csv(RESULTS_DIR / "video_results.csv")


def load_system_bench() -> List[Dict[str, Any]]:
    return _load_csv(RESULTS_DIR / "system_bench.csv")


def load_system_bench_template() -> List[Dict[str, Any]]:
    return _load_csv(RESULTS_DIR / "system_bench_template.csv")


def has_results() -> Dict[str, bool]:
    """Check which result sets have actual data."""
    return {
        "llm": len(load_llm_results()) > 0,
        "image": len(load_image_results()) > 0,
        "video": len(load_video_results()) > 0,
        "system": len(load_system_bench()) > 0,
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _finite(value: Any) -> bool:
    """True for real measurement numbers: rules out None and NaN/inf."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute mean, std dev, min, max, CV for a list of numeric values."""
    clean = [v for v in values if _finite(v)]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None, "cv": None, "n": 0}
    n = len(clean)
    mean = sum(clean) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in clean) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    cv = (std / mean * 100) if mean != 0 else None
    return {
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "cv": round(cv, 1) if cv is not None else None,
        "n": n,
    }


def flag_noisy(cv: Optional[float], threshold: float = 10.0) -> bool:
    """Return True if coefficient of variation exceeds threshold."""
    return cv is not None and cv > threshold


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

def analyze_llm(rows: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Comprehensive LLM benchmark analysis."""
    if rows is None:
        rows = load_llm_results()
    if not rows:
        return {"models": [], "engines": [], "rankings": {}, "raw": []}

    # Group by (model, quant, engine)
    groups: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("model", ""), str(r.get("quant", "")), r.get("engine", ""))
        groups[key].append(r)

    summaries = []
    for (model, quant, engine), group_rows in groups.items():
        decode_vals = [r["decode_tok_s"] for r in group_rows if r.get("decode_tok_s")]
        prefill_vals = [r["prefill_tok_s"] for r in group_rows if r.get("prefill_tok_s")]
        mem_vals = [r["peak_mem_gb"] for r in group_rows if r.get("peak_mem_gb")]
        cpu_vals = [r["avg_cpu_pct"] for r in group_rows if r.get("avg_cpu_pct")]

        decode_stats = stats(decode_vals)
        prefill_stats = stats(prefill_vals)
        mem_stats = stats(mem_vals)

        # Composite: efficiency = decode tok/s per GB of RAM
        efficiency = None
        if decode_stats["mean"] and mem_stats["mean"] and mem_stats["mean"] > 0:
            efficiency = round(decode_stats["mean"] / mem_stats["mean"], 2)

        # Model spec lookup
        spec = LLMS.get(model, {})
        approx_gb = spec.get("approx_gb", {}).get(quant)
        total_params = spec.get("family", "")

        summaries.append({
            "model": model,
            "quant": quant,
            "engine": engine,
            "family": spec.get("family", model),
            "vendor": spec.get("vendor", ""),
            "approx_gb": approx_gb,
            "decode": decode_stats,
            "prefill": prefill_stats,
            "peak_mem": mem_stats,
            "avg_cpu": stats(cpu_vals),
            "efficiency_tok_per_gb": efficiency,
            "noisy_decode": flag_noisy(decode_stats.get("cv")),
            "noisy_prefill": flag_noisy(prefill_stats.get("cv")),
        })

    # Rankings
    decode_ranked = sorted(
        [s for s in summaries if s["decode"]["mean"]],
        key=lambda s: s["decode"]["mean"],
        reverse=True,
    )
    efficiency_ranked = sorted(
        [s for s in summaries if s.get("efficiency_tok_per_gb")],
        key=lambda s: s["efficiency_tok_per_gb"],
        reverse=True,
    )

    # Engine comparison: for each (model, quant), compute deltas between engines
    engine_comparison = {}
    by_model_quant: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for s in summaries:
        by_model_quant[(s["model"], s["quant"])].append(s)

    for (model, quant), engine_results in by_model_quant.items():
        if len(engine_results) > 1:
            best = max(engine_results, key=lambda s: s["decode"]["mean"] or 0)
            comparison = []
            for er in engine_results:
                delta_pct = None
                if er["decode"]["mean"] and best["decode"]["mean"]:
                    delta_pct = round(
                        (er["decode"]["mean"] / best["decode"]["mean"] - 1) * 100, 1
                    )
                comparison.append({
                    "engine": er["engine"],
                    "decode_tok_s": er["decode"]["mean"],
                    "delta_vs_best_pct": delta_pct,
                })
            engine_comparison[f"{model}/{quant}"] = {
                "best_engine": best["engine"],
                "engines": comparison,
            }

    all_models = sorted(set(s["model"] for s in summaries))
    all_engines = sorted(set(s["engine"] for s in summaries))

    return {
        "summaries": summaries,
        "models": all_models,
        "engines": all_engines,
        "rankings": {
            "by_decode_speed": [
                {"rank": i + 1, "model": s["model"], "quant": s["quant"],
                 "engine": s["engine"], "decode_tok_s": s["decode"]["mean"]}
                for i, s in enumerate(decode_ranked)
            ],
            "by_efficiency": [
                {"rank": i + 1, "model": s["model"], "quant": s["quant"],
                 "engine": s["engine"], "tok_per_gb": s["efficiency_tok_per_gb"]}
                for i, s in enumerate(efficiency_ranked)
            ],
        },
        "engine_comparison": engine_comparison,
        "raw": rows,
    }


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

def analyze_image(rows: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Image generation benchmark analysis."""
    if rows is None:
        rows = load_image_results()
    if not rows:
        return {"summaries": [], "raw": []}

    # Group by (model, runner, resolution, steps). Resolution has to be part
    # of the key: a 512px and a 1024px run of the same model are different
    # amounts of work, and averaging them produces neither.
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("model", ""), r.get("runner", ""),
               r.get("resolution", 1024), r.get("steps", 28))
        groups[key].append(r)

    summaries = []
    for (model, runner, resolution, steps), group_rows in groups.items():
        time_vals = [r["seconds"] for r in group_rows if _finite(r.get("seconds"))]
        mem_vals = [r["peak_mem_gb"] for r in group_rows if _finite(r.get("peak_mem_gb"))]

        time_stats = stats(time_vals)
        mem_stats = stats(mem_vals)

        spec = IMAGE_MODELS.get(model, {})

        # Megapixels per second
        mpx_per_sec = None
        if time_stats["mean"] and time_stats["mean"] > 0:
            mpx = (resolution * resolution) / 1_000_000
            mpx_per_sec = round(mpx / time_stats["mean"], 4)

        summaries.append({
            "model": model,
            "runner": runner,
            "family": spec.get("family", model),
            "vendor": spec.get("vendor", ""),
            "resolution": resolution,
            "steps": steps,
            "time": time_stats,
            "peak_mem": mem_stats,
            "mpx_per_sec": mpx_per_sec,
            "noisy": flag_noisy(time_stats.get("cv")),
            "n_runs": len(group_rows),
        })

    return {"summaries": summaries, "raw": rows}


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

def analyze_video(rows: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Video generation benchmark analysis."""
    if rows is None:
        rows = load_video_results()
    if not rows:
        return {"summaries": [], "raw": []}

    summaries = []
    for r in rows:
        spec = VIDEO_MODELS.get(r.get("model", ""), {})
        summaries.append({
            "model": r.get("model"),
            "family": spec.get("family", r.get("model", "")),
            "vendor": spec.get("vendor", ""),
            "resolution": r.get("resolution"),
            "frames": r.get("frames"),
            "total_seconds": r.get("seconds"),
            "seconds_per_frame": r.get("seconds_per_frame"),
            "peak_mem_gb": r.get("peak_mem_gb"),
            "sys_peak_mem_gb": r.get("sys_peak_mem_gb"),
        })

    return {"summaries": summaries, "raw": rows}


# ---------------------------------------------------------------------------
# Composite analysis (cross-leg)
# ---------------------------------------------------------------------------

def full_analysis() -> Dict[str, Any]:
    """Run all analyses and combine into a single result dict."""
    from system_info import collect_system_info

    return {
        "system": collect_system_info(),
        "llm": analyze_llm(),
        "image": analyze_image(),
        "video": analyze_video(),
        "has_data": has_results(),
    }


def generate_summary_json(output_path: Optional[Path] = None) -> Path:
    """Generate a comprehensive JSON summary of all results."""
    if output_path is None:
        output_path = RESULTS_DIR / "analysis_summary.json"
    data = full_analysis()
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Analysis summary written to {output_path}")
    return output_path


if __name__ == "__main__":
    result = full_analysis()
    print(json.dumps(result, indent=2, default=str))

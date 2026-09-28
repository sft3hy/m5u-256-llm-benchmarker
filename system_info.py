#!/usr/bin/env python3
"""
Auto-collect system hardware and software specs on macOS.

Uses sysctl, system_profiler, and sw_vers to gather:
  - Chip name (e.g. "Apple M5 Ultra")
  - Total RAM (GB)
  - GPU core count
  - macOS version
  - SSD model/capacity
  - CPU core counts (P-cores, E-cores)

All results returned as a flat dict, suitable for embedding in report
headers, dashboard footers, and blog article metadata.
"""

import json
import re
import subprocess
import sys
from typing import Any, Dict, Optional


def _run(cmd: list[str], default: str = "") -> str:
    """Run a command and return stripped stdout, or default on failure."""
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def _sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key])


def collect_system_info() -> Dict[str, Any]:
    """Gather a comprehensive system info dict. macOS only."""
    info: Dict[str, Any] = {}

    if sys.platform != "darwin":
        info["platform"] = sys.platform
        info["note"] = "System info collection only supported on macOS"
        return info

    # --- macOS version ---
    info["macos_version"] = _run(["sw_vers", "-productVersion"])
    info["macos_build"] = _run(["sw_vers", "-buildVersion"])

    # --- Chip / CPU ---
    info["chip"] = _sysctl("machdep.cpu.brand_string")
    # Apple Silicon doesn't always populate brand_string usefully
    if not info["chip"] or "Apple" not in info["chip"]:
        chip_name = _run(
            ["system_profiler", "SPHardwareDataType"],
        )
        m = re.search(r"Chip:\s*(.+)", chip_name)
        if m:
            info["chip"] = m.group(1).strip()

    # Core counts
    total_cores = _sysctl("hw.ncpu")
    perf_cores = _sysctl("hw.perflevel0.logicalcpu")
    eff_cores = _sysctl("hw.perflevel1.logicalcpu")
    info["cpu_cores_total"] = int(total_cores) if total_cores.isdigit() else None
    info["cpu_cores_performance"] = int(perf_cores) if perf_cores.isdigit() else None
    info["cpu_cores_efficiency"] = int(eff_cores) if eff_cores.isdigit() else None

    # --- RAM ---
    memsize = _sysctl("hw.memsize")
    if memsize.isdigit():
        info["ram_gb"] = round(int(memsize) / (1024**3))
    else:
        info["ram_gb"] = None

    # --- GPU cores (Apple Silicon) ---
    gpu_info = _run(["system_profiler", "SPDisplaysDataType"])
    gpu_cores_match = re.search(r"Total Number of Cores:\s*(\d+)", gpu_info)
    if gpu_cores_match:
        info["gpu_cores"] = int(gpu_cores_match.group(1))
    else:
        info["gpu_cores"] = None

    # GPU chipset name
    gpu_chipset_match = re.search(r"Chipset Model:\s*(.+)", gpu_info)
    if gpu_chipset_match:
        info["gpu_chipset"] = gpu_chipset_match.group(1).strip()

    # --- Neural Engine cores ---
    # Apple doesn't expose this via sysctl; hard-code known values
    ne_map = {
        "M1": 16,
        "M1 Pro": 16,
        "M1 Max": 16,
        "M1 Ultra": 32,
        "M2": 16,
        "M2 Pro": 16,
        "M2 Max": 16,
        "M2 Ultra": 32,
        "M3": 16,
        "M3 Pro": 16,
        "M3 Max": 16,
        "M3 Ultra": 32,
        "M4": 16,
        "M4 Pro": 16,
        "M4 Max": 16,
        "M4 Ultra": 32,
        "M5": 16,
        "M5 Pro": 16,
        "M5 Max": 16,
        "M5 Ultra": 32,
    }
    chip_name = info.get("chip", "")
    for key, val in sorted(ne_map.items(), key=lambda x: len(x[0]), reverse=True):
        if key in chip_name:
            info["neural_engine_cores"] = val
            break

    # --- SSD ---
    disk_info = _run(["system_profiler", "SPNVMeDataType"])
    ssd_model_match = re.search(r"Model:\s*(.+)", disk_info)
    ssd_capacity_match = re.search(r"Capacity:\s*(.+)", disk_info)
    if ssd_model_match:
        info["ssd_model"] = ssd_model_match.group(1).strip()
    if ssd_capacity_match:
        info["ssd_capacity"] = ssd_capacity_match.group(1).strip()

    # --- Memory bandwidth (known Apple Silicon values, GB/s) ---
    bw_map = {
        # M1 Series
        "M1": 68.25,
        "M1 Pro": 200,
        "M1 Max": 400,
        "M1 Ultra": 800,
        # M2 Series
        "M2": 100,
        "M2 Pro": 200,
        "M2 Max": 400,
        "M2 Ultra": 800,
        # M3 Series
        "M3": 100,
        "M3 Pro": 150,
        "M3 Max": 400,
        "M3 Ultra": 800,
        # M4 Series
        "M4": 120,
        "M4 Pro": 273,
        "M4 Max": 546,
        # "M4 Ultra": 819, # Apple skipped the M4 Ultra entirely, moving straight to M5 Ultra.
        # M5 Series
        "M5": 153,
        "M5 Pro": 307,
        "M5 Max": 614,
        "M5 Ultra": 1228,
    }
    for key, val in sorted(bw_map.items(), key=lambda x: len(x[0]), reverse=True):
        if key in chip_name:
            info["memory_bandwidth_gbs"] = val
            break

    return info


def format_system_header(info: Dict[str, Any]) -> str:
    """Format system info as a human-readable header block."""
    lines = ["System Configuration"]
    lines.append("═" * 50)

    if info.get("chip"):
        lines.append(f"  Chip:          {info['chip']}")
    if info.get("ram_gb"):
        lines.append(f"  Memory:        {info['ram_gb']} GB Unified")
    if info.get("memory_bandwidth_gbs"):
        lines.append(f"  Bandwidth:     {info['memory_bandwidth_gbs']} GB/s")
    if info.get("cpu_cores_total"):
        parts = [f"{info['cpu_cores_total']} total"]
        if info.get("cpu_cores_performance"):
            parts.append(
                f"{info['cpu_cores_performance']}P + {info.get('cpu_cores_efficiency', '?')}E"
            )
        lines.append(f"  CPU Cores:     {', '.join(parts)}")
    if info.get("gpu_cores"):
        lines.append(f"  GPU Cores:     {info['gpu_cores']}")
    if info.get("neural_engine_cores"):
        lines.append(f"  Neural Engine: {info['neural_engine_cores']} cores")
    if info.get("gpu_chipset"):
        lines.append(f"  GPU:           {info['gpu_chipset']}")
    if info.get("ssd_model"):
        cap = f" ({info['ssd_capacity']})" if info.get("ssd_capacity") else ""
        lines.append(f"  SSD:           {info['ssd_model']}{cap}")
    if info.get("macos_version"):
        build = f" ({info['macos_build']})" if info.get("macos_build") else ""
        lines.append(f"  macOS:         {info['macos_version']}{build}")

    lines.append("═" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    info = collect_system_info()
    print(format_system_header(info))
    print()
    print(json.dumps(info, indent=2))

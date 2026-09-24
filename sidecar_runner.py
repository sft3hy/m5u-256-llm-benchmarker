#!/usr/bin/env python3
"""
Sidecar runner and telemetry module for M5 Ultra benchmark kit.

Provides:
  1. BtopSidecar: Launches btop in a sidecar window (tmux split pane or macOS
     Terminal window) so the user can visually monitor CPU, memory, and GPU
     usage in real time while benchmarks run.
  2. ResourceMonitor: A high-frequency background sampling monitor tracking:
     - Process tree Resident Set Size (RSS) in GB (parent + child engines like
       llama-bench, mlx_lm, mflux-generate, etc.).
     - Process tree CPU % (sum of active core utilization).
     - System-wide unified memory usage (GB) via native Mach host statistics.
     - System-wide CPU load percentage.
  3. CLI runner: Run any benchmark wrapped with the sidecar via:
     python3 sidecar_runner.py -- python3 bench_llm.py --models qwen3.8-27b
"""

import argparse
import ctypes
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# macOS Mach kernel bindings for system memory and CPU (zero external deps)
# ---------------------------------------------------------------------------

_LIBC = ctypes.CDLL(None) if sys.platform == "darwin" else None


class _VMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _HostCpuLoadInfo(ctypes.Structure):
    _fields_ = [("cpu_ticks", ctypes.c_uint32 * 4)]  # user, sys, idle, nice


_PAGE_SIZE = 16384
if _LIBC and hasattr(_LIBC, "host_page_size"):
    _pz = ctypes.c_size_t()
    if _LIBC.host_page_size(_LIBC.mach_host_self(), ctypes.byref(_pz)) == 0:
        _PAGE_SIZE = _pz.value


def get_system_memory_used_gb() -> float:
    """Return currently used system RAM in GB using Mach host statistics."""
    if not _LIBC:
        return 0.0
    try:
        vm_stat = _VMStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(_VMStatistics64) // 4)
        HOST_VM_INFO64 = 4
        ret = _LIBC.host_statistics64(
            _LIBC.mach_host_self(),
            HOST_VM_INFO64,
            ctypes.byref(vm_stat),
            ctypes.byref(count),
        )
        if ret == 0:
            used_pages = (
                vm_stat.active_count
                + vm_stat.wire_count
                + vm_stat.compressor_page_count
            )
            return round((used_pages * _PAGE_SIZE) / (1024**3), 2)
    except Exception:
        pass
    return 0.0


def _get_cpu_ticks() -> Optional[List[int]]:
    if not _LIBC:
        return None
    try:
        info = _HostCpuLoadInfo()
        count = ctypes.c_uint32(ctypes.sizeof(_HostCpuLoadInfo) // 4)
        HOST_CPU_LOAD_INFO = 3
        ret = _LIBC.host_statistics(
            _LIBC.mach_host_self(),
            HOST_CPU_LOAD_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if ret == 0:
            return list(info.cpu_ticks)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Visual Sidecar: BtopSidecar
# ---------------------------------------------------------------------------

class BtopSidecar:
    """
    Manages a live btop sidecar window or pane.

    - In tmux: splits the current window horizontally with `btop`.
    - In macOS GUI: opens a new Terminal.app window running `btop`.
    - Headless / non-GUI: logs a notice and continues without breaking.
    """

    def __init__(self, enabled: bool = True, close_on_exit: bool = True):
        self.enabled = enabled
        self.close_on_exit = close_on_exit
        self.tmux_pane_id: Optional[str] = None
        self.terminal_window_id: Optional[str] = None
        self._started = False

    def start(self) -> bool:
        if not self.enabled:
            return False

        if not shutil.which("btop"):
            print("  [sidecar] btop not found in PATH; skipping visual sidecar.")
            return False

        # 1. Check if inside a tmux session
        if os.environ.get("TMUX") and shutil.which("tmux"):
            try:
                out = subprocess.check_output(
                    ["tmux", "split-window", "-h", "-P", "-F", "#{pane_id}", "btop"],
                    text=True,
                ).strip()
                self.tmux_pane_id = out
                self._started = True
                print(f"  [sidecar] btop running in tmux sidecar pane ({self.tmux_pane_id})")
                return True
            except Exception as e:
                print(f"  [sidecar] Could not split tmux window: {e}")

        # 2. Check if macOS desktop Terminal can be launched
        if sys.platform == "darwin" and shutil.which("osascript"):
            term_program = os.environ.get("TERM_PROGRAM", "")
            has_gui = bool(os.environ.get("DISPLAY") or term_program or os.environ.get("SSH_TTY") is None)
            if has_gui:
                try:
                    script = 'tell application "Terminal" to do script "btop"'
                    out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
                    m = re.search(r"window id (\d+)", out)
                    if m:
                        self.terminal_window_id = m.group(1)
                    self._started = True
                    print(f"  [sidecar] btop running in macOS Terminal window (id {self.terminal_window_id or 'unknown'})")
                    return True
                except Exception as e:
                    print(f"  [sidecar] Could not launch Terminal window: {e}")

        print("  [sidecar] Note: Visual btop sidecar requires an active tmux or macOS Terminal session.")
        return False

    def stop(self):
        if not self._started or not self.close_on_exit:
            return

        if self.tmux_pane_id and shutil.which("tmux"):
            try:
                subprocess.run(
                    ["tmux", "kill-pane", "-t", self.tmux_pane_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"  [sidecar] Closed tmux btop pane ({self.tmux_pane_id})")
            except Exception:
                pass
            self.tmux_pane_id = None

        if self.terminal_window_id and sys.platform == "darwin" and shutil.which("osascript"):
            try:
                script = f'tell application "Terminal" to close (window id {self.terminal_window_id})'
                subprocess.run(
                    ["osascript", "-e", script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"  [sidecar] Closed macOS Terminal btop window ({self.terminal_window_id})")
            except Exception:
                pass
            self.terminal_window_id = None

        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ---------------------------------------------------------------------------
# Telemetry Monitor: ResourceMonitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """
    High-resolution background resource sampling monitor.

    Monitors:
      - Process tree RSS memory (GB) & CPU % for target PID and its children
      - Optional named target processes (e.g. 'omlx', 'llama-bench')
      - System-wide RAM used (GB) on macOS unified memory
      - System-wide CPU load (%)

    Usage:
      with ResourceMonitor() as monitor:
          # run benchmark workload
          pass
      metrics = monitor.metrics
    """

    def __init__(
        self,
        parent_pid: Optional[int] = None,
        target_names: Optional[List[str]] = None,
        sample_interval: float = 0.15,
        record_timeseries: bool = True,
    ):
        self.parent_pid = parent_pid or os.getpid()
        self.target_names = [n.lower() for n in (target_names or [])]
        self.sample_interval = sample_interval
        self.record_timeseries = record_timeseries

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Sample storage
        self._peak_proc_rss_kb = 0
        self._proc_cpu_samples: List[float] = []
        self._peak_proc_cpu_pct = 0.0

        self._peak_sys_mem_gb = 0.0
        self._sys_cpu_samples: List[float] = []
        self._peak_sys_cpu_pct = 0.0

        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._last_cpu_ticks = _get_cpu_ticks()

        # Time-series storage (for dashboard timeline charts)
        self._timeseries: List[Dict[str, Any]] = []
        self._throttle_events: List[Dict[str, Any]] = []

    def start(self):
        self._stop_event.clear()
        self._peak_proc_rss_kb = 0
        self._proc_cpu_samples.clear()
        self._peak_proc_cpu_pct = 0.0
        self._peak_sys_mem_gb = get_system_memory_used_gb()
        self._sys_cpu_samples.clear()
        self._peak_sys_cpu_pct = 0.0
        self._last_cpu_ticks = _get_cpu_ticks()
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        self._end_time = time.time()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _sample_loop(self):
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.sample_interval)

    def _sample_once(self):
        sample_time = time.time()
        elapsed = sample_time - self._start_time

        # 1. Process tree RSS and CPU via ps
        proc_rss_kb, proc_cpu = self._collect_process_tree_metrics()
        if proc_rss_kb > self._peak_proc_rss_kb:
            self._peak_proc_rss_kb = proc_rss_kb
        if proc_cpu > self._peak_proc_cpu_pct:
            self._peak_proc_cpu_pct = proc_cpu
        self._proc_cpu_samples.append(proc_cpu)

        # 2. System memory via Mach host_statistics64
        sys_mem = get_system_memory_used_gb()
        if sys_mem > self._peak_sys_mem_gb:
            self._peak_sys_mem_gb = sys_mem

        # 3. System CPU % via Mach ticks
        sys_cpu_pct = 0.0
        curr_ticks = _get_cpu_ticks()
        if curr_ticks and self._last_cpu_ticks:
            delta = [curr_ticks[i] - self._last_cpu_ticks[i] for i in range(4)]
            total = sum(delta)
            if total > 0:
                busy = total - delta[2]  # user + sys + nice
                sys_cpu_pct = (busy / total) * 100.0
                self._sys_cpu_samples.append(sys_cpu_pct)
                if sys_cpu_pct > self._peak_sys_cpu_pct:
                    self._peak_sys_cpu_pct = sys_cpu_pct
        self._last_cpu_ticks = curr_ticks

        # 4. Record time-series sample
        if self.record_timeseries:
            proc_rss_gb = round(proc_rss_kb / (1024 * 1024), 3)
            self._timeseries.append({
                "t": round(elapsed, 3),
                "proc_mem_gb": proc_rss_gb,
                "proc_cpu_pct": round(proc_cpu, 1),
                "sys_mem_gb": sys_mem,
                "sys_cpu_pct": round(sys_cpu_pct, 1),
            })

        # 5. Throttle detection: if process CPU drops >40% from its
        #    rolling average over the last 20 samples, flag it.
        if len(self._proc_cpu_samples) >= 20:
            recent_avg = sum(self._proc_cpu_samples[-20:]) / 20
            if recent_avg > 50 and proc_cpu < recent_avg * 0.6:
                self._throttle_events.append({
                    "t": round(elapsed, 3),
                    "expected_cpu": round(recent_avg, 1),
                    "actual_cpu": round(proc_cpu, 1),
                    "drop_pct": round((1 - proc_cpu / recent_avg) * 100, 1),
                })

    def _collect_process_tree_metrics(self) -> Tuple[int, float]:
        """
        Samples RSS (in KB) and CPU % of target PID, all its child processes,
        plus any processes matching `target_names`.
        """
        try:
            out = subprocess.check_output(
                ["ps", "-ax", "-o", "pid=,ppid=,rss=,%cpu=,comm="],
                text=True,
            )
        except Exception:
            return 0, 0.0

        by_ppid: Dict[int, List[int]] = {}
        proc_data: Dict[int, Tuple[int, float, str]] = {}

        for line in out.strip().splitlines():
            parts = line.strip().split(None, 4)
            if len(parts) >= 4:
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                    rss = int(parts[2])
                    cpu = float(parts[3])
                    comm = parts[4].lower() if len(parts) > 4 else ""
                    proc_data[pid] = (rss, cpu, comm)
                    by_ppid.setdefault(ppid, []).append(pid)
                except (ValueError, IndexError):
                    continue

        target_pids: Set[int] = set()

        # Add parent and all recursive descendants
        stack = [self.parent_pid]
        while stack:
            curr = stack.pop()
            target_pids.add(curr)
            if curr in by_ppid:
                stack.extend(by_ppid[curr])

        # Also add any process matching target_names (e.g. omlx server daemon)
        if self.target_names:
            for pid, (rss, cpu, comm) in proc_data.items():
                for tname in self.target_names:
                    if tname in comm:
                        target_pids.add(pid)
                        break

        total_rss_kb = 0
        total_cpu = 0.0
        for pid in target_pids:
            if pid in proc_data:
                rss, cpu, _ = proc_data[pid]
                total_rss_kb += rss
                total_cpu += cpu

        return total_rss_kb, total_cpu

    @property
    def metrics(self) -> Dict[str, Optional[float]]:
        """
        Calculates and returns the aggregated metrics summary.
        """
        duration = max(0.001, (self._end_time or time.time()) - self._start_time)
        peak_proc_gb = round(self._peak_proc_rss_kb / (1024 * 1024), 2)
        avg_proc_cpu = (
            round(sum(self._proc_cpu_samples) / len(self._proc_cpu_samples), 1)
            if self._proc_cpu_samples
            else 0.0
        )
        avg_sys_cpu = (
            round(sum(self._sys_cpu_samples) / len(self._sys_cpu_samples), 1)
            if self._sys_cpu_samples
            else 0.0
        )

        return {
            "peak_mem_gb": peak_proc_gb if peak_proc_gb > 0 else None,
            "avg_cpu_pct": avg_proc_cpu,
            "peak_cpu_pct": round(self._peak_proc_cpu_pct, 1),
            "sys_peak_mem_gb": round(self._peak_sys_mem_gb, 2) if self._peak_sys_mem_gb > 0 else None,
            "sys_avg_cpu_pct": avg_sys_cpu,
            "duration_s": round(duration, 2),
            "samples_count": len(self._proc_cpu_samples),
            "throttle_events": len(self._throttle_events),
        }

    @property
    def time_series(self) -> List[Dict[str, Any]]:
        """Return the full time-series telemetry trace."""
        return list(self._timeseries)

    @property
    def throttle_events(self) -> List[Dict[str, Any]]:
        """Return detected thermal throttling events."""
        return list(self._throttle_events)

    def export_timeseries(self, output_path: Optional[Path] = None,
                          label: str = "benchmark") -> Optional[Path]:
        """
        Export time-series data to a JSON sidecar file.

        The JSON file contains:
          - metadata: label, start_time, duration, sample_count
          - samples: array of {t, proc_mem_gb, proc_cpu_pct, sys_mem_gb, sys_cpu_pct}
          - throttle_events: array of detected throttle drops
          - summary: same as self.metrics
        """
        if not self._timeseries:
            return None

        if output_path is None:
            results_dir = Path(__file__).parent / "results" / "timeseries"
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"{label}_{ts}.json"

        import json
        data = {
            "metadata": {
                "label": label,
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S",
                                             time.localtime(self._start_time)),
                "duration_s": round((self._end_time or time.time()) - self._start_time, 2),
                "sample_count": len(self._timeseries),
                "sample_interval_s": self.sample_interval,
            },
            "samples": self._timeseries,
            "throttle_events": self._throttle_events,
            "summary": self.metrics,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path


# ---------------------------------------------------------------------------
# CLI Command Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sidecar runner: runs btop sidecar and monitors benchmark resource usage."
    )
    parser.add_argument(
        "--no-btop",
        action="store_true",
        help="Disable the visual btop sidecar window.",
    )
    parser.add_argument(
        "--keep-btop",
        action="store_true",
        help="Keep the btop window/pane open after the benchmark finishes.",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Benchmark command to execute (e.g., python3 bench_llm.py --models ...)",
    )

    args = parser.parse_args()

    if not args.cmd or (args.cmd and args.cmd[0] == "--" and len(args.cmd) == 1):
        parser.print_help()
        sys.exit(1)

    cmd = args.cmd[1:] if args.cmd[0] == "--" else args.cmd

    sidecar = BtopSidecar(enabled=not args.no_btop, close_on_exit=not args.keep_btop)
    sidecar.start()

    print(f"\n[sidecar_runner] Executing: {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.time()
    monitor = ResourceMonitor(sample_interval=0.15)
    monitor.start()

    exit_code = 0
    try:
        proc = subprocess.Popen(cmd)
        monitor.parent_pid = proc.pid
        proc.wait()
        exit_code = proc.returncode
    except KeyboardInterrupt:
        print("\n[sidecar_runner] Interrupted by user.")
        exit_code = 130
    finally:
        monitor.stop()
        sidecar.stop()

    metrics = monitor.metrics
    elapsed = time.time() - t0

    print("\n" + "=" * 50)
    print("Sidecar Resource Summary")
    print("=" * 50)
    print(f"  Duration:         {elapsed:.2f}s")
    print(f"  Peak Process RAM: {metrics['peak_mem_gb']} GB")
    print(f"  Avg Process CPU:  {metrics['avg_cpu_pct']}%")
    print(f"  Peak Process CPU: {metrics['peak_cpu_pct']}%")
    if metrics.get("sys_peak_mem_gb"):
        print(f"  Peak System RAM:  {metrics['sys_peak_mem_gb']} GB")
    print("=" * 50 + "\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

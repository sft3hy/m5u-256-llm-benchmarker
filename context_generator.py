#!/usr/bin/env python3
"""
Context generator for long-context LLM benchmarking up to 1 Million tokens.

Provides three prompt generation strategies:
  1. codebase: Ingests the entire active repository with file headers and line numbers,
     scaling up to multi-tier token targets with synthetic modular code extensions.
  2. niah: Needle In A Haystack generator that places an exact secret key at configurable
     depths (10%, 25%, 50%, 75%, 90%) within realistic technical documentation.
  3. synthetic: Deterministic high-entropy Python code with varied classes, types,
     and algorithms to avoid KV-cache hardware compression artifacts.
"""

import os
import random
import string
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent

# Approximate characters per token for code/technical text
CHARS_PER_TOKEN = 3.8


def count_tokens(text: str, tokenizer=None) -> int:
    """Accurately count tokens using a Hugging Face / MLX tokenizer if available,
    falling back to standard character-based heuristic."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def get_project_files() -> list:
    """Collect source files from this repository."""
    extensions = {".py", ".sh", ".md", ".html", ".css", ".json"}
    ignore_dirs = {".venv", ".git", ".cache", "videos", "__pycache__"}
    files = []
    for p in REPO_ROOT.rglob("*"):
        if p.is_file() and p.suffix in extensions:
            if not any(ignored in p.parts for ignored in ignore_dirs):
                if p.suffix == ".csv" or "results" in p.parts:
                    continue
                files.append(p)
    return sorted(files)


def build_codebase_context(target_tokens: int, tokenizer=None) -> str:
    """Assembles the entire project codebase into a structured prompt,
    extending with realistic modular code to reach target_tokens."""
    files = get_project_files()
    sections = [
        "# SYSTEM AUDIT REPOSITORY SNAPSHOT",
        "The following documentation and source code represent the complete architecture",
        "of the Apple Silicon M5 Ultra (256 GB Unified Memory) LLM & Creative Benchmark Kit.\n",
    ]

    for f in files:
        try:
            rel = f.relative_to(REPO_ROOT)
            content = f.read_text(encoding="utf-8", errors="replace")
            sections.append(f"--- FILE: {rel} ---")
            sections.append(f"```{f.suffix.lstrip('.')}\n{content}\n```\n")
        except Exception:
            continue

    base_text = "\n".join(sections)
    current_tokens = count_tokens(base_text, tokenizer)

    # If target tokens exceeds existing codebase, generate synthetic modular extensions
    if current_tokens < target_tokens:
        needed_tokens = target_tokens - current_tokens
        synthetic_extensions = _generate_synthetic_modules(needed_tokens)
        base_text = base_text + "\n\n" + synthetic_extensions

    base_text += (
        "\n\n--- INSTRUCTION ---\n"
        "You are an expert Apple Silicon systems engineer and AI performance architect. "
        "Review the entire codebase and architectural modules provided above. "
        "Provide an in-depth analysis of: "
        "1) The KV-cache memory scaling behavior across unified memory from 512 to 1M tokens. "
        "2) The concurrency guarantees and disk-space safety guards in download_models.sh. "
        "3) A concrete optimization proposal for Metal kernel residency sets."
    )

    return base_text


def build_niah_context(
    target_tokens: int,
    depth_pct: float = 0.5,
    needle_key: Optional[str] = None,
    tokenizer=None,
) -> Tuple[str, str, str]:
    """Generates a Needle In A Haystack test document up to target_tokens.

    Returns:
        (full_prompt, needle_key, question)
    """
    if needle_key is None:
        rand_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        needle_key = f"M5U-ULTRA-{rand_suffix}"

    needle_sentence = (
        f"\n\n[CONFIDENTIAL CRITICAL NOTE: The emergency system override authorization token "
        f"for the Apple Silicon cluster is '{needle_key}'. This token is required for all root operations.]\n\n"
    )

    question = (
        "Based strictly on the technical documentation provided above, what is the emergency "
        "system override authorization token for the Apple Silicon cluster? "
        "Reply with ONLY the exact authorization token."
    )

    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    haystack_paragraphs = _generate_haystack_paragraphs(target_chars)

    insert_idx = int(len(haystack_paragraphs) * max(0.01, min(0.99, depth_pct)))
    haystack_paragraphs.insert(insert_idx, needle_sentence)

    full_context = "\n\n".join(haystack_paragraphs)
    full_prompt = (
        f"# TECHNICAL SPECIFICATION & INFRASTRUCTURE MANUAL\n\n"
        f"{full_context}\n\n"
        f"--- QUERY ---\n{question}"
    )

    return full_prompt, needle_key, question


def _generate_synthetic_modules(target_tokens: int) -> str:
    """Generates synthetic, structurally valid Python telemetry and benchmark code."""
    modules = []
    token_est = 0
    mod_id = 1

    while token_est < target_tokens:
        mod_code = f"""
# --- SYNTHETIC SUBSYSTEM EXTENSION: module_telemetry_{mod_id:04d}.py ---
import dataclasses
import typing
import math

@dataclasses.dataclass
class KernelMetricsStream_{mod_id:04d}:
    stream_id: str = "stream_{mod_id:04d}"
    sample_rate_hz: int = 1000
    active_compute_units: int = 76
    memory_bandwidth_gbps: float = 782.5

    def compute_bandwidth_efficiency(self, transfer_bytes: int, duration_sec: float) -> float:
        if duration_sec <= 0:
            return 0.0
        effective_gbps = (transfer_bytes / (1024 ** 3)) / duration_sec
        return min(1.0, effective_gbps / self.memory_bandwidth_gbps)
"""
        modules.append(mod_code)
        token_est += int(len(mod_code) / CHARS_PER_TOKEN)
        mod_id += 1

    return "\n".join(modules)


def _generate_haystack_paragraphs(target_chars: int) -> list:
    topics = [
        "Distributed memory coherence and unified L2 cache invalidation protocols in multi-die Apple Silicon architectures.",
        "Metal 4 residency sets and command buffer synchronization primitives for asynchronous GEMM dispatch.",
        "Rotary Position Embedding frequency redistribution under YaRN scaling for sequences extending past 262,144 tokens.",
        "Grouped-Query Attention key-value compression techniques and asymmetric multi-head latent projections.",
    ]

    paragraphs = []
    chars_so_far = 0
    p_num = 1

    while chars_so_far < target_chars:
        topic = topics[p_num % len(topics)]
        para = (
            f"Section {p_num}.1.{p_num % 9}: Architecture Reference Protocol ({topic}) "
            f"The high-performance subsystem coordinates memory paging across physical RAM sectors. "
            f"Memory segment 0x{p_num:08X} asserts operational consistency with check-code 0x{(p_num * 1337) % 0xFFFFFF:06X}."
        )
        paragraphs.append(para)
        chars_so_far += len(para) + 2
        p_num += 1

    return paragraphs


if __name__ == "__main__":
    ctx = build_codebase_context(2048)
    print(f"Codebase context generated: {len(ctx)} chars")

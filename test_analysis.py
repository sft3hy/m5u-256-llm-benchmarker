#!/usr/bin/env python3
"""Regression tests for the results pipeline (stdlib unittest, no new deps).

Run from the repo root:
    python3 -m unittest test_analysis -v

These lock down bugs that used to corrupt benchmark results silently:
  * a non-finite number (inf/NaN) in a results CSV either crashed the loader
    or poisoned every mean/CV/ranking downstream;
  * image runs at different resolutions were averaged into one bucket, and
    the MPix/s figure used whichever resolution happened to come first;
  * the mlx-lm parsers only see output if --verbose is on the command line.
"""

import math
import tempfile
import unittest
from pathlib import Path

import analysis
import bench_context
import bench_llm


CSV_FIXTURE = """\
model,quant,engine,decode_tok_s,peak_mem_gb
foo,4bit,mlx-lm,nan,
bar,4bit,omlx,inf,12.0
baz,4bit,llama.cpp,,
"""

# Exactly what `mlx_lm generate --verbose` prints (checked against mlx-lm
# 0.31.3). Without --verbose the CLI prints none of these lines, so every
# rate would silently fall back to a wall-clock figure.
MLX_VERBOSE_OUTPUT = (
    "==========\n"
    "Some generated text.\n"
    "==========\n"
    "Prompt: 512 tokens, 950.195 tokens-per-sec\n"
    "Generation: 128 tokens, 45.256 tokens-per-sec\n"
    "Peak memory: 15.344 GB\n"
)


class TestLoadCsv(unittest.TestCase):
    def test_non_finite_values_do_not_crash_and_become_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text(CSV_FIXTURE, encoding="utf-8")
            rows = analysis._load_csv(path)

        self.assertEqual(len(rows), 3)
        for row in rows:
            for key in ("decode_tok_s", "peak_mem_gb"):
                value = row[key]
                if value is not None:
                    self.assertTrue(math.isfinite(value), f"{key}={value!r}")
        # 'inf' used to raise OverflowError before it got this far.
        self.assertIsNone(rows[1]["decode_tok_s"])

    def test_clean_numbers_keep_their_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y.csv"
            path.write_text("a,b,c\n1.0,2.5,hello\n", encoding="utf-8")
            row = analysis._load_csv(path)[0]

        # Integral floats stay ints: context_tokens, resolution and steps
        # are looked up and labelled as integers elsewhere in the kit.
        self.assertIsInstance(row["a"], int)
        self.assertAlmostEqual(row["b"], 2.5)
        self.assertEqual(row["c"], "hello")


class TestStats(unittest.TestCase):
    def test_nan_is_excluded_from_statistics(self):
        result = analysis.stats([10.0, float("nan"), 12.0])
        self.assertTrue(math.isfinite(result["mean"]), result)
        self.assertEqual(result["n"], 2)

    def test_all_non_finite_yields_empty_statistics(self):
        result = analysis.stats([float("nan"), float("inf")])
        self.assertIsNone(result["mean"])
        self.assertEqual(result["n"], 0)


class TestLlmAnalysis(unittest.TestCase):
    def test_non_finite_decode_time_is_not_ranked(self):
        rows = [
            {"model": "foo", "quant": "4bit", "engine": "mlx-lm",
             "decode_tok_s": float("nan"), "peak_mem_gb": 10.0},
            {"model": "bar", "quant": "4bit", "engine": "mlx-lm",
             "decode_tok_s": 40.0, "peak_mem_gb": 10.0},
        ]
        result = analysis.analyze_llm(rows=rows)

        ranked = {r["model"] for r in result["rankings"]["by_decode_speed"]}
        self.assertNotIn("foo", ranked, "a NaN decode rate still ranked")
        self.assertIn("bar", ranked, "the only real rate lost its rank slot")


class TestImageAnalysis(unittest.TestCase):
    def test_resolutions_are_not_averaged_together(self):
        rows = [
            {"model": "flux2-dev", "runner": "diffusers", "resolution": 512,
             "steps": 20, "seconds": 3.0, "peak_mem_gb": 20.0},
            {"model": "flux2-dev", "runner": "diffusers", "resolution": 1024,
             "steps": 20, "seconds": 24.0, "peak_mem_gb": 22.0},
        ]
        summaries = analysis.analyze_image(rows=rows)["summaries"]

        self.assertEqual(len(summaries), 2, "both resolutions merged into one")
        by_res = {s["resolution"]: s for s in summaries}
        # 1024x1024 is four times the pixels of 512x512 at a fraction of the
        # throughput. If the resolution leaks between buckets, both figures
        # collapse onto the resolution of whichever row came first.
        self.assertAlmostEqual(by_res[512]["mpx_per_sec"], 0.08738, places=4)
        self.assertAlmostEqual(by_res[1024]["mpx_per_sec"], 0.04369, places=4)


class TestMlxOutputParsing(unittest.TestCase):
    """The benchmarkers shell out to `mlx_lm generate`, and these parsers are
    the only thing that turns its stdout into prefill/decode/memory columns,
    so a change in that output format is worth failing on."""

    def test_parses_rates_and_peak_memory(self):
        for module in (bench_llm, bench_context):
            with self.subTest(module=module.__name__):
                out = MLX_VERBOSE_OUTPUT
                self.assertAlmostEqual(module._extract_tps(out, "Prompt"), 950.195)
                self.assertAlmostEqual(module._extract_tps(out, "Generation"), 45.256)
                self.assertAlmostEqual(module._extract_peak_mem(out), 15.344)
                self.assertIsNone(module._extract_peak_mem("no telemetry here"))

    def test_mlx_commands_request_verbose_output(self):
        # --verbose gates every line the parsers above read, so a command
        # without it records timings that look real but measure nothing.
        for module in (bench_llm, bench_context):
            with self.subTest(module=module.__name__):
                source = Path(module.__file__).read_text(encoding="utf-8")
                self.assertIn('"--verbose"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

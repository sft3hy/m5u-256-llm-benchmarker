# Nearest Claude Model, Per Open-Weight Model You're Benchmarking

Researched Sept 22, 2026. Every model here was checked against live sources
(release announcements, HF model cards, and third-party benchmark
aggregators) rather than pulled from memory, since most of this list
released in the last 6-8 weeks.

**The one important caveat, up front:** almost every public benchmark
number below is for the *full-precision, often cloud-hosted* version of
these models. You'll be running quantized versions (1-bit for GLM-5.3,
4/8-bit for everything else) on a single machine. Quantization loss is
real and roughly correlates with how aggressive the quant is — treat the
numbers here as a **ceiling**, and treat your own `bench_llm.py` quality
spot-checks (not just tok/s) as the number that actually matters for your
setup. This is exactly why a 256GB local rig is an interesting benchmark
subject: it's the first Mac configuration where you can run the 744B-class
open models at all, even if only at a quant low enough to blunt their edge.

---

## Agentic coding models

### Qwen3.8-27B → closest to **Claude Sonnet 5**
Your own instinct in the original ask was right. Head-to-head data across
five shared benchmarks has Sonnet 5 ahead on average (roughly an 8-9 point
gap), but the picture is mixed rather than a clean win: Sonnet 5 leads
DeepSWE and Terminal-Bench 2.1 by meaningful margins, while Qwen3.8-27B
actually edges ahead on OSWorld-Verified (computer-use agent tasks). For a
27B dense model running locally against a frontier cloud model, that's
closer than parameter count alone would predict — this pair is a good
benchmark target for your kit specifically because the gap is small enough
that quantization choice (4-bit vs 8-bit) could plausibly flip individual
benchmark results.

### Qwen3.8-Flash-Next → closest to **Claude Sonnet 5 / Haiku 4.5**, unsettled
This is explicitly a Qwen4 architecture preview (new attention, MoE
routing, and a large n-gram embedding table), not a finished flagship —
Alibaba's own materials frame it as an early look, not a benchmark
flex. No mature independent benchmark suite exists yet. Expect it to land
somewhere between Haiku 4.5 and Sonnet 5 on raw capability, but treat any
placement here as provisional until more third-party evals land.

### GLM-5.3 → closest to **Claude Mythos 5.1 / Opus-tier**, at full precision
This is the headline result of your whole list: Zhipu's own benchmark
claims put GLM-5.3 at 84.5 on CyberGym, ahead of both Anthropic's Mythos 5
(83.8) and OpenAI's GPT-5.6 Sol (83.6) — a claim about vulnerability-finding
and cyber-defense specifically, not general capability, but it's a genuine
frontier-tier result from an open-weight model. **This is also why Zhipu
delayed the open-weight release** — they cited the model's cyber-offense
capability as the reason for holding weights back two weeks for safety
review. The 744B/40B-active model card also shows real Terminal-Bench 3.0
gains (4.6 → 28.3) over GLM-5.2 from post-training alone, no architecture
change. **None of this transfers cleanly to the 1-bit quant you'll run** —
this is the single biggest quality-vs-quantization gap in your whole list,
since you're going from FP8 to roughly 1/8th the bits per weight on a
744B-parameter model specifically to fit 256GB. Expect a large, visible
quality drop versus these numbers; the interesting benchmark question is
*how much* it drops, not whether it drops.

### GLM-5.3-Flash → closest to **Claude Haiku 4.5**, unconfirmed
Released Aug 26, 2026 (previewed anonymously as "Ox Alpha"), MIT-licensed,
natively multimodal, roughly a tenth of GLM-5.3's API price — which points
at a mid-size model, but I couldn't confirm an exact parameter count from
public sources during this research pass. Check the Hugging Face model
card before you finalize which Claude tier to compare it against.

### DeepSeek-V4-Flash-0731 → closest to **Claude Haiku 4.5 / low Sonnet**
Coverage frames it as delivering performance comparable to OpenAI's
GPT-5.6 Luna (a mid-tier model) at a meaningfully lower cost, and DeepSeek
itself claims V4-Flash now outperforms its own larger V4-Pro flagship on
published agent benchmarks — a "smaller model, better post-training" story
similar to GLM-5.3's. At 284B total / 13B active, it's MoE-efficient
enough that decode speed on your Mac should be good even though the
on-disk footprint is large. Reminder from the README: don't attempt the
8-bit quant on a 256GB machine — it doesn't fit.

---

## American models

### Muse Glimmer → closest to **Claude Haiku 4.5**, below it
Meta's framing of its own release is candid: a 30B distillation of the
closed Muse Spark model, explicitly positioned as a local-agent play
rather than a frontier competitor, and Meta's own comparison charts show
it trailing Qwen3.6-27B on OSWorld-Verified and Terminal-Bench 2.1 while
leading on MCP-Atlas and DeepSearch QA. Read as: strong for its size,
built for tool-use reliability over raw benchmark ceiling, not a Haiku
replacement in general capability.

### Gemma4-31B → closest to **Claude Haiku 4.5**
Google's own release materials cite a #3 placement on Arena AI's text
leaderboard for the 31B dense variant — a genuinely strong showing for an
Apache 2.0 model this size. In Meta's Muse Glimmer comparison charts,
Gemma4-31B is the stronger all-rounder of that pair. Treat it as
competitive with Haiku 4.5 on general knowledge and reasoning, with
coding likely a notch behind.

### gpt-oss-120b → closest to **Claude Haiku 4.5**, trailing on hard benchmarks
Direct head-to-head data: Haiku 4.5 beats gpt-oss-120b on every measured
category (knowledge, coding, math, reasoning), by roughly 15 points overall
(57 vs 42). gpt-oss-120b's real advantage is architectural efficiency —
~5.1B active parameters out of 117B total, natively shipped in MXFP4 — so
on your machine it should be one of the fastest decode speeds in the whole
list relative to its benchmark tier. Good "best tokens/sec per capability
point" data point for your kit.

---

## Quick-reference table

| Open model | Params (total/active) | Nearest Claude model | Confidence |
|---|---|---|---|
| Qwen3.8-27B | 27B dense | Sonnet 5 | Solid — direct benchmark data |
| Qwen3.8-Flash-Next | 125B/6B (+51B n-gram) | Sonnet 5 / Haiku 4.5 | Low — too new for independent evals |
| GLM-5.3 (full precision) | 744B/40B | Mythos 5.1 / Opus tier | Solid on cited benchmark, narrow (cyber) scope |
| GLM-5.3 (1-bit, as you'll run it) | 744B/40B | Unknown — untested territory | Speculative, this is your kit's job to find out |
| GLM-5.3-Flash | Unconfirmed | Haiku 4.5 | Low — size unconfirmed |
| DeepSeek-V4-Flash-0731 | 284B/13B | Haiku 4.5 / low Sonnet | Moderate — comparative claims, not shared-benchmark table |
| Muse Glimmer | 30B dense | Below Haiku 4.5 | Moderate — vendor's own comparison data |
| Gemma4-31B | 31B dense | Haiku 4.5 | Moderate — vendor leaderboard claim |
| gpt-oss-120b | 117B/5.1B | Below Haiku 4.5 | Solid — direct benchmark data |

## Suggested benchmark angle given all this

The most interesting story your kit can produce isn't "how does open model
X compare to Claude" in the abstract — public benchmark sites already do
that, imperfectly. It's **how much of that gap survives quantization on
this specific machine**. Two things worth measuring that a cloud benchmark
can't tell you:

1. **GLM-5.3 at 1-bit vs its own FP8 cited scores.** If you can run even a
   handful of CyberGym-style or coding tasks against both the API (full
   precision) and your local 1-bit build, the delta is a genuinely new data
   point — nobody's published a 1-bit-on-256GB-unified-memory result for a
   744B model yet as far as this research turned up.
2. **Qwen3.8-27B at 4-bit vs 8-bit vs Sonnet 5**, since the public
   comparison above is presumably against a higher-precision Qwen release.
   If your 8-bit local run closes most of the gap to Sonnet 5 on
   OSWorld-Verified specifically, that's a concrete "this workload is
   viable to run locally instead of via API" finding.

---

## Sources consulted

- gigazine.net coverage of Qwen3.8, Qwen3.8-27B, Qwen3.8-Flash-Next, GLM-5.3, GLM-5.3 open-weight release, DeepSeek-V4-Flash-0731
- datalearner.com model cards and benchmark comparison pages (Qwen3.8-Max, Qwen3.8-27B vs Claude Sonnet 5)
- geotoolbox.ai explainer on GLM-5.3 vs GLM-5.3-Flash naming
- aiweekly.co / mindstudio.ai coverage of GLM-5.3's open-weight delay and CyberGym score
- theregister.com, thenextweb.com, phoronix.com, datacamp.com, betanews.com, mer.vin coverage of Muse Glimmer
- en.wikipedia.org "Gemma (language model)" and ai.google.dev release notes for Gemma 4
- letsdatascience.com coverage of the Gemma 4 release
- benchlm.ai head-to-head pages (Qwen3.8-27B vs Claude Sonnet 5/4.5, Haiku 4.5 vs GPT-OSS 120B)
- anotherwrapper.com pricing/benchmark comparison pages (Claude Haiku tiers vs GPT-OSS 120B)
- en.wikipedia.org "Flux (text-to-image model)", alternativeto.net, cloudflare blog on Flux.2
- macrumors.com, macworld.com, guru3d.com, iclarified.com on the M5 Ultra Mac Studio launch

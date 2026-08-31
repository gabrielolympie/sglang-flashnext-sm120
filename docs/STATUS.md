# Qwen3.8-Flash-Next NVFP4 — local TP1 deployment (RTX PRO 6000 Blackwell, 96 GB)

**Status (2026-08-31, after the 8h optimization run): DONE — serving, heavily optimized, validated.**
Official sglang `qwen4-main-squashed` branch + local commits on `sm120-wy` (see git log in
`../sglang-official`). No Docker. **Beats jpezzulli/sglang-rtxpro6000's published figures by ~35-45%.**

## Result (stable `serve_best.sh` build, warm)
| | jpezzulli | ours (temp 0.6) | ours (greedy = lossless) |
|---|---|---|---|
| Decode C1 | 171 tok/s | **231.0 median / 234.6 best** | 202.9 / 205.0 |
| Decode C4 aggregate | 428 | **620 / 628.7** | 549 / 560 |
| Decode C8 aggregate | — | **758** | |
| Prefill (2.5K) | ~10-12K | ~10.4K tok/s (17.8K cached) | |
| TTFT | — | ~139 ms | |
| 16K-context decode | — | ~265 tok/s (no degradation) | |
| 20-min soak + 60-min burn-in | — | 2.38M toks total, 0 errors, VRAM/RAM flat | |
| MTP accept length | 2.58 | 2.5-3.0 (relaxed 0.3) | 2.1-2.2 |
Context: 131072 default (KV pool ~411K fp8 tokens is VRAM-bound — same pool as 32K, same speed).
76K-needle stress: edges + most middle depths retrieve; occasional middle-depth misses are
model/QSA-inherent (reproduced with bf16 KV — not our fp8).
Correctness gates (all PASS, final build): greedy arithmetic/fact · 3× needles in 7.3K prompt ·
cached-prefix identical · 5-8× GSM-style @0.6 · code spot · French. VRAM peak 95.5 GB.

## Run
```bash
./serve_best.sh                         # persistent user service `qwen-sglang`, ready in ~5 min
curl -s http://127.0.0.1:8001/health    # 200 when ready
systemctl --user stop qwen-sglang       # stop
tail -f logs/serve.log
```
Endpoint **http://localhost:8001/v1**, model **`pennyroyal`** (OpenAI-compatible; thinking on by
default → tokens in `delta.reasoning_content`). Or from the laptop: `omega --update` then
`omega --serve qwen3.8-flash-next`.

## What made it fast (2026-08-31 run, in order applied)
1. **sm120 Triton low-M GEMM** (commit 086a37f): cuBLAS-under-capture served the decode dense
   GEMMs at 20-75% of DRAM bandwidth; a split-free (1,16,128) Triton kernel reaches ~90%.
   Env `SGLANG_ENABLE_SM120_LOWM_BF16_GEMM` (default on for sm120).
2. **W8A16 fp8 weight-only** (284d7fe): per-row-scaled fp8e4m3 copies of every dense weight
   ≥4 MB, served by a (1,32,128) kernel at 89% of the halved floor. `SGLANG_SM120_LOWM_FP8_WEIGHT=1`,
   costs ~3.6 GB VRAM → `MEMFRAC=0.95`.
3. **fp8 HC mix + fp8 lm_head** (85e7da1): HC persistent kernel 14.5→10.8 µs/mix; lm_head GEMV
   (and every MTP draft step's logits) byte-halved. `SGLANG_SM120_LM_HEAD_FP8=1`.
4. **Relaxed MTP acceptance** (config): `SPEC_ACCEPT_SINGLE/ACC=0.3` force-accepts draft tokens
   the target gives ≥30% prob. Lossy at temp>0 (sharpens sampling; exact at temp 0).
   Ladder (C1 @0.6): 1.0 lossless=179 · 0.5=203 · **0.3=231**. All quality gates pass at 0.3.
5. **FR-Spec 64K hot-token map** (config): draft lm_head scores a 64K subset of the 248K vocab
   (`hot_tokens_64k.pt` = 32K base BPE + code-corpus top tokens + specials). Verify stays exact.
6. **8-way concurrency** (config): `MAXREQ=8 CUDAGRAPH_MAXBS=8 MAMBA_CACHE=48` — the mamba cache
   must be ~6× MAXREQ or spec graphs silently cap at bs4 (C8 was *slower* than C4 before this).

## Knobs (env → serve.sh)
`SPEC_ACCEPT_SINGLE/ACC` (0.3; 1.0 = lossless) · `SPEC_TOKEN_MAP` (path or `none`) ·
`SGLANG_SM120_LOWM_FP8_WEIGHT` / `SGLANG_SM120_LM_HEAD_FP8` (fp8 off ⇒ pure-bf16 kernels) ·
`MAXREQ/CUDAGRAPH_MAXBS/MAMBA_CACHE` (8/8/48) · `MEMFRAC` (0.95) · `CTX` (131072; native 262144) ·
`KVDTYPE=fp8_e4m3` · `LINEAR_BACKEND=flashinfer` · `GDN_MTP_CACHE_MODE=none` (WY/RecoverSSM).

## Engine patches (branch `sm120-wy` @ ../sglang-official)
`0002` fp8-QSA dequant · `0003` sm120 fp32 prefill state · `0001b` RecoverSSM/WY port (+1316) ·
`086a37f` sm120 Triton low-M GEMM · `284d7fe` fp8 weight-only · `85e7da1` fp8 HC + lm_head.
Re-apply on a fresh checkout: see patches/README.md + cherry-pick the three commits.

## Hard-won gotchas
Desktop crash = full-range graph capture → keep CUDAGRAPH_MAXBS small. RAM thrash = uncapped
cicc JIT → MAX_JOBS=4 + systemd MemoryMax=112G. `pkill -f sglang` self-matches the tool shell —
kill by PID / `systemctl --user stop qwen-sglang`. First bench after restart is JIT-polluted —
always warm first. Profiler CPU-annotation windows lie about GPU time (async) — attribute
kernels by correlation ID; micro-benches must rotate HBM-cold weights AND serialize by stream
order (independent kernels in one graph run concurrently). torch.compile is a dead end here
(custom fused ops raise NotImplementedError in forward_native).

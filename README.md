# sglang-flashnext-sm120

**Qwen3.8-Flash-Next (180B MoE, NVFP4) at 231 tok/s single-stream on a single RTX PRO 6000 Blackwell (96 GB, sm120).**

Patches, launch scripts and benchmarks for serving `RadixArk/Qwen3.8-Flash-Next-NVFP4` at TP1
on the official SGLang `qwen4-main-squashed` branch — no Docker, no fork. Builds on the sm120
groundwork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000),
then goes ~35-45% past its published numbers.

## Results

| | jpezzulli | this repo (temp 0.6) | this repo (greedy, lossless) |
|---|---|---|---|
| Decode, 1 stream | 171 tok/s | **231 median / 243 best** | 203 |
| Decode, 4 streams | 428 | **620–657** | 549 |
| Decode, 8 streams | — | **758** | |
| Prefill (2.5K) | 10–12K | ~10.4K tok/s | |
| TTFT | — | ~135 ms | |
| Context | 32K (524K w/ YaRN) | **262144** (full native window, no speed cost) | |

Validated with greedy/needle/cached-prefix/GSM/code gates and 2.4M tokens of soak testing
(0 errors, flat VRAM/RAM).

## Contents

```
patches/            six patches against sgl-project/sglang @ qwen4-main-squashed
scripts/            serve.sh (knobbed launcher) · serve_best.sh · bench · hot-token map gen
docs/               STATUS.md (ops guide) · PERF_CEILING.md (analysis + dead-ends)
results/            benchmark JSONs, baseline -> final
hot_tokens_64k.pt   FR-Spec draft-vocab map
```

## The optimizations

**Patches** (0001–0003 unblock sm120; 0004–0006 are the speed work, all env-gated):
1. `0001b` — RecoverSSM + WY output-only MTP verify on FlashInfer for sm120.
2. `0002` — FP8-KV tile dequant for the QSA sparse prefill (2× KV capacity).
3. `0003` — fp32 prefill state for the sm120 FlashInfer GDN kernel.
4. `0004` — Triton low-M GEMM: cuBLAS-under-graph-capture runs the decode projections at
   20–75% of DRAM bandwidth on sm120; this kernel reaches ~90%.
5. `0005` — W8A16 fp8 weight-only serving of the dense bf16 stack (85% of per-step traffic,
   untouched by the NVFP4 checkpoint). Runtime-only: the checkpoint is never modified.
6. `0006` — same fp8 treatment for the HyperConnection mix and the lm_head (also halves
   every MTP draft step's logits).

**Config levers** (in `scripts/serve.sh`, each documented inline with its measured ladder):
- Relaxed MTP acceptance `0.3` (C1 179 → 231; exact at temp 0, set `1.0` for lossless sampling).
- FR-Spec: draft head scores a 64K hot-token subset of the 248K vocab (verify stays exact).
- 8-way concurrency: `--max-mamba-cache-size` must be ~6× max-running-requests or the
  speculative CUDA graphs silently cap at bs=4 (8-way used to run *slower* than 4-way).

## Reproduce

```bash
# model checkpoint (~135 GB; the ~50 GB PLE n-gram table is served from host RAM)
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir Qwen3.8-Flash-Next-NVFP4
# note: hf_xet can stall on the largest shards; scripts/serve.sh's docs and
# docs/STATUS.md describe the curl fallback that resumes reliably.

git clone -b qwen4-main-squashed https://github.com/sgl-project/sglang sglang-official
cd sglang-official && bash ../scripts/do_build.sh
git apply ../patches/0002-fp8-qsa-tile-dequant.patch --exclude='test/*'
git apply ../patches/0003-sm120-fp32-prefill-state.patch
git apply ../patches/0001b-recoverssm-wy-sm120-PORTED.patch
git am    ../patches/0004*.patch ../patches/0005*.patch ../patches/0006*.patch
# point scripts/serve.sh at your paths, then:
./scripts/serve_best.sh        # OpenAI API on :8001, ~5 min to ready
```

Single-GPU-with-display safety knobs (learned the hard way): keep `--cuda-graph-max-bs`
small, cap JIT compilation with `MAX_JOBS=4`, run under a systemd `MemoryMax` cage.
Details and every explored dead-end: `docs/`.

## Credits

- [sgl-project/sglang](https://github.com/sgl-project/sglang), branch `qwen4-main-squashed`
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) — the sm120
  RecoverSSM/WY and fp8-QSA work this repo ports and builds on

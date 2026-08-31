# Performance ceiling analysis — Qwen3.8-Flash-Next NVFP4, TP1, RTX PRO 6000 Blackwell (Max-Q, 300 W)

*Draft 2026-08-30 20:50; revised 2026-08-31 during the 8h optimization run — see "8h run findings" below.*

## 8h run findings (2026-08-31, supersedes parts of the analysis below)
**The "GPU 70% idle / host-bound" diagnosis was WRONG** — an artifact of the profile window
containing two eager 123ms tiny-prefill steps (the profiler's own 5-token prompts) plus
profiler CPU inflation. Correlation-ID attribution shows steady-state MTP decode is
**GPU-bound at ~99% busy**: verify graph 14.43 ms + draft 2.67 + draft_extend 1.28 ≈ 18.4 ms
GPU per step (wall ~15.9 ms, some overlap). Lever A (host overhead) is mostly closed for C1.

Verify-graph composition (14.43 ms, 2143 kernels/replay):
bf16 GEMMs 5.91 ms (41%; lm_head 1.57 at 94% BW — fine) · NVFP4 MoE+glue 3.72 (26%, at floor)
· HC 1.71 (12%, 55-63% of floor, block-retune flat → only FP8 shrink helps) · QSA 0.62 · GDN 0.46
· tiny-op soup ~1.44 (elementwise/fills/memops ~830 kernels).

Shape correction: the "n=4120 at 19%" read was wrong — the fused GDN in_proj is
**n=16480, k=2560** (84.4 MB/layer, grid.x=8 was N-tiling not split-K), so cuBLAS-in-graph was
at ~75% (65.9 µs vs 49.3 floor), not 19%. Mid shapes (n=2560·k=1536 ×103/step, n=1280, n=512)
were genuinely bad (30-50%).

**WIN #1 (committed 086a37f): sm120 Triton low-M GEMM** in `bf16_gemm_dispatch` +
`UnquantizedLinearMethod.apply` — (1,16,128,w4,st4) hits 90% BW on n=16480 (54.7 µs).
Verify 14.43 → 13.79 ms; C1 greedy median 142.8 → 147.9 tok/s (accept ~2.2).
**IN FLIGHT: W8A16 fp8 weight-only** in the same path (`SGLANG_SM120_LOWM_FP8_WEIGHT`):
per-row-scaled fp8e4m3 copies cached at data_ptr+version, (1,32,128) tactic = 89% of the
halved floor (27.8 µs on n=16480). Needs MEMFRAC 0.96 (+~3.3 GB fp8 copies).

Closed levers (do not retry): `--enable-torch-compile` (fused ops raise NotImplementedError
in forward_native under dynamo → capture fails); HC mix block retune (61-63% across all
configs, barrier/latency-bound); CUBLAS_WORKSPACE_CONFIG (no effect on in-graph kernel picks);
micro-benching GEMMs with a reused weight (L2-resident, reads 180% of "bandwidth") or with
eager per-launch event timing (measures the ~30 µs Python launch gap, not the kernel).
In-server kernels run ~15% slower than standalone graph replays (sustained 300 W vs burst).

## FINAL (2026-08-31 ~03:30, end of run) — see STATUS.md for the full table
C1 231 median @0.6 (lossless greedy 203) · C4 620 · C8 758 · accept 2.5-3.0 · all gates pass.
Wins after the GEMM kernel: fp8 weight-only dense (284d7fe), fp8 HC + lm_head (85e7da1),
relaxed acceptance 0.3 (+13%+), FR-Spec 64K hot-token draft head (+9%), 8-way profile
(MAMBA_CACHE=48 unlocked spec graphs >bs4; C8 487→758).

Final per-step GPU composition (13.3 ms under profiler, ~11 ms real; verify+draft+extend):
fp8w GEMMs 3.43 (89% of fp8 floor) · NVFP4 MoE 3.21 (at floor) · HC 1.64 (barrier-bound) ·
MoE TRT-LLM glue 0.88 · small bf16 GEMMs 0.86 (latency-bound) · GDN 0.76 · aten misc 0.76 ·
QSA 0.46 · rest ~1.2. ~2300 kernels/step.

Revised ceiling with the fp8/NVFP4 byte diet: ~7 ms/step floor → ideal ≈ 305 tok/s at greedy
accept 2.15 / ≈ 385 at relaxed accept 2.7. **We finish at ~66% (lossless) / ~60% (relaxed) of
the quantization-adjusted practical ceiling**, vs ~37% of the (obsolete) bf16 ceiling at start.
What the remaining ~40% would take: fused/persistent decode megakernels (the 2300-kernel step
has ~2-3 ms of pure inter-kernel overhead at ~1 µs each), a W4 dense stack (quality risk),
MTP head retrain toward accept 3.5+ (weeks), or an sm120 tcgen05-class GEMM ISA that doesn't
exist. Realistic next +10-15%: hand-fusing the aten/glue soup and the MoE prologue.

## Hardware ceiling
- GDDR7 14001 MHz (28 Gbps effective) × 512-bit = **1,792 GB/s** HBM bandwidth. PCIe **Gen4 x16** (~32 GB/s) — the 51 GB PLE
  n-gram table lives in host RAM and is gathered per token over this link.
- Single-stream decode is memory-bandwidth-bound: every step re-reads all *active* weights.

## Bytes per decode step (measured from safetensors headers, `weight_bytes_by_category.json`)
| component | on disk | read per step (bs=1) | dtype |
|---|---|---|---|
| routed experts (512, top-10) | 72.98 GB | **1.43 GB** | NVFP4 |
| GDN linear-attention projections | 4.17 GB | 4.17 GB | **bf16** |
| QSA full attention (12 layers) | 1.34 GB | 1.34 GB | **bf16** |
| HyperConnection (mHC) | 1.32 GB | 1.32 GB | **bf16** |
| shared expert + router | 0.61 GB | 0.61 GB | **bf16** |
| lm_head (vocab 248,320) | 1.27 GB | 1.27 GB | **bf16** |
| embed_tokens / PLE | 1.27 / 51.27 GB | row lookups only (PLE from host) | bf16 / FP8 |
| MTP draft layer (MoE! 512 experts) | 5.21 GB | ~0.25 GB per draft step | bf16 |
| **total per step** | | **10.14 GB** | 85% of it bf16 |

→ **Ceiling ≈ 10.14 GB ÷ 1.792 TB/s = 5.66 ms/step ≈ 177 tok/s (no MTP)**; with MTP at accept-len 2.46 the ideal is ≈ 435 tok/s.

## Where we are
| | measured | ceiling | efficiency |
|---|---|---|---|
| decode C1, no MTP | 88 tok/s | 177 | **50%** |
| decode C1, MTP (accept 2.46) | 154 median / 173 best | ~435 | 35–40% |
| decode C4 aggregate | 418–441 | (weights amortized over 4 streams) | — |
| prefill 2.5K | ~10.4K tok/s | compute-bound; at target | — |
jpezzulli's published 171 tok/s is the same ~39% of this ceiling — we are at parity with the best known result, and the
remaining 60% is *not* something more config can reach. Margin > 20% ⇒ deeper optimization is justified.

## Levers (ranked by upside ÷ effort)
1. **Deeper MTP drafting** (config): `--speculative-num-steps 4–5`, `num-draft-tokens 5–6`. Verify is nearly free
   (weights read once per step) so decode ≈ 177 × accept_len. Cheap sweep once RecoverSSM (WY) verify is validated.
2. **Quantize the bf16 dense stack to FP8 weight-only** (GDN + HC + shared + lm_head = 7.2 GB → 3.6 GB): per-step
   10.14 → ~6.5 GB ⇒ ceiling 177 → **~275 tok/s (+55%)**. Untouched by every public checkpoint (the community
   "NVFP4-FP8" only does the 12 QSA layers, −0.67 GB). Needs: ModelOpt weight-only FP8 PTQ (no calibration data) +
   sglang FP8_PB_WO dispatch for the custom RadixLinearAttention / HC modules. Effort: days. Quality risk: low (W8).
3. **Close the 50% efficiency gap** (kernel/launch overhead — see profiling): at bs=1 the work is thousands of tiny
   kernels per step (48 hybrid layers × GDN + QSA + HC + PLE gather over PCIe + MoE grouped-GEMM at M=1). Candidates:
   fused GDN decode, PLE prefetch pipelining, lm_head GEMV, CUDA-graphing the MTP draft path fully. This is the
   "agentic kernel optimization" track; profile first, then target the top 3 kernels/gaps.
4. **MTP head retraining**: accept-len 2.46 → 3.0–3.5 would give +20–40% decode. But the head is a **5.2 GB, 512-expert
   MoE layer (~2.6B params)** — retraining it needs target hidden states from the 180B model (bf16 = 360 GB, doesn't fit;
   NVFP4 forward via a custom pipeline) plus MoE training on 96 GB. Weeks-scale project; alternatively train a *small*
   EAGLE-3 dense draft head (~0.5B) on the model's own outputs — days, and sglang supports EAGLE3 drafts.
5. **Concurrency**: `--max-running-requests 8` + `--cuda-graph-max-bs 8` for agentic multi-session use (aggregate
   throughput scales; KV 799K tokens is ample). Cheap.

## Profiling — live C1 decode, torch profiler (`profiles/wy_decode/`, config = final WY/RecoverSSM + fp8 KV + MTP)
**GPU utilization during decode: 29.7% busy / 70.3% idle** (15,866 kernels in a 339 ms window; 8,711 inter-kernel gaps
totalling 239 ms; 253 gaps > 200 µs totalling 91 ms). ~3,000 kernel launches per MTP step, median kernel **2.0 µs**,
mean 7 µs, 47 kernels/ms. Stack-tracing inflates CPU time, so real idle is ≈ 45–55% — independently consistent with the
50% bandwidth-efficiency number above. **Single-stream decode is host/launch-bound first, kernel-efficiency-bound second.**

Where the GPU time goes (kernel table, decode):
| family | GPU share | note |
|---|---|---|
| **bf16 dense GEMM/GEMV** (`cutlass_80_wmma` bf16 kernels 16.1+9.3+8.5+1.8%, cuBLAS gemv 10.5%, splitK 1.9%) | **~48%** | the *unquantized* GDN/HC/shared projections + **lm_head GEMV = 10.5% alone**; SM80-generation WMMA kernels at M=1–4, not Blackwell-tuned |
| NVFP4 MoE grouped GEMM (SM120 BlockScaled CUTLASS) | 16.5% | the part the checkpoint quantized — efficient |
| `_hc_mix_persistent_kernel` (HyperConnection mix, Triton) | 8.3% | 630 launches; FlashInfer's HC path is an SM100-only PR4266 kernel, sm120 falls back to Triton (jpezzulli too) |
| QSA attention (`kernel_mha`) + RoPE/KV-write | ~4% | fine |
| GDN decode/verify (`cutlass_gdn_decode_bf16state_mtp`) | 1.2% | FlashInfer, fine |
| router / hc_combine / elementwise copies / misc | ~6% | hundreds of tiny launches each |

### What this means for the levers (revised)
| lever | expected gain | effort | comment |
|---|---|---|---|
| **A. Cut host/launch overhead** — bigger CUDA graphs across the MTP draft→verify→recover chain, fewer sync points, batching the per-token PLE host gather, `--enable-torch-compile` on the dense path | **up to ~1.6×** (70%→~45% idle) | days, sglang-internal | this is the *dominant* loss; nothing else reaches it |
| **B. FP8 weight-only quant of the bf16 dense stack** (GDN + HC + shared + lm_head, 7.2 → 3.6 GB) | ~1.3–1.5× on the kernel half (bytes −40% *and* leaves the `cutlass_80` WMMA path for FP8 kernels) | days: ModelOpt FP8 PTQ + sglang dispatch for RadixLinearAttention/HC modules | untouched by every public checkpoint |
| **C. lm_head** — FP8 or NVFP4 GEMV (1.27 GB, 10.5% of GPU time) | ~1.1× | 1–2 days | subset of B; also `--enable-fp32-lm-head` is off, good |
| **D. Deeper MTP drafts** (`SPEC_STEPS=4` → 5 draft tokens) | — | **CLOSED** (tested 2026-08-30) | server refuses: `Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4): the pending index-key ring holds one group; got 5`. 3 steps / 4 draft tokens is the architectural maximum on this branch; going deeper needs a QSA index-key ring that holds >1 group (kernel/code change), not a knob. |
| E. HC mix on FlashInfer for sm120 | ≤1.08× | uncertain (SM100-only kernel) | jpezzulli left it on Triton |
| F. MTP head retraining | 1.2–1.4× | weeks (5.2 GB, 512-expert MoE head) | prefer a small EAGLE-3 dense draft (days) |
| G. Concurrency 8 for agentic multi-session | aggregate ↑ | minutes | C4 already 425–450 tok/s aggregate |
A + B compound: ~2× single-stream is a realistic engineering target (≈300 tok/s C1 with MTP), i.e. we currently cover
~35–40% of the ideal MTP ceiling and ~50% of the practical one; the >20% margin is real and mostly on the host side.

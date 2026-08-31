#!/usr/bin/env bash
# BEST VALIDATED CONFIG (2026-08-30) — Qwen3.8-Flash-Next NVFP4, TP1, RTX PRO 6000 Blackwell (sm120).
# Measured (2026-08-31 8h-run, temp0.6 warm): C1 231.5 median/233.8 best tok/s, C4 621 agg, C8 ~740 agg, prefill ~10.4K, TTFT ~135ms.
# BEATS jpezzulli (171/428) on the OFFICIAL branch + sm120-wy commits 086a37f+284d7fe (Triton low-M GEMM + fp8 weight-only).
# + patches/0002 (FP8 QSA dequant) + patches/0003 (sm120 fp32 prefill state). See patches/README.md.
#
# Runs as a persistent user service (survives Claude/terminal sessions):
#   ./serve_best.sh            # start
#   systemctl --user stop qwen-sglang   # stop
#   journalctl --user -u qwen-sglang -f ; tail -f logs/serve.log
# OpenAI-compatible endpoint: http://localhost:8001/v1  (model name: pennyroyal). Thinking is ON by default
# (tokens stream in delta.reasoning_content); pass chat_template_kwargs {"enable_thinking": false} to disable.
set -euo pipefail
cd /home/golympie/ai-toolbox/models/qwen38fn
systemctl --user stop qwen-sglang 2>/dev/null || true
for i in $(seq 1 40); do
  [ "$(systemctl --user show qwen-sglang -p LoadState --value 2>/dev/null)" = "not-found" ] && break
  systemctl --user reset-failed qwen-sglang 2>/dev/null || true; sleep 2
done
mkdir -p logs; rm -f logs/serve.log
systemd-run --user --unit=qwen-sglang \
  --property=MemoryMax=112G --property=MemorySwapMax=0 \
  --setenv=MEMFRAC=0.95 --setenv=CTX="${CTX:-131072}" --setenv=MAXREQ=8 \
  --setenv=LINEAR_BACKEND=flashinfer --setenv=SSM_DTYPE=bfloat16 --setenv=MAMBA_RADIX=extra_buffer \
  --setenv=KVDTYPE=fp8_e4m3 --setenv=SPEC=1 --setenv=HICACHE=0 \
  --setenv=GDN_MTP_CACHE_MODE=none \
  --setenv=SGLANG_SM120_LOWM_FP8_WEIGHT=1 \
  --setenv=SGLANG_SM120_LM_HEAD_FP8=1 \
  --setenv=CUDAGRAPH_MAXBS=8 --setenv=CPU_OFFLOAD_GB=0 \
  --setenv=MAMBA_CACHE=48 \
  --setenv=AUTOTUNE=1 --setenv=MAX_JOBS=4 --setenv=FLASHINFER_NINJA_JOBS=4 --setenv=FLASHINFER_NVCC_THREADS=2 \
  --working-directory=/home/golympie/ai-toolbox/models/qwen38fn \
  -- bash -c 'exec bash serve.sh > logs/serve.log 2>&1'
echo "qwen-sglang started (unit active: $(systemctl --user is-active qwen-sglang)). Ready in ~5 min (first ever start: ~20 min autotune)."
echo "Wait for:  curl -s http://127.0.0.1:8001/health   -> HTTP 200"

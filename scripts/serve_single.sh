#!/usr/bin/env bash
# SINGLE-SESSION LONG-CONTEXT PROFILE — Qwen3.8-Flash-Next NVFP4, TP1, RTX PRO 6000 (sm120).
# Trades the fp8 dense-weight copies (~3.6 GB VRAM, ~20% decode speed) for the biggest
# possible KV pool, and extends rope with YaRN factor 3.0 -> 786,432-token context window.
# Use serve_best.sh instead for the max-throughput / 8-way concurrency profile.
#   ./serve_single.sh            # start (unit qwen-sglang, :8001, model pennyroyal)
#   systemctl --user stop qwen-sglang
set -euo pipefail
cd /home/golympie/ai-toolbox/models/qwen38fn
systemctl --user stop qwen-sglang 2>/dev/null || true
for i in $(seq 1 40); do
  [ "$(systemctl --user show qwen-sglang -p LoadState --value 2>/dev/null)" = "not-found" ] && break
  systemctl --user reset-failed qwen-sglang 2>/dev/null || true; sleep 2
done
mkdir -p logs; rm -f logs/serve.log
ROPE='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":3.0,"original_max_position_embeddings":262144}}}'
systemd-run --user --unit=qwen-sglang \
  --property=MemoryMax=112G --property=MemorySwapMax=0 \
  --setenv=MEMFRAC=0.98 --setenv=CTX="${CTX:-786432}" --setenv=MAXREQ=2 \
  --setenv=LINEAR_BACKEND=flashinfer --setenv=SSM_DTYPE=bfloat16 --setenv=MAMBA_RADIX=extra_buffer \
  --setenv=KVDTYPE=fp8_e4m3 --setenv=SPEC=1 --setenv=HICACHE=0 \
  --setenv=GDN_MTP_CACHE_MODE=none \
  --setenv=SGLANG_SM120_LOWM_FP8_WEIGHT=0 \
  --setenv=CUDAGRAPH_MAXBS=2 --setenv=MAMBA_CACHE=12 --setenv=CPU_OFFLOAD_GB=0 \
  --setenv=ROPE_OVERRIDE="$ROPE" \
  --setenv=AUTOTUNE=1 --setenv=MAX_JOBS=4 --setenv=FLASHINFER_NINJA_JOBS=4 --setenv=FLASHINFER_NVCC_THREADS=2 \
  --working-directory=/home/golympie/ai-toolbox/models/qwen38fn \
  -- bash -c 'exec bash serve.sh > logs/serve.log 2>&1'
echo "qwen-sglang started (single-session long-context profile). Wait for: curl -s http://127.0.0.1:8001/health -> 200"

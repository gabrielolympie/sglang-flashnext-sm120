#!/usr/bin/env python3
"""TP1 benchmark for Qwen3.8-Flash-Next on sglang (OpenAI endpoint).

Mirrors sweep_vllm_nvfp4.py: C=1 and C=4 concurrent, decode rate measured with
the first (prefill-bound) token excluded, best+median over REPS. Also a long-
prompt cold-prefill throughput probe. Reports peak VRAM. Compares to jpezzulli
targets: 171 tok/s C1, 428 tok/s C4 aggregate, ~10-12K tok/s 64K prefill.
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, threading, time, urllib.request

PORT = 8001
MODEL = "pennyroyal"
PROMPT = ("You are a coding agent. Implement a thread-safe LRU cache in C++20 with a "
          "templated API (get/put/size/capacity), an intrusive doubly-linked list plus "
          "unordered_map for O(1) ops, a shared_mutex for concurrent reads, and a unit "
          "test exercising eviction order. Output only the code with brief comments.")

def gpu_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0

def one_request(max_new: int, temp: float, prompt: str = PROMPT) -> dict | None:
    body = json.dumps({"model": MODEL, "messages": [{"role":"user","content":prompt}],
                       "max_tokens": max_new, "temperature": temp, "top_p": 0.95, "top_k": 20,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type":"application/json"})
    t0 = time.time(); first = None; n = 0; usage = None; prompt_toks = None
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for raw in resp:
            line = raw.decode("utf-8","replace").strip()
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            try: chunk = json.loads(payload)
            except json.JSONDecodeError: continue
            if chunk.get("usage"):
                usage = chunk["usage"]; prompt_toks = usage.get("prompt_tokens")
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                # count both visible content and thinking (reasoning_content) tokens
                if d.get("content") or d.get("reasoning_content"):
                    if first is None: first = time.time()
                    n += 1
    end = time.time()
    if first is None or n < 2: return None
    comp = (usage or {}).get("completion_tokens") or n
    return {"ttft_ms": round((first-t0)*1000,1),
            "decode_tok_s": round((comp-1)/(end-first),2),
            "prefill_tok_s": round((prompt_toks or 0)/(first-t0),1) if prompt_toks else None,
            "completion_tokens": comp, "prompt_tokens": prompt_toks, "total_s": round(end-t0,2)}

def concurrent(C: int, max_new: int, temp: float) -> dict:
    results = [None]*C
    def worker(i): results[i] = one_request(max_new, temp)
    peak = [0]; stop = [False]
    def sampler():
        while not stop[0]:
            peak[0] = max(peak[0], gpu_used_mib()); time.sleep(0.3)
    s = threading.Thread(target=sampler); s.start()
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time()-t0
    stop[0] = True; s.join()
    ok = [r for r in results if r]
    agg = round(sum(r["completion_tokens"]-1 for r in ok)/wall,2) if ok else 0
    return {"C": C, "n_ok": len(ok), "agg_decode_tok_s": agg,
            "per_req_decode": [r["decode_tok_s"] for r in ok],
            "ttft_ms_median": round(statistics.median([r["ttft_ms"] for r in ok]),1) if ok else None,
            "peak_vram_mib": peak[0]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.6)
    args = ap.parse_args()

    print(f"idle VRAM: {gpu_used_mib()} MiB")
    # warmup
    print("warmup..."); one_request(64, args.temp)

    out = {}
    for C in (1, 4):
        runs = [concurrent(C, args.max_new, args.temp) for _ in range(args.reps)]
        best = max(runs, key=lambda r: r["agg_decode_tok_s"])
        med = round(statistics.median([r["agg_decode_tok_s"] for r in runs]),2)
        out[f"C{C}"] = {"best_agg": best["agg_decode_tok_s"], "median_agg": med,
                        "ttft_ms": best["ttft_ms_median"], "peak_vram_mib": best["peak_vram_mib"],
                        "per_req_decode_best": best["per_req_decode"]}
        print(f"C{C}: best_agg={best['agg_decode_tok_s']} tok/s  median={med}  "
              f"ttft={best['ttft_ms_median']}ms  peakVRAM={best['peak_vram_mib']}MiB")

    # save C1/C4 before the prefill probe (probe can be fragile on some kernels)
    open("/home/golympie/ai-toolbox/models/qwen38fn/results_sglang.json","w").write(json.dumps(out, indent=2))
    # cold-prefill probe: moderate prompt (~2K tokens; avoid huge prefills that hit fp8 kernel limits)
    long_prompt = ("Summarize the following.\n" + ("The quick brown fox jumps over the lazy dog. " * 250))
    try:
        r = one_request(16, args.temp, long_prompt)
    except Exception as e:
        print(f"prefill probe skipped/failed: {e}"); r = None
    if r:
        out["prefill_probe"] = {"prompt_tokens": r["prompt_tokens"], "prefill_tok_s": r["prefill_tok_s"], "ttft_ms": r["ttft_ms"]}
        print(f"prefill probe: {r['prompt_tokens']} toks @ {r['prefill_tok_s']} tok/s (ttft {r['ttft_ms']}ms)")

    open("/home/golympie/ai-toolbox/models/qwen38fn/results_sglang.json","w").write(json.dumps(out, indent=2))
    print("\nTargets (jpezzulli): C1 171 tok/s, C4 428 tok/s agg, 64K prefill ~10-12K tok/s")

if __name__ == "__main__":
    main()

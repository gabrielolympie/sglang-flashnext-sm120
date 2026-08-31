#!/usr/bin/env python3
"""Build the FR-Spec hot-token map used by --speculative-token-map.

The draft (MTP) lm_head then scores only this subset of the 248,320-token
vocab, making each draft-step logits GEMV ~4x cheaper. Verification against
the target model stays exact, so only draft-proposal quality depends on the
subset. Recipe: all of the first 32K BPE ids (bytes + highest-rank merges =
the bulk of general text in any language) + the most frequent tokens of a
local code/text corpus + all special tokens, padded to 65,536 ids.

Usage: python make_hot_tokens.py <model_dir> <corpus_glob>... -o hot_tokens_64k.pt
"""

import argparse
import collections
import glob

import torch
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("corpus_globs", nargs="+", help="e.g. 'src/**/*.py' '**/*.md'")
parser.add_argument("-o", "--out", default="hot_tokens_64k.pt")
parser.add_argument("--size", type=int, default=65536)
parser.add_argument("--base-ids", type=int, default=32768)
parser.add_argument("--max-corpus-bytes", type=int, default=40_000_000)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
vocab_size = 248320

cnt = collections.Counter()
nbytes = 0
for pattern in args.corpus_globs:
    for f in glob.glob(pattern, recursive=True):
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        nbytes += len(s)
        for tid in tok(s, add_special_tokens=False)["input_ids"]:
            cnt[tid] += 1
        if nbytes > args.max_corpus_bytes:
            break
print(f"corpus {nbytes/1e6:.0f} MB, {sum(cnt.values())/1e6:.1f}M tokens, {len(cnt)} distinct")

hot = set(i for i, _ in cnt.most_common(args.size - args.base_ids))
hot |= set(range(args.base_ids))
hot |= set(tok.all_special_ids)
for i in sorted(cnt, key=lambda x: -cnt[x]):
    if len(hot) >= args.size:
        break
    hot.add(i)
i = args.base_ids
while len(hot) < args.size and i < vocab_size:
    hot.add(i)
    i += 1
hot = sorted(x for x in hot if x < vocab_size)
cov = sum(v for k, v in cnt.items() if k in set(hot)) / max(1, sum(cnt.values()))
print(f"hot set: {len(hot)} ids; corpus coverage {cov*100:.2f}%")
torch.save(hot, args.out)
print(f"saved {args.out}")

#!/usr/bin/env python3
"""Build a trace whose prompts share a hot prefix.

The recorded traces (Dolly, CNN/DailyMail) have *no* shared prefixes -- every
64-token head is unique -- so neither the cluster nor the simulator can show a
cache-affinity effect on them: measured 2026-09-16, ``cache_aware`` differed
from ``load`` by 1.3% on Dolly and 0.2% on the paced CNN trace.  This generator
takes ``--hot`` prompts from a pool and gives every emitted request one of them
as a shared head (identical text *and* identical token ids, so both the real
engine's block hashing and the simulator's prefix profiler see the reuse) plus
a unique tail taken from a different pool row.

    python3 tests/make_hot_prefix_trace.py \
        --input workloads/cnndm-long-pool-qwen3-8b.jsonl \
        --output workloads/cnndm-hot-prefix-06rps-qwen3-8b.jsonl \
        --hot 2 --prefix-tokens 1024 --rate 0.6 --duration 400

The tail is taken from the *same* character offset in a different row so the
concatenated prompt stays close in length to the pool's own prompts.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hot", type=int, default=2,
                        help="How many distinct shared heads to use.")
    parser.add_argument("--prefix-tokens", type=int, default=1024,
                        help="Tokens (and roughly characters) of shared head.")
    parser.add_argument("--rate", type=float, default=0.6,
                        help="Arrival rate inside the single phase.")
    parser.add_argument("--duration", type=float, default=400.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = pathlib.Path(args.input)
    rows = [json.loads(line) for line in source.open(encoding="utf-8")
            if line.strip()]
    if len(rows) < args.hot + 1:
        raise SystemExit("pool is smaller than the number of hot heads")
    if "input_text" not in rows[0]:
        raise SystemExit("pool needs input_text (the real client sends text)")
    heads = rows[:args.hot]
    # Characters per token varies per row; scale the head by the row's own
    # ratio so `prefix_tokens` means what it says.
    head_text, head_ids = [], []
    for row in heads:
        ratio = max(1.0, len(row["input_text"]) / max(1, row["input_toks"]))
        chars = int(args.prefix_tokens * ratio)
        head_text.append(row["input_text"][:chars])
        head_ids.append(list(row["input_tok_ids"][:args.prefix_tokens]))

    wanted = max(1, int(round(args.rate * args.duration)))
    emitted = []
    for index in range(wanted):
        hot = index % args.hot
        tail_row = rows[(index % (len(rows) - args.hot)) + args.hot]
        ratio = max(1.0, len(tail_row["input_text"]) / max(1, tail_row["input_toks"]))
        skip_chars = int(args.prefix_tokens * ratio)
        tail_text = tail_row["input_text"][skip_chars:]
        tail_ids = list(tail_row["input_tok_ids"][args.prefix_tokens:])
        if not tail_text or not tail_ids:
            continue
        emitted.append({
            "input_text": head_text[hot] + tail_text,
            "input_tok_ids": head_ids[hot] + tail_ids,
            "input_toks": len(head_ids[hot]) + len(tail_ids),
            "output_toks": int(tail_row.get("output_toks", 16)),
            "arrival_time_ns": int(round(index / max(1e-9, args.rate) * 1e9)),
            "hotspot_id": f"hot{hot}",
            "source_id": tail_row.get("source_id"),
            "category": "hot-prefix",
            "slo_ttft_ms": tail_row.get("slo_ttft_ms"),
            "slo_tpot_ms": tail_row.get("slo_tpot_ms"),
        })

    out = pathlib.Path(args.output)
    with out.open("w", encoding="utf-8") as handle:
        for row in emitted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "generator": "make_hot_prefix_trace",
        "source": args.input,
        "requests": len(emitted),
        "hot_heads": args.hot,
        "prefix_tokens": args.prefix_tokens,
        "rate_rps": args.rate,
        "span_s": (emitted[-1]["arrival_time_ns"] / 1e9) if emitted else 0.0,
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

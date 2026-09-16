"""CNN/DailyMail -> LLMServingSim JSONL.

The third classic dataset, chosen for the axis the other two do not cover:
**long prompts**.  Dolly is single-turn but short (~100 tokens), ShareGPT is
multi-turn with reusable prefixes, while a news article is one long, unique
context.  That matters for a disaggregated deployment because the cross-domain
KV budget is denominated in bytes: one 1024-token prompt carries ~150 MB of
KV, so at the measured 16 Gbps fabric a single Prefill can only push ~13 such
requests per second across a domain boundary.  Any router that spills long
prompts onto a remote Decode therefore saturates the link, and any router that
keeps them local does not -- which is exactly the behaviour under test.

Source: ``abisee/cnn_dailymail`` (Apache-2.0), the standard summarization
benchmark; the ``3.0.0`` validation split is distributed as Parquet, so this
generator reads the file directly instead of the (long-broken) HF loading
script.  Only tokenization, length filtering and pacing happen here -- the
article text is not rewritten.

Output rows match the other generators and additionally carry ``category``
(always ``summarization``) and ``source_id`` (the dataset's ``id`` column) so a
trace can be traced back to the exact article.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="Tokenizer path or HF id (must match the deployment).")
    p.add_argument("--source", default="abisee/cnn_dailymail",
                   help="HuggingFace dataset id, a local .parquet or .jsonl path.")
    p.add_argument("--split", default="validation")
    p.add_argument("--num-reqs", type=int, required=True, dest="num_reqs")
    p.add_argument("--sps", type=float, required=True,
                   help="Arrival rate (requests / sec), Poisson inter-arrivals.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--first-arrival-sec", type=int, default=0,
                   dest="first_arrival_sec")
    p.add_argument("--min-input-toks", type=int, default=0, dest="min_input_toks")
    p.add_argument("--max-input-toks", type=int, default=16384, dest="max_input_toks")
    p.add_argument("--min-output-toks", type=int, default=0, dest="min_output_toks")
    p.add_argument("--max-output-toks", type=int, default=16384, dest="max_output_toks")
    p.add_argument("--max-kv-toks", type=int, default=16384, dest="max_kv_toks")
    p.add_argument("--head-toks", type=int, default=0, dest="head_toks",
                   help="Keep at most this many leading article tokens before "
                        "the length filters run (0 = do not truncate). Head "
                        "truncation keeps a valid article prefix and is the "
                        "only text customization this generator performs.")
    p.add_argument("--pack-toks", type=int, default=0, dest="pack_toks",
                   help="Concatenate consecutive articles (separated by a blank "
                        "line) until the prompt reaches this many tokens, then "
                        "truncate to exactly this length.  CNN/DailyMail "
                        "articles average ~780 tokens, so this is the only way "
                        "to drive the cross-domain KV budget into saturation "
                        "with this corpus.  0 disables it.")
    p.add_argument("--max-rows", type=int, default=0, dest="max_rows",
                   help="Cap on source rows read (0 = all).")
    p.add_argument("--emit-text", action="store_true", default=False,
                   dest="emit_text",
                   help="Also write ``input_text`` for the real-cluster replay.")
    p.add_argument("--slo-ttft-ms", type=float, default=0.0, dest="slo_ttft_ms")
    p.add_argument("--slo-tpot-ms", type=float, default=0.0, dest="slo_tpot_ms")
    p.add_argument("--context-full-toks", type=int, default=0,
                   dest="context_full_toks")
    p.add_argument("--slo-long-ttft-ms", type=float, default=0.0,
                   dest="slo_long_ttft_ms")


def _load_rows(source: str, split: str, cap: int):
    path = Path(source)
    if path.exists():
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            for index, row in enumerate(table.to_pylist()):
                if cap and index >= cap:
                    return
                yield row
            return
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if cap and index >= cap:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    from datasets import load_dataset  # type: ignore
    for index, row in enumerate(load_dataset(source, split=split)):
        if cap and index >= cap:
            return
        yield row


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                              trust_remote_code=True)
    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_load_rows(args.source, args.split, args.max_rows))
    rng.shuffle(rows)

    written = 0
    time_ns = int(args.first_arrival_sec) * 1_000_000_000
    # ``--pack-toks`` glues consecutive articles together.  It walks a cursor
    # over the same shuffled corpus, so a packed prompt is still "the next
    # unused articles" and two requests never share a prefix.
    pack_cursor = -1

    def packed_article(first_article: str) -> str:
        nonlocal pack_cursor
        pieces = [first_article]
        total = len(tokenizer(first_article, add_special_tokens=False)["input_ids"])
        while total < args.pack_toks and pack_cursor + 1 < len(rows):
            pack_cursor += 1
            nxt = str(rows[pack_cursor].get("article") or "").strip()
            if not nxt:
                continue
            pieces.append(nxt)
            total += len(tokenizer("\n\n" + nxt, add_special_tokens=False)["input_ids"])
        return "\n\n".join(pieces)

    with out_path.open("w", encoding="utf-8") as out:
        for row_index, row in enumerate(rows):
            if written >= args.num_reqs:
                break
            article = str(row.get("article") or "").strip()
            highlights = str(row.get("highlights") or "").strip()
            if not article or not highlights:
                continue
            if args.pack_toks:
                pack_cursor = max(pack_cursor, row_index)
                article = packed_article(article)
            in_ids = tokenizer(article, add_special_tokens=False)["input_ids"]
            if args.head_toks and len(in_ids) > args.head_toks:
                in_ids = in_ids[:args.head_toks]
            if args.pack_toks and len(in_ids) > args.pack_toks:
                in_ids = in_ids[:args.pack_toks]
            out_ids = tokenizer(highlights, add_special_tokens=False)["input_ids"]
            if not (args.min_input_toks <= len(in_ids) <= args.max_input_toks):
                continue
            if not (args.min_output_toks <= len(out_ids) <= args.max_output_toks):
                continue
            if len(in_ids) + len(out_ids) > args.max_kv_toks:
                continue

            time_ns += int(rng.expovariate(max(args.sps, 1e-9)) * 1e9)
            record = {
                "input_toks": len(in_ids),
                "output_toks": len(out_ids),
                "arrival_time_ns": int(time_ns),
                "input_tok_ids": list(in_ids),
                "output_tok_ids": list(out_ids),
                "category": "summarization",
                "source_id": str(row.get("id") or ""),
            }
            if args.emit_text:
                record["input_text"] = tokenizer.decode(in_ids,
                                                        skip_special_tokens=False)
            if args.slo_ttft_ms > 0:
                ttft = float(args.slo_ttft_ms)
                if (args.context_full_toks and args.slo_long_ttft_ms > 0
                        and len(in_ids) > args.context_full_toks):
                    ttft = float(args.slo_long_ttft_ms)
                record["slo_ttft_ms"] = ttft
            if args.slo_tpot_ms > 0:
                record["slo_tpot_ms"] = float(args.slo_tpot_ms)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} requests -> {out_path}")
    return 0

"""Databricks Dolly-15k -> LLMServingSim JSONL.

The second classic dataset in this project, chosen because its shape is the
opposite of ShareGPT's: single-turn instructions with (mostly) short prompts
and almost no prefix reuse between requests.  That isolates the
heterogeneity/capacity half of CASR from the prefix-affinity half -- if the
advantage only existed because of cache hits, it would disappear here.

Source: ``databricks/databricks-dolly-15k`` (CC-BY-SA-3.0), a JSONL with
``instruction`` / ``context`` / ``response`` / ``category``.  This generator
only tokenizes, filters and paces; it does not rewrite the text.

Output rows match the other generators and additionally carry ``category`` so
later analysis can slice by task type.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="Tokenizer path or HF id (must match the deployment).")
    p.add_argument("--source", default="databricks/databricks-dolly-15k",
                   help="HuggingFace dataset id or a local .jsonl path.")
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
    p.add_argument("--max-rows", type=int, default=0, dest="max_rows",
                   help="Cap on source rows read (0 = all).")
    p.add_argument("--emit-text", action="store_true", default=False,
                   dest="emit_text",
                   help="Also write ``input_text`` for the real-cluster replay.")
    p.add_argument("--slo-ttft-ms", type=float, default=0.0, dest="slo_ttft_ms")
    p.add_argument("--slo-tpot-ms", type=float, default=0.0, dest="slo_tpot_ms")
    p.add_argument("--context-full-toks", type=int, default=0,
                   dest="context_full_toks",
                   help="Rows whose prompt exceeds this many tokens get the "
                        "``--slo-long-ttft-ms`` budget instead of "
                        "``--slo-ttft-ms`` (0 = single tier).")
    p.add_argument("--slo-long-ttft-ms", type=float, default=0.0,
                   dest="slo_long_ttft_ms")


def _load_rows(source: str, cap: int):
    path = Path(source)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if cap and index >= cap:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    from datasets import load_dataset  # type: ignore
    for index, row in enumerate(load_dataset(source, split="train")):
        if cap and index >= cap:
            return
        yield row


def _prompt_of(row) -> str:
    instruction = str(row.get("instruction") or "").strip()
    context = str(row.get("context") or "").strip()
    return f"{instruction}\n\n{context}" if context else instruction


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                              trust_remote_code=True)
    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_load_rows(args.source, args.max_rows))
    rng.shuffle(rows)

    written = 0
    time_ns = int(args.first_arrival_sec) * 1_000_000_000
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            if written >= args.num_reqs:
                break
            prompt = _prompt_of(row)
            response = str(row.get("response") or "").strip()
            if not prompt or not response:
                continue
            in_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            out_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
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
                "category": str(row.get("category") or ""),
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

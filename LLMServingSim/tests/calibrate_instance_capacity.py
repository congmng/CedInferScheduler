#!/usr/bin/env python3
"""Measure each instance's own service rate, for the router's cost model.

The router ranks candidates by ``inflight / capacity`` and the local-prefill
decision divides by ``speed``.  Both were declared from hardware ratios until
2026-09-13, and the declared numbers were badly wrong for the shared A100 node
(declared 550 prefill req/s, measured 51): the least-loaded baseline then piled
95% of prefills onto the slowest host and the comparison stopped being about
policy.  This script produces the measured inputs instead.

Two sweeps per instance:

* prefill -- ``max_tokens=1`` so the request is dominated by prompt processing;
* decode  -- a small output budget so the request is dominated by generation.

Both fire ``--concurrency`` requests at once and report req/s plus the median
latency.  Feed the ratios back into ``router_config.json`` (``capacity`` and
``speed``) keeping one instance as the scale reference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
import uuid

import httpx

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_WORKLOAD = REPO / "workloads" / "cnndm-real-qwen3-8b-600-sps14.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", required=True,
                        help="comma separated name=host:port entries")
    parser.add_argument("--role", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--workload", default=str(DEFAULT_WORKLOAD))
    parser.add_argument("--chars", type=int, default=700)
    parser.add_argument("--model", default="qwen3-8b",
                        help="served model name on the instances")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


async def one(client: httpx.AsyncClient, url: str, text: str, rid: str,
              max_tokens: int, model: str) -> float:
    started = time.perf_counter()
    response = await client.post(
        f"http://{url}/v1/chat/completions",
        json={"model": model,
              "messages": [{"role": "user", "content": text}],
              "max_tokens": max_tokens, "temperature": 0.0, "stream": False},
        headers={"X-Request-Id": rid})
    response.raise_for_status()
    return time.perf_counter() - started


async def bench(url: str, text: str, concurrency: int, max_tokens: int,
                model: str):
    limits = httpx.Limits(max_connections=concurrency * 2)
    async with httpx.AsyncClient(timeout=300.0, trust_env=False,
                                 limits=limits) as client:
        await one(client, url, text + " [warm]", f"w-{uuid.uuid4().hex[:6]}",
                  max_tokens, model)
        started = time.perf_counter()
        latencies = await asyncio.gather(*[
            one(client, url, text + f" [{i}]", f"b-{uuid.uuid4().hex[:6]}",
                max_tokens, model)
            for i in range(concurrency)])
        wall = time.perf_counter() - started
    return concurrency / wall, statistics.median(latencies)


def main() -> int:
    args = parse_args()
    article = ""
    with open(args.workload) as handle:
        for index, line in enumerate(handle):
            if index == 5:
                article = json.loads(line)["input_text"]
                break
    text = (f"Summarize in one sentence:\n\n{article[:args.chars]}\n\nSummary:")
    max_tokens = 1 if args.role == "prefill" else 16

    results = {}
    for entry in args.endpoints.split(","):
        name, _, url = entry.partition("=")
        try:
            rps, median = asyncio.run(
                bench(url, text, args.concurrency, max_tokens, args.model))
            results[name.strip()] = {
                "endpoint": url.strip(), "role": args.role,
                "requests_per_s": round(rps, 1),
                "median_latency_ms": round(median * 1000, 1),
            }
            print(f"{name.strip():10s} {rps:7.1f} req/s  "
                  f"median {median * 1000:7.1f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep sweeping the rest
            results[name.strip()] = {"endpoint": url.strip(),
                                     "error": repr(exc)[:120]}
            print(f"{name.strip():10s} FAILED {exc!r}", file=sys.stderr,
                  flush=True)

    payload = {"role": args.role, "concurrency": args.concurrency,
               "prompt_chars": args.chars, "instances": results}
    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

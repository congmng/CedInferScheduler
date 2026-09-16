#!/usr/bin/env python3
"""Drive the multi-domain P/D router with a prefix-reuse workload.

Each virtual client owns one long shared prefix and repeats it many times, so a
prefix-affinity router can keep hitting the same Prefill cache while a
round-robin / least-loaded router scatters the same prefix across instances.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

WORDS = ("caching systems route requests across heterogeneous accelerators and "
         "must balance prefix reuse against queueing and cross-domain transfer")

# The word list above is repetitive English; the Qwen3 BPE tokenizer yields
# about 6.7 characters per token for it (~4 chars/token would underestimate the
# repeats needed and silently cap the prefix far below ``--prefix-tokens``).
CHARS_PER_TOKEN = 6.7


def build_prefix(index, target_tokens):
    unit = f"client-{index:03d} " + " ".join(WORDS.split() * 8)
    repeats = max(1, round(target_tokens * CHARS_PER_TOKEN / len(unit)))
    return (unit + " ") * repeats


def prefix_index_for(index, num_clients, hot_prefixes, drift_index):
    """Which prompt a request uses, given the hotspot and drift shape.

    ``hot_prefixes`` clients share one prompt, so a value below
    ``num_clients`` concentrates traffic (and a Prefill's KV-egress link) on
    fewer cached prefixes.  Once ``index`` reaches ``drift_index`` every client
    moves to a second, brand-new prompt in the next slice of the pool.
    """
    phase = 1 if index >= drift_index else 0
    return (index % num_clients) % hot_prefixes + phase * hot_prefixes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--num-reqs", type=int, default=96)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--hot-prefixes", type=int, default=0,
                        help="distinct prefixes shared by the clients (default: "
                             "one per client).  Set it below --num-clients to make "
                             "several clients hammer the same cached prefix and "
                             "push a single Prefill's KV-egress link towards "
                             "saturation.")
    parser.add_argument("--drift-after-fraction", type=float, default=0.0,
                        help="after this fraction of dispatched requests each "
                             "client moves to a brand-new prefix, so the hot set "
                             "drifts and the controller has to migrate caches. "
                             "0 disables the drift.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


async def main():
    args = parse_args()
    hot_prefixes = args.hot_prefixes if args.hot_prefixes > 0 else args.num_clients
    phases = 2 if 0.0 < args.drift_after_fraction < 1.0 else 1
    drift_index = (int(args.num_reqs * args.drift_after_fraction)
                   if phases == 2 else args.num_reqs + 1)
    prefixes = [build_prefix(i, args.prefix_tokens)
                for i in range(hot_prefixes * phases)]
    semaphore = asyncio.Semaphore(args.concurrency)
    records = []
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        async def one(index):
            client_id = index % args.num_clients
            phase = 1 if index >= drift_index else 0
            prefix_index = prefix_index_for(index, args.num_clients,
                                            hot_prefixes, drift_index)
            prompt = prefixes[prefix_index] + f" question {index} please answer briefly."
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.output_tokens,
                "temperature": 0.0,
                "stream": False,
            }
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(
                        f"{args.url}/v1/chat/completions", json=payload,
                        headers={"X-Request-Id": f"md-{index}"})
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    usage = {}
                    if response.status_code == 200:
                        try:
                            usage = response.json().get("usage", {}) or {}
                        except Exception:
                            usage = {}
                    record = {"index": index, "client": client_id,
                              "phase": phase, "prefix_index": prefix_index,
                              "latency_ms": round(elapsed_ms, 3),
                              "status": response.status_code,
                              "prompt_tokens": usage.get("prompt_tokens"),
                              "completion_tokens": usage.get("completion_tokens"),
                              "ts": time.time()}
                except Exception as exc:  # noqa: BLE001
                    record = {"index": index, "client": client_id, "status": 0,
                              "phase": 1 if index >= drift_index else 0,
                              "error": repr(exc), "ts": time.time()}
            async with lock:
                records.append(record)

        await asyncio.gather(*(one(i) for i in range(args.num_reqs)))

    records.sort(key=lambda item: item["index"])
    with open(args.output, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    ok = [item for item in records if item["status"] == 200]
    latencies = [item["latency_ms"] for item in ok]
    if latencies:
        latencies_sorted = sorted(latencies)
        def pct(value):
            return latencies_sorted[min(len(latencies_sorted) - 1,
                                        int(value * len(latencies_sorted)) - 1)]
        summary = {
            "requests": len(records), "ok": len(ok),
            "latency_mean_ms": round(statistics.mean(latencies), 3),
            "latency_p50_ms": round(pct(0.50), 3),
            "latency_p95_ms": round(pct(0.95), 3),
        }
        if phases == 2:
            for phase in (0, 1):
                phase_latencies = [item["latency_ms"] for item in ok
                                   if item.get("phase") == phase]
                if phase_latencies:
                    summary[f"phase{phase}_mean_ms"] = round(
                        statistics.mean(phase_latencies), 3)
        print(json.dumps(summary))
    else:
        print(json.dumps({"requests": len(records), "ok": 0}))


if __name__ == "__main__":
    asyncio.run(main())

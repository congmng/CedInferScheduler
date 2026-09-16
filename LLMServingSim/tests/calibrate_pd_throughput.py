#!/usr/bin/env python3
"""Sweep offered concurrency on one real Prefill->Decode pair.

``calibrate_real_pd.py`` measures a single hand-off; the flow solver instead
needs the *saturation* point of an instance, because ``capacity =
max_num_seqs / service_ms`` assumes the engine's throughput scales linearly
with batch size, and on a real GPU it does not.  This script offers an
increasing number of concurrent hand-offs to a fixed pair and reports the
achieved throughput and latency at each level, so the knee can be read off and
written back into ``router_config.json``.

The workload mirrors ``real_multidomain_client.py``: a few long shared prefixes
repeated many times, so a Prefill's cache hit rate is realistic.
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
CHARS_PER_TOKEN = 6.7


def build_prefix(index, target_tokens):
    unit = f"client-{index:03d} " + " ".join(WORDS.split() * 8)
    repeats = max(1, round(target_tokens * CHARS_PER_TOKEN / len(unit)))
    return (unit + " ") * repeats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", required=True, help="host:port")
    parser.add_argument("--decode", required=True, help="host:port")
    parser.add_argument("--prefill-host-out", default="")
    parser.add_argument("--init-port", type=int, default=55555)
    parser.add_argument("--alloc-port", type=int, default=55556)
    parser.add_argument("--query-port", type=int, default=55557)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--levels", default="1,2,4,8,16,24,32")
    parser.add_argument("--reqs", type=int, default=32, help="requests per level")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--unique-prefixes", action="store_true", default=False,
                        help="give every request a distinct prefix so the "
                             "Prefill pays a full uncached prompt; this is "
                             "what the classic-dataset traces look like")
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--warm-reqs", type=int, default=2)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


async def main():
    args = parse_args()
    receiver_host = args.prefill_host_out or args.decode.split(":")[0]
    prefixes = [build_prefix(i, args.prefix_tokens) for i in range(args.clients)]
    levels = [int(part) for part in args.levels.split(",") if part.strip()]

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        async def handoff(index):
            # ``--unique-prefixes`` gives every request its own prefix, so the
            # Prefill pays a full (uncached) prompt instead of a cache hit.
            # That is what the real ShareGPT trace looks like, and it is the
            # only way to measure how many requests one chunked-prefill step
            # can actually hold (the shared-prefix default made every request
            # after the first a cache hit and overstated the step budget).
            client_id = index if args.unique_prefixes else index % args.clients
            request_id = f"cal-{index}-{int(time.time() * 1e6)}"
            if args.unique_prefixes:
                prompt = build_prefix(client_id, args.prefix_tokens) + \
                    f" probe {index} answer briefly."
            else:
                prompt = prefixes[client_id] + f" probe {index} answer briefly."
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.output_tokens,
                "temperature": 0.0,
                "stream": False,
            }
            prefill_payload = dict(payload)
            prefill_payload["max_tokens"] = 1
            prefill_payload["kv_transfer_params"] = {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "disagg_spec": {
                    "req_id": request_id,
                    "receiver_host": receiver_host,
                    "receiver_init_port": [args.init_port],
                    "receiver_alloc_port": [args.alloc_port],
                    "receiver_query_port": [args.query_port],
                },
            }
            headers = {"X-Request-Id": request_id}
            started = time.perf_counter()
            try:
                prefill_response = await client.post(
                    f"http://{args.prefill}/v1/chat/completions",
                    json=prefill_payload, headers=headers)
                prefill_response.raise_for_status()
                decode_started = time.perf_counter()
                response = await client.post(
                    f"http://{args.decode}/v1/chat/completions",
                    json=payload, headers=headers)
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                response.raise_for_status()
                return {"index": index, "status": 200,
                        "prefill_ms": round((decode_started - started) * 1000.0, 3),
                        "decode_ms": round(decode_ms, 3)}
            except Exception as exc:  # noqa: BLE001
                return {"index": index, "status": 0, "error": repr(exc)}

        for warm in range(max(0, args.warm_reqs)):
            await handoff(10_000 + warm)

        results = []
        for level in levels:
            semaphore = asyncio.Semaphore(level)
            order = list(range(args.reqs))

            async def one(index):
                async with semaphore:
                    return await handoff(index)

            started = time.perf_counter()
            records = await asyncio.gather(*(one(i) for i in order))
            wall_ms = (time.perf_counter() - started) * 1000.0
            ok = [r for r in records if r.get("status") == 200]
            latencies = sorted(r["decode_ms"] for r in ok)
            totals = sorted(r["prefill_ms"] + r["decode_ms"] for r in ok)
            def pct(values, fraction):
                if not values:
                    return None
                return round(values[min(len(values) - 1, int(fraction * len(values)) - 1)], 3)
            results.append({
                "concurrency": level,
                "requests": len(records),
                "ok": len(ok),
                "wall_ms": round(wall_ms, 3),
                "throughput_rps": round(len(ok) / (wall_ms / 1000.0), 3) if wall_ms else None,
                "decode_p50_ms": pct(latencies, 0.5),
                "decode_p95_ms": pct(latencies, 0.95),
                "decode_mean_ms": round(statistics.mean(latencies), 3) if latencies else None,
                "total_p50_ms": pct(totals, 0.5),
            })
            print(json.dumps(results[-1]))

    summary = {"prefill": args.prefill, "decode": args.decode,
               "prefix_tokens": args.prefix_tokens,
               "output_tokens": args.output_tokens, "levels": results}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
    print(json.dumps({"knee_guess": _knee(results)}))


def _knee(results):
    best = None
    for item in results:
        offered = item["throughput_rps"] or 0.0
        if best is None or offered > best["throughput_rps"]:
            best = item
    return best


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Measure real Prefill->Decode KV handoff cost for one P/D pair.

The router's link model must use measured transfer cost, not NIC line rate.
This script fixes a Prefill and a Decode instance and reports the per-stage
latency distribution, including the KV push performed by LMCache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

PREFIX = ("prefix calibration payload for measuring inter-domain kv transfer "
          "latency across heterogeneous accelerators ")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", required=True, help="host:port")
    parser.add_argument("--decode", required=True, help="host:port")
    parser.add_argument("--prefill-host-out", default="", help="receiver_host advertised to sender")
    parser.add_argument("--init-port", type=int, default=55555)
    parser.add_argument("--alloc-port", type=int, default=55556)
    parser.add_argument("--query-port", type=int, default=55557)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--num-reqs", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1,
                        help="requests to send and discard first.  The first "
                             "handoff on a P/D pair pays a one-off JIT cost "
                             "(~50 s, see docs/五台异构实验环境部署记录.md), so a "
                             "measurement that keeps it is off by an order of "
                             "magnitude.")
    return parser.parse_args()


async def main():
    args = parse_args()
    receiver_host = args.prefill_host_out or args.decode.split(":")[0]
    body_prefix = PREFIX * max(1, args.prefix_tokens // max(1, len(PREFIX) // 4))
    records = []

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        for index in range(args.num_reqs + args.warmup):
            request_id = f"cal-{index}-{int(time.time()*1000)}"
            payload = {
                "model": args.model,
                "messages": [{"role": "user",
                              "content": body_prefix + f" probe {index}"}],
                "max_tokens": args.output_tokens,
                "temperature": 0.0,
                "stream": False,
            }
            prefill_payload = dict(payload)
            prefill_payload["stream"] = False
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
            prefill_response = await client.post(
                f"http://{args.prefill}/v1/chat/completions",
                json=prefill_payload, headers=headers)
            prefill_ms = (time.perf_counter() - started) * 1000.0
            prefill_response.raise_for_status()
            decode_started = time.perf_counter()
            response = await client.post(
                f"http://{args.decode}/v1/chat/completions",
                json=payload, headers=headers)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if index < args.warmup:
                # Warm the pair, then drop it: this is the request that pays
                # the one-off remote-KV code-path compilation.
                continue
            usage = {}
            try:
                usage = response.json().get("usage") or {}
            except Exception:
                usage = {}
            records.append({"index": index, "status": response.status_code,
                            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                            "prefill_ms": round(prefill_ms, 3),
                            "decode_ms": round(decode_ms, 3),
                            "total_ms": round(prefill_ms + decode_ms, 3)})

    ok = [r for r in records if r["status"] == 200]
    def stats(values):
        s = sorted(values)
        return {"mean": round(statistics.mean(s), 3),
                "p50": round(s[len(s) // 2], 3),
                "min": round(s[0], 3), "max": round(s[-1], 3)}

    summary = {"prefill": args.prefill, "decode": args.decode,
               "ok": len(ok), "requests": len(records),
               "prompt_tokens": (stats([r["prompt_tokens"] for r in ok])["p50"]
                                 if ok else 0),
               "prefill_ms": stats([r["prefill_ms"] for r in ok]) if ok else {},
               "decode_ms": stats([r["decode_ms"] for r in ok]) if ok else {},
               "total_ms": stats([r["total_ms"] for r in ok]) if ok else {}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Calibrate the cross-domain KV cost on vLLM's native NixlConnector path.

``calibrate_real_pd.py`` speaks LMCache's PD protocol (``disagg_spec`` with the
receiver's alloc/query ports), which is no longer what the cluster runs.  This
script drives the officially supported flow instead:

    Prefill: POST with kv_transfer_params {do_remote_decode: true,
                                           do_remote_prefill: false}
    Decode:  POST the same payload with the ``kv_transfer_params`` the Prefill
             returned (remote block ids / engine id), as the reference proxy
             does.

Sweeping the prompt length on one (Prefill, Decode) pair separates the fixed
part of a handoff (queueing + handshake, i.e. the ``rtt_ms`` term) from the
per-token part (the transfer itself, which is what the occupancy model needs).
The slope is reported as ``ms_per_1k_tokens`` and converted to an effective
bandwidth assuming 147 KB/token of bf16 KV for Qwen3-8B.
"""

from __future__ import annotations

import argparse
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
KV_BYTES_PER_TOKEN = 147 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", required=True, help="host:port")
    parser.add_argument("--decode", required=True, help="host:port")
    parser.add_argument("--workload", default=str(DEFAULT_WORKLOAD))
    parser.add_argument("--model", default="qwen3-8b",
                        help="served model name on both endpoints")
    parser.add_argument("--chars", default="400,1600,4000,6600",
                        help="comma separated article prefixes to sweep")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    article = ""
    with open(args.workload) as handle:
        article = json.loads(handle.readline())["input_text"]
    # Use a different base article for the sweep so the Decode's cache from a
    # previous run cannot satisfy the first measured request.
    with open(args.workload) as handle:
        for index, line in enumerate(handle):
            if index == 7:
                article = json.loads(line)["input_text"]
                break

    client = httpx.Client(timeout=args.timeout, trust_env=False)
    points = []
    for chars in (int(c) for c in args.chars.split(",")):
        samples = []
        for _ in range(args.requests):
            # A unique marker per request keeps the Decode's prefix cache from
            # satisfying the next sample: without it every sample after the
            # first is a local cache hit and the sweep measures nothing.
            text = (f"Sweep {uuid.uuid4().hex[:8]}: summarize in one "
                    f"sentence:\n\n{article[:chars]}\n\nSummary:")
            request_id = f"cal-native-{uuid.uuid4().hex[:8]}"
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "stream": False,
            }
            prefill_payload = dict(payload)
            prefill_payload["max_tokens"] = 1
            prefill_payload["kv_transfer_params"] = {
                "do_remote_decode": True, "do_remote_prefill": False}
            headers = {"X-Request-Id": request_id}
            started = time.perf_counter()
            prefill_response = client.post(
                f"http://{args.prefill}/v1/chat/completions",
                json=prefill_payload, headers=headers)
            prefill_response.raise_for_status()
            prefill_ms = (time.perf_counter() - started) * 1000.0
            decode_payload = dict(payload)
            remote_params = (prefill_response.json() or {}).get(
                "kv_transfer_params")
            if remote_params:
                decode_payload["kv_transfer_params"] = remote_params
            decode_started = time.perf_counter()
            response = client.post(
                f"http://{args.decode}/v1/chat/completions",
                json=decode_payload, headers=headers)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            usage = (response.json() or {}).get("usage") or {}
            samples.append({
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "total_ms": round(prefill_ms + decode_ms, 2),
            })
        point = {
            "chars": chars,
            "prompt_tokens": int(statistics.median(
                s["prompt_tokens"] for s in samples)),
            "total_ms": round(statistics.median(
                s["total_ms"] for s in samples), 2),
            "prefill_ms": round(statistics.median(
                s["prefill_ms"] for s in samples), 2),
            "decode_ms": round(statistics.median(
                s["decode_ms"] for s in samples), 2),
        }
        points.append(point)
        print(json.dumps(point), file=sys.stderr, flush=True)

    # Least squares fit of total_ms against prompt_tokens over the sweep.
    xs = [p["prompt_tokens"] for p in points]
    ys = [p["total_ms"] for p in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    ms_per_1k = slope * 1000.0
    effective_mb_s = (KV_BYTES_PER_TOKEN * 1000.0) / (ms_per_1k * 1000.0) \
        if ms_per_1k > 0 else None
    summary = {
        "prefill": args.prefill,
        "decode": args.decode,
        "points": points,
        "fit": {
            "intercept_ms": round(intercept, 2),
            "ms_per_1k_tokens": round(ms_per_1k, 3),
            "effective_mb_s": (round(effective_mb_s, 1)
                               if effective_mb_s else None),
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

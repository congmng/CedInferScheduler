#!/usr/bin/env python3
"""Replay a classic-dataset trace against the multi-domain P/D router.

Unlike ``real_multidomain_client.py`` (which builds synthetic prefixes), this
client replays a tokenized trace produced by ``workloads.generators`` with
``--emit-text``.  It sends the raw prompt through ``/v1/completions`` so the
engine tokenizes exactly what the simulator was given (no chat template), and
it paces requests at the trace's own arrival times (open loop) with a cap on
in-flight requests.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import time

import httpx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--trace", required=True,
                        help="JSONL with input_text / arrival_time_ns.")
    parser.add_argument("--num-reqs", type=int, default=0,
                        help="Limit the number of replayed requests (0 = all).")
    parser.add_argument("--max-output-tokens", type=int, default=16,
                        help="Cap max_tokens per request; dataset replies are "
                             "often 500+ tokens and would dominate the run.")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Maximum in-flight requests.")
    parser.add_argument("--pacing", choices=("closed", "trace"), default="closed",
                        help="How the client paces itself.  'closed' (default) is "
                             "the real client's semaphore: a request starts at its "
                             "trace arrival *and* when a slot frees, so a server "
                             "that is slower than the trace throttles its own "
                             "arrival stream and the run lands in one of two "
                             "regimes (measured 2026-09-16: the same arm gave "
                             "718 ms and 4720 ms p50 TTFT this way).  'trace' "
                             "submits purely on the trace's clock, which makes two "
                             "arms comparable -- use it whenever the offered rate "
                             "is below the server's capacity, otherwise the "
                             "backlog grows without bound.")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Divide the trace's arrival offsets by this "
                             "(>1 compresses the run).")
    parser.add_argument("--request-timeout-s", type=float, default=180.0,
                        help="Per-request timeout.  A stalled P/D handoff can "
                             "hang forever otherwise, which would block the "
                             "whole phase instead of being recorded as a "
                             "failure.")
    parser.add_argument("--stream", action="store_true", default=False,
                        help="Use SSE streaming so per-request TTFT and TPOT "
                             "can be measured.  TTFT is the client-visible "
                             "time to the first generated token (prefill + KV "
                             "handoff + decode queue); without streaming only "
                             "the end-to-end latency is observable.")
    parser.add_argument("--slo-ttft-ms", type=float, default=0.0,
                        help="Fallback TTFT SLO for requests whose trace row "
                             "carries no ``slo_ttft_ms`` (0 = no bound).")
    parser.add_argument("--slo-tpot-ms", type=float, default=0.0,
                        help="Fallback TPOT SLO for trace rows without one.")
    parser.add_argument("--ignore-trace-slo", action="store_true", default=False,
                        help="Drop the per-request SLO entirely (no headers, no "
                             "local verdict).  This is the ablation switch that "
                             "turns the algorithm's SLO term off while the same "
                             "trace is still being replayed.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sse_update(line, state):
    """Fold one SSE line into ``state`` (``first_token_ms``/``tokens``/``usage``).

    Pure so the parser can be unit tested without a live engine.
    """
    if not line.startswith("data:"):
        return state
    body = line[len("data:"):].strip()
    if not body or body == "[DONE]":
        return state
    try:
        event = json.loads(body)
    except ValueError:
        return state
    if event.get("usage"):
        state["usage"] = event["usage"]
    for choice in event.get("choices") or ():
        if choice.get("text"):
            state["tokens"] = int(state.get("tokens", 0)) + 1
            if state.get("first_token_ms") is None:
                state["first_token_ms"] = state.get("now_ms")
    return state


def load_trace(path, limit):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("input_text"):
                raise SystemExit(f"{path} has no input_text; regenerate the "
                                 "trace with the generator's --emit-text")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def row_slo(row, args):
    """Per-request SLO: the trace row wins over the client's fallback."""
    if getattr(args, "ignore_trace_slo", False):
        return None, None
    ttft = float(row.get("slo_ttft_ms") or args.slo_ttft_ms or 0.0)
    tpot = float(row.get("slo_tpot_ms") or args.slo_tpot_ms or 0.0)
    return (ttft or None), (tpot or None)


def slo_ok(ttft_ms, tpot_ms, slo_ttft_ms, slo_tpot_ms):
    """Whether a finished request met every bound that applied to it.

    Returns ``None`` when a bound was set but the metric was not measured
    (a non-streaming request has no TTFT), so the aggregator can tell
    "violated" apart from "unknown" instead of silently counting a miss.
    """
    if slo_ttft_ms is not None:
        if ttft_ms is None:
            return None
        if ttft_ms > slo_ttft_ms:
            return False
    if slo_tpot_ms is not None:
        if tpot_ms is None:
            return None
        if tpot_ms > slo_tpot_ms:
            return False
    return True


async def main():
    args = parse_args()
    rows = load_trace(args.trace, args.num_reqs)
    if not rows:
        raise SystemExit("empty trace")
    origin_ns = rows[0]["arrival_time_ns"]
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    records = []
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        async def one(index, row):
            delay_s = (row["arrival_time_ns"] - origin_ns) / 1e9 / max(1e-9, args.time_scale)
            await asyncio.sleep(max(0.0, delay_s - (time.perf_counter() - started)))
            payload = {
                "model": args.model,
                "prompt": row["input_text"],
                "max_tokens": min(int(row.get("output_toks", args.max_output_tokens)),
                                  args.max_output_tokens),
                "temperature": 0.0,
                "stream": bool(args.stream),
            }
            if args.stream:
                # Ask for the exact completion token count in a final SSE chunk;
                # TPOT needs it, and counting chunks would be an approximation.
                payload["stream_options"] = {"include_usage": True}
            slo_ttft_ms, slo_tpot_ms = row_slo(row, args)
            headers = {"X-Request-Id": f"ds-{index}"}
            if slo_ttft_ms is not None:
                headers["X-SLO-TTFT-MS"] = str(slo_ttft_ms)
            if slo_tpot_ms is not None:
                headers["X-SLO-TPOT-MS"] = str(slo_tpot_ms)
            # ``trace`` pacing sends on the trace's clock only; the semaphore is
            # what makes the real client's arrival stream depend on its own
            # completions.
            gate = (semaphore if args.pacing == "closed"
                    else contextlib.nullcontext())
            async with gate:
                request_started = time.perf_counter()
                try:
                    if args.stream:
                        record = {"index": index, "input_toks": row.get("input_toks"),
                                  "ts": time.time()}
                        state = {"first_token_ms": None, "tokens": 0, "usage": {}}
                        async with client.stream(
                                "POST", f"{args.url}/v1/completions", json=payload,
                                headers=headers,
                                timeout=args.request_timeout_s) as response:
                            record["status"] = response.status_code
                            async for line in response.aiter_lines():
                                state["now_ms"] = (
                                    (time.perf_counter() - request_started) * 1000.0)
                                sse_update(line, state)
                        elapsed_ms = (time.perf_counter() - request_started) * 1000.0
                        usage = state["usage"] or {}
                        tokens = int(usage.get("completion_tokens")
                                     or state["tokens"])
                        ttft_ms = state["first_token_ms"]
                        record.update({
                            "latency_ms": round(elapsed_ms, 3),
                            "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
                            "completion_tokens": tokens,
                            "prompt_tokens": usage.get("prompt_tokens"),
                        })
                        if ttft_ms is not None and tokens > 0:
                            record["tpot_ms"] = round(
                                (elapsed_ms - ttft_ms) / max(1, tokens - 1), 3)
                        record["slo_ttft_ms"] = slo_ttft_ms
                        record["slo_tpot_ms"] = slo_tpot_ms
                        record["slo_ok"] = slo_ok(record.get("ttft_ms"),
                                                  record.get("tpot_ms"),
                                                  slo_ttft_ms, slo_tpot_ms)
                        record["ts"] = time.time()
                    else:
                        response = await client.post(
                            f"{args.url}/v1/completions", json=payload,
                            headers=headers,
                            timeout=args.request_timeout_s)
                        elapsed_ms = (time.perf_counter() - request_started) * 1000.0
                        usage = {}
                        if response.status_code == 200:
                            try:
                                usage = response.json().get("usage", {}) or {}
                            except Exception:
                                usage = {}
                        record = {"index": index,
                                  "input_toks": row.get("input_toks"),
                                  "latency_ms": round(elapsed_ms, 3),
                                  "status": response.status_code,
                                  "prompt_tokens": usage.get("prompt_tokens"),
                                  "completion_tokens": usage.get("completion_tokens"),
                                  "ts": time.time()}
                except Exception as exc:  # noqa: BLE001
                    record = {"index": index, "status": 0, "error": repr(exc),
                              "ts": time.time()}
            async with lock:
                records.append(record)

        started = time.perf_counter()
        await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))

    records.sort(key=lambda item: item["index"])
    with open(args.output, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    ok = [item for item in records if item["status"] == 200]
    latencies = sorted(item["latency_ms"] for item in ok)
    if latencies:
        def pct(values, value):
            return values[min(len(values) - 1, int(value * len(values)) - 1)]
        summary = {
            "requests": len(records), "ok": len(ok),
            "latency_mean_ms": round(statistics.mean(latencies), 3),
            "latency_p50_ms": round(pct(latencies, 0.50), 3),
            "latency_p95_ms": round(pct(latencies, 0.95), 3),
        }
        for field in ("ttft_ms", "tpot_ms"):
            values = sorted(item[field] for item in ok if item.get(field) is not None)
            if values:
                summary[f"{field}_mean"] = round(statistics.mean(values), 3)
                summary[f"{field}_p50"] = round(pct(values, 0.50), 3)
                summary[f"{field}_p95"] = round(pct(values, 0.95), 3)
        print(json.dumps(summary))
    else:
        print(json.dumps({"requests": len(records), "ok": 0}))


if __name__ == "__main__":
    asyncio.run(main())

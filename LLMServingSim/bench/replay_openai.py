"""Replay a token-pinned workload against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def _request(endpoint: str, model: str, row: dict, request_id: int,
             start_ns: int, base_arrival_ns: int) -> dict:
    target_ns = start_ns + row["arrival_time_ns"] - base_arrival_ns
    while time.perf_counter_ns() < target_ns:
        time.sleep(0.0005)
    arrival_ns = time.perf_counter_ns()
    payload = json.dumps({
        "model": model,
        "prompt": row["input_tok_ids"],
        "max_tokens": row["output_toks"],
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }).encode()
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    first_ns = None
    last_ns = None
    token_count = 0
    error = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                chunk = json.loads(data)
                text = chunk.get("choices", [{}])[0].get("text", "")
                if text:
                    now_ns = time.perf_counter_ns()
                    first_ns = first_ns or now_ns
                    last_ns = now_ns
                    token_count += 1
    except Exception as exc:
        error = repr(exc)
    return {
        "request_id": request_id,
        "input_toks": row["input_toks"],
        "output_toks": row["output_toks"],
        "arrival_ns": arrival_ns,
        "first_token_ns": first_ns,
        "last_token_ns": last_ns,
        "stream_chunks": token_count,
        "error": error,
    }


def _run_once(endpoint: str, model: str, rows: list[dict], workers: int) -> list[dict]:
    base_arrival_ns = min(row["arrival_time_ns"] for row in rows)
    start_ns = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(
            _request, endpoint, model, row, request_id, start_ns, base_arrival_ns
        ) for request_id, row in enumerate(rows)]
        records = [future.result() for future in futures]
    return sorted(records, key=lambda record: record["request_id"])


def _summarize(records: list[dict]) -> dict:
    valid = [record for record in records
             if record["first_token_ns"] and record["last_token_ns"]]
    latency = [(r["last_token_ns"] - r["arrival_ns"]) / 1e6 for r in valid]
    ttft = [(r["first_token_ns"] - r["arrival_ns"]) / 1e6 for r in valid]
    tpot = [
        (r["last_token_ns"] - r["first_token_ns"])
        / max(1, r["output_toks"] - 1) / 1e6
        for r in valid
    ]

    def metric(values: list[float]) -> dict:
        return {
            "mean_ms": statistics.fmean(values) if values else None,
            "p50_ms": _quantile(values, 0.50) if values else None,
            "p95_ms": _quantile(values, 0.95) if values else None,
            "p99_ms": _quantile(values, 0.99) if values else None,
        }

    return {
        "requests": len(records),
        "completed": len(valid),
        "errors": len(records) - len(valid),
        "latency": metric(latency),
        "ttft": metric(ttft),
        "tpot": metric(tpot),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for repeat in range(args.repeats):
        records = _run_once(args.endpoint, args.model, rows, args.workers)
        summary = _summarize(records)
        summary["repeat"] = repeat + 1
        runs.append(summary)
        (args.output_dir / f"requests-{repeat + 1:02d}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        print(json.dumps(summary, sort_keys=True), flush=True)

    complete = [run for run in runs if run["completed"] == run["requests"]]
    aggregate = {
        "repeats": len(runs),
        "successful_repeats": len(complete),
        "runs": runs,
        "model": args.model,
        "endpoint": args.endpoint,
        "dataset": str(args.dataset),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return 0 if len(complete) == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

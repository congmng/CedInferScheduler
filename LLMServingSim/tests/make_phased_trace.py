#!/usr/bin/env python3
"""Re-pace an existing real-text trace into a low -> high -> low phase profile.

The dataset generators (``workloads/generators``) emit one constant arrival
rate.  Structural-elasticity experiments need the opposite: a sustained
overload phase long enough that the controller can observe the deficit, decide
on ``+P``, start a Prefill container and actually route to it.  Rewriting
``arrival_time_ns`` over an unchanged row set keeps the experiment honest --
the prompts, token counts and SLOs stay exactly what the dataset produced, and
only the arrival process changes.

Pacing inside a phase is uniform (``1 / rate`` apart) rather than Poisson: the
earlier ``cnndm-short-elastic`` trace was built that way, so the two elastic
traces stay comparable.

Usage:
    python3 tests/make_phased_trace.py \
        --input workloads/cnndm-long-pool-qwen3-8b.jsonl \
        --rates 1,3.5,1 --durations 20,180,20 \
        --names low_a,high,low_c \
        --output workloads/cnndm-long-elastic-real-qwen3-8b.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Source JSONL (one request per line, ordered).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rates", required=True,
                        help="Comma separated arrival rates (requests/second), "
                             "one per phase.")
    parser.add_argument("--durations", required=True,
                        help="Comma separated phase durations in seconds.")
    parser.add_argument("--names", default="",
                        help="Optional comma separated phase labels; defaults "
                             "to phase0/phase1/...")
    parser.add_argument("--max-reqs", type=int, default=0, dest="max_reqs",
                        help="Truncate the emitted trace to this many requests "
                             "(0 = use exactly the rows the phases ask for).")
    return parser.parse_args()


def _floats(raw: str, flag: str) -> list[float]:
    try:
        values = [float(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:  # noqa: BLE001 - CLI validation
        raise SystemExit(f"{flag}: expected numeric comma separated values") from exc
    if not values or any(value <= 0 for value in values):
        raise SystemExit(f"{flag}: every value must be positive")
    return values


def main() -> int:
    args = parse_args()
    rates = _floats(args.rates, "--rates")
    durations = _floats(args.durations, "--durations")
    if len(rates) != len(durations):
        raise SystemExit("--rates and --durations must have the same length")
    names = [part.strip() for part in args.names.split(",") if part.strip()]
    if names and len(names) != len(rates):
        raise SystemExit("--names must have one label per phase")
    labels = names or [f"phase{index}" for index in range(len(rates))]

    source = pathlib.Path(args.input)
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]

    # One request every 1/rate seconds inside a phase, restarting the clock at
    # each boundary so the phases do not bleed into each other.
    wanted = sum(max(1, int(round(rate * duration))) for rate, duration in zip(rates, durations))
    if len(rows) < wanted:
        raise SystemExit(f"source has {len(rows)} rows but the phase profile "
                         f"asks for {wanted}; generate a bigger pool first")
    if args.max_reqs:
        wanted = min(wanted, args.max_reqs)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    now_ns = 0
    written = 0
    per_phase: list[dict] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for label, rate, duration in zip(labels, rates, durations):
            count = max(1, int(round(rate * duration)))
            step_ns = int(1_000_000_000 / rate)
            emitted = 0
            for _ in range(count):
                if written >= wanted:
                    break
                row = dict(rows[cursor])
                cursor += 1
                row["arrival_time_ns"] = int(now_ns)
                row["phase"] = label
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                now_ns += step_ns
                emitted += 1
                written += 1
            per_phase.append({"phase": label, "rate_rps": rate,
                              "duration_s": duration, "requests": emitted})
    span_s = 0.0 if written <= 1 else (now_ns - int(1_000_000_000 / rates[-1])) / 1e9

    meta = {"generator": "make_phased_trace",
            "source": str(source),
            "requests": written,
            "phases": per_phase,
            "span_s": round(span_s, 2)}
    pathlib.Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

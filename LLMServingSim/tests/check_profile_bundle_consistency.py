#!/usr/bin/env python3
"""Report whether several hardware bundles were collected the same way.

Cross-domain simulation compares one card against another, which only means
anything if the bundles behind them are the same measurement.  On 2026-09-18
that turned out to be false: 4090 and 5090 had been profiled on vLLM 0.27.1
while 3090 and A100 used 0.29.0, and A100's attention grid stopped at
``max_kv=4096`` where the other three reached 16384.  Nothing in the repo
noticed, because each bundle only describes itself.

This walks the ``meta.yaml`` files and prints the fields that have to agree,
flagging the ones that do not.  It is a report, not a test, because a bundle
may differ on purpose -- ``--root`` can point at any tree, so an A/B of two
engines stays inspectable instead of tripping an assertion.

    python3 tests/check_profile_bundle_consistency.py --model Qwen/Qwen3-8B
    python3 tests/check_profile_bundle_consistency.py --root /tmp/other-bundles
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]

# Fields a cross-domain comparison depends on.  ``skew_profile`` and
# ``measurement_iterations`` matter for noise level; the attention grid knobs
# decide which shapes were visited at all.
FIELDS = (
    ("vllm_version", lambda m: m.get("vllm_version")),
    ("cuda_version", lambda m: str(m.get("cuda_version"))),
    ("tp_degrees", lambda m: tuple(m.get("tp_degrees") or ())),
    ("measurement_iterations", lambda m: m.get("measurement_iterations")),
    ("attention.max_kv", lambda m: (m.get("attention_grid") or {}).get("max_kv")),
    ("attention.chunk_factor",
     lambda m: (m.get("attention_grid") or {}).get("chunk_factor")),
    ("attention.kv_factor",
     lambda m: (m.get("attention_grid") or {}).get("kv_factor")),
    ("engine.max_num_seqs",
     lambda m: (m.get("engine_effective") or {}).get("max_num_seqs")),
    ("engine.max_num_batched_tokens",
     lambda m: (m.get("engine_effective") or {}).get("max_num_batched_tokens")),
)


def collect(root: pathlib.Path, model: str, variant: str) -> dict:
    found = {}
    for meta in sorted(root.glob(f"*/{model}/{variant}/meta.yaml")):
        hardware = meta.relative_to(root).parts[0]
        try:
            found[hardware] = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - report, do not crash
            print(f"  {hardware}: unreadable meta.yaml ({exc})")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="profiler/perf")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--variant", default="bf16")
    args = parser.parse_args()

    root = REPO / args.root
    bundles = collect(root, args.model, args.variant)
    if not bundles:
        print(f"no bundles under {root} for {args.model}/{args.variant}")
        return 1

    print(f"{len(bundles)} bundle(s) under {args.root} for {args.model}/{args.variant}")
    names = list(bundles)
    for label, getter in FIELDS:
        values = {name: getter(bundles[name]) for name in names}
        distinct = {repr(value) for value in values.values()}
        flag = "ok " if len(distinct) == 1 else "DIFFERS"
        print(f"  [{flag}] {label}")
        if len(distinct) > 1:
            for name in names:
                print(f"            {name:<10} {values[name]}")

    print("\nper-bundle provenance:")
    for name in names:
        meta = bundles[name]
        print(f"  {name:<10} profiled_at={meta.get('profiled_at')} "
              f"gpu={meta.get('gpu')} shards={meta.get('shards') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

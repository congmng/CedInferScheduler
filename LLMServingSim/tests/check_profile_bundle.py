#!/usr/bin/env python3
"""Check that a measured profile bundle contains the requested TP data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--tp", default="1,2")
    args = parser.parse_args()

    root = Path("profiler/perf") / args.hardware / args.model / args.variant
    meta_path = root / "meta.yaml"
    if not meta_path.is_file():
        raise SystemExit(f"missing meta.yaml: {meta_path}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    requested = [int(value) for value in args.tp.split(",") if value]
    declared = set(meta.get("tp_degrees", []))
    missing = [tp for tp in requested if tp not in declared]
    if missing:
        raise SystemExit(f"TP missing from metadata: {missing}")

    for tp in requested:
        folder = root / f"tp{tp}"
        for name in ("dense.csv", "per_sequence.csv", "attention.csv"):
            path = folder / name
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            with path.open(newline="", encoding="utf-8") as stream:
                rows = sum(1 for _ in csv.DictReader(stream))
            if rows == 0:
                raise SystemExit(f"empty {path}")
            print(f"{args.hardware} {args.model} tp{tp} {name}: {rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

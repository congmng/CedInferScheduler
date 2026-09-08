"""Compare two simulator CSV outputs using the common CASR metrics."""
import csv
import statistics
import sys


def summary(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no request rows in {path}")
    def mean(name):
        values = [float(row[name]) for row in rows]
        return statistics.fmean(values) / 1_000_000
    return {
        "requests": len(rows),
        "latency_ms": mean("latency"),
        "ttft_ms": mean("TTFT"),
        "decode_instances": sorted({row.get("decode_instance_id", "") for row in rows}),
        "prefix_hit_tokens": sum(int(float(row.get("npu_hit_tokens", 0) or 0)) for row in rows),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tests/compare_casr.py BASELINE.csv CASR.csv")
    base, casr = map(summary, sys.argv[1:])
    print("metric                 baseline       casr")
    for key in ("requests", "latency_ms", "ttft_ms", "prefix_hit_tokens"):
        print(f"{key:22s} {base[key]:12.3f} {casr[key]:12.3f}")
    print(f"{'decode_instances':22s} {base['decode_instances']} {casr['decode_instances']}")

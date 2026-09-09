"""Compare two simulator CSV outputs using the common CASR metrics."""
import csv
import statistics
import sys


def summary(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no request rows in {path}")
    def scale(name):
        return sorted(float(row[name]) / 1_000_000 for row in rows)
    def quantile(values, q):
        if not values:
            return 0.0
        index = min(len(values) - 1, int(q * (len(values) - 1)))
        return values[index]

    latency = scale("latency")
    ttft = scale("TTFT")
    tpot = scale("TPOT")
    decode_counts = {}
    for row in rows:
        decode_counts[row.get("decode_instance_id", "")] = decode_counts.get(
            row.get("decode_instance_id", ""), 0) + 1
    return {
        "requests": len(rows),
        "latency_mean_ms": statistics.fmean(latency) if latency else 0.0,
        "latency_p50_ms": quantile(latency, 0.50),
        "latency_p95_ms": quantile(latency, 0.95),
        "latency_p99_ms": quantile(latency, 0.99),
        "ttft_mean_ms": statistics.fmean(ttft) if ttft else 0.0,
        "ttft_p95_ms": quantile(ttft, 0.95),
        "tpot_mean_ms": statistics.fmean(tpot) if tpot else 0.0,
        "tpot_p95_ms": quantile(tpot, 0.95),
        "decode_instances": sorted(decode_counts),
        "decode_counts": decode_counts,
        "prefix_hit_tokens": sum(int(float(row.get("npu_hit_tokens", 0) or 0)) for row in rows),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tests/compare_casr.py BASELINE.csv CASR.csv")
    base, casr = map(summary, sys.argv[1:])
    print("metric                 baseline       casr")
    for key in ("requests", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                "latency_p99_ms", "ttft_mean_ms", "ttft_p95_ms",
                "tpot_mean_ms", "tpot_p95_ms", "prefix_hit_tokens"):
        print(f"{key:22s} {base[key]:12.3f} {casr[key]:12.3f}")
    print(f"{'decode_instances':22s} {base['decode_instances']} {casr['decode_instances']}")
    print(f"{'decode_counts':22s} {base['decode_counts']} {casr['decode_counts']}")

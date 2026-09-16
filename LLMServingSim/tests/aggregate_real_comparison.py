#!/usr/bin/env python3
"""Aggregate multi-round multi-domain P/D router comparisons.

Each round directory produced by ``run_real_multidomain_comparison.sh`` holds a
``metrics-<policy>.jsonl`` with one record per request.  This script reports, per
policy, the per-round mean/P50/P95 latency plus the cross-round mean and a 95%
confidence interval, so ablation and baseline claims are not read off a single
lucky run.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rounds", nargs="+", help="round directories")
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--baseline", default="load", help="policy to compare against")
    parser.add_argument("--pairs", action="store_true",
                        help="also report the share of traffic each (prefill, decode) pair served")
    parser.add_argument("--slo-ttft-ms", type=float, default=None,
                        help="TTFT SLO in ms; enables SLO attainment and goodput")
    parser.add_argument("--slo-tpot-ms", type=float, default=None,
                        help="TPOT SLO in ms; enables SLO attainment and goodput")
    parser.add_argument("--window-s", type=float, default=None,
                        help="measurement window for goodput; defaults to the "
                             "span between the first and last recorded request")
    parser.add_argument("--glitch-ms", type=float, default=5000.0,
                        help="count requests slower than this as transfer glitches")
    parser.add_argument("--slo-grid-ttft", default="",
                        help="comma-separated TTFT bounds for a sensitivity grid")
    parser.add_argument("--slo-grid-tpot", default="",
                        help="comma-separated TPOT bounds for a sensitivity grid")
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)
    return ordered[max(0, index)]


def trimmed_mean(values, fraction=0.01):
    """Mean after dropping the slowest ``fraction`` of requests.

    A stalled KV handoff can add a 30-50 s outlier that dominates a 600-request
    mean; the trimmed mean keeps the comparison about the routing policy rather
    than about one unlucky transfer.
    """
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    drop = int(len(ordered) * fraction)
    kept = ordered[:len(ordered) - drop] if drop else ordered
    return statistics.mean(kept)


def load_round(directory):
    per_policy = {}
    duplicates = {}
    for path in sorted(pathlib.Path(directory).glob("metrics-*.jsonl")):
        policy = path.stem.split("metrics-", 1)[1]
        # One request id must count once.  Re-running a policy into the same
        # result directory appends a second block to ``metrics-<policy>.jsonl``
        # and silently doubles those requests in every mean/percentile, so keep
        # the first occurrence and surface how many rows were dropped.
        rows, seen = [], set()
        dup = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = record.get("request_id")
            if request_id is not None:
                if request_id in seen:
                    dup += 1
                    continue
                seen.add(request_id)
            if record.get("status") == 200 and "total_ms" in record:
                rows.append(record)
        if rows:
            per_policy[policy] = rows
        if dup:
            duplicates[policy] = dup
    return per_policy, duplicates


def slo_satisfied(row, slo_ttft_ms, slo_tpot_ms):
    """Whether one request meets every configured SLO.

    A request whose TTFT/TPOT was not measured (a non-streaming round) cannot
    be shown to meet the SLO, so it counts as a miss rather than being dropped:
    silently ignoring unmeasured requests would inflate attainment.
    """
    if slo_ttft_ms is not None:
        value = row.get("ttft_ms")
        if value is None or value > slo_ttft_ms:
            return False
    if slo_tpot_ms is not None:
        value = row.get("tpot_ms")
        if value is None or value > slo_tpot_ms:
            return False
    return True


def row_slo_satisfied(row, slo_ttft_ms, slo_tpot_ms):
    """Per-request verdict first, CLI thresholds as the fallback.

    The router records ``slo_ok`` from the bound the *request* carried (trace
    row or client fallback), so a run that mixes budgets is judged per request.
    Only rows without a recorded verdict fall back to one global threshold.
    """
    recorded = row.get("slo_ok")
    if recorded is not None:
        return bool(recorded)
    return slo_satisfied(row, slo_ttft_ms, slo_tpot_ms)


def main():
    args = parse_args()
    loaded = [load_round(directory) for directory in args.rounds]
    rounds = [per_policy for per_policy, _dup in loaded]
    duplicates = {}
    for index, (_per_policy, dup) in enumerate(loaded):
        if dup:
            duplicates[str(args.rounds[index])] = dup
    if duplicates:
        # A duplicated request id means the same policy ran twice into one
        # directory; the extra rows were dropped, but the reader should know.
        print(f"WARNING: duplicate request ids dropped: {duplicates}\n")
    policies = args.policies or sorted({p for r in rounds for p in r})
    summary = {}
    trace_slo = any(row.get("slo_ok") is not None
                    for per_policy in rounds for rows in per_policy.values()
                    for row in rows)
    # Rounds produced before the per-request verdict existed (or by a router
    # without it) have metrics but no ``slo_ok``.  Without a fallback they
    # would be judged against "no bound configured" and silently count as
    # hits, inflating attainment.  Adopt the bound the trace declares.
    if args.slo_ttft_ms is None:
        args.slo_ttft_ms = next(
            (row["slo_ttft_ms"] for per_policy in rounds
             for rows in per_policy.values() for row in rows
             if row.get("slo_ttft_ms") is not None), None)
    if args.slo_tpot_ms is None:
        args.slo_tpot_ms = next(
            (row["slo_tpot_ms"] for per_policy in rounds
             for rows in per_policy.values() for row in rows
             if row.get("slo_tpot_ms") is not None), None)
    slo_enabled = (args.slo_ttft_ms is not None or args.slo_tpot_ms is not None
                   or trace_slo)
    header = (f"{'policy':14} {'rounds':>6} {'reqs':>5} {'mean':>9} "
              f"{'mean*':>9} {'p50':>9} {'p95':>9}")
    if slo_enabled:
        header += f" {'ttftP95':>8} {'tpotP95':>8} {'slo%':>6} {'goodput':>8}"
    print(header)
    for policy in policies:
        round_means, round_p50, round_p95, total, nreq = [], [], [], [], 0
        round_trimmed = []
        ttfts, tpots, glitches, slo_hits = [], [], 0, 0
        # Goodput must be summed *per round*: the gap between two rounds is not
        # measurement time, and using one global span understates it by orders
        # of magnitude.
        slo_window = 0.0
        for per_policy in rounds:
            rows = per_policy.get(policy)
            if not rows:
                continue
            latencies = [row["total_ms"] for row in rows]
            round_means.append(statistics.mean(latencies))
            round_trimmed.append(trimmed_mean(latencies))
            round_p50.append(percentile(latencies, 0.50))
            round_p95.append(percentile(latencies, 0.95))
            total.extend(latencies)
            nreq += len(rows)
            glitches += sum(1 for value in latencies if value > args.glitch_ms)
            round_ts = []
            for row in rows:
                if row.get("ttft_ms") is not None:
                    ttfts.append(row["ttft_ms"])
                if row.get("tpot_ms") is not None:
                    tpots.append(row["tpot_ms"])
                if row.get("ts") is not None:
                    round_ts.append(row["ts"])
                if slo_enabled and row_slo_satisfied(row, args.slo_ttft_ms,
                                                     args.slo_tpot_ms):
                    slo_hits += 1
            if len(round_ts) > 1:
                slo_window += max(round_ts) - min(round_ts)
        if not round_means:
            continue
        mean = statistics.mean(round_means)
        trimmed = statistics.mean(round_trimmed)
        spread = statistics.stdev(round_means) if len(round_means) > 1 else 0.0
        ci95 = 1.96 * spread / math.sqrt(len(round_means)) if len(round_means) > 1 else 0.0
        summary[policy] = {
            "rounds": len(round_means), "requests": nreq,
            "mean_ms": round(mean, 1), "mean_ci95_ms": round(ci95, 1),
            "mean_trimmed_ms": round(trimmed, 1),
            "p50_ms": round(statistics.mean(round_p50), 1),
            "p95_ms": round(statistics.mean(round_p95), 1),
            "round_means": [round(value, 1) for value in round_means],
            "glitches_gt_ms": args.glitch_ms,
            "glitch_count": glitches,
        }
        print(f"{policy:14} {len(round_means):>6} {nreq:>5} "
              f"{mean:>9.1f} {trimmed:>9.1f} "
              f"{statistics.mean(round_p50):>9.1f} {statistics.mean(round_p95):>9.1f}",
              end="")
        if ttfts:
            summary[policy]["ttft_p95_ms"] = round(percentile(ttfts, 0.95), 1)
        if tpots:
            summary[policy]["tpot_p95_ms"] = round(percentile(tpots, 0.95), 1)
        if slo_enabled:
            window = (args.window_s * len(round_means) if args.window_s
                      else slo_window)
            attainment = slo_hits / nreq if nreq else 0.0
            summary[policy]["slo"] = {
                "source": "per-request" if trace_slo else "cli",
                "ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms,
                "attainment": round(attainment, 4), "hits": slo_hits,
                "requests": nreq, "window_s": round(window, 3) if window else None,
                "goodput_rps": round(slo_hits / window, 4) if window else None,
            }
            print(f" {summary[policy].get('ttft_p95_ms', float('nan')):>8.1f}"
                  f" {summary[policy].get('tpot_p95_ms', float('nan')):>8.1f}"
                  f" {100 * attainment:>6.1f}"
                  f" {(slo_hits / window if window else 0.0):>8.2f}", end="")
        print()

    if args.pairs:
        print("\ntraffic share per (prefill -> decode) pair:")
        for policy in policies:
            counts = collections.Counter(
                (row.get("prefill"), row.get("decode"))
                for per_policy in rounds for row in per_policy.get(policy, []))
            total = sum(counts.values())
            if not total:
                continue
            ranked = "  ".join(f"{p}->{d} {100 * n / total:.0f}%"
                               for (p, d), n in counts.most_common())
            print(f"  {policy:14} {ranked}")

    baseline = summary.get(args.baseline)
    if baseline:
        print(f"\ndeltas vs {args.baseline}:")
        for policy, stats in summary.items():
            if policy == args.baseline:
                continue
            dmean = (stats["mean_ms"] - baseline["mean_ms"]) / baseline["mean_ms"] * 100
            dp95 = (stats["p95_ms"] - baseline["p95_ms"]) / baseline["p95_ms"] * 100
            print(f"  {policy:14} mean {dmean:+.1f}%  p95 {dp95:+.1f}%")

    grid_ttft = [float(v) for v in args.slo_grid_ttft.split(",") if v.strip()]
    grid_tpot = [float(v) for v in args.slo_grid_tpot.split(",") if v.strip()]
    if grid_ttft and grid_tpot:
        # One threshold is a choice; the ranking must survive the grid.  Report
        # SLO attainment (%) for every (TTFT, TPOT) pair so a single lucky bound
        # cannot carry the conclusion.
        print("\nSLO attainment (%) over the threshold grid:")
        header = f"  {'policy':14}" + "".join(
            f"  ttft<={int(t)}/tpot<={int(p):<3}" for t in grid_ttft for p in grid_tpot)
        print(header)
        for policy in policies:
            rows = [row for per_policy in rounds
                    for row in per_policy.get(policy, [])]
            if not rows:
                continue
            cells = []
            for ttft in grid_ttft:
                for tpot in grid_tpot:
                    hits = sum(1 for row in rows
                               if slo_satisfied(row, ttft, tpot))
                    cells.append(f"  {100 * hits / len(rows):>17.1f}")
            print(f"  {policy:14}" + "".join(cells))

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

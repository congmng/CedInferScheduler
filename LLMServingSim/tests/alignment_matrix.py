#!/usr/bin/env python3
"""One table for every recorded run and its simulator replay.

The alignment work is spread over a dozen recorded bundles (short prompts,
long prompts, forced handoff, paced arrivals, a hot-prefix workload).  Each
replay writes an ``alignment.json`` next to its CSV; this tool collects them
into a single markdown table so the current state of the simulator is one
command away, and so a stale row is visible as such: every file carries the
simulator revision that produced it, and rows from other revisions are shown
with their revision rather than silently mixed in.

    python3 tests/alignment_matrix.py --root /mnt/home/casr/results
    python3 tests/alignment_matrix.py --root ... --run --out docs/对齐矩阵.md

``--run`` (re)generates the canonical set, which takes ~20 minutes: it drives
``replay_real_run.py`` with the placement and pacing of each recording, so the
comparison is "does the model execute the cluster's own decisions at the
cluster's own arrival rate", never "does the controller pick the same thing"
-- that question has its own runs (see docs/模拟器与真机一致性核查.md).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
# The arm every recorded bundle below ran: forced handoff, no structural
# actions, small cluster (A100 and 3090b out).
ARM_CONFIG = "configs/cluster/casr_real_small3_forced_transfer.json"

# bundle -> policies; the flags match how the bundle was recorded.
# Every replay pins the cluster's own (Prefill, Decode) placement *and* replays
# its recorded submission times: that isolates "does the model execute this
# placement at this arrival rate" from "does the controller pick the same
# thing".  Leaving the routing free is a different (and much slower) question --
# a policy that spreads onto the slower cards builds an open-loop backlog the
# cluster never had, so those runs belong in their own experiment, not in this
# table.
CANONICAL = (
    # (bundle, policies, client model)
    #   pacing -- the cluster's own submission times, replayed open-loop (the
    #             delivered traces were recorded with CLIENT_PACING=trace)
    #   closed -- the cluster's concurrency cap, which is what the saturated
    #             bundles were recorded with; replaying those open-loop would
    #             build a backlog the cluster's own client never had
    ("small3-paced1rps", "casr_lp casr_full", "pacing"),
    ("small3-paced06", "load casr_lp", "pacing"),
    ("small3-hotprefix", "load cache_aware", "pacing"),
    ("small3-long-xfer", "casr_lp", "closed"),
    ("small3-elastic-heavy", "casr_lp", "closed"),
)


def revision():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unknown"


def run_canonical(root, out_dir, timeout_s=1800):
    out_dir.mkdir(parents=True, exist_ok=True)
    for bundle, policies, mode in CANONICAL:
        real_dir = pathlib.Path(root) / bundle
        if not real_dir.is_dir():
            print(f"   (skip {bundle}: no recorded bundle)")
            continue
        command = [sys.executable, str(REPO / "tests" / "replay_real_run.py"),
                   "--real-dir", str(real_dir),
                   "--cluster-config", ARM_CONFIG,
                   "--policies", *policies.split(),
                   "--out", str(out_dir / bundle)]
        command += ["--replay-placement"]
        if mode == "pacing":
            command += ["--replay-client-pacing"]
        print(f"== {bundle} [placement+{mode}] {policies}", flush=True)
        try:
            subprocess.run(command, cwd=REPO, check=False,
                           timeout=timeout_s if timeout_s else None)
        except subprocess.TimeoutExpired:
            print(f"   (timed out after {timeout_s}s -- skipped)")


def collect(roots, current_revision_only):
    rows = []
    paths = []
    for root in roots:
        paths += sorted(pathlib.Path(root).glob("**/alignment.json"))
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        meta = report.get("_meta") or {}
        if current_revision_only and meta.get("simulator_revision") != revision():
            continue
        for policy, entry in sorted(report.items()):
            if policy.startswith("_") or "real" not in entry or "sim" not in entry:
                continue
            real, sim = entry["real"], entry["sim"]
            rows.append({
                # A replay lives in ``<bundle>/replay*/``; label the row with
                # the *bundle* so the de-dup below keeps one row per recording
                # instead of one per historical replay directory.
                "bundle": (path.parent.parent.name
                           if path.parent.name.startswith(("replay", "alignment"))
                           else path.parent.name),
                "generated_at": meta.get("generated_at", ""),
                "policy": policy,
                "mode": ("pacing" if meta.get("replay_client_pacing")
                         else "placement" if meta.get("replay_placement") else "-"),
                "revision": meta.get("simulator_revision", "-"),
                "n": real.get("n"),
                "real_ttft": real.get("ttft_p50"),
                "sim_ttft": sim.get("ttft_p50"),
                "real_e2e": real.get("e2e_p50"),
                "sim_e2e": sim.get("e2e_p50"),
                "real_tpot": real.get("tpot_p50"),
                "sim_tpot": sim.get("tpot_p50"),
            })
    # One row per (bundle, policy): keep the newest replay, so a stale file
    # left behind by an earlier pass cannot silently appear next to a fresh one.
    newest = {}
    for row in rows:
        key = (row["bundle"], row["policy"])
        if key not in newest or row["generated_at"] > newest[key]["generated_at"]:
            newest[key] = row
    return [newest[key] for key in sorted(newest)]


def render(rows):
    out = ["| bundle | policy | mode | n | TTFT real/sim (ms) | E2E real/sim (ms) | ratio | TPOT real/sim | rev |",
           "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        ratio = (row["sim_e2e"] / row["real_e2e"]
                 if row["real_e2e"] and row["sim_e2e"] else float("nan"))
        out.append(
            f"| {row['bundle']} | {row['policy']} | {row['mode']} | {row['n']} "
            f"| {row['real_ttft']:.0f} / {row['sim_ttft']:.0f} "
            f"| {row['real_e2e']:.0f} / {row['sim_e2e']:.0f} "
            f"| {ratio:.2f}x "
            f"| {row['real_tpot']:.1f} / {row['sim_tpot']:.1f} "
            f"| {row['revision']} |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/home/casr/results")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-dir", default="",
                        help="where --run writes its replays (default <root>/alignment-matrix)")
    parser.add_argument("--scan", default="",
                        help="extra directory to scan for alignment.json")
    parser.add_argument("--timeout-s", type=int, default=1800,
                        help="per-replay wall-clock cap when running")
    parser.add_argument("--run", action="store_true",
                        help="(re)generate the canonical replays first (~20 min)")
    parser.add_argument("--all-revisions", action="store_true",
                        help="include rows produced by older simulator revisions")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        pathlib.Path(args.root) / "alignment-matrix")
    if args.run:
        run_canonical(args.root, out_dir, timeout_s=args.timeout_s)
    roots = [str(out_dir), args.root] + ([args.scan] if args.scan else [])
    rows = collect(roots, current_revision_only=not args.all_revisions)
    if not rows:
        print("no alignment.json for this revision; pass --all-revisions or --run")
        return 1
    table = render(rows)
    print(table)
    if args.out:
        path = pathlib.Path(args.out)
        path.write_text(
            f"# 对齐矩阵（模拟器 revision {revision()}）\n\n"
            "由 `tests/alignment_matrix.py` 生成：每条记录都是"
            "「灌入真机放置/到达流 → 模拟器执行 → 与真机并排」。\n\n"
            + table + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

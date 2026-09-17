#!/usr/bin/env python3
"""Regenerate the figures used by the CASR report / paper draft.

Simulator numbers come from the archived `docs/arena-summaries/*.json`, so the
figures move with the artifacts.  Real-cluster numbers are quoted here with the
section of `docs/实验结果汇总.md` they come from, because that document is the
authoritative record (it carries the round counts and the artifact paths).

    python3 tests/plot_report_figures.py --out ../docs/figs

Axis labels stay in English: the CJK fallback font on this machine renders, but
English labels make the figures reusable in an English submission without a
second pass.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
SUMS = REPO.parent / "docs" / "arena-summaries"

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = (
    "#3b6db3", "#e08a1e", "#3f9c5a", "#c0392b", "#7f8c8d", "#7d5ba6")


def _load(name):
    return json.loads((SUMS / name).read_text())


def _mean_p95(name, arms):
    """(means, p95s) in seconds, from a run summary (values are ms)."""
    data = _load(name)
    return ([data[a]["e2e_mean"] / 1000 for a in arms],
            [data[a]["e2e_p95"] / 1000 for a in arms])


def fig_arms_hetero(out):
    """Six arms on the hardware-heterogeneous board, mean and p95."""
    arms = ["load", "cache_aware", "kv_aware", "rr", "casr_lp", "casr_full"]
    labels = ["load", "cache_aware", "kv_aware", "rr", "casr_lp", "casr_full"]
    means, p95 = _mean_p95("zamba2-hetero5090x2-4090x4-1250tok-8rps.json", arms)
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - 0.2, means, 0.4, label="E2E mean", color=BLUE)
    ax.bar(x + 0.2, p95, 0.4, label="E2E p95", color=ORANGE)
    ax.set_yscale("log")
    ax.set_ylabel("latency (s, log)")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_title("Zamba2-1.2B on 2xRTX5090 + 4xRTX4090, 1250-token prompts, 8 rps")
    for xi, v in zip(x - 0.2, means):
        ax.text(xi, v * 1.08, f"{v:.0f}", ha="center", fontsize=7)
    for xi, v in zip(x + 0.2, p95):
        ax.text(xi, v * 1.08, f"{v:.0f}", ha="center", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-arms-hetero.png", dpi=200)
    plt.close(fig)


def fig_waterfall(out):
    """Where the 9x comes from: metric fix, planning, elasticity."""
    stages = [("load /\ncache_aware", 403.0, BLUE),
              ("+ binding-resource\nmetric", 179.7, GREEN),
              ("+ byte-budget\nplanning", 65.3, ORANGE),
              ("+ structural\nelasticity", 45.2, RED)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    prev = stages[0][1]
    for i, (label, value, colour) in enumerate(stages):
        ax.bar(i, value, 0.55, color=colour)
        ax.text(i, value + 12, f"{value:.0f} s", ha="center", fontsize=8)
        if i:
            drop = prev / value
            ax.annotate("", xy=(i - 0.28, value), xytext=(i - 0.72, prev),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1))
            ax.text(i - 0.5, (value + prev) / 2 - 30, f"{drop:.2f}x",
                    ha="center", fontsize=8, color="#555")
        prev = value
    ax.set_xticks(range(len(stages)), [s[0] for s in stages], fontsize=8)
    ax.set_ylabel("E2E mean latency (s)")
    ax.set_ylim(0, 470)
    ax.set_title("Decomposing the gap: 403 s -> 45 s (9.0x)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-waterfall.png", dpi=200)
    plt.close(fig)


def fig_kv_size_flip(out):
    """The link-cost term flips sign with the KV each request has to move.

    Real cluster, 2P x 3D, same dataset and arrival rate, only the prompt head
    changes.  Penalty is the `casr_nonetwork` arm (the planner with the link
    term removed) relative to the full planner: negative means dropping the
    term *helps*.  Source: docs/实验结果汇总.md 5.8 / 5.9.
    """
    kv_mb = [37.6, 184.0, 560.0]
    penalty = [-5.6, 0.6, 31.8]            # % vs full planner
    cross_gbps = [0.15, 0.5, 3.8]          # bandwidth the chosen placement needs
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(0, color="#555", lw=0.9)
    ax.plot(kv_mb, penalty, "o-", color=RED, lw=2)
    for x, y, bw in zip(kv_mb, penalty, cross_gbps):
        ax.annotate(f"{y:+.1f}%\n(needs {bw:g} GB/s)", (x, y),
                    textcoords="offset points",
                    xytext=(-6, 14) if y >= 0 else (16, -8),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(28, 760)
    ax.set_ylim(-11, 45)
    ax.set_xticks(kv_mb, [f"{v:.0f}" for v in kv_mb])
    ax.set_xlabel("KV per request (MB)")
    ax.set_ylabel("penalty of dropping the link term (%)")
    ax.set_title("The link term's value flips sign with KV size")
    ax.text(600, -9, "fabric capacity ~2 GB/s", fontsize=7.5, color="#555",
            ha="right", va="bottom")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-kv-size-flip.png", dpi=200)
    plt.close(fig)


def fig_hotprefix(out):
    """Hot-prefix workload, LAN vs WAN fabric."""
    arms = ["load", "cache_aware", "rr", "casr_lp"]
    lan = _load("zamba2-homo4090-hotprefix-lan.json")
    wan = _load("zamba2-homo4090-hotprefix-wan.json")
    lan_v = [lan[a]["e2e_mean"] / 1000 for a in arms]
    wan_v = [wan[a]["e2e_mean"] / 1000 for a in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - 0.2, lan_v, 0.4, label="LAN 0.11 GB/s, 48 ms", color=BLUE)
    ax.bar(x + 0.2, wan_v, 0.4, label="WAN 0.05 GB/s, 80 ms", color=ORANGE)
    ax.set_yscale("log")
    ax.set_ylabel("E2E mean (s, log)")
    ax.set_xticks(x, arms, rotation=10)
    for xi, v in zip(x - 0.2, lan_v):
        ax.text(xi, v * 1.15, f"{v:.2f}" if v < 1 else f"{v:.0f}", ha="center", fontsize=7)
    for xi, v in zip(x + 0.2, wan_v):
        ax.text(xi, v * 1.15, f"{v:.2f}" if v < 1 else f"{v:.0f}", ha="center", fontsize=7)
    ax.set_title("Hot-prefix workload: CASR is invariant to the fabric, blind routing is not")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-hotprefix-lan-wan.png", dpi=200)
    plt.close(fig)


def fig_slo(out):
    """Real cluster: share of requests meeting TTFT <= 500 ms.

    Source: docs/实验结果汇总.md 5.16 (multi-threshold table).
    """
    loads = ["CNN-short", "Dolly", "ShareGPT\ndeep", "CNN-XL", "5-domain"]
    baseline = [59.2, 69.8, 16.7, 0.0, 57.1]      # `load`
    casr = [100.0, 100.0, 100.0, 30.0, 100.0]
    x = np.arange(len(loads))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x - 0.2, baseline, 0.4, label="load (best baseline)", color=GREY)
    ax.bar(x + 0.2, casr, 0.4, label="casr_lp", color=RED)
    ax.set_ylabel("requests with TTFT <= 500 ms (%)")
    ax.set_xticks(x, loads)
    ax.set_ylim(0, 112)
    for xi, v in zip(x + 0.2, casr):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Real cluster (4P x 3D): TTFT SLO attainment")
    fig.tight_layout()
    fig.savefig(out / "fig-slo-attainment.png", dpi=200)
    plt.close(fig)


def fig_models(out):
    """Same board, same trace, only the model changes."""
    arms = ["load", "cache_aware", "kv_aware", "rr", "casr_lp", "casr_full"]
    zamb = _load("zamba2-homo4090-1250tok-8rps.json")
    qwen = _load("qwen3-8b-homo4090-1250tok-8rps.json")
    qwen["kv_aware"] = _load("qwen3-8b-kv-aware-only.json")["kv_aware"]
    q_v = [qwen[a]["e2e_mean"] / 1000 for a in arms]
    z_v = [zamb[a]["e2e_mean"] / 1000 for a in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.bar(x - 0.2, q_v, 0.4, label="Qwen3-8B (full attention)", color=PURPLE)
    ax.bar(x + 0.2, z_v, 0.4, label="Zamba2-1.2B (real hybrid)", color=GREEN)
    ax.set_yscale("log")
    ax.set_ylabel("E2E mean (s, log)")
    ax.set_xticks(x, arms, rotation=12)
    ax.set_title("6xRTX4090, 1250-token, 8 rps: the hybrid flips the baselines' ranking")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-model-compare.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO.parent / "docs" / "figs"))
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig_arms_hetero, fig_waterfall, fig_kv_size_flip,
               fig_hotprefix, fig_slo, fig_models):
        fn(out)
        print("wrote", fn.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

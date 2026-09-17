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
    """Where the 9.7x comes from, on one board, with every level measured.

    The review of the draft pointed out that the middle step was mis-attributed:
    the jump from `load` to `kv_aware` is reproduced almost exactly by `load_rr`
    (the *same* score with a round-robin tie-break), so it is the tie-break, not
    the binding-resource metric, that recovers it.  Levels are `load` ->
    `load_rr` -> `casr_lp` -> `casr_full` on 6xRTX4090 (r9-tiebreak.json).
    """
    data = _load("r9-tiebreak.json")
    stages = [("load", "load", GREY),
              ("+ round-robin\ntie-break", "load_rr", "#8fbcd4"),
              ("+ byte-budget\nplanning", "casr_lp", ORANGE),
              ("+ structural\nelasticity", "casr_full", RED)]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    prev = data[stages[0][1]]["e2e_mean_ms"] / 1000
    for i, (label, arm, colour) in enumerate(stages):
        value = data[arm]["e2e_mean_ms"] / 1000
        ax.bar(i, value, 0.55, color=colour)
        ax.text(i, value + 12, f"{value:.0f} s", ha="center", fontsize=8)
        if i:
            ax.annotate("", xy=(i - 0.28, value), xytext=(i - 0.72, prev),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1))
            ax.text(i - 0.5, (value + prev) / 2 - 34, f"{prev / value:.2f}x",
                    ha="center", fontsize=8, color="#555")
        prev = value
    ax.set_xticks(range(len(stages)), [s_[0] for s_ in stages], fontsize=8)
    ax.set_ylabel("E2E mean (s)")
    ax.set_ylim(0, 500)
    ax.set_title("One board, four measured levels: 419 s -> 43 s (9.7x)")
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


def _r8_rows(name, order_key):
    """Parse an R8 archive into {(setting, arm): record}, sorted by setting."""
    data = _load(name)
    rows = []
    for key, value in data.items():
        tag, arm = key.split("/")
        rows.append((order_key(tag), tag, arm, value))
    return sorted(rows)


def _rps(tag):
    return int(tag.split("rps")[1])


def _outlen(tag):
    return int(tag.split("-out")[1])


def fig_q2_output_length(out):
    """Output length alone does not move the Q2 decision on this board."""
    rows = _r8_rows("r8-output-length.json", _outlen)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for arm, colour, marker in (("load", GREY, "s"), ("casr_lp", RED, "o")):
        pts = [(tag, rec["local_share"] * 100) for _, tag, a, rec in rows if a == arm]
        ax.plot([_outlen(t) for t, _ in pts], [v for _, v in pts],
                marker=marker, color=colour, label=arm, lw=2)
    ax.set_xscale("log")
    ax.set_ylim(-5, 110)
    ax.set_xticks([16, 64, 128, 256, 512], ["16", "64", "128", "256", "512"])
    ax.set_xlabel("output length, tokens (forced)")
    ax.set_ylabel("requests served locally (%)")
    ax.set_title("Q2 vs output length: no crossover, the Decode never saturates")
    ax.text(20, 50, "6 Decodes x 16 slots ~ 48 req/s of headroom,\n"
                    "offered load is 8 req/s", fontsize=7.5, color="#555")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-q2-output-length.png", dpi=200)
    plt.close(fig)


def fig_q2_decode_load(out):
    """The substitution fires once the Decode queue is priced.

    Two rules on the same traces and the same board: ``cap=3`` is the
    deployment-derived reading (the externality was capped at 3 *milliseconds*,
    i.e. inert against a 731 ms transfer), ``cap=50`` treats the cap as an
    amplification factor and also charges the producer's egress backlog.
    """
    inert = _r8_rows("r8-decodeload-cap3.json", _rps)
    fixed = _r8_rows("r8-decodeload-cap50.json", _rps)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for rows, style, label in ((inert, "--", "queue term inert (cap = 3 ms)"),
                               (fixed, "-", "queue term priced (cap = 50x)")):
        for arm, colour, marker in (("load", GREY, "s"), ("casr_lp", RED, "o")):
            pts = [(_rps(tag), rec["local_share"] * 100)
                   for _, tag, a, rec in rows if a == arm]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                    marker=marker, color=colour, lw=1.8,
                    label=f"{arm}, {label}")
    ax.set_ylim(-5, 110)
    ax.set_xlabel("offered load (req/s)  --  512-token outputs, 2 Decodes")
    ax.set_ylabel("requests served locally (%)")
    ax.set_title("Q2 switches only when the Decode queue is in the price")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-q2-decode-load.png", dpi=200)
    plt.close(fig)


def fig_tiebreak(out):
    """Is the 735/0 skew the metric's fault, or the tie-break's?"""
    data = _load("r9-tiebreak.json")
    arms = ["load", "load_rr", "kv_aware", "casr_lp", "casr_full"]
    means = [data[a]["e2e_mean_ms"] / 1000 for a in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    colours = [GREY, "#8fbcd4", BLUE, ORANGE, RED]
    ax.bar(x, means, 0.6, color=colours)
    for xi, arm, value in zip(x, arms, means):
        ax.text(xi, value + 8, f"{value:.0f} s", ha="center", fontsize=8)
    # The landing goes under the arm name rather than inside the bars: the
    # spread is the point (735/0 against 368/367), and in-bar text either
    # overflows the short bars or collides with the tall one.
    ax.set_xticks(x, [f"{arm}\n" + " / ".join(str(v) for v in data[arm]["prefills"].values())
                      for arm in arms], fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 480)
    ax.set_ylabel("E2E mean (s)")
    ax.set_title("The 735/0 skew is the tie-break: load_rr recovers 2.56x\n"
                 "(Prefill landing under each arm)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-tiebreak.png", dpi=200)
    plt.close(fig)


def fig_elastic_horizon(out):
    """Structural elasticity pays only after the peak outlives the 45 s boot."""
    data = _load("r7-peak-duration.json")
    points = sorted((int(key.split("-peak")[1].split("/")[0]),
                     key.split("/")[1], value) for key, value in data.items())
    peaks = sorted({peak for peak, _, _ in points})
    lp = [v["e2e_mean_ms"] / 1000 for p, arm, v in points if arm == "casr_lp"]
    full = [v["e2e_mean_ms"] / 1000 for p, arm, v in points if arm == "casr_full"]
    gains = [(a - b) / a * 100 for a, b in zip(lp, full)]

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(len(peaks))
    ax.bar(x - 0.2, lp, 0.4, label="casr_lp (static pool)", color=ORANGE)
    ax.bar(x + 0.2, full, 0.4, label="casr_full (+P allowed)", color=RED)
    for xi, gain, a, b in zip(x, gains, lp, full):
        ax.text(xi, max(a, b) + 3, f"{gain:+.1f}%", ha="center", fontsize=8.5)
    ax.axvspan(-0.5, 0.5, color="#bbb", alpha=0.25)
    ax.text(0.08, 62, "peak shorter than the 45 s\ncontainer boot: no gain",
            ha="center", fontsize=7.5, color="#444")
    ax.set_xticks(x, [f"{p} s" for p in peaks])
    ax.set_xlabel("peak duration  (1250-token prompts, 8 req/s, 45 s boot)")
    ax.set_ylabel("E2E mean (s)")
    ax.set_ylim(0, 100)
    ax.set_title("Elasticity's break-even: the peak has to outlive the boot")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig-elastic-horizon.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO.parent / "docs" / "figs"))
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures = (fig_arms_hetero, fig_waterfall, fig_kv_size_flip,
               fig_hotprefix, fig_slo, fig_models,
               fig_q2_output_length, fig_q2_decode_load, fig_tiebreak,
               fig_elastic_horizon)
    for fn in figures:
        fn(out)
        print("wrote", fn.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

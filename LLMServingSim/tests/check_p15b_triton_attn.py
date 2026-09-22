#!/usr/bin/env python3
"""Step 6 gate: the Triton state branch must reproduce the torch one.

The two paths compute the same joint softmax over window + selected states; the
Triton path does the state half in a kernel and merges the two branches in log
space (``design/dsv4_ref/triton_sparse_attn.merge``).  If the merge were a
weighted average instead of a log-sum-exp shift, the outputs would differ by
more than rounding -- that is the exact bug the design doc records, so this gate
checks the output rather than the implementation.

Also reports the forward time of each path, which is the number step 6 is
actually about (the profiler re-measures the whole attention category for the
same reason; this is the cheap in-process version).

    docker run --rm --gpus '"device=0"' \
        -e PYTHONPATH=/work/deploy/vllm_p15b \
        --entrypoint python3 -v "$PWD":/work -w /work \
        vllm/vllm-openai:casr029 tests/check_p15b_triton_attn.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.vllm_p15b.config import P15BConfig              # noqa: E402
from deploy.vllm_p15b.model import (                        # noqa: E402
    P15BAttention,
    STATE_KERNEL_ENV,
)


def _layer(block: str, device: str):
    hf = json.loads((REPO / "configs" / "model" / "casr"
                     / f"P15B-{block}.json").read_text(encoding="utf-8"))
    cfg = P15BConfig.from_hf(hf)
    torch.manual_seed(0)
    return P15BAttention(cfg, cfg.compress_ratios[0]).to(device).eval()


def _run(layer, x, positions):
    with torch.no_grad():
        return layer(x, positions)


def _time(layer, x, positions, iters: int) -> float:
    for _ in range(3):                       # warm-up: JIT + kernel autotune
        _run(layer, x, positions)
    if x.is_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        _run(layer, x, positions)
        if x.is_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--blocks", default="r4,r128")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("no CUDA device; this gate needs a GPU")
        return 1

    failures = []
    for block in [b for b in args.blocks.split(",") if b]:
        layer = _layer(block, args.device)
        x = torch.randn(1, args.tokens, layer.cfg.hidden_size, device=args.device,
                        dtype=torch.float32)
        positions = torch.arange(args.tokens, device=args.device)[None]

        os.environ[STATE_KERNEL_ENV] = "torch"
        out_torch = _run(layer, x, positions)
        os.environ[STATE_KERNEL_ENV] = "triton"
        out_triton = _run(layer, x, positions)
        os.environ.pop(STATE_KERNEL_ENV)

        delta = (out_torch - out_triton).abs().max().item()
        scale = out_torch.abs().mean().item()
        ok = delta <= 1e-4 * max(scale, 1e-6) + 1e-5
        os.environ[STATE_KERNEL_ENV] = "torch"
        us_torch = _time(layer, x, positions, args.iters)
        os.environ[STATE_KERNEL_ENV] = "triton"
        us_triton = _time(layer, x, positions, args.iters)
        os.environ.pop(STATE_KERNEL_ENV)
        print(f"{block:<5} tokens={args.tokens:<6} max|Δ|={delta:.3e} "
              f"(mean|out|={scale:.3f})  {'ok' if ok else 'MISMATCH'}   "
              f"torch {us_torch:8.1f} us  triton {us_triton:8.1f} us  "
              f"({us_triton / us_torch:.2f}x)")
        if not ok:
            failures.append(block)

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

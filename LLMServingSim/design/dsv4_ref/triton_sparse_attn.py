#!/usr/bin/env python3
"""The one DSV4 operator with no stock-PyTorch equivalent, written in Triton.

Attending to a *sparse, index-selected* set of compressed states is what the
official stack does with FlashMLA / FlashInfer-sparse, and both are built for
Hopper and newer.  Everything else in the block is plain matmul / norm / rope,
so this is the operator that decides whether "we compile the missing parts
ourselves" is real.  This file is the smallest version that can answer that:
one kernel, no autotuning, no fp8, checked numerically against the PyTorch
path on whatever GPU it runs on.

    python3 -m design.dsv4_ref.triton_sparse_attn --tokens 64 --states 96 \
            --topk 32 --heads 4 --head-dim 512

Run it on sm80 / sm86 / sm89 / sm120 and compare the reported max error: if the
same source compiles and matches on all four, the Triton half of the plan in
docs/DSV4_10-30B跨卡设计.md §2.3 is de-risked.
"""

from __future__ import annotations

import argparse
import math

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - reported by the caller
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = exc
else:
    TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _sparse_state_attn(
        q_ptr,            # (T, H, D)
        s_ptr,            # (N, D)  -- compressed states, already projected
        idx_ptr,          # (T, K)  -- which state each query selected
        valid_ptr,        # (T, K)  -- 1 if that state is at or before the query
        out_ptr,          # (T, H, D)
        T, K,
        scale,
        H: tl.constexpr, D: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """One program per (query, head): online softmax over top-k states."""
        t = tl.program_id(0)
        h = tl.program_id(1)
        d = tl.arange(0, D)
        q = tl.load(q_ptr + t * H * D + h * D + d).to(tl.float32)

        # Finite floor rather than -inf: an early block whose top-k is entirely
        # in the future would otherwise make `running_max - new_max` NaN.
        running_max = -1e30
        running_sum = 0.0
        acc = tl.zeros((D,), dtype=tl.float32)

        for start in range(0, K, BLOCK_K):
            k_off = start + tl.arange(0, BLOCK_K)
            k_mask = k_off < K
            rows = tl.load(idx_ptr + t * K + k_off, mask=k_mask, other=0)
            ok = tl.load(valid_ptr + t * K + k_off, mask=k_mask, other=0) > 0
            # gather the selected states: (BLOCK_K, D)
            block = tl.load(s_ptr + rows[:, None] * D + d[None, :],
                            mask=k_mask[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(block * q[None, :], axis=1) * scale
            scores = tl.where(ok, scores, -1e30)
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            alpha = tl.exp(running_max - new_max)
            weights = tl.where(ok, tl.exp(scores - new_max), 0.0)
            acc = acc * alpha + tl.sum(weights[:, None] * block, axis=0)
            running_sum = running_sum * alpha + tl.sum(weights, axis=0)
            running_max = new_max

        out = acc / tl.maximum(running_sum, 1e-20)
        tl.store(out_ptr + t * H * D + h * D + d, out.to(out_ptr.dtype.element_ty))


def reference(q, states, idx, valid):
    """Same computation in plain PyTorch -- the thing the kernel must match."""
    selected = states[idx]                                   # (T, K, D)
    scores = torch.einsum("thd,tkd->thk", q, selected) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~valid[:, None, :], float("-inf")).softmax(-1)
    scores = scores * valid[:, None, :].any(-1, keepdim=True).to(scores.dtype)
    return torch.einsum("thk,tkd->thd", scores, selected)


def triton_forward(q, states, idx, valid, block_k=64):
    t, h, d = q.shape
    k = idx.shape[1]
    out = torch.empty_like(q)
    _sparse_state_attn[(t, h)](
        q, states, idx, valid.to(torch.int8), out,
        t, k, 1.0 / math.sqrt(d),
        H=h, D=d, BLOCK_K=block_k,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--states", type=int, default=96)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-k", type=int, default=64)
    args = parser.parse_args()

    if triton is None:
        print(f"triton unavailable: {TRITON_IMPORT_ERROR}")
        return 1

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    device = args.device
    q = torch.randn(args.tokens, args.heads, args.head_dim,
                    device=device, dtype=dtype)
    states = torch.randn(args.states, args.head_dim, device=device, dtype=dtype)
    idx = torch.randint(0, args.states, (args.tokens, args.topk), device=device)
    # a causal-ish mask: roughly a quarter of the top-k may be in the future
    valid = torch.rand(args.tokens, args.topk, device=device) > 0.25
    valid[:, 0] = True

    got = triton_forward(q, states, idx, valid, args.block_k)
    torch_bf16 = reference(q, states, idx, valid)
    # float64 reference on the same inputs: the yardstick for both paths
    exact = reference(q.double(), states.double(), idx, valid)

    def report(label, tensor):
        d = (tensor.double() - exact).abs()
        print(f"{label:<22} max {d.max().item():.6f}  mean {d.mean().item():.6f}")

    scale = exact.abs().mean().item()
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    cap = torch.cuda.get_device_capability(0) if device == "cuda" else None
    print(f"device       {name}")
    print(f"capability   sm{cap[0]}{cap[1]}" if cap else "capability   n/a")
    print(f"triton       {triton.__version__}")
    print(f"shape        T={args.tokens} H={args.heads} D={args.head_dim} "
          f"K={args.topk} N={args.states} BLOCK_K={args.block_k} {args.dtype}")
    print(f"error vs fp64 (|out| mean = {scale:.4f}):")
    report("  torch bf16 path", torch_bf16)
    report("  triton kernel", got)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""The second portable operator: inverse-RoPE, written in Triton.

`docs/DSV4_10-30B跨卡设计.md` §2.3 lists two operators we have to write
ourselves.  The sparse-state attention was the first (it has no stock PyTorch
equivalent because official code ships it as FlashMLA / FlashInfer-sparse).
This is the second, and it is the *other* half of the official `o_proj`:

    inv-RoPE -> split into o_groups chunks ->  wo_a[g]  ->  wo_b

The grouped matmuls are stock (and portable on every card we care about); the
part that has to be written by hand is the inverse rotation, because the
official op folds it into an fp8 DeepGEMM kernel (`fused_inv_rope_fp8_quant` +
`fp8_einsum`) that Ampere and Ada cannot run.  A candidate that dequantises or
ignores the rotation silently produces wrong numbers, so this file checks the
round trip `inv_rope(rope(x)) == x` rather than only testing the rotation.

    python3 -m design.dsv4_ref.triton_o_proj --tokens 128 --heads 20 \
            --head-dim 512 --rope-dim 64
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
    def _inv_rope_kernel(
        x_ptr, pos_ptr, out_ptr,
        T, half,
        log_theta,
        H: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Undo RoPE on the last ROPE channels of every (t, h).

        One program handles BLOCK channels of one (t, h) pair; `half` is
        ROPE // 2 and the pairs are (i, i + half) inside the rope slice.
        """
        pid = tl.program_id(0)
        h = tl.program_id(1)
        t = tl.program_id(2)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        mask = i < half

        base = t * H * D + h * D + (D - ROPE)
        r1 = tl.load(x_ptr + base + i, mask=mask, other=0.0).to(tl.float32)
        r2 = tl.load(x_ptr + base + half + i, mask=mask, other=0.0).to(tl.float32)

        pos = tl.load(pos_ptr + t).to(tl.float32)
        # theta ** (-i / half) == exp(-i * 2 / ROPE * ln theta): Triton has no
        # scalar ** tensor, so ln(theta) is taken on the host and passed in.
        freq = tl.exp(-(i.to(tl.float32)) * (2.0 / ROPE) * log_theta)
        angle = pos * freq
        cos = tl.cos(angle)
        sin = tl.sin(angle)

        a = r1 * cos + r2 * sin
        b = -r1 * sin + r2 * cos
        tl.store(out_ptr + base + i, a.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + base + half + i, b.to(out_ptr.dtype.element_ty),
                 mask=mask)


def rope_torch(x, positions, rope_dim, theta=10000.0):
    """Forward RoPE, mirroring ``model.rope`` exactly."""
    if rope_dim <= 0:
        return x
    nope, part = x[..., :-rope_dim], x[..., -rope_dim:]
    half = rope_dim // 2
    freqs = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float()[:, :, None] * freqs[None, None, :]
    cos, sin = angles.cos(), angles.sin()
    a, b = part[..., :half], part[..., half:]
    rotated = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return torch.cat([nope, rotated], dim=-1)


def inv_rope_torch(x, positions, rope_dim, theta=10000.0):
    """Inverse of :func:`rope_torch`, on the same layout (B, T, H, D)."""
    if rope_dim <= 0:
        return x
    nope, part = x[..., :-rope_dim], x[..., -rope_dim:]
    half = rope_dim // 2
    freqs = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float()[:, :, None] * freqs[None, None, :]
    cos, sin = angles.cos(), angles.sin()
    r1, r2 = part[..., :half], part[..., half:]
    rotated = torch.cat([r1 * cos + r2 * sin, -r1 * sin + r2 * cos], dim=-1)
    return torch.cat([nope, rotated], dim=-1)


def inv_rope_triton(x, positions, rope_dim, theta=10000.0, block=64):
    """Triton version.  ``x`` is (T, H, D); ``positions`` is (T,)."""
    if rope_dim <= 0:
        return x
    t, h, d = x.shape
    half = rope_dim // 2
    # The kernel only rewrites the rope slice; the no-RoPE channels are copied
    # by identity rather than left as empty_like garbage.
    out = x.clone()
    grid = (triton.cdiv(half, block), h, t)
    _inv_rope_kernel[grid](
        x, positions.to(x.device), out,
        t, half, math.log(theta),
        H=h, D=d, ROPE=rope_dim, BLOCK=block,
    )
    return out


def o_proj_torch(attn_out, positions, wo_a, wo_b, rope_dim):
    """inv-RoPE -> grouped wo_a -> wo_b, the portable reading of the official op."""
    x = inv_rope_torch(attn_out, positions, rope_dim)
    flat = x.reshape(x.shape[0], -1)
    chunks = torch.chunk(flat, len(wo_a), dim=-1)
    grouped = torch.cat([chunk @ w for chunk, w in zip(chunks, wo_a)], dim=-1)
    return grouped @ wo_b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--o-lora", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block", type=int, default=64)
    args = parser.parse_args()

    if triton is None:
        print(f"triton unavailable: {TRITON_IMPORT_ERROR}")
        return 1

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    dev = "cuda"
    q_dim = args.heads * args.head_dim
    x = torch.randn(args.tokens, args.heads, args.head_dim, device=dev, dtype=dtype)
    positions = torch.arange(args.tokens, device=dev)

    # 1) the round trip is what a dequantising port gets wrong
    rotated = rope_torch(x, positions[:, None], args.rope_dim)
    back_torch = inv_rope_torch(rotated, positions[:, None], args.rope_dim)
    back_triton = inv_rope_triton(rotated, positions, args.rope_dim, block=args.block)
    print(f"device       {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"capability   sm{cap[0]}{cap[1]}")
    print(f"triton       {triton.__version__}")
    print(f"round trip vs fp32 input, max abs error:")
    print(f"  torch   {(back_torch.float() - x.float()).abs().max().item():.6f}")
    print(f"  triton  {(back_triton.float() - x.float()).abs().max().item():.6f}")
    print(f"triton vs torch inv-rope, max abs error: "
          f"{(back_triton.float() - back_torch.float()).abs().max().item():.6f}")

    # 2) the composed op, which is what a model would call
    chunk = q_dim // args.groups
    wo_a = [torch.randn(chunk, args.o_lora, device=dev, dtype=torch.float32)
            * (1.0 / math.sqrt(chunk)) for _ in range(args.groups)]
    wo_b = torch.randn(args.groups * args.o_lora, args.hidden,
                       device=dev, dtype=torch.float32) * 0.02
    attn_out = torch.randn(args.tokens, args.heads, args.head_dim,
                           device=dev, dtype=torch.float32)
    want = o_proj_torch(attn_out, positions[:, None], wo_a, wo_b, args.rope_dim)
    got_rot = inv_rope_triton(attn_out, positions, args.rope_dim, block=args.block)
    flat = got_rot.float().reshape(args.tokens, -1)
    chunks = torch.chunk(flat, args.groups, dim=-1)
    got = torch.cat([c @ w
                     for c, w in zip(chunks, wo_a)], dim=-1) @ wo_b
    print(f"o_proj (inv-rope + grouped wo_a + wo_b), max abs error vs torch: "
          f"{(got - want).abs().max().item():.6f}")
    print(f"|out| mean   {want.abs().mean().item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

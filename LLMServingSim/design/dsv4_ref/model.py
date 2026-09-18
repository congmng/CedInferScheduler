"""Plain-PyTorch DeepSeek-V4-style block, written to run on any CUDA GPU.

Everything here is deliberately simple and slow: no custom kernels, no fp8, no
tilelang, no mHC.  The point is to fix the *design* -- shapes, layer pattern,
compression, sparse selection, KV accounting -- and to be numerically identical
on sm86/sm89/sm120, which is what the later Triton port has to match.

One block, in order:

    x = x + Attention(RMSNorm(x))          # MLA + optional CSA/HCA compression
    x = x + FFN_or_MoE(RMSNorm(x))         # standard pre-norm residual (no mHC)

Attention has three parts, following the released config's semantics:

1. **MLA latent** every layer keeps: ``qkv_down`` produces ``[q_latent | kv_latent]``,
   q goes up to ``num_heads x head_dim``, the kv latent stays ``head_dim`` wide and
   is what a full layer caches (``head_dim + rope`` per token).
2. **Sliding window** over the most recent ``coff * ratio`` tokens' latents.
3. **Compressed states** for layers with ``compress_ratio > 0``: one state every
   ``ratio`` tokens, each ``2 * coff * head_dim`` wide (kv_state + score_state),
   of which a sparse indexer picks the top-k by score.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DSV4RefConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def rope(x: torch.Tensor, positions: torch.Tensor, rope_dim: int,
         theta: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to the last ``rope_dim`` channels of ``x`` (B, T, H, D)."""
    if rope_dim <= 0:
        return x
    nope, rope_part = x[..., :-rope_dim], x[..., -rope_dim:]
    half = rope_dim // 2
    # Angles are built in fp32 (that is where the precision is), then cast back
    # to the activation dtype: leaving them fp32 would silently upcast the whole
    # attention path and make the bf16 model fail at the first o_lora_a matmul.
    angles_dtype = rope_part.dtype
    freqs = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float()[:, :, None] * freqs[None, None, :]      # (B,T,half)
    cos = angles.cos()[:, :, None, :].to(angles_dtype)
    sin = angles.sin()[:, :, None, :].to(angles_dtype)
    a, b = rope_part[..., :half], rope_part[..., half:]
    rotated = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return torch.cat([nope, rotated], dim=-1)


class Indexer(nn.Module):
    """Sparse selector: one score per (query, compressed state), keep top-k.

    Simplification: the released "lightning indexer" is multi-head with its own
    ``index_head_dim``; we score with a single learned key/query projection and
    document it, because the *selection* is what the design needs to fix, not the
    head count.  ``index_topk`` (and the causal filter) are kept.
    """

    def __init__(self, cfg: DSV4RefConfig, state_dim: int):
        super().__init__()
        self.head_dim = cfg.index_head_dim
        self.topk = cfg.index_topk
        self.q_proj = nn.Linear(cfg.q_lora_rank, self.head_dim, bias=False)
        self.k_proj = nn.Linear(state_dim, self.head_dim, bias=False)

    def forward(self, q_latent, score_state):
        """(B,T,ql), (B,N,coff*hd) -> indices (B,T,K) and the scores' shape."""
        b, t, _ = q_latent.shape
        n = score_state.shape[1]
        if n == 0:
            return torch.zeros(b, t, 0, dtype=torch.long, device=q_latent.device), 0
        k = min(self.topk, n)
        q = self.q_proj(q_latent)                                   # (B,T,di)
        kk = self.k_proj(score_state)                               # (B,N,di)
        sim = torch.einsum("btd,bnd->btn", q, kk) / math.sqrt(self.head_dim)
        return sim.topk(k, dim=-1).indices, sim.shape[-1]


class Compressor(nn.Module):
    """hidden -> one (kv_state | score_state) per ``ratio`` tokens.

    The window is ``coff * ratio`` tokens wide and advances by ``ratio`` tokens,
    so overlapping (coff=2, the ratio-4 CSA layers) reads two windows' worth of
    hidden states but still stores one state per ``ratio`` tokens -- which is
    exactly what makes the per-token KV arithmetic ``2*coff*head_dim/ratio``.
    """

    def __init__(self, cfg: DSV4RefConfig, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.coff = cfg.coff(ratio)
        self.state = 2 * self.coff * cfg.head_dim
        self.window = self.coff * ratio
        self.norm = RMSNorm(self.state, cfg.rms_norm_eps)
        # Design choice: the released config exposes the ratio, the state width
        # and the op order but not the compressor's internals.  A dense
        # projection of the whole window would cost hidden*window*state -- for
        # HCA (window 128, hidden 1024) that is 134M parameters *per layer*,
        # which no released config would afford.  We pool the window and project
        # once, so the compressor's cost is hidden*state regardless of m.
        self.proj = nn.Linear(cfg.hidden_size, self.state, bias=False)

    def forward(self, x, positions):
        """x (B,T,H) -> ((B,N,coff*hd), (B,N,coff*hd), (B,N)) positions."""
        b, t, h = x.shape
        n = t // self.ratio
        if n == 0:
            empty = x.new_zeros(b, 0, self.state // 2)
            return empty, empty, positions[:, :0]
        # left-pad so the j-th window ends at (j+1)*ratio
        xp = F.pad(x, (0, 0, self.window - self.ratio, 0))
        win = xp.unfold(1, self.window, self.ratio)          # (B,N,H,window)
        pooled = win.mean(dim=-1)                            # (B,N,H)
        state = self.norm(self.proj(pooled))
        kv_state, score_state = state.split(self.state // 2, dim=-1)
        end = (torch.arange(n, device=x.device) + 1) * self.ratio
        return kv_state, score_state, positions[:, end - 1]


class Attention(nn.Module):
    """MLA attention with a sliding window and (optionally) compressed states."""

    def __init__(self, cfg: DSV4RefConfig, ratio: int):
        super().__init__()
        self.cfg = cfg
        self.ratio = ratio
        self.coff = cfg.coff(ratio) if ratio else 1
        self.window = cfg.coff(ratio) * ratio if ratio else cfg.max_position_embeddings
        hd, nh, rope = cfg.head_dim, cfg.num_attention_heads, cfg.qk_rope_head_dim
        self.qkv_down = nn.Linear(cfg.hidden_size, cfg.q_lora_rank + hd + rope, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.kv_norm = RMSNorm(hd + rope, cfg.rms_norm_eps)
        self.q_up = nn.Linear(cfg.q_lora_rank, nh * hd, bias=False)
        self.head_dim = hd
        self.n_heads = nh
        self.rope_dim = rope
        # grouped output LoRA: ng x [q_dim/ng -> o_lora] then [o_lora -> hidden]
        self.o_groups = cfg.o_groups
        self.o_lora_a = nn.ModuleList([
            nn.Linear(nh * hd // cfg.o_groups, cfg.o_lora_rank, bias=False)
            for _ in range(cfg.o_groups)])
        # vLLM's wo_b consumes all groups' LoRA outputs flattened: (ng * ol) -> h
        self.o_lora_b = nn.Linear(cfg.o_groups * cfg.o_lora_rank,
                                  cfg.hidden_size, bias=False)
        self.compressor = Compressor(cfg, ratio) if ratio else None
        self.indexer = (Indexer(cfg, cfg.coff(ratio) * hd) if ratio else None)
        self.kv_state_proj = (nn.Linear(cfg.coff(ratio) * hd, hd, bias=False)
                              if ratio else None)

    def _qkv(self, x, positions):
        b, t, _ = x.shape
        q_lat, kv = self.qkv_down(x).split(
            [self.cfg.q_lora_rank, self.head_dim + self.rope_dim], dim=-1)
        q = self.q_up(self.q_norm(q_lat)).view(b, t, self.n_heads, self.head_dim)
        kv = self.kv_norm(kv).view(b, t, 1, self.head_dim + self.rope_dim)
        q = rope(q, positions, self.rope_dim)
        kv = rope(kv, positions, self.rope_dim)
        self._last_kv = kv
        return q, kv, q_lat

    def forward(self, x, positions):
        """Returns (attn_out (B,T,H), kv_summary dict for the cache accounting)."""
        b, t, _ = x.shape
        q, kv, q_lat = self._qkv(x, positions)
        info = {"raw_tokens": t, "states": 0, "selected": 0, "window": min(self.window, t)
                if self.ratio else t}

        # keys/values the window sees: the raw latents (causal + banded)
        base = kv[:, :, :, :self.head_dim].squeeze(2)                    # (B,T,hd)
        logits = torch.einsum("bthd,bsd->bhts", q, base) / math.sqrt(self.head_dim)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        if self.ratio and self.window < t:
            causal = causal & torch.ones(t, t, dtype=torch.bool,
                                         device=x.device).triu(1 - self.window)
        weights = logits.masked_fill(~causal[None, None], float("-inf")).softmax(-1)
        values = base.unsqueeze(1).expand(b, t, t, self.head_dim)        # per-query view

        # compressed states: append the top-k selected values behind the window
        if self.compressor is not None and t >= self.ratio:
            kv_state, score_state, state_pos = self.compressor(x, positions)
            idx, _ = self.indexer(q_lat, score_state)                    # (B,T,K)
            proj = self.kv_state_proj(kv_state)                          # (B,N,hd)
            sel = torch.gather(proj[:, None].expand(b, t, -1, -1), 2,
                               idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            gathered_pos = torch.gather(
                state_pos[:, None].expand(b, t, -1), 2, idx)      # (B,T,K)
            valid = gathered_pos <= positions.unsqueeze(-1)
            extra = torch.einsum("bthd,btkd->bhtk", q, sel) / math.sqrt(self.head_dim)
            # A query whose whole top-k lands in the future has no valid entry:
            # mask with a large finite value (not -inf, which would make the
            # softmax NaN) and zero the block's contribution afterwards.
            extra = extra.masked_fill(~valid[:, None], -1e30).softmax(-1)
            extra = extra * valid[:, None].any(-1, keepdim=True).to(extra.dtype)
            weights = torch.cat([weights, extra], dim=-1)
            values = torch.cat([values, sel], dim=2)
            info["states"] = score_state.shape[1]
            info["selected"] = int(valid.sum(-1).float().mean().item())

        out = torch.einsum("bhts,btsd->bthd", weights, values)
        out = out.reshape(b, t, self.n_heads * self.head_dim)
        chunks = torch.chunk(out, self.o_groups, dim=-1)
        out = torch.cat([proj_(chunk) for proj_, chunk in zip(self.o_lora_a, chunks)], dim=-1)
        return self.o_lora_b(out), info


class MoE(nn.Module):
    """Small top-k MoE with SwiGLU clamping (the stability device we keep).

    Hash routing (the official model's first layers) is omitted; it is a routing
    trick, not part of the KV design.
    """

    swiglu_limit = 10.0

    def __init__(self, cfg: DSV4RefConfig):
        super().__init__()
        h, e, mi = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
        self.h, self.e, self.mi = h, e, mi
        self.topk = cfg.num_experts_per_tok
        self.gate = nn.Linear(h, e, bias=False)
        self.up = nn.Parameter(torch.randn(e, h, 2 * mi) * 0.02)
        self.down = nn.Parameter(torch.randn(e, mi, h) * 0.02)

    def forward(self, x):
        b, t, h = x.shape
        probs = self.gate(x).softmax(-1)
        w, idx = probs.topk(self.topk, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        flat = x.reshape(-1, h)
        w = w.reshape(-1, self.topk)
        idx = idx.reshape(-1, self.topk)
        out = torch.zeros_like(flat)
        for e in range(self.e):
            assigned = idx == e
            rows = assigned.any(-1)
            if not rows.any():
                continue
            tok = torch.nonzero(rows, as_tuple=False).squeeze(-1)
            slot = assigned[rows].float().argmax(-1)
            gate = w[rows].gather(1, slot[:, None]).squeeze(-1)
            # SwiGLU with clamping on the gate branch
            gu = flat[tok] @ self.up[e]                       # (n, 2*mi)
            gate_b, up_b = gu.chunk(2, dim=-1)
            act = F.silu(gate_b.clamp(max=self.swiglu_limit)) * up_b
            y = act @ self.down[e]
            out.index_add_(0, tok, y * gate[:, None])
        return out.view(b, t, h)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DSV4RefConfig, ratio: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg, ratio)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn = MoE(cfg) if cfg.is_moe else nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(cfg.ffn_hidden, cfg.hidden_size, bias=False))

    def forward(self, x, positions):
        h, info = self.attn(self.attn_norm(x), positions)
        x = x + h                                   # standard residual, no mHC
        x = x + self.ffn(self.ffn_norm(x))
        return x, info


class DSV4RefModel(nn.Module):
    """Embedding + N decoder layers.  No mHC anywhere by construction."""

    def __init__(self, cfg: DSV4RefConfig):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.rng_seed)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(cfg, r) for r in cfg.compress_ratios)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, token_ids):
        b, t = token_ids.shape
        positions = torch.arange(t, device=token_ids.device)[None].expand(b, t)
        x = self.embed(token_ids)
        infos = []
        for layer in self.layers:
            x, info = layer(x, positions)
            infos.append(info)
        return self.lm_head(self.norm(x)), infos

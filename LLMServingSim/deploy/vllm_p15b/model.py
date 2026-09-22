"""The P-15B module tree, one class per canonical layer name.

Semantics are the reference implementation's (``design/dsv4_ref/model.py``), so
the two can be compared parameter-for-parameter; the difference is packaging:
here each piece is its own class so the profiler's catalog can bind it by name
(``P15BQKVDown`` vs ``P15BQUp``, ``P15BQNorm`` vs ``P15BKVNorm`` -- a generic
``nn.Linear``/``RMSNorm`` would make those pairs indistinguishable).

Parameter shapes are deliberately identical to the reference; ``tests/
check_p15b_shapes.py`` pins the per-layer totals:

    full MLA layer   497,383,616
    CSA (ratio 4)    503,365,824
    HCA (ratio 128)  500,415,680

Step 2 of the plan swaps the linears for vLLM's parallel layers (so TP shards
per rank) and registers the model in ``ModelRegistry``; keeping construction in
``_linear()``/``_norm()`` is what makes that a local change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _linear(in_features: int, out_features: int) -> nn.Module:
    """The one place linears are built -- vLLM parallel layers drop in here."""
    return nn.Linear(in_features, out_features, bias=False)


def _embedding(num_embeddings: int, embedding_dim: int) -> nn.Module:
    return nn.Embedding(num_embeddings, embedding_dim)


# ---------------------------------------------------------------------------
# Norms: one class per canonical slot
# ---------------------------------------------------------------------------


class P15BRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class P15BInputNorm(P15BRMSNorm):
    """Pre-attention norm of a decoder layer."""


class P15BFfnNorm(P15BRMSNorm):
    """Pre-FFN norm of a decoder layer."""


class P15BQNorm(P15BRMSNorm):
    """Norm on the q LoRA latent, inside the attention block."""


class P15BKVNorm(P15BRMSNorm):
    """Norm on the kv latent, inside the attention block."""


class P15BFinalNorm(P15BRMSNorm):
    """Model-level final norm."""


def rope(x, positions, rope_dim, theta=10000.0):
    """RoPE on the trailing ``rope_dim`` channels of (B, T, H, D)."""
    if rope_dim <= 0:
        return x
    nope, part = x[..., :-rope_dim], x[..., -rope_dim:]
    half = rope_dim // 2
    dtype = part.dtype
    freqs = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float()[:, :, None] * freqs[None, None, :]
    cos = angles.cos()[:, :, None, :].to(dtype)
    sin = angles.sin()[:, :, None, :].to(dtype)
    a, b = part[..., :half], part[..., half:]
    rotated = torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return torch.cat([nope, rotated], dim=-1)


# ---------------------------------------------------------------------------
# Attention pieces
# ---------------------------------------------------------------------------


class P15BQKVDown(nn.Module):
    """hidden -> [q_lora | kv_latent(head_dim + rope)]."""

    def __init__(self, cfg):
        super().__init__()
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_width = cfg.head_dim + cfg.qk_rope_head_dim
        self.proj = _linear(cfg.hidden_size, self.q_lora_rank + self.kv_width)

    def forward(self, x):
        return self.proj(x).split([self.q_lora_rank, self.kv_width], dim=-1)


class P15BQUp(nn.Module):
    """q_lora -> n_heads x head_dim."""

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.proj = _linear(cfg.q_lora_rank, self.n_heads * self.head_dim)

    def forward(self, x):
        return self.proj(x).view(*x.shape[:-1], self.n_heads, self.head_dim)


class P15BOProjA(nn.Module):
    """One group of the output LoRA: heads/groups x head_dim -> o_lora."""

    def __init__(self, chunk: int, o_lora: int):
        super().__init__()
        self.proj = _linear(chunk, o_lora)

    def forward(self, x):
        return self.proj(x)


class P15BOProjB(nn.Module):
    """groups x o_lora -> hidden."""

    def __init__(self, groups: int, o_lora: int, hidden: int):
        super().__init__()
        self.proj = _linear(groups * o_lora, hidden)

    def forward(self, x):
        return self.proj(x)


class P15BCompressor(nn.Module):
    """Window-pool then project: ``hidden -> 2*coff*head_dim`` per `ratio` tokens.

    Pooling before projecting is not an optimisation -- projecting the whole
    window costs ``hidden * window * state``, which at HCA's window of 128 is
    134M parameters *per layer*.
    """

    def __init__(self, cfg, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.coff = cfg.coff(ratio)
        self.state = 2 * self.coff * cfg.head_dim
        self.window = self.coff * ratio
        self.norm = P15BKVNorm(self.state, cfg.rms_norm_eps)
        self.proj = _linear(cfg.hidden_size, self.state)

    def forward(self, x):
        # x: (B, T, H) -> ((B,N,coff*hd), (B,N,coff*hd), (B,N) state end positions)
        b, t, _ = x.shape
        n = t // self.ratio
        if n == 0:
            empty = x.new_zeros(b, 0, self.state // 2)
            return empty, empty, x.new_zeros(b, 0, dtype=torch.long)
        padded = F.pad(x, (0, 0, self.window - self.ratio, 0))
        pooled = padded.unfold(1, self.window, self.ratio).mean(dim=-1)
        state = self.norm(self.proj(pooled))
        kv_state, score_state = state.split(self.state // 2, dim=-1)
        ends = (torch.arange(n, device=x.device) + 1) * self.ratio - 1
        return kv_state, score_state, ends[None].expand(b, n)


class P15BIndexer(nn.Module):
    """Scores each compressed state for this query and keeps the top-k."""

    def __init__(self, cfg, state_width: int):
        super().__init__()
        self.head_dim = cfg.index_head_dim
        self.topk = cfg.index_topk
        self.q_proj = _linear(cfg.q_lora_rank, self.head_dim)
        self.k_proj = _linear(state_width, self.head_dim)

    def forward(self, q_latent, score_state):
        if score_state.shape[1] == 0:
            shape = (*q_latent.shape[:2], 0)
            return q_latent.new_zeros(shape, dtype=torch.long)
        k = min(self.topk, score_state.shape[1])
        q = self.q_proj(q_latent)
        kk = self.k_proj(score_state)
        sim = torch.einsum("btd,bnd->btn", q, kk) / math.sqrt(self.head_dim)
        return sim.topk(k, dim=-1).indices


class P15BKVStateProj(nn.Module):
    """Project a stored kv state (coff*head_dim) to head_dim."""

    def __init__(self, cfg, state_width: int):
        super().__init__()
        self.proj = _linear(state_width, cfg.head_dim)

    def forward(self, x):
        return self.proj(x)


class P15BSparseAttention(nn.Module):
    """The attention op: a banded window over raw latents plus the selected
    compressed states, in **one joint softmax** (normalising the two sets
    separately and adding them silently gives each set weight 1).
    """

    def __init__(self, cfg, ratio: int):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.ratio = ratio
        self.window = cfg.coff(ratio) * ratio if ratio else cfg.max_position_embeddings

    def forward(self, q, latents, state_values=None, state_pos=None, positions=None):
        b, t, n_heads, head_dim = q.shape
        logits = torch.einsum("bthd,bsd->bhts", q, latents) / math.sqrt(head_dim)
        causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        if self.ratio and self.window < t:
            causal &= torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1 - self.window)
        masked = logits.masked_fill(~causal[None, None], float("-inf"))
        values = latents[:, None].expand(b, t, t, head_dim)
        if state_values is not None and state_values.shape[1]:
            # state_values already carries one gathered row per query: (B,T,K,D)
            scores = torch.einsum("bthd,btkd->bhtk", q, state_values) / math.sqrt(head_dim)
            # state_pos is already (B,T,K); no extra axis here, the mask is
            # widened to heads only when it is applied to the (B,H,T,K) scores.
            valid = state_pos <= positions.unsqueeze(-1)
            masked = torch.cat([masked, scores.masked_fill(~valid[:, None], -1e30)], dim=-1)
            values = torch.cat([values, state_values], dim=2)
        weights = masked.softmax(-1)
        return torch.einsum("bhts,btsd->bthd", weights, values)


class P15BAttention(nn.Module):
    """MLA latent + sliding window + compressed states (ratio 0 -> window only)."""

    def __init__(self, cfg, ratio: int):
        super().__init__()
        self.cfg = cfg
        self.ratio = ratio
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.o_groups = cfg.o_groups
        self.qkv_down = P15BQKVDown(cfg)
        self.q_norm = P15BQNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.kv_norm = P15BKVNorm(cfg.head_dim + cfg.qk_rope_head_dim, cfg.rms_norm_eps)
        self.q_up = P15BQUp(cfg)
        chunk = cfg.num_attention_heads * cfg.head_dim // cfg.o_groups
        self.o_lora_a = nn.ModuleList(
            P15BOProjA(chunk, cfg.o_lora_rank) for _ in range(cfg.o_groups))
        self.o_lora_b = P15BOProjB(cfg.o_groups, cfg.o_lora_rank, cfg.hidden_size)
        if ratio:
            coff = cfg.coff(ratio)
            self.compressor = P15BCompressor(cfg, ratio)
            self.indexer = P15BIndexer(cfg, coff * cfg.head_dim)
            self.kv_state_proj = P15BKVStateProj(cfg, coff * cfg.head_dim)
        else:
            self.compressor = None
            self.indexer = None
            self.kv_state_proj = None
        self.attn = P15BSparseAttention(cfg, ratio)

    def forward(self, x, positions):
        b, t, _ = x.shape
        q_latent, kv = self.qkv_down(x)
        q = self.q_up(self.q_norm(q_latent))
        kv = self.kv_norm(kv).view(b, t, 1, self.head_dim + self.rope_dim)
        q = rope(q, positions, self.rope_dim)
        kv = rope(kv, positions, self.rope_dim)
        latents = kv[:, :, 0, :self.head_dim]

        state_values = state_pos = None
        if self.compressor is not None and t >= self.ratio:
            kv_state, score_state, ends = self.compressor(x)
            idx = self.indexer(q_latent, score_state)
            projected = self.kv_state_proj(kv_state)
            state_values = torch.gather(
                projected[:, None].expand(b, t, -1, -1), 2,
                idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            state_pos = torch.gather(ends[:, None].expand(b, t, -1), 2, idx)

        out = self.attn(q, latents, state_values, state_pos, positions)
        out = out.reshape(b, t, self.n_heads * self.head_dim)
        chunks = torch.chunk(out, self.o_groups, dim=-1)
        out = torch.cat([proj(c) for proj, c in zip(self.o_lora_a, chunks)], dim=-1)
        return self.o_lora_b(out)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class P15BMoE(nn.Module):
    """Top-k routed MoE with SwiGLU clamping; no shared expert, no hash layers.

    The routing trick the official model uses on its first layers is omitted --
    it is a training device, not part of the KV design.
    """

    swiglu_limit = 10.0

    def __init__(self, cfg):
        super().__init__()
        h, e, mi = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
        self.h, self.e, self.mi = h, e, mi
        self.topk = cfg.num_experts_per_tok
        self.gate = _linear(h, e)
        self.up = nn.Parameter(torch.randn(e, h, 2 * mi) * 0.02)
        self.down = nn.Parameter(torch.randn(e, mi, h) * 0.02)

    def forward(self, x):
        b, t, h = x.shape
        probs = self.gate(x).softmax(-1)
        weights, idx = probs.topk(self.topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        flat = x.reshape(-1, h)
        weights = weights.reshape(-1, self.topk)
        idx = idx.reshape(-1, self.topk)
        out = torch.zeros_like(flat)
        for expert in range(self.e):
            assigned = idx == expert
            rows = assigned.any(-1)
            if not rows.any():
                continue
            tokens = torch.nonzero(rows, as_tuple=False).squeeze(-1)
            slot = assigned[rows].float().argmax(-1)
            gate = weights[rows].gather(1, slot[:, None]).squeeze(-1)
            gu = flat[tokens] @ self.up[expert]
            gate_branch, up_branch = gu.chunk(2, dim=-1)
            act = F.silu(gate_branch.clamp(max=self.swiglu_limit)) * up_branch
            out.index_add_(0, tokens, (act @ self.down[expert]) * gate[:, None])
        return out.view(b, t, h)


# ---------------------------------------------------------------------------
# Decoder / model
# ---------------------------------------------------------------------------


class P15BDecoderLayer(nn.Module):
    def __init__(self, cfg, ratio: int):
        super().__init__()
        self.input_layernorm = P15BInputNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = P15BAttention(cfg, ratio)
        self.post_attention_layernorm = P15BFfnNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = P15BMoE(cfg)

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class P15BForCausalLM(nn.Module):
    """Embedding + N decoder layers + norm (+ head).  No mHC anywhere.

    ``with_lm_head=False`` drops the ``nn.Linear`` head and returns hidden
    states: that is the shape vLLM wants, since it supplies its own
    ``ParallelLMHead`` (TP-sharded) plus ``LogitsProcessor``.
    """

    def __init__(self, cfg, with_lm_head: bool = True):
        super().__init__()
        self.cfg = cfg
        self.with_lm_head = with_lm_head
        self.embed_tokens = _embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            P15BDecoderLayer(cfg, ratio) for ratio in cfg.compress_ratios)
        self.norm = P15BFinalNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = _linear(cfg.hidden_size, cfg.vocab_size) if with_lm_head else None

    def forward(self, input_ids, positions=None, inputs_embeds=None):
        """One code path for the parity gate and for vLLM: the latter passes
        ``inputs_embeds`` alongside ``input_ids``, and takes precedence."""
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
        b, t = x.shape[0], x.shape[1]
        if positions is None:
            positions = torch.arange(t, device=input_ids.device)[None].expand(b, t)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.norm(x)
        return self.lm_head(x) if self.lm_head is not None else x

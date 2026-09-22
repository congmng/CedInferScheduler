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
import os

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


_METADATA_REPORTED: set[str] = set()


def _report_forward_metadata(prefix: str) -> None:
    """Once per layer, report what vLLM handed the forward pass.

    Whether the engine files attention metadata under our layer's name decides
    how the attention can read the paged cache (path B's remaining piece), and
    that is not visible from the outside: the metadata arrives through a
    context variable set by the runner.  Gated on ``P15B_DEBUG_META`` so it
    costs nothing in a real run.
    """
    if not os.environ.get("P15B_DEBUG_META") or prefix in _METADATA_REPORTED:
        return
    _METADATA_REPORTED.add(prefix)
    try:
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
    except Exception as exc:  # pragma: no cover - probe only
        print(f"[p15b.meta] {prefix}: no forward context ({exc})", flush=True)
        return
    metadata = getattr(context, "attn_metadata", None)
    if isinstance(metadata, dict):
        keys = list(metadata)
        print(f"[p15b.meta] {prefix}: attn_metadata keys={keys[:4]} "
              f"(ours present: {prefix in metadata})", flush=True)
        entry = metadata.get(prefix)
        if entry is not None:
            fields = [f for f in ("block_table", "slot_mapping", "block_size")
                      if hasattr(entry, f)]
            print(f"[p15b.meta]   {type(entry).__name__} fields={fields}", flush=True)
    else:
        print(f"[p15b.meta] {prefix}: attn_metadata={type(metadata).__name__}", flush=True)


class P15BRotary(nn.Module):
    """Module wrapper around :func:`rope`.

    RoPE here is a function, but the profiler can only bind *modules* to
    canonical layer names -- without this, rotary would be charged nowhere and
    the per-layer total would silently under-count.  It carries no parameters.
    """

    def __init__(self, rope_dim: int, theta: float = 10000.0):
        super().__init__()
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, x, positions):
        return rope(x, positions, self.rope_dim, self.theta)


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


# The profiler's catalog matches on (class name, ancestor class name) and
# rejects two canonical names resolving to the same pair.  P-15B's compressed
# pieces differ in *shape* between CSA (ratio 4) and HCA (ratio 128), so each
# gets its own class; that is what lets one bundle carry both, which the
# simulator's per-block-type pipelines need (docs/P15B-profile路径B实现计划.md §4).
class P15BCompressorCSA(P15BCompressor):
    """CSA compressor: state 2*2*head_dim, one state per 4 tokens."""


class P15BCompressorHCA(P15BCompressor):
    """HCA compressor: state 2*1*head_dim, one state per 128 tokens."""


class P15BIndexerCSA(P15BIndexer):
    """Indexer scoring CSA states (key width coff*head_dim = 2*head_dim)."""


class P15BIndexerHCA(P15BIndexer):
    """Indexer scoring HCA states (key width head_dim)."""


class P15BKVStateProjCSA(P15BKVStateProj):
    """CSA state -> head_dim."""


class P15BKVStateProjHCA(P15BKVStateProj):
    """HCA state -> head_dim."""


class P15BSparseAttention(nn.Module):
    """The attention op: a banded window over raw latents plus the selected
    compressed states, in **one joint softmax** (normalising the two sets
    separately and adding them silently gives each set weight 1).

    Deliberately *not* the KV spec holder.  vLLM collects one cacheable module
    per layer from ``static_forward_context``, and ``kv_cache.P15BCache`` owns
    that (the same split the official compressor uses); this class only
    computes.  Declaring a spec here as well would put two cacheable modules
    on every layer.
    """

    def __init__(self, cfg, ratio: int, prefix: str = ""):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.ratio = ratio
        self.window = cfg.coff(ratio) * ratio if ratio else cfg.max_position_embeddings
        #: Namespace of this layer's KV cache holder; the key its attention
        #: metadata is filed under (see ``kv_cache.P15BCache``).
        self.prefix = prefix

    def forward(self, q, latents, state_values=None, state_pos=None, positions=None,
                block: int = 128):
        """Blocked sparse attention, in one joint softmax.

        Written as a loop over query blocks rather than a dense ``(B,H,T,T)``
        score matrix: at the profiler's kv lengths a dense tensor is tens of
        gigabytes (it OOM'd at 18 GiB on the first attempt) and, more to the
        point, it is not the op we are designing.  Each query attends to its
        own window (or the whole prefix, for ratio 0) plus its own top-k
        states, so peak memory is ``block x (window | context) x heads``.

        Numerically this is the same computation as the dense form -- the
        chunking only changes summation order, which is why the parity gate
        allows a tolerance rather than bit-exactness.
        """
        b, t, n_heads, head_dim = q.shape
        if positions is None:
            positions = torch.arange(t, device=q.device)[None].expand(b, t)
        _report_forward_metadata(self.prefix)
        scale = 1.0 / math.sqrt(head_dim)
        windowed = bool(self.ratio) and self.window < t
        offsets = torch.arange(self.window, device=q.device) - (self.window - 1)
        out = torch.empty_like(q)
        for start in range(0, t, block):
            end = min(start + block, t)
            q_block = q[:, start:end]
            pos_block = positions[:, start:end]
            q_index = torch.arange(start, end, device=q.device)
            if windowed:
                # Window keys are the `window` rows ending at each query.  They
                # are indexed by *sequence offset*, which equals vLLM's
                # `positions` for the single-sequence shots we profile; a
                # chunked-prefill run starting mid-sequence would need the
                # positions tensor here instead.
                raw = q_index[:, None] + offsets[None, :]            # (qb,w)
                keep = raw >= 0
                keys = latents[:, raw.clamp(min=0)]                  # (B,qb,w,D)
            else:
                raw = torch.arange(t, device=q.device)[None, :].expand(end - start, t)
                keep = raw <= q_index[:, None]                       # (qb,t)
                keys = latents[:, :t].unsqueeze(1).expand(b, end - start, t, head_dim)
            scores = torch.einsum("bqhd,bqkd->bhqk", q_block, keys) * scale
            scores = scores.masked_fill(~keep[None, None], float("-inf"))
            values = keys
            if state_values is not None and state_values.shape[1]:
                sel = state_values[:, start:end]                     # (B,qb,K,D)
                sel_pos = state_pos[:, start:end]                    # (B,qb,K)
                sel_scores = torch.einsum("bqhd,bqkd->bhqk", q_block, sel) * scale
                ok = sel_pos <= pos_block[:, :, None]
                scores = torch.cat(
                    [scores, sel_scores.masked_fill(~ok[:, None], -1e30)], dim=-1)
                values = torch.cat([values, sel], dim=2)
            out[:, start:end] = torch.einsum(
                "bhqk,bqkd->bqhd", scores.softmax(-1), values)
        return out


class P15BAttention(nn.Module):
    """MLA latent + sliding window + compressed states (ratio 0 -> window only)."""

    def __init__(self, cfg, ratio: int, prefix: str = "", with_kv_cache: bool = False,
                 dtype: torch.dtype = torch.bfloat16):
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
        self.rotary_emb = P15BRotary(cfg.qk_rope_head_dim, cfg.rope_theta)
        chunk = cfg.num_attention_heads * cfg.head_dim // cfg.o_groups
        self.o_lora_a = nn.ModuleList(
            P15BOProjA(chunk, cfg.o_lora_rank) for _ in range(cfg.o_groups))
        self.o_lora_b = P15BOProjB(cfg.o_groups, cfg.o_lora_rank, cfg.hidden_size)
        if ratio:
            coff = cfg.coff(ratio)
            # Per-ratio subclasses so the profiler's catalog can name them
            # separately (see the note above the subclass definitions).
            csa = ratio == 4
            self.compressor = (P15BCompressorCSA if csa else P15BCompressorHCA)(cfg, ratio)
            self.indexer = (P15BIndexerCSA if csa else P15BIndexerHCA)(
                cfg, coff * cfg.head_dim)
            self.kv_state_proj = (P15BKVStateProjCSA if csa
                                  else P15BKVStateProjHCA)(cfg, coff * cfg.head_dim)
        else:
            self.compressor = None
            self.indexer = None
            self.kv_state_proj = None
        self.attn = P15BSparseAttention(cfg, ratio, prefix=f"{prefix}.attn" if prefix else "")
        # Only under vLLM: declaring a spec needs a live vllm_config, and the
        # standalone gates (parity, parameter counts) run without one.
        self.cache = None
        if with_kv_cache:
            from .kv_cache import P15BCache

            self.cache = P15BCache(cfg, ratio, f"{prefix}.kv_cache", dtype)

    def forward(self, x, positions):
        b, t, _ = x.shape
        q_latent, kv = self.qkv_down(x)
        q = self.q_up(self.q_norm(q_latent))
        kv = self.kv_norm(kv).view(b, t, 1, self.head_dim + self.rope_dim)
        q = self.rotary_emb(q, positions)
        kv = self.rotary_emb(kv, positions)
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

    Two backends, one class name (so the profiler catalog can bind it either
    way):

    * **standalone** (``vllm_config is None``) -- plain torch, the form the
      parameter/parity gates build and compare against ``design/dsv4_ref``;
    * **vLLM** -- vLLM's ``FusedMoE`` behind a ``ReplicatedLinear`` gate, the
      same arrangement ``Qwen3MoeSparseMoeBlock`` uses.

    The second is not optional for profiling: the profiler's moe category
    drives the experts through ``collective_rpc`` and finds the layer with
    ``isinstance(m, FusedMoE)`` ("Expected exactly one FusedMoE layer in the
    test model, got 0").  It is also the only way this cost is measured as a
    kernel rather than as our own python loop.

    Weight layout differs between the two: the reference stores ``up``
    ``(E, H, 2I)`` / ``down`` ``(E, I, H)``, FusedMoE stores ``w13``
    ``(E, 2I, H)`` / ``w2`` ``(E, H, I)`` -- transposed.  Parameter *counts*
    are identical, so the shape gate is unaffected.
    """

    swiglu_limit = 10.0

    def __init__(self, cfg, vllm_config=None, prefix: str = "p15b", quant_config=None):
        super().__init__()
        h, e, mi = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size
        self.h, self.e, self.mi = h, e, mi
        self.topk = cfg.num_experts_per_tok
        self.fused = None
        if vllm_config is None:
            self.gate = _linear(h, e)
            self.up = nn.Parameter(torch.randn(e, h, 2 * mi) * 0.02)
            self.down = nn.Parameter(torch.randn(e, mi, h) * 0.02)
        else:
            from vllm.model_executor.layers.fused_moe import FusedMoEFactory
            from vllm.model_executor.layers.linear import ReplicatedLinear

            self.gate = ReplicatedLinear(h, e, bias=False, quant_config=quant_config,
                                         prefix=f"{prefix}.gate")
            self.fused = FusedMoEFactory(
                shared_experts=None, gate=self.gate, num_experts=e, top_k=self.topk,
                hidden_size=h, intermediate_size=mi,
                # The reference renormalises top-k weights after selection.
                renormalize=True, quant_config=quant_config,
                prefix=f"{prefix}.experts", enable_eplb=False,
                num_redundant_experts=0, is_sequence_parallel=False,
                is_fused_checkpoint_transposed=False)

    def forward(self, x):
        if self.fused is not None:
            # FusedMoE takes 2D (tokens, hidden); the tree works in (B, T, H).
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            out = self.fused(hidden_states=flat, router_logits=flat)
            return out.reshape(shape)
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
    def __init__(self, cfg, ratio: int, prefix: str = "", with_kv_cache: bool = False,
                 dtype: torch.dtype = torch.bfloat16, vllm_config=None,
                 quant_config=None):
        super().__init__()
        self.input_layernorm = P15BInputNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = P15BAttention(cfg, ratio, prefix=prefix,
                                       with_kv_cache=with_kv_cache, dtype=dtype)
        self.post_attention_layernorm = P15BFfnNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = P15BMoE(cfg, vllm_config=vllm_config,
                           prefix=f"{prefix}.mlp" if prefix else "p15b.mlp",
                           quant_config=quant_config)

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        return x + self.mlp(self.post_attention_layernorm(x))


class P15BForCausalLM(nn.Module):
    """Embedding + N decoder layers + norm (+ head).  No mHC anywhere.

    ``with_lm_head=False`` drops the ``nn.Linear`` head and returns hidden
    states: that is the shape vLLM wants, since it supplies its own
    ``ParallelLMHead`` (TP-sharded) plus ``LogitsProcessor``.
    """

    def __init__(self, cfg, with_lm_head: bool = True, with_kv_cache: bool = False,
                 prefix: str = "p15b", dtype: torch.dtype = torch.bfloat16,
                 vllm_config=None, quant_config=None):
        super().__init__()
        self.cfg = cfg
        self.with_lm_head = with_lm_head
        self.embed_tokens = _embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            P15BDecoderLayer(cfg, ratio, prefix=f"{prefix}.layers.{index}",
                             with_kv_cache=with_kv_cache, dtype=dtype,
                             vllm_config=vllm_config, quant_config=quant_config)
            for index, ratio in enumerate(cfg.compress_ratios))
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

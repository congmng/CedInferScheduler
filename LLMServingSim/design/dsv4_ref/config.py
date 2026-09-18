"""Configurations and the two estimators the design has to satisfy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def default_ratios(n_layer: int) -> list[int]:
    """The released family's pattern: 0,0 then alternating 4/128, then 0.

    V4-Flash (43 layers) is ``[0, 0] + [4,128]*20 + [0]`` and V4-Pro (61) is the
    same shape with one layer at ratio 0; only the first two and the last layer
    keep a full MLA cache, everything between alternates CSA (4) and HCA (128).
    """
    if n_layer < 4:
        return [0] * n_layer
    middle = n_layer - 3
    pattern = [4, 128] * ((middle + 1) // 2)
    return [0, 0] + pattern[:middle] + [0]


@dataclass
class DSV4RefConfig:
    # Shape
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    head_dim: int = 512              # official MLA head dim
    qk_rope_head_dim: int = 64       # official
    q_lora_rank: int = 256
    o_lora_rank: int = 256
    o_groups: int = 4
    # Sparse selector (indexer)
    index_head_dim: int = 128
    index_n_heads: int = 16
    index_topk: int = 256
    # Compression
    compress_ratios: list[int] = field(default_factory=list)
    # FFN: a dense FFN keeps the reference readable; MoE is a separate knob
    ffn_hidden: int = 0              # 0 -> is_moe
    n_routed_experts: int = 32
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 512
    n_shared_experts: int = 1
    # Bookkeeping
    vocab_size: int = 65536
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rng_seed: int = 0

    def __post_init__(self):
        if not self.compress_ratios:
            self.compress_ratios = default_ratios(self.num_hidden_layers)
        assert len(self.compress_ratios) == self.num_hidden_layers, (
            f"compress_ratios has {len(self.compress_ratios)} entries for "
            f"{self.num_hidden_layers} layers"
        )
        assert all(r in (0, 4, 128) for r in self.compress_ratios)

    # ------------------------------------------------------------------
    # Estimators -- these are what the design has to hit
    # ------------------------------------------------------------------

    @property
    def is_moe(self) -> bool:
        return self.ffn_hidden == 0

    def coff(self, ratio: int) -> int:
        return 2 if ratio == 4 else 1

    def layer_bytes_per_token(self, ratio: int, kv_bytes: int = 2) -> float:
        """KV bytes one token occupies in this layer.

        Full layer (ratio 0): a single latent of ``head_dim + rope`` values.
        Compressed layer: ``2 * coff * head_dim`` values every ``ratio`` tokens
        (``kv_state`` + ``score_state``).
        """
        if ratio == 0:
            return (self.head_dim + self.qk_rope_head_dim) * kv_bytes
        return 2 * self.coff(ratio) * self.head_dim * kv_bytes / ratio

    def kv_bytes_per_token(self, kv_bytes: int = 2) -> float:
        return sum(self.layer_bytes_per_token(r, kv_bytes)
                   for r in self.compress_ratios)

    def layer_params(self, ratio: int) -> dict[str, int]:
        h, hd, rope = self.hidden_size, self.head_dim, self.qk_rope_head_dim
        ql, ol, ng = self.q_lora_rank, self.o_lora_rank, self.o_groups
        nh = self.num_attention_heads
        out = {
            # MLA: down-projection to (q_latent | kv_latent), q up-projection,
            # q/k norm weights, grouped output LoRA (ng x [q_dim -> ol] + [ol -> h])
            # The kv latent is stored as nope(512) + rope(64) per layer, which is
            # the 576 B/token the engine reported for a full layer.
            "qkv_down": h * (ql + hd + rope),
            "q_up": ql * nh * hd,
            "qk_norm": hd + hd,
            "o_lora_a": ng * (nh * hd // ng) * ol,
            "o_lora_b": ng * ol * h,
            # indexer: query projection + key projection
            "indexer": ql * (self.index_n_heads * self.index_head_dim)
                       + h * (self.index_n_heads * self.index_head_dim),
        }
        if ratio:
            coff = self.coff(ratio)
            out["compressor"] = h * (2 * coff * hd)   # window-pooled projection
        else:
            out["compressor"] = 0
        if self.is_moe:
            per_expert = 3 * h * self.moe_intermediate_size
            out["moe"] = (h * self.n_routed_experts
                          + self.n_routed_experts * per_expert
                          + self.n_shared_experts * per_expert)
        else:
            out["moe"] = 2 * h * self.ffn_hidden
        return out

    def activated_params_per_layer(self, ratio: int) -> int:
        p = self.layer_params(ratio)
        attn = (p["qkv_down"] + p["q_up"] + p["qk_norm"] + p["o_lora_a"]
                + p["o_lora_b"] + p["indexer"] + p["compressor"])
        if self.is_moe:
            per_expert = 3 * self.hidden_size * self.moe_intermediate_size
            attn += (self.hidden_size * self.n_routed_experts
                     + self.num_experts_per_tok * per_expert
                     + self.n_shared_experts * per_expert)
        else:
            attn += p["moe"]
        return attn

    def summary(self) -> dict:
        total = sum(sum(self.layer_params(r).values())
                    for r in self.compress_ratios)
        total += 2 * self.vocab_size * self.hidden_size       # embed + lm_head
        activated = sum(self.activated_params_per_layer(r)
                        for r in self.compress_ratios)
        return {
            "layers": self.num_hidden_layers,
            "hidden": self.hidden_size,
            "ratio_mix": {r: self.compress_ratios.count(r) for r in set(self.compress_ratios)},
            "total_params_B": round(total / 1e9, 3),
            "activated_params_B": round(activated / 1e9, 3),
            "kv_kb_per_token_bf16": round(self.kv_bytes_per_token(2) / 1024, 2),
            "kv_kb_per_token_fp8": round(self.kv_bytes_per_token(1) / 1024, 2),
        }


# ---------------------------------------------------------------------------
# The two configurations we talk about
# ---------------------------------------------------------------------------

def _small() -> DSV4RefConfig:
    """Smoke-scale: this is the one we actually run end to end."""
    return DSV4RefConfig(
        hidden_size=1024, num_hidden_layers=12, num_attention_heads=16,
        head_dim=512, qk_rope_head_dim=64, q_lora_rank=256, o_lora_rank=256,
        o_groups=4, index_head_dim=128, index_n_heads=16, index_topk=256,
        n_routed_experts=32, num_experts_per_tok=2, moe_intermediate_size=512,
    )


def _p15b() -> DSV4RefConfig:
    """The target design: 28 layers, measured 14.9B total / 2.15B activated."""
    return DSV4RefConfig(
        hidden_size=2560, num_hidden_layers=28, num_attention_heads=20,
        head_dim=512, qk_rope_head_dim=64, q_lora_rank=640, o_lora_rank=512,
        o_groups=8, index_head_dim=128, index_n_heads=20, index_topk=512,
        n_routed_experts=48, num_experts_per_tok=3, moe_intermediate_size=1280,
        vocab_size=65536,
    )


REF_CONFIGS: dict[str, DSV4RefConfig] = {"small": _small(), "p15b": _p15b()}

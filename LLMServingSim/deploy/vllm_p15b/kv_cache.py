"""Per-layer KV cache for P-15B: a spec-declaring module, not an attention impl.

The profiler's attention category sweeps ``(prefill_chunk, kv_prefill,
n_decode, kv_decode)`` and expects the layer's work to follow the cache those
lengths imply.  Our sparse attention is our own op, so reusing vLLM's
attention backends wholesale is wrong -- but we do not need to: the official
DeepSeek-V4 compressor solves the same problem by declaring a spec and binding
the allocated tensor itself (``CompressorStateCache`` in
``vllm/models/deepseek_v4/compressor.py``, whose backend is a handful of
static methods and no attention impl at all).  This mirrors that shape.

Two things we get to fix by construction, having measured the official path's
mistake: a compressed layer is declared with ``block_size = tokens_per_state =
ratio``, so one page holds exactly one state and is charged to `ratio` tokens.
The official ``CompressorStateCache`` leaves ``tokens_per_state`` at 1, which
is what made its pages cost 4-8 KB/token (docs/DSV4_10-30B跨卡设计.md 3.1.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport,
                                       AttentionMetadata,
                                       AttentionMetadataBuilder,
                                       CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec


@dataclass
class P15BCacheMetadata(AttentionMetadata):
    """What our attention needs to address the cache: block table + slots."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int


class P15BCacheMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = self.kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> P15BCacheMetadata:
        return P15BCacheMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
        )


class P15BCacheBackend(AttentionBackend):
    """Cache-only backend: no attention impl, the layer reads the cache itself.

    ``get_impl_cls`` is deliberately absent -- the same reason the official
    compressor's backend omits it: nothing ever runs a vLLM attention kernel
    for this layer.
    """

    @staticmethod
    def get_name() -> str:
        return "P15BCacheBackend"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_builder_cls() -> type[P15BCacheMetadataBuilder]:
        return P15BCacheMetadataBuilder


class P15BCache(torch.nn.Module, AttentionLayerBase):
    """Declares the layer's KV spec and receives the allocated tensor.

    Layout, per layer type (``tokens_per_state`` is the whole point):

    * ``ratio == 0`` -- one latent of ``head_dim + rope`` per token;
    * ``ratio > 0``   -- one state of ``2*coff*head_dim`` every `ratio` tokens.
    """

    def __init__(self, cfg, ratio: int, prefix: str, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.ratio = ratio
        self.coff = type(cfg).coff(ratio) if ratio else 1
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # -- AttentionLayerBase ------------------------------------------------
    def get_attn_backend(self) -> type[AttentionBackend]:
        return P15BCacheBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if self.ratio == 0:
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=self.cfg.head_dim + self.cfg.qk_rope_head_dim,
                dtype=self.dtype,
                tokens_per_state=1,
            )
        return MLAAttentionSpec(
            # One state per page, charged to `ratio` tokens -- see the module
            # docstring; this is the accounting the official path gets wrong.
            block_size=self.ratio,
            num_kv_heads=1,
            head_size=2 * self.coff * self.cfg.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.ratio,
        )

    def forward(self): ...

"""A Transformers ``PretrainedConfig`` for ``model_type: p15b``.

Without this, vLLM's ``ModelConfig`` rejects the checkpoint before it ever
looks at ``architectures``: "Transformers does not recognize this
architecture".  Registering the type is the documented extension point and
keeps one config file serving both consumers -- the profiler resolves
``profiler/models/<model_type>.yaml`` from it, vLLM resolves the model class
from ``architectures[0]``.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class P15BHFConfig(PretrainedConfig):
    model_type = "p15b"

    def __init__(
        self,
        hidden_size: int = 2560,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 640,
        o_lora_rank: int = 512,
        o_groups: int = 8,
        index_head_dim: int = 128,
        index_topk: int = 512,
        intermediate_size: int = 1280,
        moe_intermediate_size: int = 1280,
        n_routed_experts: int = 48,
        num_experts_per_tok: int = 3,
        n_shared_experts: int = 0,
        vocab_size: int = 65536,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        compress_ratios: list[int] | None = None,
        layers_block_type: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.compress_ratios = list(compress_ratios or [])
        self.layers_block_type = list(layers_block_type or [])


def register() -> None:
    from transformers import AutoConfig

    AutoConfig.register("p15b", P15BHFConfig, exist_ok=True)

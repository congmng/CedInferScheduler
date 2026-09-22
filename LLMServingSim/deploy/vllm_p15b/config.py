"""P-15B hyper-parameters, from an HF config dict or the reference dataclass.

Kept separate from ``model.py`` so the module tree has no opinion about where
its numbers came from: the profiler hands us the JSON under ``configs/model/``,
and the shape gate hands us the same numbers via ``design/dsv4_ref``.
"""

from __future__ import annotations

from dataclasses import fields


class P15BConfig:
    """Attribute view over a plain HF-style config mapping."""

    #: Every field the module tree reads.  Anything missing raises here rather
    #: than silently defaulting into a differently-shaped model.
    FIELDS = (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "head_dim",
        "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups",
        "index_head_dim", "index_topk", "n_routed_experts",
        "num_experts_per_tok", "moe_intermediate_size", "vocab_size",
        "rms_norm_eps", "max_position_embeddings", "compress_ratios",
        "rope_theta",
    )

    def __init__(self, **values):
        missing = [name for name in self.FIELDS if name not in values]
        if missing:
            raise ValueError(f"P15BConfig is missing {missing}")
        for name in self.FIELDS:
            setattr(self, name, values[name])
        self.compress_ratios = [int(r) for r in self.compress_ratios]
        if len(self.compress_ratios) != int(self.num_hidden_layers):
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries for "
                f"{self.num_hidden_layers} layers")
        #: Expert width as the *engine's* TP sharding sees it.  The profiler
        #: emulates TP by dividing ``intermediate_size`` (its SHARD_FIELDS) and
        #: never touches ``moe_intermediate_size``, so the MoE has to read this
        #: one or a tp=2 profile would measure unsharded experts.  The shipped
        #: configs set both to 1280, so tp=1 numbers are unchanged.
        self.intermediate_size = int(values.get("intermediate_size")
                                     or self.moe_intermediate_size)

    @classmethod
    def from_hf(cls, hf) -> "P15BConfig":
        """From a HF config mapping (the dict vLLM reads from config.json)."""
        get = hf.get if hasattr(hf, "get") else (lambda k, d=None: getattr(hf, k, d))
        values = {name: get(name) for name in cls.FIELDS}
        values["intermediate_size"] = get("intermediate_size")
        if values.get("compress_ratios") is None:
            # vLLM hands back a config object that may not carry our extra key;
            # fall back to the alternating pattern the design specifies.
            values["compress_ratios"] = _default_ratios(int(values["num_hidden_layers"]))
        if values.get("rope_theta") is None:
            values["rope_theta"] = 10000.0
        for name in ("rms_norm_eps", "max_position_embeddings", "vocab_size"):
            if values.get(name) is None:
                raise ValueError(f"HF config is missing {name}")
        return cls(**values)

    @classmethod
    def from_reference(cls, ref) -> "P15BConfig":
        """From ``design.dsv4_ref.config.DSV4RefConfig`` (the shape gate)."""
        names = {f.name for f in fields(ref)}
        values = {name: getattr(ref, name) for name in cls.FIELDS}
        missing = [name for name in cls.FIELDS if name not in names]
        if missing:
            raise ValueError(f"reference config lacks {missing}")
        return cls(**values)

    @staticmethod
    def coff(ratio: int) -> int:
        """Overlapping compressor windows: 2 for ratio 4, else 1."""
        return 2 if ratio == 4 else 1


def _default_ratios(n_layer: int) -> list[int]:
    """`[0, 0] + [4, 128] * k + [0]` -- the released family's pattern."""
    pattern, middle = [0, 0], n_layer - 3
    pattern += ([4, 128] * ((middle + 1) // 2))[:middle]
    return pattern + [0]

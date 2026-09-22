"""vLLM entry point for P-15B: the glue that makes the tree engine-loadable.

Deliberately thin.  The module tree in ``model.py`` is plain torch and is
pinned to ``design/dsv4_ref`` by ``tests/check_p15b_shapes.py``; everything
here exists only because vLLM needs it:

* ``vllm_config`` / ``prefix`` constructor,
* a TP-sharded ``ParallelLMHead`` + ``LogitsProcessor`` instead of our
  ``nn.Linear`` head,
* ``compute_logits`` / ``load_weights`` hooks.

Profiling never loads a checkpoint (``load_format: dummy``), so
``load_weights`` only has to exist.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from .config import P15BConfig
from .model import P15BForCausalLM as P15BTree


class P15BVllmForCausalLM(nn.Module):
    """Registered under the ``P15BForCausalLM`` architecture string."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf = vllm_config.model_config.hf_config
        self.cfg = P15BConfig.from_hf(hf)
        self.tree = P15BTree(self.cfg, with_lm_head=False, with_kv_cache=True,
                             prefix=prefix or "p15b")
        self.lm_head = ParallelLMHead(self.cfg.vocab_size, self.cfg.hidden_size)
        self.logits_processor = LogitsProcessor(self.cfg.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_ignored,
    ):
        """Adapt vLLM's flat layout to the tree's ``(B, T, H)``.

        The V1 runner hands the model flattened tensors -- ``input_ids`` of
        shape ``(num_tokens,)``, not ``(B, T)`` -- and expects flattened hidden
        states back, where ``num_tokens`` spans every sequence in the batch.
        The tree is written per-sequence, so this reshapes to one sequence.

        That is exact for the profiler's shots (one sequence each) and is the
        assumption to revisit before this model serves real batching: with
        several sequences concatenated, attention must not cross the
        boundaries, which needs a segment id rather than a reshape.
        """
        flat = input_ids is not None and input_ids.dim() == 1
        if flat:
            tokens = input_ids.shape[0]
            input_ids = input_ids.reshape(1, tokens)
            positions = positions.reshape(1, tokens)
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.reshape(1, tokens, -1)
        hidden = self.tree(input_ids, positions, inputs_embeds=inputs_embeds)
        return hidden.reshape(-1, hidden.shape[-1]) if flat else hidden

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM's ``VllmModel`` protocol requires this; the embedding lives in
        the tree rather than on the wrapper."""
        return self.tree.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ``load_format: dummy`` never calls this; a real checkpoint would map
        # onto the tree's names, which is step 2's follow-up.
        return set()


def register() -> None:
    """Bind the architecture string to this class (called from sitecustomize)."""
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model("P15BForCausalLM", P15BVllmForCausalLM)

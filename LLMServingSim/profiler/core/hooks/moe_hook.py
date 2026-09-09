"""MoE forced-routing hook.

To profile the MoE block cleanly across the (tokens, activated_experts)
grid we need to control which experts receive tokens — relying on the
(dummy-weighted) learned gate would give unpredictable activation
patterns and poor grid coverage.

This module provides:

* ``ExpertRoute.forge``: build the ``(topk_weights, topk_ids)`` tensors
  that would force a given number of experts to be activated over a
  given number of tokens.

* ``force_moe_routing``: a context manager that monkey-patches
  ``FusedMoE.forward_native`` for the duration of the block so that
  ``self.router.select_experts`` returns our forged tensors instead
  of whatever the actual learned gate produces. The patch is
  reverted on exit.

The patch targets (``FusedMoE.forward_native`` and
``self.router.select_experts`` / ``self.router._compute_routing``)
are internal vLLM APIs. A vLLM version bump may require updating the
monkey-patch to match renamed or restructured symbols.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Iterator

import torch


@dataclass
class ExpertRoute:
    """Precomputed tensors that pin routing to a specific expert set.

    Attributes:
        layer_name: Identifier of the FusedMoE layer this route targets.
            The runtime patch only applies when the currently-forwarding
            layer's ``layer_name`` matches.
        weights: Tensor of shape (num_tokens, top_k); each row is a
            uniform ``1/top_k`` distribution over the chosen experts.
        ids: Integer tensor of shape (num_tokens, top_k) naming the
            experts each token is routed to.
    """

    layer_name: str
    weights: torch.Tensor
    ids: torch.Tensor

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def forge(
        cls,
        layer,
        num_tokens: int,
        activated_experts: int,
    ) -> "ExpertRoute":
        """Allocate ``weights`` and ``ids`` for a single FusedMoE layer.

        Args:
            layer: The ``FusedMoE`` instance we'll patch. Used to read
                ``top_k`` and the expected indices dtype.
            num_tokens: Number of tokens being routed this call.
            activated_experts: How many distinct experts should receive
                at least one token. Must satisfy
                ``top_k <= activated_experts <= num_tokens * top_k``.
        """
        top_k = getattr(layer, "top_k", None)
        if top_k is None:
            top_k = layer.moe_config.experts_per_token
        if activated_experts < top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) must be >= "
                f"top_k ({top_k})"
            )
        if activated_experts > num_tokens * top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) cannot exceed "
                f"num_tokens*top_k ({num_tokens * top_k})"
            )

        ids_rows = _cycle_expert_ids(num_tokens, top_k, activated_experts)

        # vLLM's router cares about the dtype of topk_ids: some kernels
        # expect int32, others a specific dtype reported by the router.
        get_indices_type = getattr(layer.router, "_get_indices_type", None)
        indices_dtype = get_indices_type() if get_indices_type else None
        device = next(layer.parameters()).device

        ids = torch.tensor(
            ids_rows,
            device=device,
            dtype=torch.int32 if indices_dtype is None else indices_dtype,
        )
        # Shape sanity check — the kernel will complain later with a
        # cryptic message if this is wrong, so we prefer to fail early.
        expected_shape = (num_tokens, top_k)
        if tuple(ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch: expected {expected_shape}, "
                f"got {tuple(ids.shape)}"
            )

        weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            device=device,
            dtype=torch.float32,
        )
        return cls(
            layer_name=layer.layer_name,
            weights=weights,
            ids=ids,
        )


def _cycle_expert_ids(
    num_tokens: int,
    top_k: int,
    activated_experts: int,
) -> list[list[int]]:
    """Assign expert ids deterministically so exactly ``activated_experts``
    distinct ids appear, cycled across the token dimension.

    The specific assignment doesn't matter for latency — only the count
    of distinct activations does. We use the simplest pattern:
    ``id = (token_idx * top_k + offset) % activated_experts``.
    """
    return [
        [
            (token_idx * top_k + offset) % activated_experts
            for offset in range(top_k)
        ]
        for token_idx in range(num_tokens)
    ]


# ---------------------------------------------------------------------------
# Context manager: live FusedMoE patch
# ---------------------------------------------------------------------------

@contextmanager
def force_moe_routing(route: ExpertRoute | None) -> Iterator[None]:
    """Patch ``FusedMoE.forward_native`` to use ``route`` when called.

    If ``route`` is None the function is a no-op (useful for dense
    profile categories where we still pass through the MoE-aware
    execute path).

    The patch is layer-scoped: only the specific FusedMoE whose
    ``layer_name`` matches ``route.layer_name`` is affected; other MoE
    layers (if any) fall through to their normal forward. This matters
    if the model has multiple MoE layers and we're profiling only one
    at a time. For the single-layer test model we override to 1
    decoder layer, this is moot.
    """
    if route is None:
        yield
        return

    # Local import so that host-side code doesn't pay the vLLM import
    # cost just to read this module.
    from vllm.model_executor.layers.fused_moe import layer as fused_moe

    moe_type = getattr(fused_moe, "FusedMoE", None)
    method_name = "forward_native"
    if moe_type is None:
        moe_type = fused_moe.MoERunner
        method_name = "forward"
    original_forward = getattr(moe_type, method_name)

    @wraps(original_forward)
    def hooked_forward(self, hidden_states, router_logits, *args, **kwargs):
        # Only patch the specific layer we care about. Any other
        # FusedMoE encountered during this forward pass uses its
        # normal routing.
        if self.layer_name != route.layer_name:
            return original_forward(self, hidden_states, router_logits, *args, **kwargs)

        # We also sanity-check that our forged topk_ids matches the
        # actual per-call token count. If hidden_states is padded
        # differently than expected, bail.
        top_k = getattr(self, "top_k", None)
        if top_k is None:
            top_k = self.moe_config.experts_per_token
        expected_shape = (hidden_states.shape[0], top_k)
        if tuple(route.ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch for {self.layer_name}: "
                f"expected {expected_shape}, got {tuple(route.ids.shape)}"
            )

        # The router's public API is select_experts(...). We override
        # the underlying _compute_routing to return our forged pair,
        # restore it after the single call completes. This is cleaner
        # than swapping select_experts wholesale because
        # select_experts does normalization / validation around
        # _compute_routing that we still want to run.
        original_select_experts = self.router.select_experts
        original_compute_routing = getattr(self.router, "_compute_routing", None)

        @wraps(original_select_experts)
        def hooked_select_experts(*args, **kwargs):
            if original_compute_routing is not None:
                @wraps(original_compute_routing)
                def forced_compute_routing(*_args, **_kwargs):
                    return route.weights, route.ids

                self.router._compute_routing = forced_compute_routing
            else:
                def forced_select_experts(*_args, **_kwargs):
                    return route.weights, route.ids

                self.router.select_experts = forced_select_experts
            try:
                return original_select_experts(*args, **kwargs)
            finally:
                if original_compute_routing is not None:
                    self.router._compute_routing = original_compute_routing
                else:
                    self.router.select_experts = original_select_experts

        self.router.select_experts = hooked_select_experts
        try:
            return original_forward(self, hidden_states, router_logits, *args, **kwargs)
        finally:
            self.router.select_experts = original_select_experts

    setattr(moe_type, method_name, hooked_forward)
    try:
        yield
    finally:
        # Restore the original class method so future calls (in tests,
        # or after this profile session ends) behave normally.
        setattr(moe_type, method_name, original_forward)


# ---------------------------------------------------------------------------
# Helpers that run worker-side
# ---------------------------------------------------------------------------

def single_moe_layer(model_runner):
    """Return the model's lone ``FusedMoE`` layer.

    We run profiling with ``hf_overrides.num_hidden_layers=1`` so
    there's exactly one of every decoder sub-module — including MoE.
    If for some reason there are zero or more-than-one, raise so the
    caller can investigate rather than forge the wrong route.
    """
    from vllm.model_executor.layers.fused_moe import layer as fused_moe
    moe_type = getattr(fused_moe, "FusedMoE", fused_moe.MoERunner)

    model = model_runner.get_model()
    moe_layers = [m for m in model.modules() if isinstance(m, moe_type)]
    if len(moe_layers) != 1:
        raise RuntimeError(
            f"Expected exactly one FusedMoE layer in the test model, "
            f"got {len(moe_layers)}"
        )
    return moe_layers[0]

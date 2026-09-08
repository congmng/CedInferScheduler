"""Versioned, class-aware P/D routing plans.

The flow solver owns plan construction.  This module intentionally contains no
optimization policy; it gives the request router a small, validated data
surface that can be atomically replaced at a control tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple


def _normalise(weights: Mapping[int, float]) -> Dict[int, float]:
    positive = {int(key): float(value) for key, value in weights.items()
                if float(value) > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in positive.items()}


@dataclass(frozen=True)
class AffinityPlan:
    """A slow-layer plan used by the fast request router.

    ``prefill_weights`` is keyed by class id; ``decode_weights`` is keyed by
    ``(prefill instance id, class id)``.  IDs refer to simulator instance IDs,
    rather than list offsets, so plans remain valid when workers are inactive.
    """

    version: int
    expires_at_ns: int
    prefill_weights: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    decode_weights: Mapping[Tuple[int, str], Mapping[int, float]] = field(default_factory=dict)
    fallback_decode_ids: Mapping[Tuple[int, str], Iterable[int]] = field(default_factory=dict)

    def prefill_for(self, class_id: str) -> Dict[int, float]:
        return _normalise(self.prefill_weights.get(class_id, {}))

    def decode_for(self, prefill_id: int, class_id: str) -> Dict[int, float]:
        return _normalise(self.decode_weights.get((int(prefill_id), class_id), {}))

    def fallback_for(self, prefill_id: int, class_id: str) -> Tuple[int, ...]:
        return tuple(int(value) for value in
                     self.fallback_decode_ids.get((int(prefill_id), class_id), ()))

    def is_expired(self, current_ns: int) -> bool:
        return self.expires_at_ns >= 0 and current_ns >= self.expires_at_ns

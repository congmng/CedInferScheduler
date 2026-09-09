"""Prefix-class observation for CASR.

The simulator's cache index is block-hash based.  CASR needs a coarser and
stable control-plane view, so this profiler aggregates arrivals and lookup
results into prefix classes without retaining original prompt text.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from typing import Dict, Iterable, Optional, Tuple


def _bucket(value: int) -> str:
    value = max(1, int(value))
    lower = 1 << (value.bit_length() - 1)
    return f"{lower}-{lower * 2 - 1}"


def derive_prefix_id(model_id: str, token_ids: Optional[Iterable[int]],
                     block_size: int, max_prefix_tokens: int = 512) -> str:
    """Return a privacy-preserving ID for a block-aligned reusable prefix."""
    ids = list(token_ids or [])
    usable = min(len(ids), max(0, int(max_prefix_tokens)))
    usable -= usable % max(1, int(block_size))
    if usable == 0:
        return "none"
    digest = hashlib.blake2b(digest_size=12)
    digest.update(model_id.encode("utf-8"))
    digest.update(str(usable).encode("ascii"))
    for token in ids[:usable]:
        digest.update(int(token).to_bytes(8, "little", signed=True))
    return f"p{usable}:{digest.hexdigest()}"


def derive_class_id(model_id: str, prefix_id: str, input_tokens: int,
                    output_tokens: int) -> str:
    return f"{model_id}|{prefix_id}|in:{_bucket(input_tokens)}|out:{_bucket(output_tokens)}"


@dataclass
class PrefixState:
    request_count: int = 0
    arrival_rate_ewma: float = 0.0
    reuse_ewma: float = 0.0
    hit_tokens_ewma: float = 0.0
    npu_hit_tokens: int = 0
    storage_hit_tokens: int = 0
    requested_tokens: int = 0
    last_access_ns: int = -1
    last_arrival_ns: int = -1

    def observe_arrival(self, at_ns: int, input_tokens: int, alpha: float) -> None:
        if self.last_arrival_ns >= 0 and at_ns > self.last_arrival_ns:
            instant_rate = 1_000_000_000.0 / (at_ns - self.last_arrival_ns)
            self.arrival_rate_ewma = alpha * instant_rate + (1.0 - alpha) * self.arrival_rate_ewma
        self.reuse_ewma = alpha * 1.0 + (1.0 - alpha) * self.reuse_ewma
        self.request_count += 1
        self.requested_tokens += int(input_tokens)
        self.last_arrival_ns = at_ns
        self.last_access_ns = at_ns

    def observe_lookup(self, at_ns: int, npu_hit: int, storage_hit: int,
                       alpha: float) -> None:
        total_hit = max(0, int(storage_hit))
        self.hit_tokens_ewma = alpha * total_hit + (1.0 - alpha) * self.hit_tokens_ewma
        self.npu_hit_tokens += max(0, int(npu_hit))
        self.storage_hit_tokens += total_hit
        self.last_access_ns = at_ns


class PrefixProfiler:
    """Collect per-P, per-class observations and emit control-tick snapshots."""

    def __init__(self, block_size: int, ewma_alpha: float = 0.2,
                 max_prefix_tokens: int = 512,
                 arrival_half_life_ms: float = 1000.0):
        self.block_size = int(block_size)
        self.ewma_alpha = float(ewma_alpha)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.arrival_half_life_ns = max(1, int(float(arrival_half_life_ms) * 1_000_000))
        self._states: Dict[Tuple[int, str], PrefixState] = defaultdict(PrefixState)
        self._classes: Dict[str, Dict[str, object]] = {}
        self._representatives: Dict[str, Tuple[int, list[int]]] = {}

    def assign(self, model_id: str, input_tokens: int, output_tokens: int,
               token_ids: Optional[Iterable[int]], kv_bytes_per_request: float = 0.0) -> Tuple[str, str]:
        prefix_id = derive_prefix_id(model_id, token_ids, self.block_size,
                                     self.max_prefix_tokens)
        class_id = derive_class_id(model_id, prefix_id, input_tokens, output_tokens)
        self._classes.setdefault(class_id, {
            "class_id": class_id,
            "model_id": model_id,
            "prefix_id": prefix_id,
            "input_bucket": _bucket(input_tokens),
            "output_bucket": _bucket(output_tokens),
            "kv_bytes_per_request": max(0.0, float(kv_bytes_per_request)),
        })
        self._classes[class_id]["kv_bytes_per_request"] = max(
            self._classes[class_id]["kv_bytes_per_request"], float(kv_bytes_per_request))
        if token_ids:
            self._representatives.setdefault(class_id, (int(input_tokens), list(token_ids)))
        return class_id, prefix_id

    def warm_candidate(self, class_id):
        """Return the representative token sequence needed to seed a cache."""
        return self._representatives.get(class_id)

    def observe_arrival(self, request, prefill_instance_id: int, at_ns: int) -> None:
        self._states[(int(prefill_instance_id), request.class_id)].observe_arrival(
            int(at_ns), request.original_input, self.ewma_alpha)

    def observe_lookup(self, request, prefill_instance_id: int, at_ns: int,
                       npu_hit_tokens: int, storage_hit_tokens: int) -> None:
        self._states[(int(prefill_instance_id), request.class_id)].observe_lookup(
            int(at_ns), npu_hit_tokens, storage_hit_tokens, self.ewma_alpha)

    def snapshot(self, at_ns: int, schedulers=()) -> Dict[str, object]:
        for state in self._states.values():
            if state.last_arrival_ns >= 0 and at_ns > state.last_arrival_ns:
                elapsed = at_ns - state.last_arrival_ns
                state.arrival_rate_ewma *= 0.5 ** (elapsed / self.arrival_half_life_ns)
        cache_by_instance = {}
        for scheduler in schedulers:
            pool = scheduler.memory.npu_pool
            cache_by_instance[int(scheduler.instance_id)] = {
                "cached_blocks": len(pool.cached_block_hash_to_block),
                "cached_bytes": pool.used_bytes(),
                "free_blocks": pool.get_num_free_blocks(),
                "running": len(scheduler.running),
                "waiting": len(scheduler.waiting),
                "admission_state": scheduler.admission_state,
                "resource_gpu_ids": list(getattr(scheduler, "resource_gpu_ids", ())),
                "resource_mem_gb": getattr(scheduler, "resource_mem_gb", 0.0),
            }
        rows = []
        for (instance_id, class_id), state in sorted(self._states.items()):
            row = dict(self._classes[class_id])
            row.update(asdict(state))
            row["prefill_instance_id"] = instance_id
            row["cache"] = cache_by_instance.get(instance_id, {})
            rows.append(row)
        return {"time_ns": int(at_ns), "prefix_states": rows,
                "instances": cache_by_instance}

    @staticmethod
    def append_snapshot(path: str, snapshot: Dict[str, object]) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")

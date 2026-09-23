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
    # ``hit_tokens_ewma`` is an EWMA *per request*, so the matching denominator
    # must also be per request.  ``requested_tokens`` is the cumulative sum over
    # the whole run and dividing by it makes the observed hit ratio decay like
    # 1/N, which silently erases prefix affinity from the control plane.
    requested_tokens_ewma: float = 0.0
    last_access_ns: int = -1
    last_arrival_ns: int = -1
    # Arrivals counted since the last control tick.  The rate is derived from a
    # count over the sampling window, never from the reciprocal of an
    # inter-arrival gap: two requests dispatched in the same millisecond would
    # otherwise produce a spike of ~1000 req/s and make the LP believe the
    # offered load is several times what the clients actually send.
    arrivals_since_sample: int = 0
    arrival_sampled: bool = False
    #: Smoothed *share* of the window's arrivals (sums to 1 across classes).
    #: The demand level comes from the global counter; multiplying the two
    #: keeps ``sum_k rate_k`` from drifting above the true total.
    share_ewma: float = 0.0

    def observe_arrival(self, at_ns: int, input_tokens: int, alpha: float) -> None:
        self.arrivals_since_sample += 1
        self.reuse_ewma = alpha * 1.0 + (1.0 - alpha) * self.reuse_ewma
        self.request_count += 1
        self.requested_tokens += int(input_tokens)
        self.requested_tokens_ewma = (alpha * float(input_tokens) +
                                      (1.0 - alpha) * self.requested_tokens_ewma)
        self.last_arrival_ns = at_ns
        self.last_access_ns = at_ns

    def sample_arrivals(self, window_s: float, alpha: float,
                        total_arrivals: int | None = None) -> None:
        """Fold a completed sampling window's count into the rate estimate.

        ``window_s`` is the *accumulated* time, and the caller only folds once
        it has reached the configured horizon (see ``PrefixProfiler``).  The
        distinction matters: the real control loop differences a counter over
        1 s, so its rate is a 1 s mean.  Taking the *instantaneous* ratio every
        control tick instead makes the estimate a peak detector whenever the
        router dispatches in batches -- which it does exactly under
        back-pressure.  Measured 2026-09-23 (main matrix, 16 req/s offered,
        100 ms control interval): a burst of dispatched requests pushed a
        class whose true rate is 2 req/s to 9.75, and the pool's total from 16
        to 80.6, which the solver then treated as the offered load.
        """
        if window_s <= 0.0:
            return
        instant = self.arrivals_since_sample / window_s
        share = (self.arrivals_since_sample / total_arrivals
                 if total_arrivals else 0.0)
        if self.arrival_sampled:
            self.arrival_rate_ewma = (alpha * instant +
                                      (1.0 - alpha) * self.arrival_rate_ewma)
            self.share_ewma = alpha * share + (1.0 - alpha) * self.share_ewma
        else:
            self.arrival_rate_ewma = instant
            self.share_ewma = share
            self.arrival_sampled = True
        self.arrivals_since_sample = 0

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
                 arrival_half_life_ms: float = 1000.0,
                 arrival_window_ms: float = 1000.0):
        self.block_size = int(block_size)
        self.ewma_alpha = float(ewma_alpha)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.arrival_half_life_ns = max(1, int(float(arrival_half_life_ms) * 1_000_000))
        # How much time the arrival count is accumulated over before a rate is
        # derived.  The deployment differences a counter every 1 s; a simulator
        # tuning the control loop to 100 ms must not shrink this along with it,
        # or the estimate becomes a peak detector (see ``sample_arrivals``).
        self.arrival_window_ns = max(1, int(float(arrival_window_ms) * 1_000_000))
        self._window_elapsed_ns = 0
        #: Global arrival count for the current window, and its smoothed rate.
        #: The per-class estimates are renormalised against this: with a
        #: unique-prefix trace every request is its own class, and summing
        #: per-class EWMAs then adds each one-shot class's decaying tail to the
        #: total (measured 2026-09-23: 16 req/s offered read as 44-80).
        self._arrivals_total = 0
        self._total_rate_ewma = 0.0
        self._total_sampled = False
        self._states: Dict[Tuple[int, str], PrefixState] = defaultdict(PrefixState)
        self._classes: Dict[str, Dict[str, object]] = {}
        self._representatives: Dict[str, Tuple[int, list[int]]] = {}
        self._last_sample_ns = -1

    def assign(self, model_id: str, input_tokens: int, output_tokens: int,
               token_ids: Optional[Iterable[int]], kv_bytes_per_request: float = 0.0,
               slo_ttft_ms: Optional[float] = None,
               slo_tpot_ms: Optional[float] = None) -> Tuple[str, str]:
        """Bucket one request into a prefix class.

        ``slo_ttft_ms`` / ``slo_tpot_ms`` are the request's own budgets when
        the trace carries them.  A class keeps the **tightest** bound seen, the
        same rule the real router applies (``class_slo`` in ``/routing-state``):
        one 250 ms interactive request makes the whole class 250 ms, which is
        what stops a mixed class from being priced at the loosest of its
        members.
        """
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
        for key, value in (("slo_ttft_ms", slo_ttft_ms), ("slo_tpot_ms", slo_tpot_ms)):
            try:
                value = float(value) if value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is None or value <= 0:
                continue
            prior = self._classes[class_id].get(key)
            self._classes[class_id][key] = value if prior is None else min(prior, value)
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
        self._arrivals_total += 1

    def observe_offered_arrival(self, class_id, prefill_instance_id: int,
                                at_ns: int, input_tokens: int = 0) -> None:
        """Count a request when it is *offered*, not when it is dispatched.

        The router calls this at the front door (see
        ``Router._note_offered_arrival``).  Counting on dispatch turned the
        arrival estimate into a property of the dispatcher: a held backlog
        released in one go reads as tens of requests per window, and the plan
        then sizes itself for that phantom load.
        """
        if not class_id:
            return
        state = self._states[(int(prefill_instance_id), str(class_id))]
        state.observe_arrival(int(at_ns), int(input_tokens), self.ewma_alpha)
        self._arrivals_total += 1

    def observe_lookup(self, request, prefill_instance_id: int, at_ns: int,
                       npu_hit_tokens: int, storage_hit_tokens: int) -> None:
        self._states[(int(prefill_instance_id), request.class_id)].observe_lookup(
            int(at_ns), npu_hit_tokens, storage_hit_tokens, self.ewma_alpha)

    def snapshot(self, at_ns: int, schedulers=()) -> Dict[str, object]:
        elapsed_s = 0.0
        if self._last_sample_ns >= 0 and at_ns > self._last_sample_ns:
            elapsed_s = (at_ns - self._last_sample_ns) / 1e9
        if elapsed_s > 0.0:
            self._window_elapsed_ns += int(at_ns - self._last_sample_ns)
        # Only fold a rate once the sampling horizon is complete; until then the
        # counts keep accumulating (the caller's ticks can be much faster).
        if self._window_elapsed_ns >= self.arrival_window_ns:
            window_s = self._window_elapsed_ns / 1e9
            total = self._arrivals_total
            for state in self._states.values():
                state.sample_arrivals(window_s, self.ewma_alpha, total)
            # Level from the global counter, shares from the per-class EWMAs:
            # the shares are renormalised so the published rates sum to exactly
            # the observed total.  They need it: a class that stops arriving
            # keeps a decaying share, and new classes keep adding theirs, so the
            # raw sum drifts above 1 (measured: 16 req/s offered read as 38-80
            # when the shares were used as-is).
            instant_total = total / window_s
            if self._total_sampled:
                self._total_rate_ewma = (self.ewma_alpha * instant_total +
                                         (1.0 - self.ewma_alpha) * self._total_rate_ewma)
            else:
                self._total_rate_ewma = instant_total
                self._total_sampled = True
            share_sum = sum(state.share_ewma for state in self._states.values()
                            if state.arrival_sampled)
            scale = (self._total_rate_ewma / share_sum) if share_sum > 0.0 else 0.0
            for state in self._states.values():
                if state.arrival_sampled:
                    state.arrival_rate_ewma = scale * state.share_ewma
            if os.environ.get("ARRIVAL_DEBUG"):
                print(f"[arrival] t={at_ns/1e9:.2f}s window={window_s:.2f}s "
                      f"count={total} classes={len(self._states)} "
                      f"total_rate={self._total_rate_ewma:.2f}", flush=True)
            self._arrivals_total = 0
            self._window_elapsed_ns = 0
        self._last_sample_ns = int(at_ns)
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
        # Per-class TTFT budget for the solver's ``class_ttft_slo_ms``.  Only
        # classes whose requests carried a budget appear, so a trace without
        # SLO fields produces an empty map and the solver keeps its global
        # ``ttft_slo_ms`` (0 = the term is off).
        class_slo = {class_id: entry["slo_ttft_ms"]
                     for class_id, entry in self._classes.items()
                     if entry.get("slo_ttft_ms")}
        return {"time_ns": int(at_ns), "prefix_states": rows,
                "instances": cache_by_instance, "class_slo": class_slo}

    @staticmethod
    def append_snapshot(path: str, snapshot: Dict[str, object]) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")

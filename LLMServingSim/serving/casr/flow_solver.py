"""Dependency-free baseline allocator for CASR f[class, P, D] flows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Mapping


@dataclass(frozen=True)
class FlowAssignment:
    class_id: str
    prefill_id: int
    decode_id: int
    flow: float
    cost: float
    link_ids: tuple[str, ...] = ()
    prefill_overflow: float = 0.0
    decode_overflow: float = 0.0
    link_overflow: float = 0.0


@dataclass(frozen=True)
class SharedLink:
    link_id: str
    capacity: float
    pairs: frozenset[tuple[int, int]] = frozenset()
    capacity_bytes_per_s: float = 0.0

    def carries(self, prefill_id: int, decode_id: int) -> bool:
        return not self.pairs or (prefill_id, decode_id) in self.pairs


@dataclass(frozen=True)
class FlowSolverConfig:
    prefill_capacity: Mapping[int, float] = field(default_factory=dict)
    decode_capacity: Mapping[int, float] = field(default_factory=dict)
    shared_links: tuple[SharedLink, ...] = ()
    # Penalty per *unit of capacity* by which an instance or a shared link is
    # overloaded: overflowing a link by its own capacity costs this much,
    # regardless of whether the constraint is expressed in requests/s or
    # bytes/s.
    overflow_penalty: float = 10.0
    solver: str = "greedy"
    pair_costs: Mapping[tuple[int, int], Mapping[str, float]] = field(default_factory=dict)
    class_kv_bytes: Mapping[str, float] = field(default_factory=dict)
    default_kv_bytes: float = 0.0
    # Bytes of KV a single prompt token occupies on one accelerator.  When a
    # producer does not report ``kv_bytes_per_request`` (the simulator's trace
    # path does not), the solver derives it from the observed per-request input
    # token count.  This is what lets the shared-link budget be expressed in
    # bytes/s -- the same units the real LMCache producer-side push link is
    # measured in -- instead of an abstract requests/s.
    kv_bytes_per_token: float = 0.0
    queue_weight: float = 0.0
    use_cache_capacity: bool = True
    network_weight: float = 1.0
    # Marginal congestion price.  ``0`` keeps the objective purely linear.  A
    # positive value adds a convex (piecewise-linear) cost on each instance's
    # utilisation, so the LP spreads flow instead of stacking everything on the
    # cheapest pair -- a purely per-unit price cannot do that, because it is the
    # same for the first and the last request an instance serves.
    utilization_weight: float = 0.0
    utilization_segments: int = 8
    # -- Decode tail pricing ------------------------------------------------
    # ``queue_weight`` prices a dimensionless queue *fraction*, so it is
    # blind to how long that queue takes to drain: the same ``waiting`` count
    # costs 46 ms per step on a 5090 and 372 ms on a 4090.  Measured on the
    # 2026-09-23 peak trace (1900x 1250-token requests, 30 rps for 60 s), that
    # blind spot was the whole story -- the least-loaded baseline pinned every
    # request on the fastest Decode (p95 18.8 s) while the plan spread 20-30%
    # onto 2-3x slower ones and took p95 78-97 s.
    #
    # With a positive ``tail_weight`` the queue term becomes a *time*: the
    # number of decode steps a request has to wait, times this instance's own
    # step time for this class,
    #
    #     queue_wait_ms = queue_fraction x decode_service_ms x decode_work_k
    #
    # so routing a long-output class to a slow, deep queue costs what it
    # actually costs.  Units are seconds, the same as every other term in
    # ``c_ijk``.  0 = off (the historical objective).
    tail_weight: float = 0.0
    # A class whose offered load is below this (requests/s) is single-homed:
    # all of its flow is placed on the one Prefill the LP liked best.  A low
    # demand cannot amortise the cold first prefill a second edge costs, and
    # for such classes the LP frequently has many equal-cost vertices that
    # split the flow arbitrarily.  ``0`` disables the collapse.
    single_home_below_rps: float = 0.0
    # Per-class floor applied once to the aggregated demand (requests/s).  It
    # exists only so a class that is still in flight keeps a variable when its
    # EWMA has decayed to zero.  Must stay far below ``capacity / num_classes``
    # -- a per-*row* floor is what made the LP believe the offered load was
    # several times what the clients were sending.
    class_demand_floor_rps: float = 0.0
    # Per-instance compute cost.  Heterogeneous accelerators differ by far more
    # than their link cost, so the objective needs a term that is linear in the
    # flow routed through an instance; capacity limits alone cannot express it
    # (a slow instance simply becomes a cheap place to put overflow).  Values
    # are measured service times in milliseconds, so the term lands in the same
    # units as the network RTT/transfer costs.
    prefill_service_ms: Mapping[int, float] = field(default_factory=dict)
    decode_service_ms: Mapping[int, float] = field(default_factory=dict)
    compute_weight: float = 0.0
    # -- SLO modelling ----------------------------------------------------
    # Predicted client-visible TTFT above ``ttft_slo_ms`` makes a
    # (class, prefill, decode) triple SLO-violating.  ``slo_penalty`` is added
    # to its unit cost rather than deleting the pair: when *every* feasible
    # pair is over SLO (a capacity shortfall) the solver must still return a
    # plan, but it prefers a compliant pair whenever one exists.  This is the
    # ``p_slo`` term of ``docs/CASR调度算法实施设计.md``.  ``0`` disables it.
    ttft_slo_ms: float = 0.0
    # Per-class override, keyed by class id.  Classes missing from the map use
    # ``ttft_slo_ms``.
    class_ttft_slo_ms: Mapping[str, float] = field(default_factory=dict)
    slo_penalty: float = 0.0
    # Measured fixed overheads of the TTFT path, per instance, in ms.
    # ``prefill_ms`` observed by the router is larger than the pure compute
    # service time because it includes storing the KV chunk
    # (measured p50: 101 ms on p5090 vs 48.5 ms service), and TTFT contains one
    # decode-side contribution even when the decode queue is empty
    # (measured p50: 56 ms on d5090).  Without these two terms the predicted
    # TTFT of an idle cluster is ~50 ms while the real one is ~240 ms, so no
    # realistic SLO ever binds.  Values come from
    # ``ttft_ms - prefill_ms`` / ``prefill_ms - service_ms`` over a replay.
    prefill_overhead_ms: Mapping[int, float] = field(default_factory=dict)
    decode_ttft_ms: Mapping[int, float] = field(default_factory=dict)
    # Cost charged by ``plan_objective`` for a unit of demand whose class the
    # plan does not cover at all.  A published plan that names only the classes
    # observed in the first control tick would otherwise look *free* for every
    # later class (the loop simply never visits it), so hysteresis would keep
    # that stale plan forever and the router would serve everything new from
    # its least-loaded fallback -- the exact opposite of class-aware routing.
    plan_uncovered_penalty: float = 10.0
    # Prefix-cache working-set cap: the largest number of distinct classes one
    # Prefill should hold.  The LP prices cache *hits* per (class, Prefill) but
    # has no notion of the working set, so it can pile every class onto the
    # cheapest Prefill and pay for it in evictions -- measured 2026-09-13:
    # ``casr_lp`` put 95% of the wide trace's 537 classes on ``p5090`` and lost
    # ~6% to ``load``, which spread them (same pair, ``prefill_ms`` P50
    # 80.7 -> 105.4 ms while ``decode_ms`` did not move).  ``0`` disables the
    # post-processing cap; a positive value spreads the cheapest-to-move
    # classes until every Prefill is at or below it.
    prefill_class_limit: int = 0
    # -- prompt-length normalisation ---------------------------------------
    # ``prefill_capacity`` / ``decode_capacity`` are measured at ONE workload
    # length (the calibration basis below), but the LP used to scale nothing by
    # length: a 1250-token request consumed exactly as much of a Prefill as a
    # 200-token one.  The solver therefore reported ``prefill_overflow = 0`` on
    # a deployment whose Prefill queue was 48 deep with a ``prefill_ms`` P95 of
    # 34 s (2026-09-14, ``elastic-long-a1``).  Work is now the class's service
    # time *relative to the reference service time*, using a fixed + linear
    # per-token model fitted to two measurements on ``p5090``: 156 req/s at
    # ~180 tokens and 62.9 req/s at 1024 tokens (docs/实验结果汇总.md §5.9),
    # i.e. ``4.4 ms + 11.2 ms / 1k tokens``.  Leaving ``*_fixed_ms`` at 0 makes
    # the model purely linear in tokens.
    capacity_reference_tokens: int = 1024
    prefill_fixed_ms: float = 0.0
    prefill_ms_per_1k_tokens: float = 0.0
    decode_reference_tokens: int = 16
    decode_fixed_ms: float = 0.0
    decode_ms_per_1k_tokens: float = 0.0
    # Measured prompt-processing ceiling per Prefill, in tokens/s.  This is the
    # form that actually fits the deployment: measured 2026-09-14 with unique
    # prefixes, prefill-only, concurrency 16 --
    # ``p5090`` 11.3 req/s at 1250 tokens and 71.9 req/s at 254 tokens (14.1 vs
    # 18.3 ktok/s), ``p4090`` 8.4 and 40.8 req/s (10.4 ktok/s, flat).  The
    # declared ``prefill_capacity`` is therefore only valid up to
    # ``prefill_tokens_per_s / prefill_capacity`` tokens per request; past that
    # the token ceiling binds.  When this map is empty the fixed+linear model
    # above is used instead.
    prefill_tokens_per_s: Mapping[int, float] = field(default_factory=dict)
    # A class can never claim more than this many reference-length units of an
    # instance (a guard against a mis-parsed token count).
    work_ceiling: float = 8.0

    @classmethod
    def from_dict(cls, raw):
        raw = raw or {}

        def numeric_map(source, cast=float):
            """Parse a JSON map keyed by instance id.

            Documentation keys (``_comment``) are legitimate in these config
            blocks and must not crash the router, and a single malformed entry
            must not take the control plane down with it.
            """
            parsed = {}
            for key, value in (source or {}).items():
                try:
                    instance_id = int(key)
                except (TypeError, ValueError):
                    continue
                try:
                    parsed[instance_id] = cast(value)
                except (TypeError, ValueError):
                    continue
            return parsed

        links = []
        for index, item in enumerate(raw.get("shared_links", ())):
            pairs = frozenset((int(pair[0]), int(pair[1])) for pair in item.get("pairs", ()))
            links.append(SharedLink(str(item.get("id", f"link-{index}")),
                                    float(item.get("capacity", float("inf"))), pairs,
                                    float(item.get("capacity_bytes_per_s", 0.0))))
        pair_costs = {}
        for key, value in (raw.get("pair_costs") or {}).items():
            if isinstance(key, str):
                p_id, d_id = (int(part) for part in key.replace("/", ",").split(",", 1))
            else:
                p_id, d_id = (int(key[0]), int(key[1]))
            pair_costs[p_id, d_id] = {str(name): float(amount)
                                      for name, amount in (value or {}).items()}
        return cls(
            prefill_capacity=numeric_map(raw.get("prefill_capacity")),
            decode_capacity=numeric_map(raw.get("decode_capacity")),
            shared_links=tuple(links),
            overflow_penalty=float(raw.get("overflow_penalty", 10.0)),
            solver=str(raw.get("solver", "greedy")).lower(),
            pair_costs=pair_costs,
            class_kv_bytes={str(key): float(value)
                            for key, value in (raw.get("class_kv_bytes") or {}).items()
                            if not str(key).startswith("_")},
            default_kv_bytes=float(raw.get("default_kv_bytes", 0.0)),
            kv_bytes_per_token=float(raw.get("kv_bytes_per_token", 0.0)),
            queue_weight=float(raw.get("queue_weight", 0.0)),
            use_cache_capacity=bool(raw.get("use_cache_capacity", True)),
            network_weight=float(raw.get("network_weight", 1.0)),
            utilization_weight=float(raw.get("utilization_weight", 0.0)),
            utilization_segments=max(1, int(raw.get("utilization_segments", 8))),
            tail_weight=float(raw.get("tail_weight", 0.0)),
            single_home_below_rps=float(raw.get("single_home_below_rps", 0.0)),
            class_demand_floor_rps=float(raw.get("class_demand_floor_rps", 0.0)),
            capacity_reference_tokens=int(raw.get("capacity_reference_tokens", 1024)),
            prefill_fixed_ms=float(raw.get("prefill_fixed_ms", 0.0)),
            prefill_ms_per_1k_tokens=float(raw.get("prefill_ms_per_1k_tokens", 0.0)),
            decode_reference_tokens=int(raw.get("decode_reference_tokens", 16)),
            decode_fixed_ms=float(raw.get("decode_fixed_ms", 0.0)),
            decode_ms_per_1k_tokens=float(raw.get("decode_ms_per_1k_tokens", 0.0)),
            prefill_tokens_per_s=numeric_map(raw.get("prefill_tokens_per_s")),
            work_ceiling=float(raw.get("work_ceiling", 8.0)),
            prefill_service_ms=numeric_map(raw.get("prefill_service_ms")),
            decode_service_ms=numeric_map(raw.get("decode_service_ms")),
            compute_weight=float(raw.get("compute_weight", 0.0)),
            ttft_slo_ms=float(raw.get("ttft_slo_ms", 0.0)),
            class_ttft_slo_ms={str(key): float(value) for key, value
                               in (raw.get("class_ttft_slo_ms") or {}).items()},
            slo_penalty=float(raw.get("slo_penalty", 0.0)),
            prefill_overhead_ms=numeric_map(raw.get("prefill_overhead_ms")),
            decode_ttft_ms=numeric_map(raw.get("decode_ttft_ms")),
            plan_uncovered_penalty=float(raw.get("plan_uncovered_penalty", 10.0)),
            prefill_class_limit=int(raw.get("prefill_class_limit", 0) or 0),
        )


class CapacityAwareFlowSolver:
    """Greedy min-cost baseline with explicit P and D capacity accounting.

    It is intentionally deterministic and dependency-free.  Its output has
    the same f_ijk surface as the later LP backend, so replacing this policy
    does not affect router, scheduler, or trace integration.
    """

    def __init__(self, config: FlowSolverConfig | None = None):
        self.config = config or FlowSolverConfig()
        self.backend = "greedy"
        self.diagnostics = {}
        self._last_single_homed = []
        # Classes moved by ``_cap_class_footprint`` in the last solve:
        # ``(class_id, from_prefill, to_prefill)``.
        self._last_capped = []
        # ``(prefill_id, decode_id)`` pairs the last solve judged SLO-violating,
        # exposed through ``diagnostics`` so a run can be audited afterwards.
        self._slo_violations = set()
        self._fallback = None
        self._runtime_queue = {}
        self._runtime_link_capacity = {}
        # Output-length work per class (pure function of the class id and the
        # configured model), cached because every constraint and cost term
        # reads it during one solve.
        self._decode_work_cache = {}

    def set_telemetry(self, telemetry):
        """Install best-effort exporter queue samples for the next solve."""
        self._runtime_queue = {}
        self._runtime_link_capacity = {}
        for worker_id, metrics in (telemetry or {}).get("workers", {}).items():
            try:
                instance_id = int(worker_id)
            except (TypeError, ValueError):
                continue
            for name in ("vllm_num_requests_waiting", "queue", "queue_depth"):
                if name in metrics:
                    self._runtime_queue[instance_id] = max(0.0, float(metrics[name]))
                    break
        for link_id, metrics in (telemetry or {}).get("links", {}).items():
            for name in ("capacity_bytes_per_s", "bandwidth_bytes_per_s",
                         "link_bandwidth_bytes_per_s"):
                if name in metrics:
                    self._runtime_link_capacity[str(link_id)] = max(0.0, float(metrics[name]))
                    break

    def _link_capacity(self, link):
        return self._runtime_link_capacity.get(link.link_id,
                                               link.capacity_bytes_per_s or link.capacity)

    def _link_uses_bytes(self, link):
        return link.capacity_bytes_per_s > 0 or link.link_id in self._runtime_link_capacity

    def apply_capacity_overrides(self, prefill_capacity=None, decode_capacity=None):
        """Temporarily re-price instance capacity for the next solve.

        The plan and the structural evaluator otherwise price a Prefill by its
        *compute* capacity (9.3 rps on a 5090), while the deployment's push
        ceiling is 0.26 GB/s -- about 1.4 req/s for a 1250-token prompt.  With
        the compute number the counterfactual for one more worker shows almost
        no gain, so ``+P`` fires late: measured in the six-domain arena, the
        elastic arm took 14878 ms where the capacity derived from the egress
        took 1400 ms on the same pool floor.
        """
        updates = {}
        if prefill_capacity:
            updates["prefill_capacity"] = {**self.config.prefill_capacity,
                                           **prefill_capacity}
        if decode_capacity:
            updates["decode_capacity"] = {**self.config.decode_capacity,
                                          **decode_capacity}
        if updates:
            self.config = dataclass_replace(self.config, **updates)

    def apply_slo_overrides(self, ttft_slo_ms=None, class_ttft_slo_ms=None,
                            slo_penalty=None):
        """Publish this tick's latency budgets into the solver config.

        ``class_ttft_slo_ms`` comes from the prefix profiler (the tightest
        budget any request of that class carried); ``ttft_slo_ms`` is the
        cluster-wide default from the policy.  Mirrors what the real control
        plane does every tick, so a simulated replay and a recorded run price
        the SLO term the same way.
        """
        updates = {}
        if ttft_slo_ms is not None:
            updates["ttft_slo_ms"] = float(ttft_slo_ms)
        if slo_penalty is not None:
            updates["slo_penalty"] = float(slo_penalty)
        if class_ttft_slo_ms is not None:
            merged = {str(key): float(value)
                      for key, value in self.config.class_ttft_slo_ms.items()}
            merged.update({str(key): float(value)
                           for key, value in class_ttft_slo_ms.items()})
            updates["class_ttft_slo_ms"] = merged
        if updates:
            self.config = dataclass_replace(self.config, **updates)

    def solve(self, rows, prefill, decode, work_overrides=None):
        self._slo_violations = set()
        if self.config.solver == "lp":
            try:
                return self._solve_lp(rows, prefill, decode, work_overrides)
            except ImportError:
                self._fallback = "ortools unavailable"
        return self._solve_greedy(rows, prefill, decode, work_overrides)

    def _aggregate_rows(self, rows, prefill, work_overrides=None):
        """Aggregate demand once while retaining cache state per P and class."""
        by_id = {int(scheduler.instance_id): scheduler for scheduler in prefill}
        grouped = {}
        for row in rows:
            class_id = row["class_id"]
            entry = grouped.setdefault(class_id, {
                "class_id": class_id, "arrival_rate_ewma": 0.0,
                "hit_tokens_ewma": {}, "requested_tokens": {},
                "requested_tokens_ewma": {}, "kv_bytes_per_request": 0.0,
            })
            # Sum the *estimated* rates.  This used to floor every row at
            # ``1.0`` before adding, which does not scale: the real router
            # observes one row per (prefill, class), so a few hundred classes
            # produced a few hundred req/s of phantom demand against ~100 req/s
            # of capacity.  The LP then solved a structurally overloaded
            # problem, spilled into its overflow slacks (measured: prefill
            # overflow ~7x capacity, KV link overflow ~25x) and replayed those
            # fractions onto the real traffic.  ``class_demand_floor_rps``
            # applies the floor once per class instead, and defaults to 0.
            entry["arrival_rate_ewma"] += float(row.get("arrival_rate_ewma", 0.0))
            p_id = int(row.get("prefill_instance_id", next(iter(by_id), -1)))
            entry["hit_tokens_ewma"][p_id] = entry["hit_tokens_ewma"].get(p_id, 0.0) + float(row.get("hit_tokens_ewma", 0.0))
            entry["requested_tokens"][p_id] = entry["requested_tokens"].get(p_id, 0.0) + float(row.get("requested_tokens", 0.0))
            entry["requested_tokens_ewma"][p_id] = max(
                entry["requested_tokens_ewma"].get(p_id, 0.0),
                float(row.get("requested_tokens_ewma", 0.0) or 0.0))
            entry["kv_bytes_per_request"] = max(
                entry["kv_bytes_per_request"], float(row.get("kv_bytes_per_request", 0.0)))
        work = {}
        for class_id, entry in grouped.items():
            for scheduler in prefill:
                p_id = int(scheduler.instance_id)
                value = self._cache_work(class_id, p_id, entry)
                if work_overrides and (p_id, class_id) in work_overrides:
                    value = max(0.05, min(1.0, float(work_overrides[p_id, class_id])))
                work[p_id, class_id] = value
        return grouped, work

    def _hit_ratio(self, class_id, prefill_id, entry):
        """Observed prefix-cache hit ratio for one class on one Prefill.

        Prefers the per-request EWMA denominator when the producer supplies one
        (both the simulator profiler and the real controller do), and falls back
        to the cumulative counter for older snapshots.
        """
        hit_tokens = float(entry.get("hit_tokens_ewma", {}).get(prefill_id, 0.0))
        per_request = float(entry.get("requested_tokens_ewma", {}).get(prefill_id, 0.0) or 0.0)
        if per_request <= 0:
            per_request = float(entry.get("requested_tokens", {}).get(prefill_id, 0.0))
        return min(0.95, hit_tokens / max(1.0, per_request))

    def _cache_work(self, class_id, prefill_id, entry):
        """Reference-length units this Prefill has to compute for one request.

        Two independent factors:

        * the prefix-cache miss ratio -- how much of the prompt is not already
          resident on that Prefill, and
        * the prompt-length factor -- how many *reference-length* prompts'
          worth of service time this one request costs.

        Before 2026-09-14 only the first existed, so the LP treated a
        1250-token request and a 200-token request as the same load and could
        not see a Prefill saturated by long prompts.
        """
        if not self.config.use_cache_capacity:
            miss = 1.0
        else:
            miss = max(0.05, 1.0 - self._hit_ratio(class_id, prefill_id, entry))
        tokens = self._requested_tokens(entry, prefill_id)
        return miss * self._prefill_length_work(tokens, prefill_id)

    def _prefill_length_work(self, tokens, prefill_id):
        """How many ``prefill_capacity`` units one request of ``tokens`` costs.

        Preferred form: the measured token-rate ceiling.  With
        ``prefill_tokens_per_s[p] = T`` and ``prefill_capacity[p] = C``, the
        instance serves ``min(C, T / tokens)`` requests per second, so one
        request costs ``max(1, tokens x C / T)`` capacity units.  Below
        ``T / C`` tokens the declared capacity binds and the factor is 1.
        """
        if tokens is None:
            return 1.0
        try:
            tokens = float(tokens)
        except (TypeError, ValueError):
            return 1.0
        if tokens <= 0.0:
            return 1.0
        capacity = self.config.prefill_capacity.get(int(prefill_id))
        token_ceiling = self.config.prefill_tokens_per_s.get(int(prefill_id))
        if capacity and token_ceiling:
            factor = tokens * float(capacity) / float(token_ceiling)
            return min(max(1.0, factor), max(1.0, float(self.config.work_ceiling)))
        return self._relative_work(tokens, self.config.capacity_reference_tokens,
                                   self.config.prefill_fixed_ms,
                                   self.config.prefill_ms_per_1k_tokens)

    def decode_work(self, class_id):
        """Same normalisation for the Decode leg, keyed on *output* length.

        ``decode_capacity`` is a requests/s figure measured at
        ``decode_reference_tokens`` output tokens; a class that generates four
        times as many tokens occupies the Decode for roughly four times as
        long.  The output length is already part of the class id (``out:16-31``)
        so no extra observation is needed.
        """
        key = str(class_id)
        if key in self._decode_work_cache:
            return self._decode_work_cache[key]
        value = self._relative_work(
            self.class_output_tokens(key),
            self.config.decode_reference_tokens,
            self.config.decode_fixed_ms,
            self.config.decode_ms_per_1k_tokens)
        self._decode_work_cache[key] = value
        return value

    @staticmethod
    def class_output_tokens(class_id):
        """Output token bucket encoded in a class id (``...|out:16-31``)."""
        marker = "|out:"
        text = str(class_id)
        index = text.find(marker)
        if index < 0:
            return None
        raw = text[index + len(marker):].split("|", 1)[0].split("-", 1)[0]
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _service_seconds(tokens, fixed_ms, per_1k_tokens):
        """Fixed + linear service-time model, in milliseconds."""
        return (float(fixed_ms)
                + float(per_1k_tokens) * (float(tokens) / 1000.0))

    def _relative_work(self, tokens, reference_tokens, fixed_ms, per_1k_tokens):
        """Service time of ``tokens`` relative to the calibration length."""
        if tokens is None:
            return 1.0
        try:
            tokens = float(tokens)
        except (TypeError, ValueError):
            return 1.0
        if tokens <= 0.0:
            return 1.0
        denominator = self._service_seconds(max(1, int(reference_tokens)),
                                            fixed_ms, per_1k_tokens)
        if denominator <= 0.0:
            return 1.0
        ratio = self._service_seconds(tokens, fixed_ms, per_1k_tokens) / denominator
        return min(max(0.05, ratio), max(0.05, float(self.config.work_ceiling)))

    @staticmethod
    def _requested_tokens(entry, prefill_id):
        """Per-request prompt length observed for one (class, Prefill)."""
        for key in ("requested_tokens_ewma", "requested_tokens"):
            values = entry.get(key) or {}
            if isinstance(values, Mapping):
                value = values.get(prefill_id, values.get(int(prefill_id), 0.0))
            else:
                value = values
            try:
                value = float(value or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                return value
        return None

    def _class_kv_bytes(self, class_id, entry):
        explicit = self.config.class_kv_bytes.get(class_id)
        if explicit is not None:
            return max(0.0, explicit)
        per_request = float(entry.get("kv_bytes_per_request", 0.0) or 0.0)
        if per_request > 0:
            return per_request
        if self.config.kv_bytes_per_token > 0:
            # KV bytes are a physical property of the prompt, so use the exact
            # observation rather than a smoothed rate.  The EWMA starts at
            # ``alpha x tokens`` -- 250 for a 1250-token prompt -- which
            # under-priced the handoff five-fold and kept the producer-egress
            # constraint (the one that actually binds this fabric) invisible:
            # measured 2026-09-16, the small-cluster forced-transfer arm showed
            # ``link_overflow 0`` for every instance while the run was 20x over
            # the link's budget, so ``+P`` never looked profitable.
            per_request = 0.0
            exact = entry.get("requested_tokens")
            if isinstance(exact, Mapping):
                per_request = max((float(value) for value in exact.values()),
                                  default=0.0)
            elif exact:
                per_request = float(exact)
            if per_request <= 0:
                tokens = entry.get("requested_tokens_ewma") or {}
                if isinstance(tokens, Mapping):
                    per_request = max((float(value) for value in tokens.values()),
                                      default=0.0)
                else:
                    per_request = float(tokens)
            if per_request > 0:
                return max(0.0, per_request * self.config.kv_bytes_per_token)
        return max(0.0, self.config.default_kv_bytes)

    def _service_ms(self, scheduler, configured):
        """Measured service time for one instance in milliseconds."""
        if scheduler.instance_id in configured:
            return float(configured[scheduler.instance_id])
        return float(getattr(scheduler, "service_ms", 0.0) or 0.0)

    @staticmethod
    def _work_diagnostics(work):
        """Compact ``{class_id: {prefill_id: ratio}}`` view of the cache work."""
        view = {}
        for (prefill_id, class_id), value in work.items():
            view.setdefault(class_id, {})[prefill_id] = round(float(value), 4)
        return view

    def _pair_cost(self, prefill, decode, class_id, entry, work_ratio=None):
        config = self.config.pair_costs.get((prefill.instance_id, decode.instance_id), {})
        distance = abs(prefill.start_npu - decode.start_npu) * 0.001
        rtt_ms = config.get("rtt_ms", 0.0)
        rtt = rtt_ms / 1000.0
        bandwidth = config.get("bandwidth_bytes_per_s", 0.0)
        kv_bytes = self._class_kv_bytes(class_id, entry)
        transfer_ms = kv_bytes / bandwidth * 1000.0 if bandwidth > 0 else 0.0
        transfer = transfer_ms / 1000.0
        waiting = max(float(len(getattr(decode, "waiting", ()))),
                      self._runtime_queue.get(int(decode.instance_id), 0.0))
        running = len(getattr(decode, "running", ()))
        max_num_seqs = getattr(decode, "max_num_seqs", 1)
        queue_fraction = (waiting * 4 + running) / max(1, max_num_seqs)
        queue = self.config.queue_weight * ((waiting * 4 + running) /
                                            max(1, max_num_seqs))
        # Tail term: the same queue_fraction, converted to the seconds it takes
        # *on this instance*.  ``decode_work_k`` is the class's output-length
        # multiplier, so a class that generates four times as many tokens pays
        # four times the drain time -- and a 2-3x slower Decode pays its own
        # step time rather than the fleet average.
        tail = 0.0
        if self.config.tail_weight:
            decode_step_ms = (self._service_ms(decode, self.config.decode_service_ms)
                              * self.decode_work(class_id))
            tail = self.config.tail_weight * queue_fraction * decode_step_ms / 1000.0
        if work_ratio is None:
            work_ratio = self._cache_work(class_id, int(prefill.instance_id), entry)
        compute = 0.0
        if self.config.compute_weight:
            # Prefill cost scales with the tokens that are actually computed: a
            # class whose prefix is already cached on this Prefill only pays for
            # the missing tail.  Without this the objective sees compute and
            # cache as independent, and the LP will happily move a warm class to
            # a slightly faster cold Prefill.
            # The Decode leg scales with the class's *output* length for the same
            # reason the Prefill leg scales with its prompt length.
            service_ms = (self._service_ms(prefill, self.config.prefill_service_ms) * work_ratio +
                          self._service_ms(decode, self.config.decode_service_ms)
                          * self.decode_work(class_id))
            compute = self.config.compute_weight * service_ms / 1000.0
        cost = (self.config.network_weight * (distance + rtt + transfer) +
                queue + tail + compute)
        slo_ms = self.config.class_ttft_slo_ms.get(class_id,
                                                   self.config.ttft_slo_ms)
        if slo_ms > 0 and self._predicted_ttft_ms(
                prefill, decode, entry, work_ratio, distance, rtt_ms,
                transfer_ms, queue_fraction,
                self.decode_work(class_id)) > slo_ms:
            cost += self.config.slo_penalty
            self._slo_violations.add((prefill.instance_id, decode.instance_id))
        return cost

    def _predicted_ttft_ms(self, prefill, decode, entry, work_ratio, distance,
                           rtt_ms, transfer_ms, queue_fraction,
                           decode_work_ratio=1.0):
        """First-order prediction of the client-visible TTFT, in milliseconds.

        Mirrors ``c_ijk`` in the design doc: the Prefill computes only the
        *uncached* part of the prompt (``work_ratio``), the KV is pushed across
        the measured RTT + link, and the Decode instance contributes its own
        measured service time scaled by its queued fraction.

        The decode term prices *queueing* (``service_ms × queued fraction``),
        not the full generation time, because TTFT ends at the first token.
        It is deliberately a first-order model: it is used only to decide
        whether a pair is over the SLO, and it is calibrated against the
        measured end-to-end service times rather than against a queueing
        formula.
        """
        if work_ratio is None:
            work_ratio = 1.0
        prefill_ms = (self._service_ms(prefill, self.config.prefill_service_ms) * work_ratio
                      + float(self.config.prefill_overhead_ms.get(
                          prefill.instance_id, 0.0)))
        # One decode-side contribution is always paid, queued or not; the
        # service term then prices the queue on top of it.
        decode_ms = (self._measured_ms(decode, self.config.decode_ttft_ms,
                                       self.config.decode_service_ms,
                                       getattr(decode, "max_num_seqs", 1))
                     + self._service_ms(decode, self.config.decode_service_ms)
                     * decode_work_ratio
                     * queue_fraction)
        return (distance * 1000.0 + rtt_ms + transfer_ms + prefill_ms + decode_ms)

    @staticmethod
    def _measured_ms(scheduler, measured, service, divisor):
        """A measured per-instance constant, or one service step as fallback."""
        if scheduler.instance_id in measured:
            return float(measured[scheduler.instance_id])
        value = service.get(scheduler.instance_id)
        if value is None:
            value = getattr(scheduler, "service_ms", 0.0)
        return float(value or 0.0) / max(1, divisor)

    def _congestion_variables(self, solver, prefill, decode, flows, demand, p_work):
        """Piecewise-linear convex cost on per-instance utilisation.

        An instance's capacity ``K`` is split into ``m`` equal blocks of
        ``K / m``.  Block ``k`` (0-based) carries marginal price
        ``w * service_s * (2k+1) / (2m)``, so serving load ``L`` costs
        ``w * service_s * L^2 / (2K)``: the first request is nearly free, the
        last one costs a full service time, and the price grows linearly in
        between.  That is the missing signal which makes an unconstrained
        capacity limit spread flow -- and it mirrors the ``load / K`` term the
        greedy baseline already carries.  The last block is unbounded, so
        overload stays feasible and is still priced by the explicit overflow
        slack of the capacity constraint.

        When an instance has no measured ``service_ms`` the inverse capacity
        stands in for its speed, so the knob still balances load on a simulated
        cluster that only declares capacities.

        Returns ``[(variable, objective_coefficient), ...]``.
        """
        weight = self.config.utilization_weight
        if not weight:
            return []
        segments = max(1, int(self.config.utilization_segments))
        priced = []
        for role, instances, capacity_of, service_of, gather in (
                ("p", prefill,
                 lambda s: self.config.prefill_capacity.get(s.instance_id, float(s.max_num_seqs)),
                 lambda s: self._service_ms(s, self.config.prefill_service_ms),
                 lambda s: sum(flows[c, s.instance_id, d.instance_id] * p_work[s.instance_id, c]
                               for c in demand for d in decode)),
                ("d", decode,
                 lambda s: self.config.decode_capacity.get(s.instance_id, float(s.max_num_seqs)),
                 lambda s: self._service_ms(s, self.config.decode_service_ms),
                 lambda s: sum(flows[c, p.instance_id, s.instance_id]
                               * self.decode_work(c)
                               for c in demand for p in prefill))):
            for instance in instances:
                capacity = max(1e-6, float(capacity_of(instance)))
                step = capacity / segments
                service_ms = max(0.0, float(service_of(instance)))
                # No calibration on this cluster: use 1/capacity as the speed
                # proxy so the term still has the right ordering.
                service_s = service_ms / 1000.0 if service_ms else 1.0 / capacity
                blocks = []
                for index in range(segments):
                    bound = solver.infinity() if index == segments - 1 else step
                    block = solver.NumVar(0.0, bound, f"u_{role}_{instance.instance_id}_{index}")
                    blocks.append(block)
                    marginal = weight * service_s * (2 * index + 1) / (2.0 * segments)
                    priced.append((block, marginal))
                solver.Add(sum(blocks) == gather(instance))
        return priced

    def _congestion_price(self, load, capacity, service_ms):
        """Closed-form twin of ``_congestion_variables``.

        The LP prices an instance's utilisation with ``m`` linear blocks whose
        marginal price rises from ``w*s/(2m)`` to ``w*s*(2m-1)/(2m)``.  The
        total for a given ``load`` is reproduced here so a plan can be scored
        without rebuilding the solver, and so the hysteresis comparison uses
        exactly the objective the LP minimised.
        """
        weight = self.config.utilization_weight
        if not weight or load <= 0.0 or capacity <= 0.0:
            return 0.0
        segments = max(1, int(self.config.utilization_segments))
        # No calibration: 1/capacity stands in for the service rate, matching
        # ``_congestion_variables``.
        service_s = service_ms / 1000.0 if service_ms else 1.0 / capacity
        step = capacity / segments
        total = 0.0
        remaining = float(load)
        for index in range(segments):
            take = remaining if index == segments - 1 else min(remaining, step)
            total += weight * service_s * (2 * index + 1) / (2.0 * segments) * take
            remaining -= take
            if remaining <= 0.0:
                break
        return total

    def _plan_load_cost(self, prefill, decode, p_load, d_load, link_load):
        """Capacity, congestion and link-overflow price of a plan's loads.

        A plan can only be compared against the LP optimum if it is scored with
        the same objective.  Without the congestion and overflow terms an
        incumbent that piles every class onto the single cheapest instance
        scores *better* than the balanced optimum, and plan hysteresis then
        locks that overloaded plan in place -- which is exactly the cold-start
        behaviour that made the LP lose to the load-balancing baseline.
        """
        penalty = self.config.overflow_penalty
        total = 0.0
        for scheduler in prefill:
            instance_id = scheduler.instance_id
            capacity = max(1e-6, float(self.config.prefill_capacity.get(
                instance_id, float(scheduler.max_num_seqs))))
            load = float(p_load.get(instance_id, 0.0))
            total += self._congestion_price(
                load, capacity, self._service_ms(scheduler, self.config.prefill_service_ms))
            total += penalty * max(0.0, load / capacity - 1.0)
        for scheduler in decode:
            instance_id = scheduler.instance_id
            capacity = max(1e-6, float(self.config.decode_capacity.get(
                instance_id, float(scheduler.max_num_seqs))))
            load = float(d_load.get(instance_id, 0.0))
            total += self._congestion_price(
                load, capacity, self._service_ms(scheduler, self.config.decode_service_ms))
            total += penalty * max(0.0, load / capacity - 1.0)
        for link in self.config.shared_links:
            capacity = self._link_capacity(link)
            if capacity == float("inf"):
                continue
            capacity = max(1e-6, float(capacity))
            total += penalty * max(0.0,
                                   float(link_load.get(link.link_id, 0.0)) / capacity - 1.0)
        return total

    def plan_objective(self, plan, rows, prefill, decode, work_overrides=None):
        """Cost of an already-published weight table under current observations.

        A freshly solved optimum is not the only candidate for publication: the
        plan that is already installed has value of its own, because the router
        has warmed the caches it names.  This evaluates an existing
        ``AffinityPlan`` with the same cost model -- including the convex
        congestion price and the capacity/link overflow penalties the LP
        optimises -- so a controller can require a minimum improvement before it
        churns class->Prefill assignments.
        """
        grouped, work = self._aggregate_rows(rows, prefill, work_overrides)
        prefills = {scheduler.instance_id: scheduler for scheduler in prefill}
        decodes = {scheduler.instance_id: scheduler for scheduler in decode}
        p_load = {scheduler.instance_id: 0.0 for scheduler in prefill}
        d_load = {scheduler.instance_id: 0.0 for scheduler in decode}
        link_load = {link.link_id: 0.0 for link in self.config.shared_links}
        total = 0.0
        for class_id, entry in grouped.items():
            demand = max(float(entry["arrival_rate_ewma"]),
                         self.config.class_demand_floor_rps)
            kv_bytes = self._class_kv_bytes(class_id, entry)
            prefill_weights = plan.prefill_for(class_id)
            if not prefill_weights:
                # The plan names no Prefill for this class, so the fast router
                # serves it from the least-loaded fallback: no cache affinity,
                # no per-class SLO.  Charging it here is what lets a candidate
                # that *does* cover the class win the hysteresis comparison.
                total += demand * self.config.plan_uncovered_penalty
                continue
            for prefill_id, prefill_share in prefill_weights.items():
                prefill_sched = prefills.get(int(prefill_id))
                if prefill_sched is None:
                    continue
                decode_weights = plan.decode_for(prefill_id, class_id)
                if not decode_weights:
                    fallback = plan.fallback_for(prefill_id, class_id)
                    decode_weights = ({instance_id: 1.0 for instance_id in fallback}
                                      if fallback else {})
                share = sum(decode_weights.values()) or 1.0
                work_ratio = work.get((int(prefill_id), class_id), 1.0)
                for decode_id, decode_share in decode_weights.items():
                    decode_sched = decodes.get(int(decode_id))
                    if decode_sched is None:
                        continue
                    flow = demand * prefill_share * (decode_share / share)
                    cost = self._pair_cost(prefill_sched, decode_sched, class_id, entry,
                                           work_ratio)
                    total += flow * cost
                    p_load[int(prefill_id)] += flow * work_ratio
                    d_load[int(decode_id)] += flow * self.decode_work(class_id)
                    for link in self.config.shared_links:
                        if link.carries(int(prefill_id), int(decode_id)):
                            link_load[link.link_id] += flow * (
                                kv_bytes if self._link_uses_bytes(link) else 1.0)
        total += self._plan_load_cost(prefill, decode, p_load, d_load, link_load)
        return total

    def _solve_greedy(self, rows, prefill, decode, work_overrides=None):
        self.backend = "greedy"
        p_load = {sched.instance_id: 0.0 for sched in prefill}
        d_load = {sched.instance_id: 0.0 for sched in decode}
        link_load = {link.link_id: 0.0 for link in self.config.shared_links}
        assignments = []
        grouped, work = self._aggregate_rows(rows, prefill, work_overrides)
        for class_id, entry in sorted(grouped.items(), key=lambda item: (
                -item[1]["arrival_rate_ewma"], item[0])):
            flow = max(float(entry["arrival_rate_ewma"]),
                       self.config.class_demand_floor_rps)
            best = None
            for p_sched in prefill:
                p_work = flow * work[p_sched.instance_id, class_id]
                d_work = self.decode_work(class_id)
                p_capacity = max(1.0, self.config.prefill_capacity.get(p_sched.instance_id, float(p_sched.max_num_seqs)))
                for d_sched in decode:
                    d_capacity = max(1.0, self.config.decode_capacity.get(
                        d_sched.instance_id, float(d_sched.max_num_seqs)))
                    link_cost = self._pair_cost(p_sched, d_sched, class_id, entry,
                                                work[p_sched.instance_id, class_id])
                    p_overflow = max(0.0, p_load[p_sched.instance_id] + p_work - p_capacity)
                    d_overflow = max(0.0, d_load[d_sched.instance_id] + flow * d_work - d_capacity)
                    carried = tuple(link for link in self.config.shared_links
                                    if link.carries(p_sched.instance_id, d_sched.instance_id))
                    kv_bytes = self._class_kv_bytes(class_id, entry)
                    link_overflow = sum(max(
                        0.0, link_load[link.link_id] + flow * (kv_bytes if self._link_uses_bytes(link) else 1.0) -
                        self._link_capacity(link))
                        for link in carried)
                    cost = ((p_load[p_sched.instance_id] + p_work) / p_capacity +
                            (d_load[d_sched.instance_id] + flow * d_work) / d_capacity + link_cost +
                            self.config.overflow_penalty * (p_overflow + d_overflow + link_overflow))
                    candidate = (cost, p_sched.instance_id, d_sched.instance_id,
                                 p_sched, d_sched, carried, p_overflow, d_overflow, link_overflow)
                    if best is None or candidate[:3] < best[:3]:
                        best = candidate
            cost, _, _, p_sched, d_sched, carried, p_overflow, d_overflow, link_overflow = best
            selected_p_work = flow * work[p_sched.instance_id, class_id]
            p_load[p_sched.instance_id] += selected_p_work
            d_load[d_sched.instance_id] += flow * self.decode_work(class_id)
            for link in carried:
                link_load[link.link_id] += flow * (self._class_kv_bytes(class_id, entry)
                                                   if self._link_uses_bytes(link) else 1.0)
            assignments.append(FlowAssignment(class_id, p_sched.instance_id,
                                              d_sched.instance_id, flow, cost,
                                              tuple(link.link_id for link in carried),
                                              p_overflow, d_overflow, link_overflow))
        assignments = self._cap_class_footprint(assignments, grouped, prefill, decode)
        self.diagnostics = {"backend": self.backend, "overflow_penalty": self.config.overflow_penalty,
                            "work": self._work_diagnostics(work),
                            "objective": sum(item.flow * item.cost for item in assignments),
                            "use_cache_capacity": self.config.use_cache_capacity,
                            "network_weight": self.config.network_weight,
                            "compute_weight": self.config.compute_weight,
            "queue_weight": self.config.queue_weight,
            "tail_weight": self.config.tail_weight,
                            "capped_classes": len(self._last_capped),
                            "ttft_slo_ms": self.config.ttft_slo_ms,
                            "slo_penalty": self.config.slo_penalty,
                            "slo_violating_pairs": sorted(self._slo_violations)}
        if self._fallback is not None:
            self.diagnostics["fallback"] = self._fallback
            self._fallback = None
        return assignments

    def _solve_lp(self, rows, prefill, decode, work_overrides=None):
        """Solve continuous f_ijk with capacity and overflow slack variables."""
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            raise RuntimeError("OR-Tools GLOP backend is unavailable")
        self.backend = "ortools-glop"
        # Prefix observations are keyed by (prefill instance, class), so a
        # class may occur multiple times.  LP variables are keyed by class and
        # P/D pair; aggregate those rows before constructing variables instead
        # of silently letting the last row overwrite the demand.
        grouped, p_work = self._aggregate_rows(rows, prefill, work_overrides)
        flows = {}
        work = {}
        for class_id, row in grouped.items():
            demand = max(float(row["arrival_rate_ewma"]),
                         self.config.class_demand_floor_rps)
            work[class_id] = demand
            for p_sched in prefill:
                for d_sched in decode:
                    flows[class_id, p_sched.instance_id, d_sched.instance_id] = solver.NumVar(
                        0.0, solver.infinity(), f"f_{len(flows)}")
        for class_id, demand in work.items():
            solver.Add(sum(flows[class_id, p.instance_id, d.instance_id]
                           for p in prefill for d in decode) == demand)
        p_slack = {}
        for p_sched in prefill:
            p_slack[p_sched.instance_id] = solver.NumVar(0.0, solver.infinity(),
                                                          f"overflow_p_{p_sched.instance_id}")
            cap = max(1e-6, self.config.prefill_capacity.get(
                p_sched.instance_id, float(p_sched.max_num_seqs)))
            # Overflow is priced as a *fraction* of capacity.  Pricing raw units
            # makes the slack dominate the objective (a byte/s of link overflow
            # is ~1e8 times a per-request compute cost), so the LP chases
            # microscopic overflow reductions with large class migrations that
            # throw away warmed prefixes.
            solver.Add(sum(flows[class_id, p_sched.instance_id, d.instance_id] * p_work[p_sched.instance_id, class_id]
                           for class_id in work for d in decode) / cap
                       <= 1.0 + p_slack[p_sched.instance_id])
        d_slack = {}
        for d_sched in decode:
            d_slack[d_sched.instance_id] = solver.NumVar(0.0, solver.infinity(),
                                                          f"overflow_d_{d_sched.instance_id}")
            cap = max(1e-6, self.config.decode_capacity.get(
                d_sched.instance_id, float(d_sched.max_num_seqs)))
            solver.Add(sum(flows[class_id, p.instance_id, d_sched.instance_id]
                           * self.decode_work(class_id)
                           for class_id in work for p in prefill) / cap
                       <= 1.0 + d_slack[d_sched.instance_id])
        link_slack = {}
        for link in self.config.shared_links:
            link_slack[link.link_id] = solver.NumVar(0.0, solver.infinity(),
                                                      f"overflow_link_{link.link_id}")
            link_cap = max(1e-6, float(self._link_capacity(link)))
            solver.Add(sum(flows[class_id, p.instance_id, d.instance_id] *
                           (self._class_kv_bytes(class_id, grouped[class_id])
                            if self._link_uses_bytes(link) else 1.0)
                           for class_id in work for p in prefill for d in decode
                           if link.carries(p.instance_id, d.instance_id)) / link_cap
                       <= 1.0 + link_slack[link.link_id])
        objective = solver.Objective()
        for (class_id, p_id, d_id), variable in flows.items():
            p_sched = next(item for item in prefill if item.instance_id == p_id)
            d_sched = next(item for item in decode if item.instance_id == d_id)
            cost = self._pair_cost(p_sched, d_sched, class_id, grouped[class_id],
                                   p_work[p_id, class_id])
            objective.SetCoefficient(variable, cost)
        if os.environ.get("SIM_LP_DEBUG"):
            # Why did the plan pick that pair?  The LP minimises
            # sum(flow * pair_cost), so dumping a class's cheapest edges
            # separates "a genuine cost difference" from "a degenerate LP whose
            # simplex vertex is arbitrary" (measured 2026-09-16: the plan
            # single-homed onto a Prefill that was not the cheapest edge).
            sample = next(iter(grouped), None)
            if sample is not None:
                ranked = sorted(
                    (self._pair_cost(p, d, sample, grouped[sample],
                                     p_work[p.instance_id, sample]),
                     p.instance_id, d.instance_id)
                    for p in prefill for d in decode)
                print("[lp] class=" + sample[:44] + " cheapest="
                      + ", ".join(f"p{p}/d{d}={c:.4f}" for c, p, d in ranked[:5]),
                      flush=True)
        congestion = self._congestion_variables(solver, prefill, decode, flows, work, p_work)
        for variable, coefficient in congestion:
            objective.SetCoefficient(variable, coefficient)
        for variable in (*p_slack.values(), *d_slack.values(), *link_slack.values()):
            objective.SetCoefficient(variable, self.config.overflow_penalty)
        objective.SetMinimization()
        if solver.Solve() != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("CASR LP did not find an optimal solution")
        assignments = []
        for (class_id, p_id, d_id), variable in flows.items():
            value = variable.solution_value()
            if value <= 1e-9:
                continue
            carried = tuple(link.link_id for link in self.config.shared_links if link.carries(p_id, d_id))
            assignments.append(FlowAssignment(class_id, p_id, d_id, value,
                                              objective.Value() / max(1.0, work[class_id]), carried))
        assignments = self._collapse_low_demand(assignments, grouped, prefill, decode)
        assignments = self._cap_class_footprint(assignments, grouped, prefill, decode)
        assignments = self._enforce_egress_budget(assignments, grouped, prefill, decode)
        self.diagnostics = {
            "backend": self.backend,
            "objective": objective.Value(),
            # Offered load the LP actually solved for, against the capacity it
            # can place it on.  ``total_demand_rps`` far above
            # ``total_capacity_rps`` means the plan is structurally infeasible
            # and every "overflow" number below is an artefact of the demand
            # estimate, not of the real load.
            "total_demand_rps": sum(work.values()),
            "total_capacity_rps": sum(
                max(0.0, float(self.config.prefill_capacity.get(
                    s.instance_id, getattr(s, "max_num_seqs", 0.0))))
                for s in prefill),
            "class_demand_floor_rps": self.config.class_demand_floor_rps,
            "prefill_overflow": {key: value.solution_value() for key, value in p_slack.items()},
            "decode_overflow": {key: value.solution_value() for key, value in d_slack.items()},
            "link_overflow": {key: value.solution_value() for key, value in link_slack.items()},
            "use_cache_capacity": self.config.use_cache_capacity,
            "network_weight": self.config.network_weight,
            "utilization_weight": self.config.utilization_weight,
            "utilization_segments": self.config.utilization_segments,
            "compute_weight": self.config.compute_weight,
            "queue_weight": self.config.queue_weight,
            "tail_weight": self.config.tail_weight,
            "work": self._work_diagnostics(p_work),
            "single_homed_classes": self._last_single_homed,
            "capped_classes": len(self._last_capped),
            "ttft_slo_ms": self.config.ttft_slo_ms,
            "slo_penalty": self.config.slo_penalty,
            "slo_violating_pairs": sorted(self._slo_violations),
        }
        return assignments

    def _class_link_bytes(self, class_id, grouped, link):
        if not self._link_uses_bytes(link):
            return 1.0
        return self._class_kv_bytes(class_id, grouped[class_id])

    def _enforce_egress_budget(self, assignments, grouped, prefill, decode):
        """Move class flow off producers whose KV egress budget is exhausted.

        The LP prices the producer-side push through a *soft* slack variable, and
        a single class's flow is far too small for that slack to show: the
        overflow only appears once a few hundred classes land on the same
        producer.  Measured 2026-09-16 in the six-domain arena -- the LP's own
        flows reported ``link_overflow 0`` while the A100's push link was
        overrun ~11x in aggregate, 284 of 376 requests landed there, and the
        p95 was 37 s against the best arm's 1.3 s.  Neither disabling the
        low-demand single-homing (byte-identical result) nor raising
        ``overflow_penalty`` 100x (284 -> 280 requests) moved it.

        So: after the LP, walk the classes sitting on an over-budget link and
        move the cheapest-to-move ones to the cheapest pair that still has room,
        in the same spirit as ``_cap_class_footprint`` does for the prefix
        working set.  Whole classes move, because the router samples one
        instance per request and a fractional rescale would not change what the
        timeline does.
        """
        self._last_egress_moves = []
        if not assignments:
            return assignments
        links = [link for link in self.config.shared_links if self._link_uses_bytes(link)]
        if not links:
            return assignments
        prefills = {int(s.instance_id): s for s in prefill}
        decodes = {int(s.instance_id): s for s in decode}
        by_class = {}
        for item in assignments:
            by_class.setdefault(item.class_id, []).append(item)

        def bytes_of(item, link):
            return item.flow * self._class_link_bytes(item.class_id, grouped, link)

        for _sweep in range(4):
            load = {link.link_id: 0.0 for link in links}
            for item in assignments:
                for link in links:
                    if link.carries(item.prefill_id, item.decode_id):
                        load[link.link_id] += bytes_of(item, link)
            over = [link for link in links
                    if load[link.link_id] > self._link_capacity(link) + 1e-9]
            if os.environ.get("EGRESS_DEBUG"):
                print("[egress] " + " ".join(
                    f"{link.link_id}={load[link.link_id]/1e6:.1f}/"
                    f"{self._link_capacity(link)/1e6:.1f}MB/s" for link in links),
                    f"over={[link.link_id for link in over]}", flush=True)
            if not over:
                break
            over.sort(key=lambda link: load[link.link_id] - self._link_capacity(link),
                      reverse=True)
            moved_any = False
            for link in over:
                for class_id, items in sorted(by_class.items(),
                                              key=lambda kv: -sum(i.flow for i in kv[1])):
                    if load[link.link_id] <= self._link_capacity(link) + 1e-9:
                        break
                    mine = [item for item in items
                            if link.carries(item.prefill_id, item.decode_id)]
                    if not mine:
                        continue
                    entry = grouped.get(class_id) or {}
                    best = None
                    for candidate_p in prefills.values():
                        for candidate_d in decodes.values():
                            if candidate_p.instance_id == mine[0].prefill_id and \
                                    candidate_d.instance_id == mine[0].decode_id:
                                continue
                            spare = True
                            for other in links:
                                if not other.carries(candidate_p.instance_id,
                                                     candidate_d.instance_id):
                                    continue
                                extra = sum(bytes_of(item, other) for item in mine)
                                if load[other.link_id] + extra > self._link_capacity(other) + 1e-9:
                                    spare = False
                                    break
                            if not spare:
                                continue
                            cost = self._pair_cost(candidate_p, candidate_d, class_id,
                                                   entry)
                            if best is None or cost < best[0]:
                                best = (cost, candidate_p.instance_id,
                                        candidate_d.instance_id)
                    if best is None:
                        continue
                    _, new_p, new_d = best
                    moved = []
                    for item in mine:
                        for other in links:
                            if other.carries(item.prefill_id, item.decode_id):
                                load[other.link_id] = max(
                                    0.0, load[other.link_id] - bytes_of(item, other))
                        moved.append(dataclass_replace(item, prefill_id=new_p,
                                                       decode_id=new_d))
                    for item in moved:
                        for other in links:
                            if other.carries(item.prefill_id, item.decode_id):
                                load[other.link_id] += bytes_of(item, other)
                    by_class[class_id] = [item for item in items if item not in mine] + moved
                    self._last_egress_moves.append({
                        "class_id": class_id, "from": mine[0].prefill_id,
                        "to": new_p, "decode": new_d,
                        "link": link.link_id})
                    moved_any = True
            if not moved_any:
                break
        out = [item for items in by_class.values() for item in items]
        out.sort(key=lambda item: (item.prefill_id, item.decode_id, item.class_id))
        return out

    def _cap_class_footprint(self, assignments, grouped, prefill, decode):
        """Keep each Prefill's prefix working set at or below its cap.

        The LP prices cache *hits* per ``(class, Prefill)`` but has no notion of
        the working set, so it can pile every class onto the cheapest Prefill
        and pay for it in evictions.  This moves the classes that are cheapest
        to move (smallest increase in pair cost) until every Prefill is under
        ``prefill_class_limit``.  It is a heuristic stand-in for an exact
        fixed-charge formulation, which would need binaries and a MILP backend.
        """
        limit = int(self.config.prefill_class_limit or 0)
        self._last_capped = []
        if limit <= 0 or not assignments:
            return assignments
        prefills = {s.instance_id: s for s in prefill}
        decodes = {s.instance_id: s for s in decode}
        owners = {}
        for item in assignments:
            owners.setdefault(item.class_id, []).append(item)
        held = {}
        for class_id, items in owners.items():
            for item in items:
                held.setdefault(item.prefill_id, set()).add(class_id)
        if all(len(classes) <= limit for classes in held.values()):
            return assignments

        moved = {}
        for p_id, classes in sorted(held.items()):
            excess = len(classes) - limit
            if excess <= 0:
                continue
            candidates = []
            for class_id in sorted(classes):
                entry = grouped[class_id]
                decode_id = owners[class_id][0].decode_id
                decode_sched = decodes.get(decode_id)
                here = self._pair_cost(prefills[p_id], decode_sched, class_id, entry) \
                    if decode_sched is not None else 0.0
                best = None
                for q_id, q_sched in sorted(prefills.items()):
                    if q_id == p_id:
                        continue
                    if len(held.get(q_id, ())) >= limit:
                        continue
                    there = self._pair_cost(q_sched, decode_sched, class_id, entry) \
                        if decode_sched is not None else 0.0
                    if best is None or there < best[0]:
                        best = (there, q_id)
                if best is not None:
                    candidates.append((best[0] - here, class_id, best[1]))
            # Cheapest to move first: these lose the least cache affinity.
            for _penalty, class_id, q_id in sorted(candidates)[:excess]:
                moved[class_id] = q_id
                held[p_id].discard(class_id)
                held.setdefault(q_id, set()).add(class_id)
                self._last_capped.append((class_id, p_id, q_id))

        if not moved:
            return assignments
        capped = []
        for item in assignments:
            target = moved.get(item.class_id)
            if target is None:
                capped.append(item)
                continue
            capped.append(dataclass_replace(item, prefill_id=target))
        return capped

    def _collapse_low_demand(self, assignments, grouped, prefill, decode):
        """Single-home classes whose demand cannot amortise a second edge.

        The LP is frequently degenerate for a low-demand class: several
        ``(prefill, decode)`` edges have identical cost, so the vertex simplex
        returns can split the class across two Prefills.  Each extra Prefill
        costs one cold prefill for that class, which is pure tail latency when
        the class only has a couple of requests.  Merging the split onto the
        Prefill the LP liked best removes that cost, and the merge is skipped
        when it would push the target Prefill or one of its links over budget.
        """
        self._last_single_homed = []
        threshold = self.config.single_home_below_rps
        if threshold <= 0.0 or not assignments:
            return assignments
        p_cap = {s.instance_id: max(1.0, float(self.config.prefill_capacity.get(
            s.instance_id, getattr(s, "max_num_seqs", 1.0)))) for s in prefill}
        p_load, link_load = {}, {}
        for item in assignments:
            p_load[item.prefill_id] = p_load.get(item.prefill_id, 0.0) + item.flow
            for link in self.config.shared_links:
                if link.carries(item.prefill_id, item.decode_id):
                    link_load[link.link_id] = (link_load.get(link.link_id, 0.0) + item.flow *
                                               self._class_link_bytes(item.class_id, grouped, link))
        by_class = {}
        for item in assignments:
            by_class.setdefault(item.class_id, []).append(item)
        kept = []
        for class_id, items in by_class.items():
            demand = float(grouped[class_id].get("arrival_rate_ewma", 0.0))
            by_prefill = {}
            for item in items:
                by_prefill.setdefault(item.prefill_id, []).append(item)
            if demand >= threshold or len(by_prefill) <= 1:
                kept.extend(items)
                continue
            totals = {p_id: sum(item.flow for item in rows)
                      for p_id, rows in by_prefill.items()}
            target = max(totals, key=lambda p_id: (totals[p_id], -p_id))
            moved = [item for p_id, rows in by_prefill.items() if p_id != target
                     for item in rows]
            if not moved:
                kept.extend(items)
                continue
            added = sum(item.flow for item in moved)
            if p_load.get(target, 0.0) + added > p_cap.get(target, float("inf")) + 1e-9:
                kept.extend(items)
                continue
            overflow = False
            for item in moved:
                for link in self.config.shared_links:
                    if not link.carries(target, item.decode_id):
                        continue
                    projected = (link_load.get(link.link_id, 0.0) + item.flow *
                                 self._class_link_bytes(class_id, grouped, link))
                    if projected > self._link_capacity(link) + 1e-9:
                        overflow = True
                        break
                if overflow:
                    break
            if overflow:
                kept.extend(items)
                continue
            for item in moved:
                for link in self.config.shared_links:
                    if link.carries(item.prefill_id, item.decode_id):
                        link_load[link.link_id] = (link_load.get(link.link_id, 0.0) -
                                                   item.flow *
                                                   self._class_link_bytes(class_id, grouped, link))
            merged = {}
            for item in moved:
                key = (target, item.decode_id)
                merged[key] = merged.get(key, 0.0) + item.flow
            for (target_id, decode_id), flow in merged.items():
                kept.append(FlowAssignment(
                    class_id, target_id, decode_id, flow,
                    items[0].cost if items else 0.0,
                    tuple(link.link_id for link in self.config.shared_links
                          if link.carries(target_id, decode_id))))
            kept.extend(row for row in by_prefill[target])
            p_load[target] = p_load.get(target, 0.0) + added
            for item in moved:
                for link in self.config.shared_links:
                    if link.carries(target, item.decode_id):
                        link_load[link.link_id] = (link_load.get(link.link_id, 0.0) + item.flow *
                                                   self._class_link_bytes(class_id, grouped, link))
            self._last_single_homed.append(class_id)
        return kept

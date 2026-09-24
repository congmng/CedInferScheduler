import bisect
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace
from .logger import get_logger
from .block_pool import NONE_HASH


_MAX_PLAN_ASSIGNMENTS = 200_000


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            prefix_profiler=None,
            name_decode_at_arrival=False,
            policy_options=None,
            casr_enabled=False,
            client_concurrency=0,
            placement_override=None,
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.prefix_profiler = prefix_profiler
        self.affinity_plan = None
        #: Deficit counters for ``_select_weighted``, keyed by
        #: ``(assignment_key, instance_id)``.  They are **not** reset when a new
        #: plan is installed: the aggregate split has ~1.6 arrivals per control
        #: tick at the 16 rps peak, and restarting the balance each tick sends
        #: every one of them to the highest-weight worker.
        self._plan_assignments = defaultdict(int)
        # P/D handoffs a Decode refused because its KV pool was full; retried
        # on the next transfer call (see ``transfer_prefill_request``).
        self._pending_handoffs = []
        self._counters = {}
        # Aggregate Prefill split of the installed plan; used for classes the
        # plan cannot name (see ``install_affinity_plan``).
        self._plan_prefill_totals = {}
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        # CLI fallback for traces that carry no per-request SLO; the trace row
        # wins, exactly like the real client's ``row_slo()``.
        self._cli_slo_ttft_ms = None
        self._cli_slo_tpot_ms = None
        # -- Deadline-aware Decode placement --------------------------------
        # ``load`` ranks a Decode by *how full* it is; the same fullness costs
        # 46 ms per step on a 5090 and 372 ms on a 4090, so a mostly-empty slow
        # Decode looks cheap while it drains for seconds.  Measured on the
        # 2026-09-23 peak trace: the plan named the fast Decode, the router's
        # fallback (once that one stopped accepting) handed 8-11% of the
        # requests to 2-3x slower ones, and p95 went 18.8 s -> 78-97 s.  With
        # this on, the fallback minimises the *expected drain time*
        # (queue_fraction x this instance's step time), skipping candidates
        # whose wait already exceeds the request's budget when a compliant one
        # exists.  Off by default: it changes routing, so it must be an arm.
        options = policy_options or {}
        #: Kept so a post-hoc component (the migration arm) can read the same
        #: knobs the router was built with instead of re-parsing the config.
        self.policy_options = dict(options)
        self.deadline_aware_decode = bool(options.get("deadline_aware_decode", False))
        self.decode_reference_tokens = max(
            1.0, float(options.get("decode_reference_tokens", 16) or 16))
        self.decode_service_ms = {}
        for key, value in (options.get("decode_service_ms") or {}).items():
            try:
                self.decode_service_ms[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
        self.tail_debug = bool(os.environ.get("TAIL_SELECT_DEBUG"))
        # Only the domain-aware cluster configs name the Decode half of every
        # pair up front; see ``_decode_instance_id_for``.
        self.name_decode_at_arrival = name_decode_at_arrival
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0
        # ``inflight`` in the real router counts the requests *dispatched to* an
        # instance that have not completed yet -- it spans the whole request
        # lifetime, not just the leg that instance serves.  ``_pick_load`` ranks
        # by ``(inflight + 1) / capacity``, which is why the recorded runs pin
        # the Prefill on the largest-capacity host even while every request is
        # served locally.  Keeping the same counter is what makes a replay land
        # on the same instances (measured 2026-09-16, forced-transfer arm:
        # real 104/120 on p4090 against a flat spread here).
        self._assigned = defaultdict(int)
        self._request_pair = {}
        # P/D handoff vs local recompute.  The real router decides this per
        # request (``disagg_router.kv_exchange_decision``): moving KV costs
        # 585 ms (same host) / 1351 ms (cross host, +48 ms fixed) per 1000
        # prompt tokens while the Decode prefilling the prompt itself costs
        # ~93 ms/1k.  The simulator used to *always* hand the KV over, which
        # put it on the egress budget instead of on compute and made short
        # prompts 25x slower than the measured cluster (2026-09-16: the real
        # ``load`` arm answered 200/200 requests on the local path).
        # ``never`` (the default here) keeps the historical behaviour so
        # existing configs and comparisons are unchanged; the deployment-derived
        # configs opt in with ``casr.local_prefill: auto``.
        options = dict(policy_options or {})
        self.local_prefill_mode = str(options.get("local_prefill", "never")).lower()
        # Per-instance capacity from the deployment-derived config.  The real
        # router's ``load`` policy ranks candidates by
        # ``(inflight + 1) / capacity``; the simulator normalized by
        # ``max_num_seqs`` instead, which is the *same* number for every
        # instance and therefore degenerates into round-robin.  Measured
        # 2026-09-16 on the 3-domain run: the real Decode split was
        # 112/73/15 (proportional to 230/191/90), the simulator's 98/47/55.
        self.capacity_tables = {
            "prefill": {int(key): float(value) for key, value in
                        (options.get("prefill_capacity") or {}).items()},
            "decode": {int(key): float(value) for key, value in
                       (options.get("decode_capacity") or {}).items()},
        }
        # The real router picks the Decode of a CASR policy by *cost*
        # (``disagg_router._pick_decode_cost``: network + service + utilisation)
        # rather than by load, which is why its ``casr_lp`` arm concentrated all
        # 200 requests on the fastest Decode while its ``load`` arm spread them
        # by capacity (measured 2026-09-16, 3-domain run: 200/200 vs 112/73/15).
        # Without this the simulator's local-recompute path made ``casr_lp``
        # bit-identical to ``load``.
        self.casr_enabled = bool(casr_enabled)
        # The recorded comparisons drive the cluster with a *closed-loop*
        # client: ``real_dataset_client.py`` keeps at most ``--concurrency``
        # requests in flight, so a slow system throttles its own arrival stream.
        # Replaying a trace open-loop instead piles up every arrival the trace
        # ever scheduled: measured 2026-09-16 on the 300-request forced-transfer
        # arm, the cluster's TTFT settled at 4.7-5.5 s while the simulator's grew
        # linearly to 42 s, and the resulting 12x E2E gap was mostly this, not a
        # timing-model error.  ``0`` keeps the historical open-loop replay.
        self.client_concurrency = max(0, int(client_concurrency or 0))
        self._in_flight = 0
        # Recorded placement, keyed by the trace's request index:
        # ``{index: (prefill_instance_id, decode_instance_id)}``.  Feeding the
        # *cluster's* per-request placement into the simulator separates the two
        # questions an alignment run has to answer at once -- "does the
        # controller pick the same instances" and "does the model execute a
        # given placement at the same speed" -- so a residual gap can be
        # attributed instead of guessed.
        self.placement_override = {int(k): tuple(v)
                                   for k, v in (placement_override or {}).items()}
        self.decode_service_ms = {int(key): float(value) for key, value in
                                  (options.get("decode_service_ms") or {}).items()}
        self.prefill_service_ms = {int(key): float(value) for key, value in
                                   (options.get("prefill_service_ms") or {}).items()}
        # The real control loop's cost weights (``weights`` in
        # router_config.json): network, service, and each leg's load.
        self.cost_weights = dict(options.get("weights") or {})
        # Set by the entry point once the P/D handoff link exists: it is what
        # tells this router how much KV is already queued on a candidate pair.
        self.pd_link = None
        self.pair_rtt_ms = {}
        self.pair_bandwidth = {}
        for key, value in (options.get("pair_costs") or {}).items():
            if isinstance(key, str):
                source, target = (int(part) for part in
                                  key.replace("/", ",").split(",", 1))
            else:
                source, target = int(key[0]), int(key[1])
            self.pair_rtt_ms[(source, target)] = float(
                (value or {}).get("rtt_ms", 0.0))
            self.pair_bandwidth[(source, target)] = float(
                (value or {}).get("bandwidth_bytes_per_s", 0.0) or 0.0)
        # The fast router's own load denominator (``_pick_load``), which is the
        # deployment's ``capacity`` field rather than the LP's solver capacity.
        # Prefer it when the config carries it so a replay picks the same
        # instance the recorded run did.
        self.router_capacity = {int(key): float(value) for key, value in
                                (options.get("router_capacity") or {}).items()}
        self.local_prefill_ms_per_1k = float(
            options.get("local_prefill_ms_per_1k", 93.0) or 93.0)
        self.transfer_ms_per_1k_local = float(
            options.get("transfer_ms_per_1k_local", 585.0) or 585.0)
        self.transfer_ms_per_1k_cross = float(
            options.get("transfer_ms_per_1k_cross", 1351.0) or 1351.0)
        self.transfer_fixed_ms_cross = float(
            options.get("transfer_fixed_ms_cross", 48.0) or 48.0)
        self.kv_bytes_per_token = float(options.get("kv_bytes_per_token", 0.0) or 0.0)
        # Price moving KV from the *cluster's own* link model instead of the
        # deployment's measured constants.  The hardcoded defaults (93 ms of
        # local recompute per 1k tokens, 585 / 1351 ms to move them) describe
        # the real fabric with Qwen3-8B's 147 KB/token; on a simulated cluster
        # with a different KV size or a different link they are simply wrong in
        # direction: P-15B's 16.96 KB/token over this fabric costs ~10 ms to
        # move against ~290 ms to recompute, so the decision should be
        # "transfer", while the old constants always said "recompute".  Turning
        # this off restores the recorded behaviour for A/B.
        self.link_derived_pricing = bool(
            options.get("link_derived_pricing", True))
        self.capacity_reference_tokens = float(
            options.get("capacity_reference_tokens", 1024) or 1024)
        # Producer-side KV push ceiling (GB/s), the deployment's measured
        # NixlConnector rate.  It is what the KV-aware arm scores against.
        self.kv_egress_gbps = float(options.get("kv_egress_gbps", 0.0) or 0.0)
        self.local_prefill_queue_weight = float(
            options.get("local_prefill_queue_weight", 1.0) or 0.0)
        self.local_prefill_queue_cap = float(
            options.get("local_prefill_queue_cap", 3.0) or 0.0)

        # -- PrfaaS-style cross-DC Prefill offload --------------------------
        # "Run the Prefill in another data centre" is only safe behind a
        # transfer budget: the KV of a long prompt is what pays for it.  The
        # arm keeps the *local* pool until it is loaded past
        # ``prfaas_local_load_threshold``, then offloads to whichever producer
        # minimises max(transfer, compute wait) -- and refuses any pairing whose
        # transfer alone exceeds ``prfaas_max_offload_ms`` (0 = no gate).
        self.prfaas_local_load_threshold = float(
            options.get("prfaas_local_load_threshold", 0.5) or 0.0)
        self.prfaas_max_offload_ms = float(
            options.get("prfaas_max_offload_ms", 300.0) or 0.0)

        # -- Phase sharing (semi-PD style) ----------------------------------
        # semi-PD's claim is that the Prefill and Decode phases do not have to
        # own disjoint machines: under pressure, a request's Prefill phase can
        # run on the Decode that will generate its tokens ("unified resource
        # management").  This simulator has one knob for that -- the
        # local-recompute path, which makes the Decode run the prompt itself --
        # so the arm forces it once the chosen Prefill is past this load
        # (0 = off, the deployment's own cost comparison decides).
        self.prefill_overflow_to_decode = float(
            options.get("prefill_overflow_to_decode", 0.0) or 0.0)

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._is_init = True

        # Agentic session dependency tracking
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._next_request_id = 0        # monotonic counter for unique request IDs

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy == "CACHE_AWARE":
            # SGLang Router's ``cache_aware`` policy: longest prefix match wins,
            # and traffic overflows to the least-loaded instance once the
            # matched one is busy.  This is the SOTA *routing* baseline the real
            # cluster compares against, and it was missing here.
            self._select_instance = self._least_load_select
        elif self.routing_policy == "KV_AWARE":
            # The same least-loaded family, scored on the resource that
            # actually binds the Prefill here: its KV push, not its engine.
            self._select_instance = self._least_load_select
        elif self.routing_policy == "LOAD_RR":
            # ``load`` with a round-robin tie-break: separates "the score has
            # no signal" from "our tie-break pinned everything to id 0".
            self._select_instance = self._least_load_rr_select
        elif self.routing_policy == "PRFAAS":
            # Cross-DC Prefill offload (see ``_prfaas_select``).  Decode and any
            # other role keep the ``load`` semantics, so the arm differs from
            # ``load`` in exactly one decision.
            self._select_instance = self._least_load_select
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._custom_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, CUSTOM")
        self.logger = get_logger(self.__class__)

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, role):
        return self.decode_rr_counter if role == "decode" else self.prefill_rr_counter

    # -- deadline-aware Decode placement (see __init__) ---------------------

    def decode_step_ms(self, scheduler) -> float:
        """One decode step on this instance, in ms (0 when uncalibrated)."""
        table = getattr(self, "decode_service_ms", {}) or {}
        step = table.get(int(scheduler.instance_id))
        if step is None:
            step = float(getattr(scheduler, "service_ms", 0.0) or 0.0)
        return step

    def expected_decode_wait_ms(self, scheduler, output_tokens) -> float:
        """Milliseconds a request would spend queued on this Decode.

        ``(4 x waiting + running) / max_num_seqs`` is the same queue fraction
        the solver prices; multiplying it by *this* instance's step time (and
        by the class's output-length multiplier) turns it into the time it
        actually takes to drain, which is what a slow Decode was hiding.
        """
        step_ms = self.decode_step_ms(scheduler)
        if step_ms <= 0.0:
            return 0.0
        reference = max(1.0, float(getattr(self, "decode_reference_tokens", 16) or 16))
        multiplier = max(1.0, float(output_tokens or 0) / reference)
        queue_fraction = ((4 * len(getattr(scheduler, "waiting", ()))
                           + len(getattr(scheduler, "running", ())))
                          / max(1, getattr(scheduler, "max_num_seqs", 1)))
        return queue_fraction * step_ms * multiplier

    def _select_decode(self, eligible, req_data=None):
        """Index into ``eligible`` for the Decode of this request.

        Falls through to the active routing policy when the deadline-aware
        placement is off, so every existing arm keeps its exact behaviour.
        """
        if not getattr(self, "deadline_aware_decode", False) or not eligible:
            return self._select_instance(eligible, "decode")
        output_tokens = 0
        budget = None
        if req_data is not None:
            try:
                output_tokens = int(req_data.get("output_toks", 0)) - int(
                    req_data.get("input_toks", 0))
            except (TypeError, ValueError):
                output_tokens = 0
            budget = req_data.get("slo_ttft_ms")
        scored = [(self.expected_decode_wait_ms(sched, output_tokens),
                   int(sched.instance_id), index)
                  for index, sched in enumerate(eligible)]
        if budget:
            try:
                budget = float(budget)
            except (TypeError, ValueError):
                budget = None
        pool = scored
        if budget:
            compliant = [item for item in scored if item[0] <= budget]
            pool = compliant or scored
        wait_ms, instance_id, index = min(pool, key=lambda item: (item[0], item[1]))
        if getattr(self, "tail_debug", False):
            print("[decode-select] " + " ".join(
                f"d{instance}:{wait:.1f}ms" for wait, instance, _ in scored)
                + f" budget={budget} -> d{instance_id}", flush=True)
        return index

    def _set_counter(self, role, value):
        if role == "decode":
            self.decode_rr_counter = value
        else:
            self.prefill_rr_counter = value

    def _rr_select(self, schedulers, role):
        num_instances = len(schedulers)
        idx = self._get_counter(role) % num_instances
        self._set_counter(role, idx + 1)
        return idx

    def _rand_select(self, schedulers, role):
        return self._rnd.randrange(len(schedulers))

    def _least_load_select(self, schedulers, role):
        """vLLM-style least-loaded routing, normalized by instance capacity.

        Exactly the real router's ``_pick_load``: ``(inflight + 1) / capacity``
        with the instance id as the tie-break.  The simulator used to walk the
        candidates from a rotating cursor and keep the first strict minimum,
        which turns every *tie* into a round-robin -- measured 2026-09-16 on
        the paced trace, ``load`` spread the Prefills 160/80 (the capacity
        ratio between p5090 and p3090a) where the cluster put 240/240 on p5090.

        ``inflight`` counts requests dispatched to this instance and not
        finished yet, and the Prefill's slot is given back as soon as that leg
        returns (``release_prefill_leg``, called by the entry point when the
        Prefill batch finishes) -- the real router releases
        ``prefill.inflight`` in the dispatch handler's ``finally``.  It
        deliberately does *not* read the instance's own queue: the recorded
        ``load`` arm pins the largest-capacity host while the local-recompute
        path is serving every request elsewhere, so a queue-aware score sends a
        replay somewhere the real router never went.
        """
        if not schedulers:
            return 0
        table = getattr(self, "capacity_tables", {}).get(role, {})
        router_table = getattr(self, "router_capacity", {})
        assigned = getattr(self, "_assigned", None)

        def score(sched):
            if assigned is not None:
                inflight = assigned.get(int(sched.instance_id), 0)
            else:
                inflight = len(sched.waiting) * 4 + len(sched.running)
            capacity = (router_table.get(int(sched.instance_id))
                        or table.get(int(sched.instance_id))
                        or getattr(sched, "max_num_seqs", 0))
            if capacity not in (0, float('inf')):
                return ((inflight + 1) / capacity, int(sched.instance_id))
            return (float(inflight + 1), int(sched.instance_id))

        best = min(schedulers, key=score)
        best_idx = schedulers.index(best)
        if os.environ.get("LOAD_DEBUG"):
            print(f"[load-select:{role}] "
                  + " ".join(f"{s.instance_id}:{score(s)[0]:.6f}" for s in schedulers)
                  + f" -> {best.instance_id}", flush=True)
        self._set_counter(role, (best_idx + 1) % len(schedulers))
        return best_idx

    def _least_load_rr_select(self, schedulers, role):
        """``load`` with a round-robin tie-break instead of the instance id.

        ``_least_load_select`` ranks by ``((inflight+1)/capacity, instance_id)``,
        so whenever the score stops discriminating the lowest id wins every
        time.  The score stops discriminating exactly when the engine is fast
        enough that ``inflight`` never accumulates: measured 2026-09-17 on
        Zamba2-1.2B (31 ms Prefill), all 735 requests of the arena trace landed
        on instance 0 while Qwen3-8B (185 ms Prefill) spread 375/360.

        This arm keeps the score and changes *only* the tie-break, so a skew
        that survives it is not a tie-break artifact.  That separation is the
        point: the review of the paper draft asked whether 735/0 is a property
        of the load metric or of our deterministic tie-breaking.
        """
        if not schedulers:
            return 0
        table = getattr(self, "capacity_tables", {}).get(role, {})
        router_table = getattr(self, "router_capacity", {})
        assigned = getattr(self, "_assigned", None)

        def score(sched):
            if assigned is not None:
                inflight = assigned.get(int(sched.instance_id), 0)
            else:
                inflight = len(sched.waiting) * 4 + len(sched.running)
            capacity = (router_table.get(int(sched.instance_id))
                        or table.get(int(sched.instance_id))
                        or getattr(sched, "max_num_seqs", 0))
            if capacity not in (0, float('inf')):
                return (inflight + 1) / capacity
            return float(inflight + 1)

        scores = [score(sched) for sched in schedulers]
        best_score = min(scores)
        tied = {index for index, value in enumerate(scores)
                if value <= best_score + 1e-12}
        start = self._get_counter(role) % len(schedulers)
        chosen = next((start + step) % len(schedulers)
                      for step in range(len(schedulers))
                      if (start + step) % len(schedulers) in tied)
        self._set_counter(role, (chosen + 1) % len(schedulers))
        return chosen

    def _custom_select(self, schedulers, role):
        raise NotImplementedError("Implement custom routing policy.")

    # -- SGLang-style cache-aware routing (prefill side) -------------------

    #: An instance whose normalized load reaches this value is "busy"; a
    #: matching-but-busy instance loses to a slightly worse match that is idle.
    CACHE_AWARE_OVERFLOW = 0.75

    def _prefix_hit_blocks(self, sched, req_data):
        """How many leading blocks of this request the instance already holds.

        Mirrors ``request_block_hashes`` (same chained hash) but stops at the
        first miss, so it is a routing-time probe rather than an allocation.
        """
        kv = getattr(sched, "kv", None)
        if kv is None or not getattr(kv, "enable_caching", False):
            return 0
        tokens = req_data.get("input_hash_ids") or []
        block_size = int(getattr(kv, "block_size", 0) or 0)
        if not tokens or block_size <= 0:
            return 0
        parent = NONE_HASH
        hits = 0
        for start in range(0, len(tokens) - block_size + 1, block_size):
            parent = hash((parent, tuple(tokens[start:start + block_size])))
            if kv.npu_pool.get_cached_block(parent) is None:
                break
            hits += 1
        return hits

    def _cache_aware_select(self, schedulers, role, req_data=None):
        """Longest prefix match with load-based overflow (SGLang semantic).

        Mirrors the real router's ``_pick_cache_aware`` exactly, including its
        tie-break: rank the owners of the longest matching prefix by the load
        this request would create (``inflight + 1`` over capacity), and only
        spill to the least-loaded instance when the best owner is busy.  The
        earlier version returned the first idle owner in list order, so the
        simulator and the real baseline could disagree on the same input.
        """
        if role != "prefill" or req_data is None:
            return self._least_load_select(schedulers, role)
        hits = [self._prefix_hit_blocks(sched, req_data) for sched in schedulers]
        best = max(hits)
        if best <= 0:
            return self._least_load_select(schedulers, role)

        def queued(idx):
            sched = schedulers[idx]
            return len(sched.waiting) * 4 + len(sched.running)

        owners = [idx for idx, hit in enumerate(hits) if hit == best]
        chosen = min(owners, key=lambda idx: (
            (queued(idx) + 1) / max(1, schedulers[idx].max_num_seqs or 1), idx))
        ratio = queued(chosen) / max(1, schedulers[chosen].max_num_seqs or 1)
        if ratio < self.CACHE_AWARE_OVERFLOW:
            return chosen
        return self._least_load_select(schedulers, role)

    # -- KV-egress-aware routing (what the SOTA routers *would* do) ----------

    def _kv_aware_select(self, schedulers, role, req_data=None, now_ns=0):
        """Least-loaded on the resource that actually binds: compute or egress.

        ``load`` / ``cache_aware`` both score ``(inflight+1)/capacity``, and
        that capacity is the *engine's* requests/s.  On a P/D-disaggregated
        deployment the Prefill's real ceiling is usually the KV push
        (``kv_egress_gbps``), not the engine: on the pack-1 arena a 1250-token
        Zamba2 handoff is 157 MB against a 0.26 GB/s producer, i.e. ~1.7 req/s
        of egress against 39 req/s of compute.  A compute-only score ties
        across instances (both look idle), the instance-id tie-break fires, and
        every request lands on the first Prefill -- measured 2026-09-17, all
        735 requests on instance 0 with a 419 s mean TTFT.

        This arm is that baseline with the blind spot removed, the way a
        KV-cache-centric router (Mooncake / LMCache-style pair pricing) scores
        a producer: predict the wait for this request's own push -- the bytes
        already queued on the producer's egress plus this handoff's
        serialisation -- and take the better of that and the compute wait
        (``max``, because whichever is larger is the term the request waits on).

        It is deliberately *not* CASR: no affinity plan, no class LP, no
        structural decision.  It exists so the comparison can answer "is the
        gap a missing planner, or just a load metric that scores the wrong
        resource?".
        """
        if not schedulers:
            return 0
        if role != "prefill":
            return self._least_load_select(schedulers, role)

        prompt_tokens = 0
        if req_data:
            prompt_tokens = len(req_data.get("input_hash_ids")
                                or req_data.get("input_tok_ids") or ())
        table = getattr(self, "capacity_tables", {}).get(role, {})
        router_table = getattr(self, "router_capacity", {})
        assigned = getattr(self, "_assigned", None)
        egress_gbps = float(getattr(self, "kv_egress_gbps", 0.0) or 0.0)
        link = getattr(self, "pd_link", None)

        def wait_ns(sched):
            instance_id = int(sched.instance_id)
            inflight = (assigned.get(instance_id, 0) if assigned is not None
                        else len(sched.waiting) * 4 + len(sched.running))
            capacity = (router_table.get(instance_id)
                        or table.get(instance_id)
                        or getattr(sched, "max_num_seqs", 0))
            compute_ns = ((inflight + 1) / capacity * 1e9
                          if capacity not in (0, None, float("inf")) else float(inflight + 1))
            if link is None or egress_gbps <= 0 or prompt_tokens <= 0:
                return (compute_ns, instance_id)
            need_bytes = self.kv_bytes_for(prompt_tokens, sched)
            push_ns = need_bytes / (egress_gbps * 1e9) * 1e9
            queued_ns = float(link.pending_ns(instance_id, now_ns))
            return (max(compute_ns, queued_ns + push_ns), instance_id)

        best = min(schedulers, key=wait_ns)
        return schedulers.index(best)

    @staticmethod
    def _least_loaded(candidates):
        """Choose an eligible scheduler, retaining vLLM-style load scoring."""
        return min(
            candidates,
            key=lambda sched: ((len(sched.waiting) * 4 + len(sched.running)) /
                               (sched.max_num_seqs if sched.max_num_seqs not in (0, float('inf'))
                                else 1), sched.instance_id),
        )

    # -- Cross-DC Prefill offload (PrfaaS-style threshold arm) --------------

    def _instance_load_score(self, role, sched):
        """``(inflight + 1) / capacity`` -- the score every ``load`` arm uses."""
        table = getattr(self, "capacity_tables", {}).get(role, {})
        router_table = getattr(self, "router_capacity", {})
        assigned = getattr(self, "_assigned", None)
        if assigned is not None:
            inflight = assigned.get(int(sched.instance_id), 0)
        else:
            inflight = len(sched.waiting) * 4 + len(sched.running)
        capacity = (router_table.get(int(sched.instance_id))
                    or table.get(int(sched.instance_id))
                    or getattr(sched, "max_num_seqs", 0))
        if capacity in (0, float('inf')):
            return float(inflight + 1)
        return (inflight + 1) / capacity

    def _pair_transfer_ms(self, prefill, decode, tokens, now_ns=0):
        """Milliseconds of KV transfer for this (Prefill, Decode) pair.

        Uses the same two sources as ``kv_exchange_decision``: the cluster's own
        link tables when they exist (``_link_derived_terms``) and the recorded
        constants otherwise, plus what is already queued on the producer.
        """
        derived = self._link_derived_terms(prefill, decode, tokens)
        if derived is not None:
            transfer_ms = float(derived[1])
        else:
            thousands = float(tokens) / 1000.0
            same_node = int(getattr(prefill, "node_id", -1)) == int(
                getattr(decode, "node_id", -2))
            if same_node:
                transfer_ms = self.transfer_ms_per_1k_local * thousands
            else:
                transfer_ms = (self.transfer_fixed_ms_cross
                               + self.transfer_ms_per_1k_cross * thousands)
        link = getattr(self, "pd_link", None)
        if link is not None and now_ns:
            transfer_ms += float(link.pending_ns(
                int(getattr(prefill, "instance_id", -1)), int(now_ns))) / 1e6
        return transfer_ms

    def node_link_cost_ms(self, source, target, tokens):
        """KV copy cost between two *instances' nodes* in milliseconds.

        ``pair_bandwidth`` / ``pair_rtt_ms`` are keyed by (Prefill, Decode)
        pairs, and every domain-aware cluster config puts both roles on the
        node, so a synthetic source Prefill on the source's node prices the
        link the migration's bytes actually cross.
        """
        if tokens <= 0:
            return 0.0
        source_node = getattr(source, "node_id", None)
        kv_bytes = float(tokens) * float(getattr(self, "kv_bytes_per_token", 0.0) or 0.0)
        if kv_bytes <= 0.0:
            return 0.0
        bandwidth, rtt_ms = 0.0, 0.0
        for scheduler in (self.prefill_schedulers or ()):
            if getattr(scheduler, "node_id", None) != source_node:
                continue
            key = (int(scheduler.instance_id), int(getattr(target, "instance_id", -1)))
            bandwidth = float(getattr(self, "pair_bandwidth", {}).get(key, 0.0) or 0.0)
            rtt_ms = float(getattr(self, "pair_rtt_ms", {}).get(key, 0.0) or 0.0)
            if bandwidth > 0.0:
                break
        if bandwidth <= 0.0:
            return 0.0
        return kv_bytes / bandwidth * 1000.0 + rtt_ms

    def _prfaas_decode_target(self):
        """The Decode this request's KV has to reach (the ``load`` arm's own)."""
        eligible = [candidate for candidate in (self.decode_schedulers or ())
                    if candidate.accepts_new_requests]
        if not eligible:
            return None
        return eligible[self._least_load_select(eligible, "decode")]

    def _prefill_pressure(self, sched):
        """How loaded a Prefill really is, for the semi-PD phase-sharing arm.

        ``(inflight+1)/capacity`` is the *routing* score and understates a
        saturated producer badly: on the P-15B WAN peak it reads 0.07-0.20 while
        the worker's own queue is backed up, because ``inflight`` is released
        the moment the Prefill leg finishes and ``capacity`` is the *compute*
        figure (4.9-13.6 req/s) rather than the push bound (~1.4 req/s).
        Queue occupancy is the signal that actually moves, so the arm takes the
        larger of the two.
        """
        slots = max(1, int(getattr(sched, "max_num_seqs", 1) or 1))
        queued = (4 * len(getattr(sched, "waiting", ()) or ())
                  + len(getattr(sched, "running", ()) or ()))
        return max(self._instance_load_score("prefill", sched), queued / slots)

    def _prfaas_select(self, schedulers, role, req_data=None, now_ns=0):
        """Prefill placement with a cross-data-centre offload gate.

        PrfaaS-style deployments run Prefill as a service in another DC and keep
        the request's own DC for Decode.  Two rules are what make that a
        *different* algorithm from our planner and from plain ``load``:

        1. **local first** -- while the local DC's Prefill pool is below
           ``prfaas_local_load_threshold``, the request is prefilled there, even
           if a remote producer is faster;
        2. **a transfer budget, not a price** -- once local is saturated, the
           remote candidates are ranked by ``max(transfer, compute wait)`` but
           any pairing whose transfer alone exceeds ``prfaas_max_offload_ms``
           is refused outright (this is the threshold the literature searches),
           and the request degrades to the least-loaded local producer instead
           of crossing the fabric.

        Both rules price the *KV bytes* of this request over the pair's measured
        bandwidth, so the arm is model-aware in the same way ``kv_aware`` is.
        """
        if role != "prefill" or not schedulers or req_data is None:
            return self._least_load_select(schedulers, role)
        tokens = len(req_data.get("input_hash_ids")
                     or req_data.get("input_tok_ids") or ())
        decode = self._prfaas_decode_target()
        if decode is None or tokens <= 0:
            return self._least_load_select(schedulers, role)
        local = [candidate for candidate in schedulers
                 if getattr(candidate, "node_id", None) is not None
                 and getattr(candidate, "node_id", None) == getattr(decode, "node_id", None)]
        if local:
            pick = min(local, key=lambda item: (
                self._instance_load_score(role, item), int(item.instance_id)))
            if self._instance_load_score(role, pick) < self.prfaas_local_load_threshold:
                self._counters["prfaas_local"] = self._counters.get("prfaas_local", 0) + 1
                return schedulers.index(pick)

        gate = self.prfaas_max_offload_ms
        scored = []
        blocked = 0
        for index, candidate in enumerate(schedulers):
            transfer_ms = self._pair_transfer_ms(candidate, decode, tokens, now_ns)
            if gate > 0.0 and transfer_ms > gate:
                blocked += 1
                continue
            compute_ms = self._instance_load_score(role, candidate) * 1000.0
            scored.append((max(transfer_ms, compute_ms), transfer_ms,
                           int(candidate.instance_id), index))
        self._counters["prfaas_gate_blocked"] = (
            self._counters.get("prfaas_gate_blocked", 0) + blocked)
        if not scored:
            # Nothing fits the transfer budget: stay in the local DC even
            # though it is loaded (the fallback the threshold arm implies).
            pool = local or schedulers
            self._counters["prfaas_local_fallback"] = (
                self._counters.get("prfaas_local_fallback", 0) + 1)
            return schedulers.index(min(pool, key=lambda item: (
                self._instance_load_score(role, item), int(item.instance_id))))
        score, transfer_ms, instance_id, index = min(scored)
        local_ids = {item.instance_id for item in local}
        if instance_id in local_ids:
            self._counters["prfaas_local_loaded"] = (
                self._counters.get("prfaas_local_loaded", 0) + 1)
        else:
            self._counters["prfaas_offload"] = (
                self._counters.get("prfaas_offload", 0) + 1)
        if os.environ.get("PRFAAS_DEBUG"):
            print(f"[prfaas] p{instance_id} transfer={transfer_ms:.1f}ms "
                  f"score={score:.1f} -> "
                  f"{'local' if instance_id in local_ids else 'offload'}",
                  flush=True)
        return index

    def install_affinity_plan(self, plan, flows=()):
        """Atomically replace the slow-layer plan used for future requests.

        ``flows`` (the solver's per-class rate assignments) are used to derive
        the aggregate Prefill split.  Per-class *shares* are the wrong statistic
        for that: classes are single-homed at low demand, so summing shares
        gives every class the same weight and the aggregate ends up near
        "all on the cheapest worker", while the plan's *rate* intent may put a
        fifth of the load on a different one.  Measured 2026-09-15: the plan
        asked for 1.81 / 8.86 req/s on the fresh worker, yet that worker stayed
        idle (run 0, wait 0) while two others queued 82 and 74 requests.
        """
        self.affinity_plan = plan
        # ``_plan_assignments`` is deliberately *not* cleared here.  It used to
        # be, and that reset the deficit balance every control tick (~1.6
        # arrivals at the 16 rps peak), so the split degenerated to "always the
        # highest-weight worker": measured 2026-09-23, a plan asking for 91 / 9
        # dispatched 737 of its 740 requests to the 91 % worker while the other
        # sat idle.  See ``_select_weighted``.
        #
        # Aggregate Prefill split of the plan.  A class the plan cannot name
        # (every request of a unique-prompt workload is its own class, so most
        # arrivals land here) still has to follow the *intended* load split --
        # otherwise the router falls back to least-loaded and the workers the
        # plan just asked for receive nothing.  Measured 2026-09-15: with three
        # workers active the plan put 20% of the flow on the newest one, yet it
        # served 0 of 200 requests and the elastic run matched the static one
        # millisecond for millisecond.
        totals = defaultdict(float)
        for flow in (flows or ()):
            value = getattr(flow, "flow", None)
            if value is None and isinstance(flow, dict):
                value = flow.get("flow")
            instance_id = getattr(flow, "prefill_id", None)
            if instance_id is None and isinstance(flow, dict):
                instance_id = flow.get("prefill_id")
            if instance_id is None or value is None or float(value) <= 0.0:
                continue
            totals[int(instance_id)] += float(value)
        if not totals:
            for weights in (getattr(plan, "prefill_weights", {}) or {}).values():
                for instance_id, weight in (weights or {}).items():
                    totals[int(instance_id)] += float(weight or 0.0)
        total = sum(totals.values())
        self._plan_prefill_totals = ({instance_id: weight / total
                                      for instance_id, weight in totals.items()}
                                     if total > 0 else {})

    def _select_weighted(self, candidates, weights, assignment_key):
        """Approximate a fractional plan with deterministic deficit routing.

        Mirrors ``deploy/real_lmcache_pd/disagg_router._select_weighted``: the
        plan sets the *ratios*, and any candidate within a small band of the
        best deficit score is ordered by the current queue.  Without the band,
        deficit routing sends a run of consecutive requests to one instance
        (invisible when idle, a p95 tail under load -- measured 2026-09-13:
        identical pair shares but p95 3973 ms for the plan replay against
        3186 ms for the queue-aware heuristic).
        """
        total = sum(self._plan_assignments[(assignment_key, sched.instance_id)]
                    for sched in candidates)
        def score(sched):
            observed = self._plan_assignments[(assignment_key, sched.instance_id)]
            desired = weights[sched.instance_id] * (total + 1)
            return desired - observed
        highest = max(score(sched) for sched in candidates)
        band = max(1.0, abs(highest)) * 0.05
        selected = self._least_loaded([sched for sched in candidates
                                       if score(sched) >= highest - band])
        self._plan_assignments[(assignment_key, selected.instance_id)] += 1
        # The counters are not reset per tick (see ``install_affinity_plan``),
        # so a unique-prompt trace adds one key per request.  Prune the
        # per-class keys when they grow past any plausible working set and keep
        # the aggregate split, which is the one that must stay balanced.
        if len(self._plan_assignments) > _MAX_PLAN_ASSIGNMENTS:
            for key in [key for key in self._plan_assignments
                        if key[0] != "prefill_aggregate"]:
                del self._plan_assignments[key]
        return selected

    def _occupancy_weights(self, candidates, weights, current_time_ns):
        """Discount a candidate Prefill by how backed-up its egress already is.

        The plan is per class, and a few hundred classes can each independently
        prefer the same producer: their flows look small one by one (the LP
        reported ``link_overflow 0`` in the six-domain arena) while the executed
        placement overran that producer's push budget ~11x and the p95 was 37 s
        against the best arm's 1.3 s.  The real router prices the link's
        occupancy per request (``link_inflight_bytes / bandwidth``), so a
        producer that is already backed up stops looking cheapest.
        """
        link = getattr(self, "pd_link", None)
        if link is None or len(candidates) <= 1:
            return weights
        now_ns = int(current_time_ns or 0)
        return {
            instance_id: float(weight)
            / (1.0 + link.pending_ns(int(instance_id), now_ns) / 1e9)
            for instance_id, weight in weights.items()}

    def _select_planned_prefill(self, req_data, current_time_ns):
        plan = self.affinity_plan
        if plan is None or plan.is_expired(current_time_ns):
            return None
        weights = plan.prefill_for(req_data['class_id'])
        candidates = [sched for sched in self.prefill_schedulers
                      if sched.accepts_new_requests and sched.instance_id in weights]
        if not candidates:
            return None
        weights = self._occupancy_weights(candidates, weights, current_time_ns)
        if not weights:
            return None
        return self._select_weighted(candidates, weights,
                                     ("prefill", req_data['class_id']))

    def _select_planned_decode(self, req, current_time_ns):
        plan = self.affinity_plan
        if plan is None or plan.is_expired(current_time_ns):
            return None
        weights = plan.decode_for(req.prefill_instance_id, req.class_id)
        candidates = [sched for sched in self.decode_schedulers
                      if sched.accepts_new_requests and sched.instance_id in weights]
        if candidates:
            return self._select_weighted(candidates, weights,
                                         ("decode", req.prefill_instance_id, req.class_id))
        fallback_ids = set(plan.fallback_for(req.prefill_instance_id, req.class_id))
        candidates = [sched for sched in self.decode_schedulers
                      if sched.accepts_new_requests and sched.instance_id in fallback_ids]
        return self._least_loaded(candidates) if candidates else None

    def kv_bytes_for(self, prompt_tokens, scheduler=None):
        """KV bytes one handoff of this prompt moves, for the network term.

        The pair cost wants the *size* of the push the candidate pair implies
        (``disagg_router._pair_cost`` divides it by the link bandwidth); a
        per-token constant is enough for the ranking, and the scheduler's own
        memory model is exact when it is available.
        """
        memory = getattr(scheduler, "memory", None)
        if memory is not None and prompt_tokens:
            try:
                return float(memory.pd_kv_bytes(int(prompt_tokens)))
            except (AttributeError, TypeError, ValueError):
                pass
        return float(self.kv_bytes_per_token or 0.0) * max(0, int(prompt_tokens or 0))

    def _decode_cost_select(self, prefill_sched, kv_bytes=None, now_ns=0):
        """Cheapest Decode for this Prefill, mirroring ``_pick_decode_cost``.

        Same three parts as the real router's ``_pair_cost``, and the same
        shape for each of them:

        * **network** -- zero for a same-domain pair, otherwise the measured
          RTT *plus the KV bytes already queued on that link* over its
          bandwidth.  It is an occupancy price, not a serialisation: the push
          overlaps with the Prefill's compute, so an idle link costs only the
          RTT.  Pricing the full transfer made the real router refuse
          profitable cross-domain pairings (measured -5.6% when that term was
          removed there); pricing *nothing* let the simulator move traffic off
          a link that was already backed up.
        * **service** -- the pair's measured Prefill + Decode service times.
        * **load** -- ``inflight / capacity`` on both legs.  This is a *wait
          time* by Little's law (``W = L / lambda`` with ``lambda`` ~ the
          calibrated requests/s), which is what makes it commensurate with the
          two terms above; multiplying it by ``service`` instead (the shape
          used here before 2026-09-16) turns it into a dimensionless
          utilisation and over-penalises the faster instance.
        """
        candidates = [candidate for candidate in self.decode_schedulers
                      if candidate.accepts_new_requests]
        if not candidates:
            return None
        capacities = getattr(self, "capacity_tables", {})
        table = capacities.get("decode", {})
        prefill_table = capacities.get("prefill", {})
        prefill_node = int(getattr(prefill_sched, "node_id", -1))
        weights = getattr(self, "cost_weights", {})
        w_network = float(weights.get("network", 1.0))
        w_service = float(weights.get("service", 1.0))
        w_prefill = float(weights.get("prefill_load", 1.0))
        w_decode = float(weights.get("decode_load", 1.0))
        prefill_service = float(getattr(self, "prefill_service_ms", {}).get(
            int(prefill_sched.instance_id), 0.0))
        prefill_inflight = len(getattr(prefill_sched, "running", ())) + len(
            getattr(prefill_sched, "waiting", ()))
        prefill_capacity = max(1.0, float(
            prefill_table.get(int(prefill_sched.instance_id))
            or getattr(prefill_sched, "max_num_seqs", 1) or 1))

        def cost(sched):
            same_node = prefill_node == int(getattr(sched, "node_id", -2))
            rtt = 0.0 if same_node else self.pair_rtt_ms.get(
                (int(prefill_sched.instance_id), int(sched.instance_id)), 0.0)
            queued_ms = 0.0
            link = getattr(self, "pd_link", None)
            if link is not None and not same_node:
                queued_ms = link.wait_ns(
                    int(prefill_sched.instance_id), prefill_node,
                    int(getattr(sched, "node_id", -2)),
                    kv_bytes or 0, now_ns) / 1e6
            network = (rtt + queued_ms) / 1000.0
            service = (prefill_service
                       + self.decode_service_ms.get(int(sched.instance_id), 0.0)) / 1000.0
            capacity = table.get(int(sched.instance_id)) or getattr(
                sched, "max_num_seqs", 1) or 1
            load = (w_prefill * prefill_inflight / prefill_capacity
                    + w_decode * (len(sched.running) + len(sched.waiting))
                    / max(1.0, float(capacity)))
            return w_network * network + w_service * service + load

        return min(candidates, key=lambda sched: (cost(sched), sched.instance_id))

    def _decode_scheduler_for(self, req_data, current_time_ns, prefill_sched):
        """Decode instance this arriving request would be paired with."""
        if not self.decode_schedulers:
            return None
        # ``_select_planned_decode`` reads the pair off a *routed* request; at
        # arrival we only have the dataset row, so hand it the two fields it
        # needs (the Prefill it would have used and the class).
        planned = self._select_planned_decode(
            SimpleNamespace(prefill_instance_id=prefill_sched.instance_id,
                            class_id=req_data.get("class_id")),
            current_time_ns)
        if planned is not None:
            return planned
        eligible = [candidate for candidate in self.decode_schedulers
                    if candidate.accepts_new_requests]
        if not eligible:
            return None
        return eligible[self._select_decode(eligible, req_data)]

    def kv_exchange_decision(self, prefill, decode, tokens, now_ns=0):
        """Local recompute vs P/D handoff, mirroring the real router.

        The real deployment decides this per request
        (``disagg_router.kv_exchange_decision``) from measured constants: moving
        KV costs 585 ms (same host) / 1351 ms + 48 ms fixed (cross host) per
        1000 prompt tokens, while the Decode prefilling the prompt itself costs
        ~93 ms/1k plus an M/G/1 queue externality on the Decode's own batch
        (capped, and zero when the Decode is idle).

        The simulator used to always hand the KV over, which made short prompts
        egress-bound and 25x slower than the measured cluster -- the real
        ``load`` arm answered 200/200 requests by recomputing locally.

        **Both sides now price their own queue**, which is what makes this a
        resource substitution rather than a service-time comparison:

        * the transfer side adds the producer's egress backlog
          (``PdHandoffLink.pending_ns``), so moving KV onto an already-backed-up
          producer is charged for the wait it creates;
        * the local side's externality is ``T_prefill * rho/(1-rho)`` capped at
          ``local_prefill_queue_cap`` -- an **amplification factor**, since the
          cost of inserting ``T_prefill`` of work into a queue at utilisation
          ``rho`` is the wait it inflicts on everything behind it, not the
          service time itself.

        Measured 2026-09-17 (2 Decode instances, 512-token outputs, 1250-token
        prompts): with the old reading of the cap -- a literal 3 *milliseconds*
        -- the externality was inert against a 731 ms transfer, so the rule
        chose local in 100% of requests at *every* Decode utilisation up to
        125%, including the runs where TTFT p50 had already reached 9.4 s.
        The cap is still 3.0 by default, so the decision on the deployment's own
        calibration is unchanged; raising it (``--local-queue-cap``) is what
        enables the substitution, and §6.9 of the paper draft contrasts the two.
        Returns ``(use_local, local_ms, transfer_ms)``.
        """
        if tokens <= 0:
            return False, 0.0, 0.0
        thousands = float(tokens) / 1000.0
        same_node = int(getattr(prefill, "node_id", -1)) == int(getattr(decode, "node_id", -2))
        derived = (self._link_derived_terms(prefill, decode, tokens)
                   if getattr(self, "link_derived_pricing", False) else None)
        if derived is not None:
            local_service_ms, transfer_ms = derived
        else:
            if same_node:
                transfer_ms = self.transfer_ms_per_1k_local * thousands
            else:
                transfer_ms = (self.transfer_fixed_ms_cross
                               + self.transfer_ms_per_1k_cross * thousands)
            local_service_ms = self.local_prefill_ms_per_1k * thousands
        link = getattr(self, "pd_link", None)
        if link is not None and now_ns:
            transfer_ms += float(link.pending_ns(
                int(getattr(prefill, "instance_id", -1)), int(now_ns))) / 1e6
        local_ms = local_service_ms
        if os.environ.get("LOCAL_PREFILL_DEBUG"):
            print(f"[exchange] p{getattr(prefill, 'instance_id', '?')}"
                  f"(node {getattr(prefill, 'node_id', '?')}) -> "
                  f"d{getattr(decode, 'instance_id', '?')}"
                  f"(node {getattr(decode, 'node_id', '?')}) tokens={tokens} "
                  f"model={'link' if derived is not None else 'constants'} "
                  f"local={local_ms:.1f}ms transfer={transfer_ms:.1f}ms "
                  f"-> {'local' if local_ms < transfer_ms else 'transfer'}",
                  flush=True)
        if self.local_prefill_queue_weight:
            budget = max(1.0, float(getattr(decode, "max_num_seqs", 1) or 1))
            inflight = float(len(getattr(decode, "running", ()) or ())
                             + len(getattr(decode, "waiting", ()) or ()))
            rho = min(1.0, inflight / budget)
            amplification = rho / max(1e-6, 1.0 - rho)
            externality = local_service_ms * min(self.local_prefill_queue_cap,
                                                 amplification)
            local_ms += self.local_prefill_queue_weight * externality
        return local_ms < transfer_ms, local_ms, transfer_ms

    def _link_derived_terms(self, prefill, decode, tokens):
        """``(local_ms, transfer_ms)`` from the cluster's own numbers, or None.

        Transfer = the request's KV over the pair's measured bandwidth plus the
        pair's RTT; local recompute = this node's own Prefill step for a prompt
        of this length.  Both come from the same tables the solver prices with
        (``pair_costs`` / ``prefill_service_ms`` / ``kv_bytes_per_token``), so
        the fast router and the plan cannot disagree about which way is
        cheaper.  Returns ``None`` when the config does not carry what is
        needed, which keeps the recorded constants as the fallback.
        """
        if float(getattr(self, "kv_bytes_per_token", 0.0) or 0.0) <= 0.0:
            return None
        pair = (int(getattr(prefill, "instance_id", -1)),
                int(getattr(decode, "instance_id", -2)))
        bandwidth = getattr(self, "pair_bandwidth", {}).get(pair, 0.0)
        if bandwidth <= 0.0:
            return None
        kv_bytes = float(tokens) * self.kv_bytes_per_token
        transfer_ms = (kv_bytes / bandwidth * 1000.0
                       + getattr(self, "pair_rtt_ms", {}).get(pair, 0.0))
        local_service_ms = self._local_prefill_ms(decode, tokens)
        if local_service_ms is None:
            return None
        return local_service_ms, transfer_ms

    def _local_prefill_ms(self, decode, tokens):
        """What recomputing this prompt costs on the Decode's own node.

        The local path runs the Prefill leg on the Decode's node, so the price
        is that node's Prefill step (the sibling Prefill's measured service
        time) scaled by the prompt length.  Without a sibling the cheapest
        measured Prefill stands in, and with no measurements at all the caller
        falls back to the recorded constants.
        """
        service_table = getattr(self, "prefill_service_ms", {}) or {}
        if not service_table:
            return None
        node_id = getattr(decode, "node_id", None)
        step_ms = None
        for scheduler in (self.prefill_schedulers or ()):
            if node_id is not None and getattr(scheduler, "node_id", None) == node_id:
                step_ms = service_table.get(int(scheduler.instance_id))
                if step_ms:
                    break
        if not step_ms:
            candidates = [value for value in service_table.values() if value]
            if not candidates:
                return None
            step_ms = min(candidates)
        reference = float(getattr(self, "capacity_reference_tokens", 1024) or 1024)
        work = max(1.0, float(tokens) / max(1.0, reference))
        return float(step_ms) * work

    def _decode_instance_id_for(self, request, current_time_ns):
        """Decode instance the arriving request is paired with, or ``None``.

        Mirrors what ``transfer_prefill_request`` does after the Prefill
        completes -- the affinity plan first, then the routing policy -- but
        runs at arrival so the Prefill graph has a real receiver NPU to send
        its KV to.
        """
        if not self.decode_schedulers:
            return None
        if getattr(self, "casr_enabled", False):
            # CASR policies pick the Decode of a pair by cost -- network +
            # measured service + wait -- exactly like the real
            # ``_pick_decode_cost``, and that is what makes their handoffs
            # concentrate on the same-domain fast Decode.  Measured 2026-09-16
            # on the forced-transfer arm: real ``casr_lp`` sent 117/120
            # handoffs to d5090 while the simulator spread them 60/19/41 and
            # took 8x the latency.
            prefill_sched = next(
                (candidate for candidate in self.prefill_schedulers
                 if candidate.instance_id == getattr(
                     request, "pair_prefill_instance_id",
                     getattr(request, "prefill_instance_id", None))), None)
            if prefill_sched is not None:
                kv_bytes = (getattr(request, "pd_kv_bytes", 0) or 0
                            or self.kv_bytes_for(getattr(request, "original_input", 0),
                                                 prefill_sched))
                chosen = self._decode_cost_select(
                    prefill_sched, kv_bytes=kv_bytes, now_ns=current_time_ns)
                if chosen is not None:
                    return chosen.instance_id
        planned = self._select_planned_decode(request, current_time_ns)
        if planned is not None:
            return planned.instance_id
        eligible = [candidate for candidate in self.decode_schedulers
                    if candidate.accepts_new_requests]
        if not eligible:
            return None
        return eligible[self._select_decode(eligible)].instance_id

    def _note_offered_arrival(self, req_data, at_ns):
        """Tell the profiler a request has been *offered*, wherever it lands.

        Attributed to the Prefill the plan prefers for that class (a held
        request has no placement yet, and the plan is the router's own opinion
        anyway); falls back to the least-loaded accepting Prefill.  The counts
        only feed the demand estimate, so being wrong about the instance costs
        nothing as long as the class is right.
        """
        profiler = self.prefix_profiler
        if profiler is None:
            return
        prefill_id = None
        plan = self.affinity_plan
        if plan is not None:
            try:
                weights = plan.prefill_for(req_data.get("class_id"))
            except Exception:
                weights = None
            if weights:
                prefill_id = max(weights.items(), key=lambda item: item[1])[0]
        if prefill_id is None:
            eligible = [s for s in self.prefill_schedulers if s.accepts_new_requests]
            if not eligible:
                return
            prefill_id = eligible[self._select_instance(eligible, "prefill")].instance_id
        profiler.observe_offered_arrival(
            req_data.get("class_id"), int(prefill_id), int(at_ns),
            int(req_data.get("input_toks", 0) or 0))

    def _decorate_req_data(self, req_data):
        if self.prefix_profiler is None:
            req_data.setdefault('class_id', 'default')
            req_data.setdefault('prefix_id', 'none')
            return req_data
        class_id, prefix_id = self.prefix_profiler.assign(
            req_data.get('model_id', self.prefill_schedulers[0].model),
            req_data['input_toks'],
            req_data['output_toks'] - req_data['input_toks'],
            req_data.get('input_hash_ids', []),
            req_data.get('kv_bytes_per_request', 0.0),
            req_data.get('slo_ttft_ms'),
            req_data.get('slo_tpot_ms'),
        )
        req_data['class_id'] = class_id
        req_data['prefix_id'] = prefix_id
        return req_data

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True,
                      max_output_tokens=0, slo_ttft_ms=None, slo_tpot_ms=None):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.

        ``max_output_tokens`` mirrors the real client's flag of the same name:
        the recorded comparisons cap generation at 16 tokens, so a replay that
        let the trace's own 26-653 token outputs run would compare a different
        workload (measured 2026-09-16: real completion_tokens were 16/16 for all
        200 requests while the simulator generated 74.7 on average).
        """
        # Relative to the repository root (this runs inside ``astra-sim``);
        # an absolute path is already unambiguous.
        if not os.path.isabs(path):
            path = f'../{path}'
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        self._max_output_tokens = int(max_output_tokens or 0)
        self._cli_slo_ttft_ms = slo_ttft_ms
        self._cli_slo_tpot_ms = slo_tpot_ms
        loaded_lines = 0

        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                row = json.loads(line)
                if 'sub_requests' in row:
                    self._load_agentic_session(row, enable_prefix_caching)
                else:
                    self._load_flat_request(row, enable_prefix_caching)
                loaded_lines += 1

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=lambda r: r['arrival_time_ns'])

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions))

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        output_toks = int(row['output_toks'])
        output_ids = list(row.get('output_tok_ids', []))
        limit = getattr(self, "_max_output_tokens", 0)
        if limit and output_toks > limit:
            output_toks = limit
            output_ids = output_ids[:limit]
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks']) + output_toks,
            'arrival_time_ns': int(row['arrival_time_ns']),
            'model_id': row.get('model_id', self.prefill_schedulers[0].model),
            'input_hash_ids': row.get('input_tok_ids', []),
            'output_hash_ids': output_ids,
            'kv_bytes_per_request': row.get('kv_bytes_per_request', 0.0),
            # Request-level latency budget: trace row first, CLI second.  The
            # solver's ``p_slo`` term prices a pair against this, and until now
            # the simulated path dropped the fields the real client forwards as
            # ``X-SLO-*`` headers.
            'slo_ttft_ms': row.get('slo_ttft_ms', self._cli_slo_ttft_ms),
            'slo_tpot_ms': row.get('slo_tpot_ms', self._cli_slo_tpot_ms),
        }
        self._pending_requests.append(self._decorate_req_data(req_data))

    def _load_agentic_session(self, row, enable_prefix_caching):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        session_id = row.get('session_id', f'session_{self._next_request_id}')
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        arrival_ns = int(row['arrival_time_ns'])

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
        }

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = {
            'index': base_id,
            'input_toks': int(first['input_toks']),
            'output_toks': int(first['input_toks'] + first['output_toks']),
            'arrival_time_ns': arrival_ns,
            'session_id': session_id,
            'sub_request_index': 0,
            'model_id': first.get('model_id', self.prefill_schedulers[0].model),
            'input_hash_ids': first.get('input_tok_ids', []),
                'output_hash_ids': first.get('output_tok_ids', []),
                'kv_bytes_per_request': first.get('kv_bytes_per_request', 0.0),
        }
        self._pending_requests.append(self._decorate_req_data(req_data))
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def route_arrived_requests(self, current_time_ns):
        """Route requests that have arrived by current_time_ns to instances.

        Called at the start of each iteration in the main simulation loop.
        Returns the number of newly routed requests.
        """
        routed = 0
        while self._pending_idx < len(self._pending_requests):
            # Closed-loop client: hold the arrival stream back while the cap is
            # reached, exactly as the real client's semaphore does.
            if (self.client_concurrency and
                    self._in_flight >= self.client_concurrency):
                break
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break
            # Offer observation happens *here*, when the request is due -- not
            # when it is finally dispatched.  Counting on dispatch makes the
            # observed arrival rate a property of the dispatcher: once the
            # router starts holding requests (which is exactly what happens
            # under back-pressure), a batch release shows up as tens of
            # arrivals in one window.  Measured 2026-09-23 (16 req/s offered):
            # a 1 s window read 44 req/s and the plan sized for it.  The real
            # router counts at the front door, before any placement.
            # Once per request: a held request is re-examined on every call, and
            # counting it again each time inflated the offer by the number of
            # retries (measured: 5x).
            if not req_data.get("_offered_noted"):
                req_data["_offered_noted"] = True
                self._note_offered_arrival(req_data, current_time_ns)

            recorded = self.placement_override.get(int(req_data['index']))
            recorded_decode = None
            if recorded is not None:
                recorded_prefill = next(
                    (candidate for candidate in self.prefill_schedulers
                     if candidate.instance_id == int(recorded[0])
                     and candidate.accepts_new_requests), None)
                recorded_decode = next(
                    (candidate.instance_id for candidate in self.decode_schedulers
                     if candidate.instance_id == int(recorded[1])), None)
                if recorded_prefill is not None and recorded_decode is not None:
                    self._counters["placement_replayed"] = (
                        self._counters.get("placement_replayed", 0) + 1)
                else:
                    recorded = None

            sched = recorded_prefill if recorded is not None else \
                self._select_planned_prefill(req_data, current_time_ns)
            self._counters["prefill_planned"] = (
                self._counters.get("prefill_planned", 0) + (1 if sched is not None else 0))
            if sched is None and self._plan_prefill_totals:
                # The plan does not name this class; follow its aggregate split
                # instead of dropping straight to least-loaded (see
                # ``install_affinity_plan``).
                candidates = [candidate for candidate in self.prefill_schedulers
                              if candidate.accepts_new_requests
                              and candidate.instance_id in self._plan_prefill_totals]
                aggregate_weights = self._occupancy_weights(
                    candidates, self._plan_prefill_totals, current_time_ns)
                if candidates and aggregate_weights:
                    sched = self._select_weighted(
                        candidates, aggregate_weights,
                        ("prefill_aggregate",))
                    self._counters["prefill_aggregate"] = (
                        self._counters.get("prefill_aggregate", 0) + 1)
                    self._counters[f"aggregate_pick_{sched.instance_id}"] = (
                        self._counters.get(f"aggregate_pick_{sched.instance_id}", 0) + 1)
                    if len(self._plan_prefill_totals) > 1:
                        self._counters["aggregate_multi_weight"] = (
                            self._counters.get("aggregate_multi_weight", 0) + 1)
            if sched is None:
                self._counters["prefill_fallback"] = (
                    self._counters.get("prefill_fallback", 0) + 1)
                eligible = [candidate for candidate in self.prefill_schedulers
                            if candidate.accepts_new_requests]
                if not eligible:
                    break
                if self.routing_policy == "CACHE_AWARE":
                    instance_id = self._cache_aware_select(eligible, "prefill", req_data)
                elif self.routing_policy == "KV_AWARE":
                    instance_id = self._kv_aware_select(
                        eligible, "prefill", req_data, current_time_ns)
                elif self.routing_policy == "PRFAAS":
                    instance_id = self._prfaas_select(
                        eligible, "prefill", req_data, current_time_ns)
                else:
                    instance_id = self._select_instance(eligible, "prefill")
                sched = eligible[instance_id]

            # P/D handoff vs local recompute.  When the Decode is going to
            # recompute the prompt anyway, dispatch the request straight to it:
            # no Prefill instance runs, no KV crosses the fabric.  This mirrors
            # the real router, whose ``load`` arm took the local path for
            # 200/200 requests of the Dolly trace (measured 2026-09-16).
            local_request = False
            prefill_sched = sched          # the pair's Prefill, kept for accounting
            if self.local_prefill_mode != "never" and self.decode_schedulers:
                # CASR policies choose the Decode by cost (that is what makes
                # their placement differ from the baselines); everything else
                # keeps the plan-then-least-loaded order.
                decode_sched = (
                    self._decode_cost_select(
                        sched,
                        kv_bytes=self.kv_bytes_for(req_data['input_toks'], sched),
                        now_ns=current_time_ns)
                    if self.casr_enabled
                    else self._decode_scheduler_for(req_data, current_time_ns, sched))
                if decode_sched is not None and decode_sched is not sched:
                    if self.local_prefill_mode == "always":
                        use_local = True
                    else:
                        use_local, _, _ = self.kv_exchange_decision(
                            sched, decode_sched, req_data['input_toks'],
                            current_time_ns)
                    # Phase sharing (semi-PD): a saturated Prefill pool hands its
                    # phase to the Decode rather than queueing behind it, even
                    # when the KV would have travelled cheaply.
                    if (not use_local and self.prefill_overflow_to_decode > 0.0
                            and self._prefill_pressure(sched)
                            >= self.prefill_overflow_to_decode):
                        use_local = True
                        self._counters["prefill_phase_shared"] = (
                            self._counters.get("prefill_phase_shared", 0) + 1)
                    if use_local:
                        sched = decode_sched
                        local_request = True

            request = sched.add_request([
                req_data['index'], sched.model,
                req_data['input_toks'], req_data['output_toks'],
                # A closed-loop client submits a request when a slot frees, and
                # measures its latency from *that* moment -- the real client's
                # semaphore does the same.  Keeping the trace's nominal arrival
                # for a request that was held back would charge it for the time
                # it spent waiting to be submitted.
                (req_data['arrival_time_ns'] if not self.client_concurrency
                 else max(req_data['arrival_time_ns'], current_time_ns)),
                sched.instance_id,
                req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                req_data['class_id'], req_data['prefix_id'],
            ], is_init=True if local_request else self._is_init)
            # Carry the request's own budget onto the Request so the output CSV
            # can record the verdict (the real router writes the same three
            # fields into ``metrics-<policy>.jsonl``).
            request.slo_ttft_ms = req_data.get('slo_ttft_ms')
            request.slo_tpot_ms = req_data.get('slo_tpot_ms')
            if local_request:
                # The Decode owns the whole request: it runs the Prefill chunk
                # itself, so the handoff path never sees it.
                request.local_prefill = True
                request.decode_instance_id = sched.instance_id
                self._counters["local_prefill"] = self._counters.get("local_prefill", 0) + 1
            elif recorded_decode is not None:
                # Replayed placement: the Decode half comes from the recording,
                # so the handoff target matches the cluster's.
                request.decode_instance_id = recorded_decode
            elif self.name_decode_at_arrival:
                # Pick the Decode half of the pair now, the way the real router
                # does (``disagg_router._pick_decode`` runs on the same request
                # before anything is dispatched).  This is not just
                # bookkeeping: the Prefill graph sends its per-layer KV to
                # *this* Decode's NPU, so the choice is what the network model
                # charges.  Leaving it to the post-Prefill handoff made every
                # KV send target the Prefill instance's own adjacent NPU -- an
                # intra-node hop -- and the cross-domain link the solver priced
                # was never actually paid in the timeline (see
                # docs/模拟器与真机一致性核查.md 附七).
                #
                # A locally-computed request keeps the instance it was routed
                # to: overwriting it here recorded a Decode that never ran the
                # request, so the per-request CSV named the wrong instance
                # (measured 2026-09-16: every local request reported
                # ``decode_instance_id=5`` while actually running on 1).
                request.decode_instance_id = self._decode_instance_id_for(
                    request, current_time_ns)
            # Dispatch accounting, mirroring the real router: both legs of the
            # chosen pair carry the request until it completes, even when the
            # Prefill is bypassed by a local recompute.  It has to run *after*
            # the Decode half is final -- for a transferring request that is
            # ``decode_instance_id`` (a locally-recomputed one already owns its
            # instance), otherwise the Decode leg is never counted and every
            # replay piles up on the first candidate.
            pair_prefill = prefill_sched.instance_id
            served = (sched.instance_id if local_request
                      else request.decode_instance_id)
            self._in_flight += 1
            request.pair_prefill_instance_id = pair_prefill
            self._assigned[pair_prefill] += 1
            if served is not None:
                self._assigned[served] += 1
                self._request_pair[request.id] = (pair_prefill, served)
            else:
                self._request_pair[request.id] = (pair_prefill,)
            # The arrival itself was counted when the request became due
            # (``_note_offered_arrival``); counting again here would double it.

            self._pending_idx += 1
            routed += 1

        return routed

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return self._pending_idx < len(self._pending_requests)

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request_id, completion_time_ns):
        """Called when a request finishes. Releases the next sub-request in
        the session chain after the tool_call duration elapses.

        For flat requests (not in a session), this is a no-op.
        """
        # Dispatch accounting first: the real router decrements ``inflight`` on
        # the Prefill and the Decode as soon as the request completes, and that
        # counter is what ``load`` ranks with.  Flat requests used to return
        # early here, which would have left the counters rising forever.
        tracker = getattr(self, "_request_pair", None)
        if tracker is not None and request_id in tracker:
            self._in_flight = max(0, self._in_flight - 1)
            for instance_id in tracker.pop(request_id):
                if instance_id is None:
                    continue          # already released when the Prefill handed over
                if instance_id in self._assigned:
                    self._assigned[instance_id] = max(0, self._assigned[instance_id] - 1)
        session_info = self._request_to_session.pop(request_id, None)
        if session_info is None:
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(sub_reqs[completed_idx].get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            req_data = {
                'index': next_id,
                'input_toks': int(next_sub['input_toks']),
                'output_toks': int(next_sub['input_toks'] + next_sub['output_toks']),
                'arrival_time_ns': release_time_ns,
                'session_id': session_id,
                'sub_request_index': next_idx,
                'model_id': next_sub.get('model_id', self.prefill_schedulers[0].model),
                'input_hash_ids': next_sub.get('input_tok_ids', []),
                'output_hash_ids': next_sub.get('output_tok_ids', []),
                'kv_bytes_per_request': next_sub.get('kv_bytes_per_request', 0.0),
            }
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(self._decorate_req_data(req_data))
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete — all sub-requests have been released
            del self._deferred_sessions[session_id]

    def _insert_pending_sorted(self, req_data):
        """Insert a request into _pending_requests maintaining arrival-time
        sort order for the not-yet-consumed portion (from _pending_idx onward)."""
        arrival = req_data['arrival_time_ns']
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_requests[mid]['arrival_time_ns'] <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there are agentic sessions with unreleased sub-requests."""
        return bool(self._deferred_sessions)

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        if self._pending_idx < len(self._pending_requests):
            return self._pending_requests[self._pending_idx]['arrival_time_ns']
        return None

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.waiting),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def release_prefill_leg(self, request_id):
        """Give the Prefill's dispatch slot back when its leg completes.

        The real router releases ``prefill.inflight`` in the dispatch handler's
        ``finally``: it awaits the Prefill POST and then returns the streaming
        response, so the counter is back to zero long before the request
        finishes, while ``decode.inflight`` is held until completion.  The
        simulator kept both until completion, so a Prefill that had already
        handed its KV over still looked busy -- with ``(inflight+1)/capacity``
        the smaller p3090a then out-competed p5090 and ``load`` spread 160/80
        where the cluster put 240/240 on p5090 (measured 2026-09-16, paced
        trace, 1 req/s).
        """
        tracker = getattr(self, "_request_pair", None)
        if tracker is None or request_id not in tracker:
            return
        pair = tracker[request_id]
        if not pair or pair[0] is None:
            return
        prefill_id = pair[0]
        if prefill_id in self._assigned:
            self._assigned[prefill_id] = max(0, self._assigned[prefill_id] - 1)
        tracker[request_id] = (None,) + tuple(pair[1:])

    def transfer_prefill_request(self, requests, current_time_ns=0):
        # A Decode whose KV pool is full refuses the handoff (``add_decode``
        # returns False); keep those requests here and retry them on the next
        # call instead of dropping them or aborting the run.  This is the
        # simulator's back-pressure: the request's KV is already in flight, so
        # the wait is charged to the Decode, exactly as the real deployment's
        # PD buffer does.
        queue = list(self._pending_handoffs) + list(requests)
        self._pending_handoffs = []
        for req in queue:
            self.release_prefill_leg(req.id)
            sched = next((candidate for candidate in self.decode_schedulers
                          if candidate.instance_id == req.decode_instance_id
                          and candidate.accepts_new_requests), None)
            if sched is None:
                sched = self._select_planned_decode(req, current_time_ns)
            if sched is None:
                eligible = [candidate for candidate in self.decode_schedulers
                            if candidate.accepts_new_requests]
                if not eligible:
                    # A Decode pool that scales (coordinated autoscaling) can be
                    # momentarily empty of ACTIVE workers -- every one DRAINING
                    # or WARMING.  The KV is already in flight, so the request
                    # waits in the pending list and is retried on the next
                    # iteration, exactly like a handoff a full Decode refused.
                    # Raising here killed the run the first time the pooled arm
                    # drained below its peak (measured 2026-09-24).
                    self._pending_handoffs.append(req)
                    self._counters["handoff_no_active_decode"] = (
                        self._counters.get("handoff_no_active_decode", 0) + 1)
                    continue
                # Handoff fallback: same deadline-aware rule, fed from the
                # request object (it carries its own budget and lengths).
                instance_id = self._select_decode(eligible, {
                    "input_toks": req.input,
                    "output_toks": req.output,
                    "slo_ttft_ms": getattr(req, "slo_ttft_ms", None),
                })
                sched = eligible[instance_id]
            if self.affinity_plan is not None:
                req.affinity_version = self.affinity_plan.version
            if not sched.add_decode(req):
                self._pending_handoffs.append(req)
                self._counters["handoff_backpressure"] = (
                    self._counters.get("handoff_backpressure", 0) + 1)
        return tuple(self._pending_handoffs)

    def retry_pending_handoffs(self, current_time_ns=0):
        """Re-attempt hand-offs whose Decode was full.

        ``transfer_prefill_request`` is only called when a Prefill *finishes*
        something, so a pending hand-off that is waiting on a full Decode would
        never be retried once the Prefill goes idle -- measured 2026-09-15: a
        long-prompt run drained to "Prefill instance 4: Waiting 59, Running 0"
        with zero throughput and the simulated clock still advancing.  The main
        loop calls this every iteration, which is what makes the back-pressure
        progress rather than deadlock.
        """
        if not self._pending_handoffs:
            return ()
        return self.transfer_prefill_request((), current_time_ns)

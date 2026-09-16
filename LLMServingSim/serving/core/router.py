import bisect
import json
import random
from collections import defaultdict
from .logger import get_logger
from .block_pool import NONE_HASH


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            prefix_profiler=None,
            name_decode_at_arrival=False,
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
        # Only the domain-aware cluster configs name the Decode half of every
        # pair up front; see ``_decode_instance_id_for``.
        self.name_decode_at_arrival = name_decode_at_arrival
        self._rnd = random.Random(seed) if seed is not None else random
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0

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
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        best_idx = 0
        best_score = float('inf')
        num_instances = len(schedulers)
        start = self._get_counter(role) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.waiting)
            running = len(sched.running)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(role, (best_idx + 1) % num_instances)
        return best_idx

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

    @staticmethod
    def _least_loaded(candidates):
        """Choose an eligible scheduler, retaining vLLM-style load scoring."""
        return min(
            candidates,
            key=lambda sched: ((len(sched.waiting) * 4 + len(sched.running)) /
                               (sched.max_num_seqs if sched.max_num_seqs not in (0, float('inf'))
                                else 1), sched.instance_id),
        )

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
        self._plan_assignments.clear()
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
        return selected

    def _select_planned_prefill(self, req_data, current_time_ns):
        plan = self.affinity_plan
        if plan is None or plan.is_expired(current_time_ns):
            return None
        weights = plan.prefill_for(req_data['class_id'])
        candidates = [sched for sched in self.prefill_schedulers
                      if sched.accepts_new_requests and sched.instance_id in weights]
        if not candidates:
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

    def _decode_instance_id_for(self, request, current_time_ns):
        """Decode instance the arriving request is paired with, or ``None``.

        Mirrors what ``transfer_prefill_request`` does after the Prefill
        completes -- the affinity plan first, then the routing policy -- but
        runs at arrival so the Prefill graph has a real receiver NPU to send
        its KV to.
        """
        if not self.decode_schedulers:
            return None
        planned = self._select_planned_decode(request, current_time_ns)
        if planned is not None:
            return planned.instance_id
        eligible = [candidate for candidate in self.decode_schedulers
                    if candidate.accepts_new_requests]
        if not eligible:
            return None
        return eligible[self._select_instance(eligible, "decode")].instance_id

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
        )
        req_data['class_id'] = class_id
        req_data['prefix_id'] = prefix_id
        return req_data

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        path = f'../{path}'
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
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
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
            'model_id': row.get('model_id', self.prefill_schedulers[0].model),
            'input_hash_ids': row.get('input_tok_ids', []),
            'output_hash_ids': row.get('output_tok_ids', []),
            'kv_bytes_per_request': row.get('kv_bytes_per_request', 0.0),
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
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break

            sched = self._select_planned_prefill(req_data, current_time_ns)
            self._counters["prefill_planned"] = (
                self._counters.get("prefill_planned", 0) + (1 if sched is not None else 0))
            if sched is None and self._plan_prefill_totals:
                # The plan does not name this class; follow its aggregate split
                # instead of dropping straight to least-loaded (see
                # ``install_affinity_plan``).
                candidates = [candidate for candidate in self.prefill_schedulers
                              if candidate.accepts_new_requests
                              and candidate.instance_id in self._plan_prefill_totals]
                if candidates:
                    sched = self._select_weighted(
                        candidates, self._plan_prefill_totals,
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
                else:
                    instance_id = self._select_instance(eligible, "prefill")
                sched = eligible[instance_id]

            request = sched.add_request([
                req_data['index'], sched.model,
                req_data['input_toks'], req_data['output_toks'],
                req_data['arrival_time_ns'], sched.instance_id,
                req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                req_data['class_id'], req_data['prefix_id'],
            ], is_init=self._is_init)
            if self.name_decode_at_arrival:
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
                request.decode_instance_id = self._decode_instance_id_for(
                    request, current_time_ns)
            if self.prefix_profiler is not None:
                self.prefix_profiler.observe_arrival(
                    request, sched.instance_id, current_time_ns)

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
            sched = next((candidate for candidate in self.decode_schedulers
                          if candidate.instance_id == req.decode_instance_id
                          and candidate.accepts_new_requests), None)
            if sched is None:
                sched = self._select_planned_decode(req, current_time_ns)
            if sched is None:
                eligible = [candidate for candidate in self.decode_schedulers
                            if candidate.accepts_new_requests]
                if not eligible:
                    raise RuntimeError("No active Decode instance can accept a Prefill handoff")
                instance_id = self._select_instance(eligible, "decode")
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

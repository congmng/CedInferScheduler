import bisect
import json
import random
from collections import defaultdict
from .logger import get_logger


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            prefix_profiler=None,
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
        self.routing_policy = routing_policy.upper()
        self.seed = seed
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

    @staticmethod
    def _least_loaded(candidates):
        """Choose an eligible scheduler, retaining vLLM-style load scoring."""
        return min(
            candidates,
            key=lambda sched: ((len(sched.waiting) * 4 + len(sched.running)) /
                               (sched.max_num_seqs if sched.max_num_seqs not in (0, float('inf'))
                                else 1), sched.instance_id),
        )

    def install_affinity_plan(self, plan):
        """Atomically replace the slow-layer plan used for future requests."""
        self.affinity_plan = plan
        self._plan_assignments.clear()

    def _select_weighted(self, candidates, weights, assignment_key):
        """Approximate a fractional plan with deterministic deficit routing."""
        total = sum(self._plan_assignments[(assignment_key, sched.instance_id)]
                    for sched in candidates)
        def score(sched):
            observed = self._plan_assignments[(assignment_key, sched.instance_id)]
            desired = weights[sched.instance_id] * (total + 1)
            return desired - observed
        highest = max(score(sched) for sched in candidates)
        selected = self._least_loaded([sched for sched in candidates
                                       if score(sched) == highest])
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
            if sched is None:
                eligible = [candidate for candidate in self.prefill_schedulers
                            if candidate.accepts_new_requests]
                if not eligible:
                    break
                instance_id = self._select_instance(eligible, "prefill")
                sched = eligible[instance_id]

            request = sched.add_request([
                req_data['index'], sched.model,
                req_data['input_toks'], req_data['output_toks'],
                req_data['arrival_time_ns'], sched.instance_id,
                req_data.get('input_hash_ids', []), req_data.get('output_hash_ids', []),
                req_data['class_id'], req_data['prefix_id'],
            ], is_init=self._is_init)
            # Decode affinity is applied at the completed-Prefill handoff in
            # ``transfer_prefill_request``.  The ASTRA-Sim P/D backend owns a
            # legacy adjacent KV receiver rank while Prefill is executing, so
            # selecting a non-adjacent Decode rank here would generate a graph
            # that the backend cannot legally dispatch.  Delaying the choice
            # keeps the physical KV protocol correct and still makes Decode
            # admission class-aware.
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
        for req in requests:
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
            sched.add_decode(req)

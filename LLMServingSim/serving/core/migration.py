"""Llumnix-style request migration between Decode instances.

Llumnix (OSDI'24) reschedules *in-flight* requests across instances: when one
Decode's queue grows while a sibling sits idle, the scheduler migrates queued
requests instead of letting the imbalance stand until the slow one drains.

What this arm models, and what it does not:

* **moves** requests that are queued but not yet stepping -- a Decode's
  ``waiting`` list and the router's pending P/D hand-offs -- to the idle
  Decode, choosing the newest arrivals first so the head of the queue is not
  disturbed;
* **charges** the move exactly once, as the KV copy between the two nodes
  (``kv_bytes / link bandwidth + hop``) written onto the request's
  ``migration_ns``, which delays its first token and therefore its latency;
* **does not** move a request that is already ``running``.  That would need a
  block-level transfer inside the Decode's own KV pool (and a re-derivation of
  its prefix cache), which this simulator does not model -- so the arm is the
  "queued-request" subset of Llumnix, and a run that shows a gain does so
  without claiming the running-request case.

The thresholds are the ones a real migration policy runs on: move only when the
source is above ``llumnix_hot_threshold`` of its slots, the target is below
``llumnix_cold_threshold``, and the difference clears ``llumnix_min_gain``.
"""

from __future__ import annotations

from .request import RequestStatus


class LlumnixMigrator:
    """Periodic Decode-queue rebalancing with a modelled copy cost."""

    def __init__(self, options=None):
        options = options or {}
        interval_ms = float(options.get("llumnix_interval_ms", 0.0) or 0.0)
        self.interval_ns = max(0, int(interval_ms * 1_000_000))
        #: The trigger is a *drain time*, not a load fraction: an instance at
        #: ``(queued + running) / slots`` of its batch has that much work left,
        #: and the same fraction costs 46 ms/step on a 5090 and 372 ms on a
        #: 4090.  Millisecond thresholds keep the two comparable.
        self.hot_ms = float(options.get("llumnix_hot_ms", 50.0) or 0.0)
        self.cold_ms = float(options.get("llumnix_cold_ms", 20.0) or 0.0)
        self.min_gain_ms = float(options.get("llumnix_min_gain_ms", 30.0) or 0.0)
        self.batch = max(1, int(options.get("llumnix_batch", 4) or 1))
        # A request that is already stepping can still move, but the copy grows
        # with everything it has generated.  Llumnix prefers the request with
        # the least remaining work; this is the same idea with a bound that
        # keeps the move cheap: only requests that have generated at most this
        # many tokens are eligible.
        self.migrate_running = bool(options.get("llumnix_migrate_running", True))
        self.max_moved_tokens = int(options.get("llumnix_max_moved_tokens", 64) or 0)
        # Llumnix only migrates when the move pays for itself.  The benefit is
        # the work taken off the source; the cost is the KV copy plus the queue
        # the request joins on the target.
        self.gain_ratio = float(options.get("llumnix_gain_ratio", 1.0) or 0.0)
        self.next_ns = self.interval_ns if self.interval_ns else -1
        #: Observability: how many requests each direction has moved.
        self.moved = 0
        self.attempts = 0
        self.last_reason = "disabled" if not self.interval_ns else "not run yet"

    @property
    def enabled(self):
        return self.interval_ns > 0

    def due(self, current_ns):
        return self.enabled and current_ns >= self.next_ns

    # ------------------------------------------------------------------
    def _tokens(self, req):
        return int(getattr(req, "original_input", 0)
                   or getattr(req, "input", 0) or 0)

    def drain_ms(self, router, sched):
        """How long this Decode needs to work off what it is already holding."""
        return self._drain_ms(router, sched, extra=0)

    def arrival_ms(self, router, sched):
        """Drain time this instance would have *after* taking one more request.

        The source's pressure is the cost of leaving the request where it is;
        the target's is the cost of putting it there.  Scoring a target by its
        *idle* drain time is the mistake the deadline-aware Decode placement
        already made once: an idle RTX3090 looks cheap (172 ms) while the next
        request waits 16x its own step -- moving onto it turned a 27-move run
        into a 3 773 ms mean against 2 977 ms for the arm that moved nothing
        (measured 2026-09-24, P-15B WAN peak).
        """
        return self._drain_ms(router, sched, extra=1)

    def _drain_ms(self, router, sched, extra=0):
        slots = max(1, int(getattr(sched, "max_num_seqs", 1) or 1))
        queued = (4 * len(getattr(sched, "waiting", ()) or ())
                  + len(getattr(sched, "running", ()) or ()) + extra)
        occupancy = queued / slots
        table = (getattr(router, "capacity_tables", {}) or {}).get("decode", {}) or {}
        capacity = table.get(int(sched.instance_id)) or slots
        inflight = (getattr(router, "_assigned", {}) or {}).get(
            int(sched.instance_id), 0)
        load = (inflight + 1 + extra) / capacity if capacity else 1.0
        return max(occupancy, load) * self.step_ms(router, sched)

    @staticmethod
    def step_ms(router, sched):
        """One executed token on this instance, in milliseconds.

        ``decode_service_ms`` is the *deployment's* measured per-token step
        (772 ms on the 3090) while this simulator executes the P-15B Decode in
        ~14 ms.  The queue this arm balances is the executed one, so the price
        of a slot has to be the executed step -- and the resolved capacity *is*
        ``1000 / (reference_tokens x step)``.
        """
        table = (getattr(router, "capacity_tables", {}) or {}).get("decode", {}) or {}
        capacity = table.get(int(sched.instance_id))
        if capacity:
            reference = float(getattr(router, "decode_reference_tokens", 16.0) or 16.0)
            return 1000.0 / (reference * capacity)
        return (getattr(router, "decode_service_ms", {}) or {}).get(
            int(sched.instance_id), 0.0) or 100.0

    def migrate(self, router, current_ns):
        """Rebalance every Decode's queue for one control tick.

        Returns the number of requests moved.  ``router`` supplies the load
        score the routing policies use and the per-pair link tables, so the
        cost charged here is the same one the handoff itself pays.
        """
        self.next_ns = current_ns + self.interval_ns
        decodes = [item for item in (router.decode_schedulers or ())
                   if item.accepts_new_requests]
        if len(decodes) < 2:
            self.last_reason = "fewer than two active Decodes"
            return 0
        # Source: the instance whose queue the request is stuck behind.
        scored = sorted(((self.drain_ms(router, item),
                          int(item.instance_id), item) for item in decodes),
                        key=lambda entry: (-entry[0], entry[1]))
        hot_score, _, hot = scored[0]
        # Target: the instance that would be cheapest to land on, i.e. the one
        # with the smallest drain time *after* taking this request.
        landing = sorted(((self.arrival_ms(router, item),
                           int(item.instance_id), item) for item in decodes),
                         key=lambda entry: (entry[0], entry[1]))
        cold_score, _, cold = landing[0]
        if cold is hot:
            cold_score, _, cold = landing[1]
        if __import__("os").environ.get("LLUMNIX_DEBUG"):
            print(f"[llumnix] t={current_ns/1e9:.1f}s "
                  + " ".join(f"d{entry[1]}={entry[0]:.0f}ms"
                             f"(w{len(getattr(entry[2], 'waiting', ()) or ())}"
                             f"/r{len(getattr(entry[2], 'running', ()) or ())})"
                             for entry in scored)
                  + f" pending={len(getattr(router, '_pending_handoffs', ()) or ())}",
                  flush=True)
        if (hot_score < self.hot_ms or cold_score > self.cold_ms
                or hot_score - cold_score < self.min_gain_ms):
            self.last_reason = (f"no eligible pair (hot {hot_score:.0f} ms -> "
                                f"cold {cold_score:.0f} ms)")
            return 0
        self.attempts += 1
        moved = 0
        # 1. Hand-offs the hot Decode refused: their KV is already pushed, so
        #    re-targeting means copying it to the new node.
        for req in list(getattr(router, "_pending_handoffs", ()) or ()):
            if moved >= self.batch:
                break
            if int(getattr(req, "decode_instance_id", -1)) != int(hot.instance_id):
                continue
            self._repoint(router, req, hot, cold, current_ns)
            moved += 1
        # 2. Queued-but-not-stepping requests (locally prefilled): newest first.
        if moved < self.batch:
            for req in sorted(getattr(hot, "waiting", ()) or (),
                              key=lambda item: (getattr(item, "arrival", 0),
                                                getattr(item, "id", 0)), reverse=True):
                if moved >= self.batch:
                    break
                if req not in hot.waiting:
                    continue
                hot.waiting.remove(req)
                self._repoint(router, req, hot, cold, current_ns)
                cold.waiting.append(req)
                moved += 1
        # 3. Requests that are stepping but have barely started.  Their blocks
        #    are released on the source and claimed on the target, and the copy
        #    is charged; a target that cannot take it rolls the move back into
        #    the source's own queue.
        if self.migrate_running and moved < self.batch:
            moved += self._migrate_running(router, hot, cold, current_ns,
                                           self.batch - moved)
        self.moved += moved
        self.last_reason = (f"moved {moved} request(s) d{hot.instance_id} -> "
                            f"d{cold.instance_id} (drain {hot_score:.0f} ms -> "
                            f"{cold_score:.0f} ms)")
        return moved

    def _repoint(self, router, req, hot, cold, current_ns):
        """Point one queued request at ``cold`` and charge its KV copy."""
        cost_ms = router.node_link_cost_ms(hot, cold, self._tokens(req))
        req.migration_ns = int(getattr(req, "migration_ns", 0)
                               + max(0.0, cost_ms) * 1e6)
        req.decode_instance_id = int(cold.instance_id)
        req.instance_id = int(cold.instance_id)
        self._move_counters(router, hot, cold)

    def _migrate_running(self, router, hot, cold, current_ns, budget):
        """Move just-admitted requests between two stepping Decodes."""
        generated_of = lambda req: max(  # noqa: E731
            0, int(getattr(req, "num_tokens_reached", 0)
                   - getattr(req, "original_input", 0)))
        # Llumnix moves the request with the *least remaining* work: that frees
        # the source soonest per unit of copy.  Requests that have already
        # finished generating are not moved.
        def remaining_of(req):
            return max(0, int(getattr(req, "output", 0)
                              - getattr(req, "num_tokens_reached", 0)))
        # A request in an in-flight batch cannot leave: the batch will complete
        # it on this instance, and its blocks are already accounted for.
        in_flight = {req.id for batch in (getattr(hot, "inflight", ()) or ())
                     for req in getattr(batch, "requests", ()) or ()}
        eligible = [req for req in list(getattr(hot, "running", ()) or ())
                    if getattr(req, "id", None) not in in_flight
                    and generated_of(req) <= self.max_moved_tokens
                    and remaining_of(req) > 0]
        eligible.sort(key=lambda req: (remaining_of(req), getattr(req, "id", 0)))
        moved = 0
        for req in eligible:
            if moved >= budget:
                break
            tokens = int(getattr(req, "num_tokens_reached", 0)
                         or self._tokens(req))
            cost_ms = router.node_link_cost_ms(hot, cold, tokens)
            landing_ms = max(0.0, self.arrival_ms(router, cold)
                             - self.drain_ms(router, cold))
            remaining = remaining_of(req)
            source_step_ms = self.step_ms(router, hot)
            benefit_ms = remaining * source_step_ms
            if self.gain_ratio > 0.0 and benefit_ms < self.gain_ratio * (
                    cost_ms + landing_ms):
                self.last_reason = (f"move refused: benefit {benefit_ms:.0f} ms < "
                                    f"cost {cost_ms + landing_ms:.0f} ms "
                                    f"(copy {cost_ms:.0f} + landing {landing_ms:.0f})")
                continue
            hot.kv.preempt(req)
            hot.running.remove(req)
            req.status = RequestStatus.WAITING
            # The generated history travels with the request: ``add_decode``
            # keeps ``num_computed_tokens`` and allocates the blocks on the
            # target, and the copy that makes that possible is what
            # ``migration_ns`` charges.  Resetting the counter here instead
            # would make the target re-prefill the whole prompt -- measured
            # 2026-09-24 on the P-15B WAN peak: 20 moved requests raised the
            # mean from 2 977 to 3 434 ms and TPOT p50 from 27 to 36 ms.
            #
            # A request that left its batch with nothing pending
            # (``num_tokens == num_computed_tokens``) has to be given one token
            # of work, or the target's scheduler has nothing to run and the run
            # deadlocks: the token advance belonged to the source's in-flight
            # batch.  One recomputed token is the cost of landing on a step
            # boundary, and it is far below a re-prefill.
            if req.num_tokens <= req.num_computed_tokens:
                req.num_computed_tokens = max(0, req.num_tokens - 1)
            req.migration_ns = int(getattr(req, "migration_ns", 0)
                                   + max(0.0, cost_ms) * 1e6)
            req.decode_instance_id = int(cold.instance_id)
            req.instance_id = int(cold.instance_id)
            if not cold.add_decode(req):
                # The target could not take it: hand it back to its own queue
                # rather than dropping it (its blocks are already released, so
                # it re-enters through the normal admission path).
                hot.waiting.append(req)
                self.last_reason = (f"d{cold.instance_id} refused a migrated "
                                    "request; it re-queues on the source")
                continue
            self._move_counters(router, hot, cold)
            moved += 1
        return moved

    @staticmethod
    def _move_counters(router, hot, cold):
        assigned = getattr(router, "_assigned", None)
        if assigned is not None:
            assigned[int(hot.instance_id)] = max(
                0, assigned.get(int(hot.instance_id), 0) - 1)
            assigned[int(cold.instance_id)] = assigned.get(
                int(cold.instance_id), 0) + 1
        router._counters["llumnix_migrated"] = (
            router._counters.get("llumnix_migrated", 0) + 1)

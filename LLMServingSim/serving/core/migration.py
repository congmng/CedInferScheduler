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


class LlumnixMigrator:
    """Periodic Decode-queue rebalancing with a modelled copy cost."""

    def __init__(self, options=None):
        options = options or {}
        interval_ms = float(options.get("llumnix_interval_ms", 0.0) or 0.0)
        self.interval_ns = max(0, int(interval_ms * 1_000_000))
        self.hot_threshold = float(options.get("llumnix_hot_threshold", 1.0) or 0.0)
        self.cold_threshold = float(options.get("llumnix_cold_threshold", 0.5) or 0.0)
        self.min_gain = float(options.get("llumnix_min_gain", 0.25) or 0.0)
        self.batch = max(1, int(options.get("llumnix_batch", 4) or 1))
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
        scored = sorted(((router._instance_load_score("decode", item),
                          int(item.instance_id), item) for item in decodes),
                        key=lambda entry: (-entry[0], entry[1]))
        hot_score, _, hot = scored[0]
        cold_score, _, cold = scored[-1]
        if (hot_score < self.hot_threshold or cold_score > self.cold_threshold
                or hot_score - cold_score < self.min_gain):
            self.last_reason = (f"no eligible pair (hot {hot_score:.2f} -> "
                                f"cold {cold_score:.2f})")
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
        self.moved += moved
        self.last_reason = (f"moved {moved} request(s) d{hot.instance_id} -> "
                            f"d{cold.instance_id} (score {hot_score:.2f} -> "
                            f"{cold_score:.2f})")
        return moved

    def _repoint(self, router, req, hot, cold, current_ns):
        """Point one queued request at ``cold`` and charge its KV copy."""
        cost_ms = router.node_link_cost_ms(hot, cold, self._tokens(req))
        req.migration_ns = int(getattr(req, "migration_ns", 0)
                               + max(0.0, cost_ms) * 1e6)
        req.decode_instance_id = int(cold.instance_id)
        req.instance_id = int(cold.instance_id)
        assigned = getattr(router, "_assigned", None)
        if assigned is not None:
            assigned[int(hot.instance_id)] = max(
                0, assigned.get(int(hot.instance_id), 0) - 1)
            assigned[int(cold.instance_id)] = assigned.get(
                int(cold.instance_id), 0) + 1
        router._counters["llumnix_migrated"] = (
            router._counters.get("llumnix_migrated", 0) + 1)

"""Simulated Prefill worker lifecycle for CASR structural actions."""

from __future__ import annotations

import math

from .resources import ResourceOrchestrator


class PrefillLifecycle:
    """Coordinate worker lifecycle with a node-level resource pool."""

    def __init__(self, config=None):
        config = config or {}
        self.min_active = max(0, int(config.get("min_active_prefill", 1)))
        self.max_active = config.get("max_active_prefill")
        self.warmup_ns = max(0, int(float(config.get("warmup_ms", 0)) * 1_000_000))
        self.resources = ResourceOrchestrator(config.get("resources"))
        self._bootstrapped = False
        self.capacity = {int(key): float(value)
                         for key, value in config.get("prefill_capacity", {}).items()}
        self._warming_until = {}

    def update(self, current_ns, rows, schedulers):
        all_schedulers = list(schedulers)
        schedulers = [s for s in all_schedulers if s.pd_type == "prefill"]
        if not schedulers:
            return ()
        events = [event.as_dict() for event in self.resources.bootstrap(all_schedulers, current_ns)] if not self._bootstrapped else []
        self._bootstrapped = True
        startup_ns = max(self.warmup_ns, self.resources.startup_ns)
        demand = sum(max(float(row["arrival_rate_ewma"]), 0.0) for row in rows)
        average_capacity = max(1.0, sum(max(1.0, self.capacity.get(s.instance_id,
                                                                    float(s.max_num_seqs))) for s in schedulers) /
                               len(schedulers))
        desired = max(self.min_active, int(math.ceil(demand / average_capacity)))
        if self.max_active is not None:
            desired = min(desired, int(self.max_active))
        desired = min(desired, len(schedulers))
        ranked = sorted(schedulers, key=lambda s: (
            0 if s.admission_state == "ACTIVE" else 1,
            len(s.running) + len(s.waiting), s.instance_id))
        wanted = {scheduler.instance_id for scheduler in ranked[:desired]}
        for scheduler in schedulers:
            state = scheduler.admission_state
            if state == "WARMING" and current_ns >= self._warming_until.get(scheduler.instance_id, current_ns + startup_ns):
                scheduler.set_admission_state("ACTIVE")
                events.append({"instance_id": scheduler.instance_id, "action": "warm_complete"})
                state = "ACTIVE"
            if state == "DRAINING" and not scheduler.running and not scheduler.waiting:
                scheduler.set_admission_state("INACTIVE")
        resource_events = self.resources.reconcile(schedulers, wanted, current_ns)
        for event in resource_events:
            events.append(event.as_dict())
            if event.action == "resource_acquire":
                scheduler = next((s for s in schedulers if s.instance_id == event.instance_id), None)
                if scheduler is not None and scheduler.admission_state == "WARMING":
                    self._warming_until[event.instance_id] = current_ns + startup_ns
                    events.append({"instance_id": event.instance_id, "action": "warm_start"})
        return tuple(events)

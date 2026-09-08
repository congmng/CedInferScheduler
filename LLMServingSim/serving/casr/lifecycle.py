"""Simulated Prefill worker lifecycle for CASR structural actions."""

from __future__ import annotations

import math


class PrefillLifecycle:
    """Apply reversible admission-state transitions to pre-created workers."""

    def __init__(self, config=None):
        config = config or {}
        self.min_active = max(0, int(config.get("min_active_prefill", 1)))
        self.max_active = config.get("max_active_prefill")
        self.warmup_ns = max(0, int(float(config.get("warmup_ms", 0)) * 1_000_000))
        self.capacity = {int(key): float(value)
                         for key, value in config.get("prefill_capacity", {}).items()}
        self._warming_until = {}

    def update(self, current_ns, rows, schedulers):
        if not schedulers:
            return ()
        demand = sum(max(float(row["arrival_rate_ewma"]), 1.0) for row in rows)
        average_capacity = max(1.0, sum(max(1.0, self.capacity.get(s.instance_id,
                                                                    float(s.max_num_seqs))) for s in schedulers) /
                               len(schedulers))
        desired = max(self.min_active, int(math.ceil(demand / average_capacity)))
        if self.max_active is not None:
            desired = min(desired, int(self.max_active))
        desired = min(desired, len(schedulers))
        events = []
        ranked = sorted(schedulers, key=lambda s: (
            0 if s.admission_state == "ACTIVE" else 1,
            len(s.running) + len(s.waiting), s.instance_id))
        wanted = {scheduler.instance_id for scheduler in ranked[:desired]}
        for scheduler in schedulers:
            state = scheduler.admission_state
            if state == "WARMING" and current_ns >= self._warming_until.get(scheduler.instance_id, current_ns):
                scheduler.set_admission_state("ACTIVE")
                events.append({"instance_id": scheduler.instance_id, "action": "warm_complete"})
                state = "ACTIVE"
            if scheduler.instance_id in wanted:
                if state == "INACTIVE":
                    scheduler.set_admission_state("WARMING" if self.warmup_ns else "ACTIVE")
                    self._warming_until[scheduler.instance_id] = current_ns + self.warmup_ns
                    events.append({"instance_id": scheduler.instance_id,
                                   "action": "warm_start" if self.warmup_ns else "activate"})
            elif state in {"ACTIVE", "WARMING"}:
                if scheduler.running or scheduler.waiting:
                    scheduler.set_admission_state("DRAINING")
                    events.append({"instance_id": scheduler.instance_id, "action": "drain_start"})
                else:
                    scheduler.set_admission_state("INACTIVE")
                    events.append({"instance_id": scheduler.instance_id, "action": "deactivate"})
            elif state == "DRAINING" and not scheduler.running and not scheduler.waiting:
                scheduler.set_admission_state("INACTIVE")
                events.append({"instance_id": scheduler.instance_id, "action": "deactivate"})
        return tuple(events)

"""Rule-based structure controllers: the SOTA autoscaling baselines.

``StructuralEvaluator`` decides a pool edit by *counterfactual gain*: it
re-solves the flow LP with the candidate worker added or removed and only acts
when the windowed benefit clears its startup cost.  That is CASR's own
mechanism, and comparing against a rule-based autoscaler is the only way to
say whether the counterfactual pays for itself.

``ThresholdScaler`` is that baseline.  It mirrors the DOPD line of work
(``docs/SOTA覆盖与模拟器基线映射.md`` R5): pool size is a function of
observed utilisation and queue depth with hysteresis, not of a re-solved plan.
It is deliberately *not* CASR -- it never looks at the objective, the prefix
classes or the counterfactual gain -- and it shares the rest of the loop
(the solver still places the load, the lifecycle still executes the edit), so
the two arms differ in exactly one decision.
"""

from __future__ import annotations

from .evaluator import StructuralDecision, StructuralEvaluator


class ThresholdScaler:
    """Scale the Prefill pool on utilisation / queue thresholds (DOPD-style).

    Rules (all evaluated on the same tick as the plan):

    * **scale out** when the pool's offered load is at or above
      ``scale_out_utilization`` of what the *active* workers can push, or when
      any active worker's queue is at ``queue_scale_out_ratio`` of its slots.
    * **scale in** when utilisation is at or below ``scale_in_utilization``
      *and* every active worker is idle (nothing running, nothing waiting).
      A worker with work in flight is never drained -- the rule has no
      model of where that work would go.
    * Both directions need the condition to hold for ``scale_ticks``
      consecutive ticks, and a scale-out is held for ``threshold_hold_ms``
      before a scale-in may undo it.

    ``min_active_prefill`` / ``max_active_prefill`` are the operator's bounds
    and always win, exactly as they do for the counterfactual evaluator.
    """

    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.scale_out_utilization = float(config.get("scale_out_utilization", 0.8))
        self.scale_in_utilization = float(config.get("scale_in_utilization", 0.3))
        self.queue_scale_out_ratio = float(config.get("queue_scale_out_ratio", 0.5))
        self.queue_scale_in_ratio = float(config.get("queue_scale_in_ratio", 0.1))
        self.required_ticks = max(1, int(config.get("scale_ticks", 3) or 1))
        self.startup_s = max(0.0, float(config.get("startup_s", 0.0)))
        dwell_ns = max(0, int(float(config.get("dwell_time_ms", 0)) * 1_000_000))
        self.dwell_ns = max(dwell_ns, int(self.startup_s * 1e9))
        self.hold_ns = max(0, int(float(config.get("threshold_hold_ms", 0)) * 1_000_000))
        max_active = int(config.get("max_active_prefill", 0) or 0)
        self.max_active_prefill = max_active if max_active > 0 else None
        self._hot_ticks = 0
        self._cold_ticks = 0
        self._hold_until_ns = -1
        #: Last tick's reading, so the run summary can show what the rule saw.
        self.last_metrics = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _capacity_of(solver, scheduler):
        table = getattr(solver.config, "prefill_capacity", {}) or {}
        value = table.get(int(scheduler.instance_id))
        if value in (None, 0):
            value = getattr(scheduler, "max_num_seqs", 0)
        return max(0.0, float(value or 0.0))

    @staticmethod
    def _queue_ratio(scheduler):
        slots = max(1, int(getattr(scheduler, "max_num_seqs", 1) or 1))
        queued = len(getattr(scheduler, "waiting", ()) or ()) * 4 \
            + len(getattr(scheduler, "running", ()) or ())
        return queued / slots

    def _metrics(self, rows, active_prefill, solver, backlog_rps):
        offered = sum(max(0.0, float(row.get("arrival_rate_ewma", 0.0) or 0.0))
                      for row in rows)
        capacity = sum(self._capacity_of(solver, item) for item in active_prefill)
        queue_ratio = max((self._queue_ratio(item) for item in active_prefill),
                          default=0.0)
        busy = any(getattr(item, "running", ()) or getattr(item, "waiting", ())
                   for item in active_prefill)
        return {
            "offered_rps": offered,
            "active_capacity_rps": capacity,
            "utilization": (offered / capacity) if capacity > 0.0 else float("inf"),
            "offered_with_backlog_rps": offered + max(0.0, float(backlog_rps or 0.0)),
            "queue_ratio": queue_ratio,
            "all_idle": not busy,
        }

    # ------------------------------------------------------------------
    def evaluate(self, snapshot, active_prefill, all_prefill, decode, solver,
                 current_ns, last_action_ns=-1, min_active=1, backlog_rps=0.0):
        rows = snapshot["prefix_states"]
        keep = lambda reason, objective=0.0: StructuralDecision(  # noqa: E731
            "keep", "none", 0.0, objective, objective,
            tuple(item.instance_id for item in active_prefill), reason)
        if not active_prefill or not decode or not rows:
            return keep("insufficient active workers or observed classes")
        if not self.enabled:
            return keep("threshold autoscaling disabled")

        metrics = self._metrics(rows, active_prefill, solver, backlog_rps)
        self.last_metrics = metrics
        # The backlog is the part of the offer the pool has *not* served yet;
        # a rule-based scaler that only watched `arrival_rate_ewma` would see a
        # saturated pool as idle (measured: the CASR controller had to recover
        # the same missing demand).  Adding it keeps the threshold honest.
        utilization = metrics["utilization"]
        if backlog_rps and metrics["active_capacity_rps"] > 0.0:
            utilization = (metrics["offered_with_backlog_rps"]
                           / metrics["active_capacity_rps"])
        queue_ratio = metrics["queue_ratio"]

        hot = (utilization >= self.scale_out_utilization
               or queue_ratio >= self.queue_scale_out_ratio)
        cold = (utilization <= self.scale_in_utilization
                and queue_ratio <= self.queue_scale_in_ratio
                and metrics["all_idle"])
        self._hot_ticks = self._hot_ticks + 1 if hot else 0
        self._cold_ticks = self._cold_ticks + 1 if cold else 0
        if not hot and not cold:
            return keep(
                f"utilization {utilization:.2f} / queue {queue_ratio:.2f} "
                "inside the hysteresis band")
        if last_action_ns >= 0 and current_ns - last_action_ns < self.dwell_ns:
            return keep("dwell time has not elapsed")

        pool = [item for item in (all_prefill or active_prefill)
                if getattr(item, "admission_state", "ACTIVE") != "INACTIVE"]
        if hot and self._hot_ticks >= self.required_ticks:
            if self.max_active_prefill is not None and len(pool) >= self.max_active_prefill:
                return keep(f"pool already at max_active_prefill "
                            f"({self.max_active_prefill})")
            inactive = [item for item in all_prefill
                        if item not in active_prefill
                        and getattr(item, "admission_state", "ACTIVE") == "INACTIVE"]
            if not inactive:
                return keep("no inactive worker available")
            # Rule-based choice: the spare that removes the most utilisation.
            candidate = max(inactive, key=lambda item: (self._capacity_of(solver, item),
                                                        -int(item.instance_id)))
            wanted = tuple(item.instance_id for item in active_prefill) \
                + (candidate.instance_id,)
            self._hold_until_ns = int(current_ns + max(self.hold_ns, self.dwell_ns))
            return StructuralDecision(
                "+P", "cold", 0.0, 0.0, 0.0, wanted,
                f"utilisation {utilization:.2f} >= {self.scale_out_utilization:.2f} "
                f"(queue {queue_ratio:.2f}) for {self._hot_ticks} tick(s)",
                ())
        if cold and self._cold_ticks >= self.required_ticks:
            if len(active_prefill) <= min_active:
                return keep(f"pool already at min_active_prefill ({min_active})")
            if self._hold_until_ns >= 0 and current_ns < self._hold_until_ns:
                return keep("scale-out still within its hold window")
            candidate = min(active_prefill,
                            key=lambda item: (self._capacity_of(solver, item),
                                              int(item.instance_id)))
            wanted = tuple(item.instance_id for item in active_prefill
                           if item is not candidate)
            return StructuralDecision(
                "-P", "none", 0.0, 0.0, 0.0, wanted,
                f"utilisation {utilization:.2f} <= {self.scale_in_utilization:.2f} "
                f"and pool idle for {self._cold_ticks} tick(s)",
                ())
        return keep(
            f"threshold not sustained yet "
            f"(hot {self._hot_ticks}, cold {self._cold_ticks} of "
            f"{self.required_ticks})")


def build_structural_evaluator(config=None):
    """Pick the structural rule the run asked for.

    ``counterfactual`` (default) keeps CASR's gain-evaluated evaluator, so no
    existing configuration changes behaviour; ``dopd``/``threshold`` selects
    the rule-based autoscaler above.
    """
    config = config or {}
    rule = str(config.get("rule", "counterfactual") or "counterfactual").lower()
    if rule in ("dopd", "threshold", "utilization", "utilisation"):
        return ThresholdScaler(config)
    if rule in ("counterfactual", "gain", "casr"):
        return StructuralEvaluator(config)
    raise ValueError(
        f"unknown structural rule {rule!r}; expected counterfactual or dopd")

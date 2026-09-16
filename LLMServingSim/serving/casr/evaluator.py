"""Counterfactual Prefill structure evaluation for CASR."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralDecision:
    action: str
    mode: str
    gain: float
    base_objective: float
    candidate_objective: float
    wanted_ids: tuple[int, ...]
    reason: str
    warm_classes: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "action": self.action,
            "mode": self.mode,
            "gain": self.gain,
            "base_objective": self.base_objective,
            "candidate_objective": self.candidate_objective,
            "wanted_ids": list(self.wanted_ids),
            "reason": self.reason,
            "warm_classes": list(self.warm_classes),
        }


class StructuralEvaluator:
    """Compare one-step P row edits using the same inner flow solver.

    Economics (added 2026-09-14 after the real-cluster measurement):

    * ``evaluation_window_ms`` is the *horizon over which the edit has to pay
      for itself* -- i.e. the expected remaining duration of the overload, not
      a 1 s control tick.
    * ``startup_s`` is how long a scaled-out Prefill needs before it can serve
      anything (measured: 45 s for a container restart plus ``/metrics``), so
      the benefit of ``+P`` only accrues over ``horizon - startup_s``.
    * ``idle_cost_fraction`` (or an explicit ``holding_cost``) charges every
      active Prefill for merely existing, which is what lets ``-P`` ever be
      profitable: without a holding cost, removing an instance could only make
      the remaining ones busier, so its counterfactual gain was mechanically
      negative (measured on the real fabric: -1.25, -0.56, -10.09, -10.20).
    * ``max_active_prefill`` caps scale-out; previously every configured spare
      could be started, one per tick, with nothing to stop at.
    """

    def __init__(self, config=None):
        config = config or {}
        self.window_ns = max(1, int(float(config.get("evaluation_window_ms", 1000)) * 1_000_000))
        self.threshold_abs = float(config.get("gain_threshold_abs", 1e-4))
        self.threshold_rel = float(config.get("gain_threshold_rel", 0.01))
        self.dwell_ns = max(0, int(float(config.get("dwell_time_ms", 0)) * 1_000_000))
        self.startup_cost = float(config.get("startup_cost", 0.0))
        self.warm_cost = float(config.get("warm_cost", 0.0))
        self.warm_top_k = max(0, int(config.get("warm_top_k", 8)))
        self.warm_budget_bytes = max(0.0, float(config.get("warm_budget_bytes", 0.0)))
        self.enabled = bool(config.get("enabled", True))
        self.enable_warm = bool(config.get("enable_warm_counterfactual", True))
        # A scale decision has to outlive its own start-up, otherwise the next
        # tick reverses it: the cooldown is the boot time unless the operator
        # asked for a longer one.
        self.startup_s = max(0.0, float(config.get("startup_s", 0.0)))
        self.dwell_ns = max(self.dwell_ns, int(self.startup_s * 1e9))
        self.holding_cost = max(0.0, float(config.get("holding_cost", 0.0)))
        self.idle_cost_fraction = max(0.0, float(config.get("idle_cost_fraction", 0.0)))
        max_active = int(config.get("max_active_prefill", 0) or 0)
        self.max_active_prefill = max_active if max_active > 0 else None
        # Predictive scale-out: start a spare once the offered load is within
        # ``prescale_utilization`` of what the active pool can actually push.
        # A counterfactual that has to *observe* the gain cannot pay for a 45 s
        # boot inside a 90 s peak -- measured in the six-domain arena, the
        # reactive ``+P`` arm settled at 14572 ms where an egress-sized pool
        # reached 1120 ms.  ``0`` disables the predictive branch.
        self.prescale_utilization = float(config.get("prescale_utilization", 0.8) or 0.0)

    def _holding_per_instance(self, instances, solver):
        """Per-second cost of keeping one Prefill active, in objective units.

        ``holding_cost`` is used verbatim when configured.  Otherwise the cost
        is a fraction of what a saturated instance of that class of hardware
        costs per second (``service_s x capacity``), averaged over the active
        set -- an explicit, tunable stand-in for the reserved-GPU opportunity
        cost rather than a magic constant.
        """
        if self.holding_cost > 0.0:
            return self.holding_cost
        if self.idle_cost_fraction <= 0.0 or not instances:
            return 0.0
        config = getattr(solver, "config", None)
        total = 0.0
        for instance in instances:
            instance_id = int(instance.instance_id)
            capacity = float(getattr(config, "prefill_capacity", {}).get(instance_id, 0.0)
                             or 0.0)
            service_ms = float(getattr(config, "prefill_service_ms", {}).get(
                instance_id, getattr(instance, "service_ms", 0.0)) or 0.0)
            if capacity > 0.0 and service_ms > 0.0:
                total += capacity * service_ms / 1000.0
            else:
                total += 1.0
        return self.idle_cost_fraction * total / max(1, len(instances))

    @staticmethod
    def _hit_work(rows, class_id):
        ratios = []
        for row in rows:
            if row["class_id"] != class_id:
                continue
            # ``hit_tokens_ewma`` is per request, so prefer the per-request
            # denominator and only fall back to the cumulative counter.
            requested = float(row.get("requested_tokens_ewma", 0.0) or 0.0)
            if requested <= 0:
                requested = float(row.get("requested_tokens", 0.0))
            ratios.append(min(0.95, float(row.get("hit_tokens_ewma", 0.0)) /
                              max(1.0, requested)))
        return max(0.05, 1.0 - max(ratios or [0.0]))

    def _warm_classes(self, rows):
        grouped = {}
        for row in rows:
            class_id = row["class_id"]
            grouped.setdefault(class_id, []).append(row)
        ranked = []
        for class_id, class_rows in grouped.items():
            demand = sum(float(row.get("arrival_rate_ewma", 0.0)) for row in class_rows)
            hit = 1.0 - self._hit_work(rows, class_id)
            bytes_per_request = max(float(row.get("kv_bytes_per_request", 0.0))
                                    for row in class_rows)
            ranked.append((demand * hit, class_id, bytes_per_request))
        selected = []
        used_bytes = 0.0
        for _, class_id, bytes_per_request in sorted(ranked, reverse=True):
            if len(selected) >= self.warm_top_k:
                break
            if self.warm_budget_bytes and used_bytes + bytes_per_request > self.warm_budget_bytes:
                continue
            selected.append(class_id)
            used_bytes += bytes_per_request
        return tuple(selected)

    def _predictive_scale_decision(self, rows, active_prefill, all_prefill, solver,
                                   decode, current_ns, base_objective, warm_classes):
        """Start a spare when the offer approaches the pool's *push* capacity.

        The gain-evaluated counterfactual has to see the overload before it acts,
        and then pays the boot inside what is left of the peak: in the six-domain
        arena (90 s at 4 req/s against a 45 s boot) the reactive arm reached only
        14572 ms while an egress-sized pool reached 1120 ms.  The signal used
        here is available *before* the backlog: the offered rate the profiler
        already reports against the capacity the solver prices, both of which
        the controller fills with the producer's 0.26 GB/s egress for 1250-token
        prompts (~1.4 req/s per worker).  Ordering by lowest cost keeps the
        choice of *which* spare consistent with the plan's own objective.
        """
        pool = [item for item in (all_prefill or active_prefill)
                if getattr(item, "admission_state", "ACTIVE") != "INACTIVE"]
        if self.max_active_prefill is not None and len(pool) >= self.max_active_prefill:
            return None
        inactive = [s for s in all_prefill
                    if s.admission_state == "INACTIVE"]
        if not inactive:
            return None
        offered = sum(max(0.0, float(row.get("arrival_rate_ewma", 0.0))) for row in rows)
        if offered <= 0.0:
            return None
        capacities = getattr(solver.config, "prefill_capacity", {})
        active_capacity = sum(
            float(capacities.get(int(s.instance_id), 0.0) or 0.0)
            for s in active_prefill)
        if active_capacity <= 0.0:
            return None
        if offered <= self.prescale_utilization * active_capacity:
            return None
        candidate = sorted(inactive, key=lambda item: item.instance_id)[0]
        return StructuralDecision(
            "+P", "cold", offered - active_capacity, base_objective, base_objective,
            tuple(item.instance_id for item in active_prefill) + (candidate.instance_id,),
            f"predictive scale-out: offered {offered:.2f} req/s is above "
            f"{self.prescale_utilization:.0%} of the active pool's "
            f"{active_capacity:.2f} req/s push capacity",
            tuple(warm_classes or ()))

    def evaluate(self, snapshot, active_prefill, all_prefill, decode, solver,
                 current_ns, last_action_ns=-1, min_active=1, backlog_rps=0.0):
        rows = snapshot["prefix_states"]
        if not active_prefill or not decode or not rows:
            return StructuralDecision("keep", "none", 0.0, 0.0, 0.0,
                                      tuple(s.instance_id for s in active_prefill),
                                      "insufficient active workers or observed classes")
        if not self.enabled:
            solver.solve(rows, active_prefill, decode)
            objective = float(solver.diagnostics.get("objective", 0.0))
            return StructuralDecision("keep", "none", 0.0, objective, objective,
                                      tuple(s.instance_id for s in active_prefill),
                                      "structural gain evaluation disabled")
        if last_action_ns >= 0 and current_ns - last_action_ns < self.dwell_ns:
            solver.solve(rows, active_prefill, decode)
            objective = float(solver.diagnostics.get("objective", 0.0))
            return StructuralDecision("keep", "none", 0.0, objective, objective,
                                      tuple(s.instance_id for s in active_prefill),
                                      "dwell time has not elapsed")

        solver.solve(rows, active_prefill, decode)
        base_objective = float(solver.diagnostics.get("objective", 0.0))
        demand_scale = max(1.0, sum(float(row.get("arrival_rate_ewma", 0.0)) for row in rows))
        candidates = []
        warm_classes = self._warm_classes(rows)
        if self.prescale_utilization > 0.0:
            predictive = self._predictive_scale_decision(
                rows, active_prefill, all_prefill, solver, decode,
                current_ns, base_objective, warm_classes)
            if predictive is not None:
                return predictive

        horizon_s = self.window_ns / 1_000_000_000.0
        # A *growing* backlog means the imbalance is not a transient: the
        # counterfactual has to be judged over the time the pressure will
        # actually last, not over one evaluation window.  Without this the
        # 45 s boot is charged against a 60 s window and only 15 s of benefit
        # remain, so ``+P`` waited until the peak was half over -- measured in
        # the six-domain arena: the structural arm settled at 14878 ms while an
        # egress-sized pool reached 1120 ms, and ``casr_lp`` with the
        # lifecycle's demand-driven sizing (which reacts immediately) reached
        # 1400 ms.
        if backlog_rps > 0.05:
            horizon_s = max(horizon_s, min(4.0 * horizon_s,
                                           horizon_s * (1.0 + backlog_rps)))
        # What one more (or one fewer) active Prefill costs per second.  It
        # appears with opposite signs on the two branches below, which is what
        # makes the criterion symmetric instead of "grow whenever possible".
        holding = self._holding_per_instance(active_prefill, solver)
        # The ceiling counts every worker the pool still pays for, not only the
        # ones already serving: a WARMING boot and a DRAINING drain both still
        # hold their GPU, so starting another on top of them would exceed the
        # operator's limit.  Counting only ``active_prefill`` let a booting
        # worker be ignored and the pool overshoot.
        pool = [item for item in (all_prefill or active_prefill)
                if getattr(item, "admission_state", "ACTIVE") != "INACTIVE"]
        can_grow = (self.max_active_prefill is None
                    or len(pool) < self.max_active_prefill)
        inactive = [s for s in all_prefill if s not in active_prefill and
                    s.admission_state == "INACTIVE"]
        if not can_grow:
            inactive = []
        for candidate in sorted(inactive, key=lambda item: item.instance_id):
            candidate_prefill = list(active_prefill) + [candidate]
            modes = [("cold", self.startup_cost,
                      {(candidate.instance_id, row["class_id"]): 1.0 for row in rows})]
            if self.enable_warm and warm_classes:
                modes.append(("warm", self.startup_cost + self.warm_cost,
                              {(candidate.instance_id, row["class_id"]):
                               (self._hit_work(rows, row["class_id"])
                                if row["class_id"] in warm_classes else 1.0)
                               for row in rows}))
            for mode, default_cost, overrides in modes:
                solver.solve(rows, candidate_prefill, decode, overrides)
                candidate_objective = float(solver.diagnostics.get("objective", 0.0))
                # The new instance only contributes after it has booted, and it
                # has to be paid for from the moment it is started.
                effective_horizon = max(0.0, horizon_s - self.startup_s)
                gain = effective_horizon * (base_objective - candidate_objective - holding)
                gain -= default_cost
                candidates.append(StructuralDecision(
                    "+P", mode, gain, base_objective, candidate_objective,
                    tuple(item.instance_id for item in candidate_prefill),
                    f"counterfactual add Prefill {candidate.instance_id}",
                    warm_classes if mode == "warm" else ()))

        # Shrinking the pool while it has work in flight is never justified by
        # the counterfactual: the solver prices *capacity*, so removing a worker
        # only ever removes headroom, and the requests it was carrying are not
        # modelled at all.  Measured 2026-09-16 on the small cluster: with a
        # light load (1.05 req/s over two Prefills) every instance is briefly
        # idle between requests, so a per-candidate check was not enough -- the
        # controller removed p5090 twice and the pool collapsed both times.
        # Gate the whole branch on the pool being quiet *and* on the candidate
        # having nothing assigned (``inflight`` is the deployment's
        # dispatched-not-finished count; the simulator only has running/waiting).
        pool_busy = any(getattr(item, "waiting", ()) or getattr(item, "running", ())
                        for item in active_prefill)
        if len(active_prefill) > min_active and not pool_busy:
            # Every active instance is a candidate: which one is least useful is
            # a question for the counterfactual, not for the instance id.
            for candidate in sorted(active_prefill, key=lambda item: item.instance_id):
                if (getattr(candidate, "running", ()) or
                        getattr(candidate, "waiting", ()) or
                        float(getattr(candidate, "inflight", 0) or 0) > 0.0):
                    continue
                candidate_prefill = [item for item in active_prefill if item is not candidate]
                solver.solve(rows, candidate_prefill, decode)
                candidate_objective = float(solver.diagnostics.get("objective", 0.0))
                # Removal is immediate (drain only delays it), so the full
                # horizon applies, and the holding cost it saves counts as a
                # benefit -- this is the only way "-P" can be positive.
                gain = horizon_s * (base_objective - candidate_objective + holding)
                candidates.append(StructuralDecision(
                    "-P", "none", gain, base_objective, candidate_objective,
                    tuple(item.instance_id for item in candidate_prefill),
                    f"counterfactual remove Prefill {candidate.instance_id}"))

        if not candidates:
            return StructuralDecision("keep", "none", 0.0, base_objective, base_objective,
                                      tuple(s.instance_id for s in active_prefill),
                                      "no eligible one-step structure edit")
        best = max(candidates, key=lambda item: (item.gain, item.action, item.mode))
        relative = best.gain / max(abs(self.window_ns / 1_000_000_000.0 * base_objective), demand_scale)
        if best.gain <= self.threshold_abs or relative <= self.threshold_rel:
            return StructuralDecision("keep", "none", best.gain, base_objective,
                                      base_objective,
                                      tuple(s.instance_id for s in active_prefill),
                                      f"best gain below threshold: {best.reason}")
        return best

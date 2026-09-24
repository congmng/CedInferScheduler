"""Deterministic slow control loop for the first CASR simulator integration.

The controller deliberately has no solver dependency.  It turns observed
prefix-class demand into an executable, capacity-aware affinity plan.  The
interface is kept separate from the policy so an OR-Tools LP can replace this
greedy baseline without changing the router or the trace path.
"""

from __future__ import annotations

from .affinity import AffinityPlan
from .plan_builder import build_affinity_plan
from .flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
from .lifecycle import PrefillLifecycle
from .policy import PolicyError, load_policy
from typing import Mapping

from .autoscalers import build_structural_evaluator
from .state import PrometheusStateCollector
from .executor import ReconfigExecutor


class CASRController:
    """Build a reproducible class-to-P/D plan at fixed simulated-time ticks."""

    def __init__(self, interval_ns: int, plan_ttl_ns: int | None = None, policy=None,
                 policy_spec="builtin"):
        self.interval_ns = max(1, int(interval_ns))
        self.plan_ttl_ns = int(plan_ttl_ns or interval_ns * 2)
        self.next_tick_ns = 0
        self.version = 0
        self.solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(policy))
        self.policy = load_policy(policy_spec, policy)
        self.policy_spec = policy_spec
        #: Clamp the offered rate to what the pool can serve before solving.
        #: On by default: above capacity the LP's choice of vertex is arbitrary
        #: and can single-home a peak onto the slowest worker (see
        #: ``_clamped_rows``).  ``demand_capacity_clamp: false``
        #: restores the raw behaviour for A/B.
        self.demand_capacity_clamp = bool(
            (policy or {}).get("demand_capacity_clamp", True))
        self.last_demand_clamp = {}
        lifecycle_policy = dict((policy or {}).get("lifecycle", {}))
        lifecycle_policy.setdefault("prefill_capacity", (policy or {}).get("prefill_capacity", {}))
        lifecycle_policy.setdefault("decode_capacity", (policy or {}).get("decode_capacity", {}))
        lifecycle_policy.setdefault("resources", (policy or {}).get("resources", {}))
        # The producer-side egress budget lives in the solver/lifecycle-facing
        # policy (``casr.shared_links``, or the cluster-level ``kv_egress_gbps``
        # that the entry point folds in).  The lifecycle's demand->worker-count
        # heuristic needs it: without the term it only sees compute capacity and
        # keeps one worker while the run is link-bound (measured 2026-09-15).
        lifecycle_policy.setdefault("shared_links", (policy or {}).get("shared_links", ()))
        lifecycle_policy.setdefault("kv_egress_gbps", (policy or {}).get("kv_egress_gbps"))
        # Prompt-length model: the lifecycle scales capacity the same way the
        # solver does, so it needs the measured token ceiling and the per-token
        # KV size (the simulator's snapshots do not fill
        # ``kv_bytes_per_request``).
        for key in ("prefill_tokens_per_s", "capacity_reference_tokens",
                    "kv_bytes_per_token"):
            if (policy or {}).get(key) is not None:
                lifecycle_policy.setdefault(key, policy[key])
        self.lifecycle = PrefillLifecycle(lifecycle_policy)
        # Coordinated autoscaling: when the config declares a Decode pool bound
        # (``decode_min_active`` / ``decode_max_active``), the same lifecycle
        # machinery also sizes the Decode pool, from the class demand weighted
        # by output length.  Absent those keys nothing changes: every existing
        # configuration keeps its fixed Decode set.
        self.decode_lifecycle = None
        if lifecycle_policy.get("decode_max_active") is not None \
                or lifecycle_policy.get("decode_min_active") is not None:
            self.decode_lifecycle = PrefillLifecycle(
                lifecycle_policy, role="decode", work_of=self.solver.decode_work)
        # ``max_active_prefill`` is declared with the lifecycle block, but it is
        # the ceiling on *structural* scale-out.  Passing only
        # ``casr.structural`` to the evaluator left it without a cap, so a pool
        # declared ``max_active_prefill=1`` still grew to three workers
        # (measured 2026-09-15: the "static one worker" arm of the elasticity
        # replay acquired Prefills 6 and 4 at t+1.8 s / t+47.3 s).
        structural_policy = dict((policy or {}).get("structural", {}))
        if not structural_policy.get("max_active_prefill"):
            cap = lifecycle_policy.get("max_active_prefill")
            if cap:
                structural_policy["max_active_prefill"] = int(cap)
        # ``structural.rule`` picks the *rule*: CASR's counterfactual gain
        # evaluation (default) or the DOPD-style threshold autoscaler that the
        # SOTA comparison needs (``autoscalers.ThresholdScaler``).
        self.evaluator = build_structural_evaluator(structural_policy)
        self.state_collector = PrometheusStateCollector((policy or {}).get("telemetry", {}))
        self.executor = ReconfigExecutor((policy or {}).get("executor", {}))
        self.last_action_ns = -1
        self.last_flows = ()
        self.last_lifecycle = ()
        self.last_decode_lifecycle = ()
        self.last_warmups = ()
        self.last_solver_diagnostics = {}
        self.last_resource_snapshot = {}
        self.last_structural_decision = {}
        self.pending_warm_classes = {}
        self.last_telemetry = {}
        self.last_execution = ()
        # Backlog-aware demand, mirroring the real controller's
        # ``_sample_backlog``: the profiler only sees the requests the router
        # already admitted, so under back-pressure the observed arrival rate
        # collapses to the *served* rate and the LP is told "demand == capacity"
        # exactly when the pool is drowning.  The unserved part of the offered
        # load is sitting in the Prefills' waiting queues, so its growth rate is
        # the missing demand.  Measured 2026-09-16 on the small-cluster
        # elasticity A/B: the simulator reported 108 req/s of capacity against
        # 9 req/s of demand while the run was 23 s deep in backlog, so every
        # ``+P`` counterfactual came out "gain below threshold" and the elastic
        # arm stayed bit-identical to the static one.
        self.alpha = float((policy or {}).get("ewma_alpha", 0.2) or 0.2)
        self.backlog_rps_multiple = max(
            0.0, float((policy or {}).get("backlog_rps_multiple", 4.0) or 0.0))
        # Recovering the offered load must not invent demand the pool could not
        # serve anyway; see ``_inflate_demand``.
        self.backlog_capacity_bound = bool(
            (policy or {}).get("backlog_capacity_bound", True))
        self._backlog_rps = 0.0
        self._last_backlog_waiting = None
        self._last_backlog_ns = 0
        self._backlog_active = False

    def due(self, current_ns: int) -> bool:
        return int(current_ns) >= self.next_tick_ns

    def _sample_backlog(self, current_ns, prefill):
        """Recover the offered load from the Prefills' waiting queues.

        Only *growth* counts: a draining queue must not subtract demand, and the
        EWMA decays the term once the backlog stops building.  Identical to the
        real controller's version (``deploy/real_lmcache_pd/casr_control.py``),
        which scrapes the same number off vLLM's metrics; here the schedulers
        are in-process.
        """
        waiting = float(sum(len(s.waiting) for s in prefill))
        # ``_last_backlog_waiting`` is the "have we sampled yet" sentinel: a
        # timestamp cannot be one, because the first control tick is at t=0.
        elapsed_s = ((current_ns - self._last_backlog_ns) / 1e9
                     if self._last_backlog_waiting is not None else 0.0)
        growth = 0.0
        if elapsed_s > 0.0 and self._last_backlog_waiting is not None:
            growth = max(0.0, waiting - self._last_backlog_waiting) / elapsed_s
        self._last_backlog_waiting = waiting
        self._last_backlog_ns = int(current_ns)
        if not self._backlog_active:
            self._backlog_rps = growth
            self._backlog_active = True
        else:
            self._backlog_rps = (self.alpha * growth +
                                 (1.0 - self.alpha) * self._backlog_rps)
        return self._backlog_rps

    def _egress_bound_prefill_capacity(self, rows, prefills):
        """Per-Prefill capacity once its KV egress is taken into account.

        The deployment's push ceiling is 0.26 GB/s (measured 239-314 MB/s), and
        a 1250-token prompt is 184 MB of KV, so one worker can push ~1.4
        requests/s no matter how fast its compute is.  The plan and the
        structural evaluator need that number: with compute capacity alone
        (9.3 rps on a 5090) the counterfactual for one more worker is
        unprofitable until the run is already deep in backlog -- measured in
        the six-domain arena, the ``+P`` arm settled at 14878 ms while the same
        pool floor with an egress-derived capacity reached 1400 ms.
        """
        budgets = {}
        for link in getattr(self.solver.config, "shared_links", ()):
            capacity = float(getattr(link, "capacity_bytes_per_s", 0.0) or 0.0)
            if capacity <= 0:
                continue
            for prefill_id, _decode_id in (getattr(link, "pairs", ()) or ()):
                # A producer's *best* pairing, not its worst.  A deployment
                # prices the two paths separately (the measured same-host push
                # is faster than the wire), and the LP's own per-link byte
                # constraints bound each pairing on its own.  Collapsing them
                # with ``min`` charged every producer at the cross-domain rate:
                # measured 2026-09-24 on the P-15B WAN environment the plan
                # capped the 5090 at 5.19 req/s while it executed 8.3, so it
                # idled while the slowest Prefill's queue grew to 203 requests.
                budgets[int(prefill_id)] = max(budgets.get(int(prefill_id), 0.0),
                                               capacity)
        if not budgets:
            return {}
        per_token = float(getattr(self.solver.config, "kv_bytes_per_token", 0.0) or 0.0)
        tokens = []
        for row in rows:
            requested = row.get("requested_tokens")
            if isinstance(requested, Mapping):
                tokens.extend(float(value) for value in requested.values() if value)
            elif requested:
                tokens.append(float(requested))
            else:
                ewma = row.get("requested_tokens_ewma")
                if isinstance(ewma, Mapping):
                    tokens.extend(float(value) for value in ewma.values() if value)
                elif ewma:
                    tokens.append(float(ewma))
        if not tokens or per_token <= 0:
            return {}
        tokens.sort()
        median_tokens = tokens[len(tokens) // 2]
        per_request = per_token * max(1.0, median_tokens)
        declared = {int(s.instance_id): float(
            self.solver.config.prefill_capacity.get(
                int(s.instance_id), getattr(s, "max_num_seqs", 1) or 1))
            for s in prefills}
        capped = {}
        for instance_id, budget in budgets.items():
            if instance_id not in declared:
                continue
            limit = budget / per_request
            capped[instance_id] = min(declared[instance_id], max(0.05, limit))
        return capped

    def _inflate_demand(self, rows):
        """Add the backlog growth back to the observed class rates.

        The queue is not class-attributed, so the missing rate is spread
        proportionally to each class's observed share (a mean-field
        approximation).

        Two caps, and the second is the one that matters:

        * ``backlog_rps_multiple`` x the observed total (a transient cannot
          invent unbounded demand), and
        * the **capacity deficit** -- the part of the observed load the pool
          cannot serve.  Recovering *offered* load only makes sense when the
          pool is actually saturated: at 16 req/s against a 26 req/s pool the
          deficit is zero, and adding the queue's growth to every class made
          the solver plan for 88 req/s (measured 2026-09-23, matrix run).  With
          demand that far above every instance, the LP minimises summed
          overflow by topping up the small instances and dumping the rest on
          the largest one -- 76 req/s onto a Decode whose capacity is 12.5 --
          which is the worst possible tail.  Above capacity *every* plan is
          infeasible, so the planner only needs to know how far above; the
          deficit is that number.
        """
        observed = sum(max(0.0, float(row.get("arrival_rate_ewma") or 0.0))
                       for row in rows)
        if observed <= 0.0 or self._backlog_rps <= 0.0:
            return rows
        backlog = min(self._backlog_rps, self.backlog_rps_multiple * observed)
        if self.backlog_capacity_bound:
            servable = self._servable_reference_units()
            if servable > 0.0:
                backlog = min(backlog, max(0.0, observed - servable))
        scale = 1.0 + backlog / observed
        for row in rows:
            rate = float(row.get("arrival_rate_ewma") or 0.0)
            row["arrival_rate_ewma"] = max(0.0, rate) * scale
        return rows

    def _servable_reference_units(self) -> float:
        """How much reference-length load the current pool can absorb.

        The binding side is whichever is smaller: the Prefills' capacity (the
        controller rewrites it each tick from the egress bound) or the Decodes'.
        Returns 0.0 when either side is unknown, which disables the cap.
        """
        config = getattr(self.solver, "config", None)
        if config is None:
            return 0.0
        prefill = [float(value) for value in (getattr(config, "prefill_capacity", {}) or {}).values()
                   if value and value > 0.0]
        decode = [float(value) for value in (getattr(config, "decode_capacity", {}) or {}).values()
                  if value and value > 0.0]
        if not prefill or not decode:
            return 0.0
        return min(sum(prefill), sum(decode))

    def _clamped_rows(self, rows, prefills):
        """Defined degradation: plan the load the pool can actually serve.

        Above the pool's capacity every plan is infeasible, and in that regime
        the LP's objective is flat in the *distribution* -- total overflow is
        the same wherever the excess lands, so whichever vertex the simplex
        reaches decides, and it may put most of the peak on the slowest card
        (measured 2026-09-23, the 32 rps Qwen3-8B cell: 68% of 977 requests on
        the 3090 Prefill, mean 67.9 s against the baseline's 25.4 s).

        Scaling every class's rate by ``servable / offered`` keeps the offer's
        *shape* and makes the problem feasible, so the LP allocates by cost and
        lands each worker at ~100% of its own capacity -- the best a scheduler
        can do once the offer exceeds the fleet.  The clamp is uniform, so it
        changes no relative ordering, and it is recorded so a run can tell
        "this pool is short" from "this policy chose badly".

        Returns ``(rows, scale)``; the caller's snapshot is left alone, so the
        state log keeps reporting the *offered* load and a later tick cannot
        inherit a previous tick's clamp.
        """
        scale = 1.0
        offered = servable = 0.0
        if self.demand_capacity_clamp:
            servable = self._servable_reference_units()
            if servable > 0.0 and prefills:
                offered = self.solver.offered_reference_units(rows, prefills)
                if offered > servable:
                    scale = servable / offered
        self.last_demand_clamp = {"offered": offered, "servable": servable,
                                  "scale": scale}
        if scale == 1.0:
            return rows, scale
        clamped = []
        for row in rows:
            copy = dict(row)
            copy["arrival_rate_ewma"] = max(
                0.0, float(row.get("arrival_rate_ewma") or 0.0)) * scale
            clamped.append(copy)
        return clamped, scale

    def build_plan(self, current_ns: int, profiler, schedulers) -> AffinityPlan:
        self.last_execution = ()
        snapshot = profiler.snapshot(current_ns, schedulers)
        # Publish this tick's latency budgets before anything prices a pair:
        # the trace's per-class bounds (tightest wins) override the policy's
        # global ``ttft_slo_ms``.  Without this the ``p_slo`` term is dead code
        # in the simulator even when the traces carry budgets.
        self.solver.apply_slo_overrides(
            class_ttft_slo_ms=snapshot.get("class_slo") or None)
        if self.state_collector.enabled:
            self.last_telemetry = self.state_collector.collect()
            snapshot["telemetry"] = self.last_telemetry
            self.solver.set_telemetry(self.last_telemetry)
        all_prefill = [s for s in schedulers if s.pd_type == "prefill"]
        # Demand recovery has to happen *before* the lifecycle reads demand.
        # Under back-pressure ``arrival_rate_ewma`` only reflects what was
        # already served, so a saturated pool looks *idle* and the lifecycle's
        # ``ceil(demand / capacity)`` shrinks the pool -- measured 2026-09-23 on
        # the 30 rps peak: the controller drained the fastest Prefill while a
        # spare was warming, leaving the RTX3090 as the only fully available
        # Prefill, and the classes on it queued for 80 s.
        backlog_prefill = [s for s in all_prefill if s.accepts_new_requests] or all_prefill
        self._sample_backlog(current_ns, backlog_prefill)
        self._inflate_demand(snapshot["prefix_states"])
        self.last_lifecycle = self.lifecycle.update(
            current_ns, snapshot["prefix_states"], schedulers,
            backlog_rps=getattr(self, "_backlog_rps", 0.0))
        if self.decode_lifecycle is not None:
            self.last_decode_lifecycle = self.decode_lifecycle.update(
                current_ns, snapshot["prefix_states"], schedulers)
        self.last_resource_snapshot = self.lifecycle.resources.snapshot()
        prefill = [s for s in all_prefill if s.accepts_new_requests]
        decode = [s for s in schedulers if s.pd_type == "decode" and s.accepts_new_requests]
        # A colocated deployment has neither role.  Treat its instances as both
        # ends so CASR observability remains useful without changing semantics.
        if not prefill:
            prefill = [s for s in schedulers if s.accepts_new_requests]
        if not decode:
            decode = [s for s in schedulers if s.accepts_new_requests]

        # Price each Prefill by what it can actually push, not by its compute
        # capacity: the producer's egress (0.26 GB/s measured) is what caps a
        # 1250-token workload at ~1.4 req/s per worker, and without this term
        # the counterfactual for one more worker shows almost no gain, so the
        # structural action fires only once the peak is half over (measured in
        # the six-domain arena: +P arm 14878 ms against 1400 ms for the same
        # pool floor when capacity came from the egress).
        egress_caps = self._egress_bound_prefill_capacity(
            snapshot["prefix_states"], all_prefill)
        if egress_caps:
            self.solver.apply_capacity_overrides(prefill_capacity=egress_caps)

        decision = self.evaluator.evaluate(
            snapshot, prefill, all_prefill, decode, self.solver, current_ns,
            self.last_action_ns, self.lifecycle.min_active,
            backlog_rps=getattr(self, "_backlog_rps", 0.0))
        self.last_structural_decision = decision.as_dict()
        # A rule-based scaler has no counterfactual gain to report, so publish
        # the reading its decision was based on (the run summary quotes it).
        rule_metrics = getattr(self.evaluator, "last_metrics", None)
        if rule_metrics:
            self.last_structural_decision["rule"] = type(self.evaluator).__name__
            self.last_structural_decision["rule_metrics"] = dict(rule_metrics)
        if decision.action != "keep":
            self.last_action_ns = int(current_ns)
            if decision.action == "+P" and decision.mode == "warm":
                new_ids = set(decision.wanted_ids) - {item.instance_id for item in prefill}
                for instance_id in new_ids:
                    self.pending_warm_classes[instance_id] = set(decision.warm_classes)
            self.last_lifecycle = self.lifecycle.update(
                current_ns, snapshot["prefix_states"], schedulers,
                wanted_override=set(decision.wanted_ids),
                action=decision.action)
            self.last_resource_snapshot = self.lifecycle.resources.snapshot()
            blocked_ids = {event["instance_id"] for event in self.last_lifecycle
                           if event.get("action") == "resource_reject"}
            self.last_execution = tuple(item.as_dict() for item in self.executor.apply(
                decision, [item.instance_id for item in prefill], blocked_ids))
            self.last_lifecycle = tuple(self.last_lifecycle) + tuple(
                {"action": "executor_" + item["action"],
                 "instance_id": item["instance_id"], "ok": item["ok"],
                 "detail": item["detail"]} for item in self.last_execution)
            prefill = [s for s in all_prefill if s.accepts_new_requests]

        by_id = {scheduler.instance_id: scheduler for scheduler in prefill}
        warmups = []
        # A structural warm decision is made before the new worker is ready.
        # Apply its selected prefixes once the worker becomes ACTIVE, even if
        # the current flow solver sends no class to it during this tick.
        for instance_id, classes in list(self.pending_warm_classes.items()):
            scheduler = by_id.get(instance_id)
            if scheduler is None:
                continue
            for class_id in tuple(classes):
                candidate = profiler.warm_candidate(class_id)
                if candidate is None:
                    classes.discard(class_id)
                    continue
                warmed_bytes = scheduler.warm_prefix(*candidate)
                requested_bytes = (int(candidate[0]) // max(1, scheduler.kv.block_size) *
                                   scheduler.kv.npu_pool.bytes_per_block)
                warmups.append({"class_id": class_id,
                                "prefill_id": instance_id,
                                "bytes": warmed_bytes,
                                "requested_bytes": requested_bytes,
                                "resident": warmed_bytes < requested_bytes})
                classes.discard(class_id)
            if not classes:
                self.pending_warm_classes.pop(instance_id, None)

        # Degrade deliberately when the offer exceeds the fleet.  This runs
        # *after* the lifecycle and the structural evaluator, which must both
        # see the true offered load (a clamp before them would cap the demand
        # at today's capacity, so ``ceil(demand / capacity)`` could never ask
        # for another worker), and *before* the solve, so the plan the router
        # installs is the feasible one.
        solve_rows, _scale = self._clamped_rows(snapshot["prefix_states"],
                                                all_prefill)
        solve_snapshot = dict(snapshot, prefix_states=solve_rows)
        proposed = self.policy.solve(solve_snapshot, prefill, decode,
                                     self.solver)
        self.last_flows = tuple(self._validate_flows(proposed, solve_snapshot,
                                                     prefill, decode))
        self.last_solver_diagnostics = dict(self.solver.diagnostics)
        self.last_solver_diagnostics["policy"] = self.policy_spec
        self.last_solver_diagnostics["structural"] = self.last_structural_decision
        self.last_warmups = tuple(warmups)

        self.version += 1
        self.next_tick_ns = int(current_ns) + self.interval_ns
        return build_affinity_plan(self.last_flows, decode, self.version,
                                   int(current_ns) + self.plan_ttl_ns)

    def _validate_flows(self, flows, snapshot, prefill, decode):
        """Reject malformed custom-policy output before it can alter routing."""
        from .flow_solver import FlowAssignment

        classes = {row["class_id"] for row in snapshot["prefix_states"]}
        p_ids = {scheduler.instance_id for scheduler in prefill}
        d_ids = {scheduler.instance_id for scheduler in decode}
        normalized = []
        for item in flows:
            if isinstance(item, dict):
                item = FlowAssignment(**item)
            if not isinstance(item, FlowAssignment):
                raise PolicyError("CASR policy must return FlowAssignment objects or dictionaries")
            if item.class_id not in classes or item.prefill_id not in p_ids or item.decode_id not in d_ids:
                raise PolicyError(f"CASR policy returned an unknown class or inactive instance: {item}")
            if item.flow < 0:
                raise PolicyError(f"CASR policy returned negative flow: {item}")
            if item.flow == 0:
                # A placeholder, not a plan: the builtin solver emits one row
                # per (class, prefill, decode) and a class whose EWMA arrival
                # rate has decayed to 0 -- with ``class_demand_floor_rps`` at
                # its default 0 -- gets exactly this.  Dropping it changes no
                # routing (it carries no flow) and keeps the conservation check
                # below meaningful; rejecting it made every builtin-solver run
                # fail on the first zero-demand class.
                continue
            normalized.append(item)
        # A prefix class can be observed on more than one Prefill worker.  The
        # solver receives one row per worker and therefore emits one flow per
        # row; validate against the aggregate class demand rather than letting
        # the last row overwrite the earlier ones.  This has to mirror
        # ``CapacityAwareFlowSolver._aggregate_rows`` exactly -- it used to
        # floor every row at 1.0 as well, which both inflated the demand the
        # solver planned against and rejected the (correct) plan once the
        # solver stopped doing it.
        floor = float(self.solver.config.class_demand_floor_rps)
        expected = {}
        for row in snapshot["prefix_states"]:
            class_id = row["class_id"]
            expected[class_id] = expected.get(class_id, 0.0) + float(
                row["arrival_rate_ewma"])
        for class_id, demand in expected.items():
            expected[class_id] = max(demand, floor)
        assigned = {class_id: 0.0 for class_id in expected}
        for item in normalized:
            assigned[item.class_id] += item.flow
        for class_id, demand in expected.items():
            if abs(assigned[class_id] - demand) > 1e-6 * max(1.0, demand):
                raise PolicyError(
                    f"CASR policy violates flow conservation for {class_id}: "
                    f"assigned={assigned[class_id]}, demand={demand}")
        return normalized

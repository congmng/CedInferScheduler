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
from .evaluator import StructuralEvaluator
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
        lifecycle_policy = dict((policy or {}).get("lifecycle", {}))
        lifecycle_policy.setdefault("prefill_capacity", (policy or {}).get("prefill_capacity", {}))
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
        self.evaluator = StructuralEvaluator(structural_policy)
        self.state_collector = PrometheusStateCollector((policy or {}).get("telemetry", {}))
        self.executor = ReconfigExecutor((policy or {}).get("executor", {}))
        self.last_action_ns = -1
        self.last_flows = ()
        self.last_lifecycle = ()
        self.last_warmups = ()
        self.last_solver_diagnostics = {}
        self.last_resource_snapshot = {}
        self.last_structural_decision = {}
        self.pending_warm_classes = {}
        self.last_telemetry = {}
        self.last_execution = ()

    def due(self, current_ns: int) -> bool:
        return int(current_ns) >= self.next_tick_ns

    def build_plan(self, current_ns: int, profiler, schedulers) -> AffinityPlan:
        self.last_execution = ()
        snapshot = profiler.snapshot(current_ns, schedulers)
        if self.state_collector.enabled:
            self.last_telemetry = self.state_collector.collect()
            snapshot["telemetry"] = self.last_telemetry
            self.solver.set_telemetry(self.last_telemetry)
        all_prefill = [s for s in schedulers if s.pd_type == "prefill"]
        self.last_lifecycle = self.lifecycle.update(current_ns, snapshot["prefix_states"], schedulers)
        self.last_resource_snapshot = self.lifecycle.resources.snapshot()
        prefill = [s for s in all_prefill if s.accepts_new_requests]
        decode = [s for s in schedulers if s.pd_type == "decode" and s.accepts_new_requests]
        # A colocated deployment has neither role.  Treat its instances as both
        # ends so CASR observability remains useful without changing semantics.
        if not prefill:
            prefill = [s for s in schedulers if s.accepts_new_requests]
        if not decode:
            decode = [s for s in schedulers if s.accepts_new_requests]

        decision = self.evaluator.evaluate(
            snapshot, prefill, all_prefill, decode, self.solver, current_ns,
            self.last_action_ns, self.lifecycle.min_active)
        self.last_structural_decision = decision.as_dict()
        if decision.action != "keep":
            self.last_action_ns = int(current_ns)
            if decision.action == "+P" and decision.mode == "warm":
                new_ids = set(decision.wanted_ids) - {item.instance_id for item in prefill}
                for instance_id in new_ids:
                    self.pending_warm_classes[instance_id] = set(decision.warm_classes)
            self.last_lifecycle = self.lifecycle.update(
                current_ns, snapshot["prefix_states"], schedulers,
                wanted_override=set(decision.wanted_ids))
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

        proposed = self.policy.solve(snapshot, prefill, decode, self.solver)
        self.last_flows = tuple(self._validate_flows(proposed, snapshot, prefill, decode))
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
            if item.flow <= 0:
                raise PolicyError(f"CASR policy returned non-positive flow: {item}")
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

"""Deterministic slow control loop for the first CASR simulator integration.

The controller deliberately has no solver dependency.  It turns observed
prefix-class demand into an executable, capacity-aware affinity plan.  The
interface is kept separate from the policy so an OR-Tools LP can replace this
greedy baseline without changing the router or the trace path.
"""

from __future__ import annotations

from .affinity import AffinityPlan
from .flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
from .lifecycle import PrefillLifecycle
from .policy import PolicyError, load_policy


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
        self.lifecycle = PrefillLifecycle(lifecycle_policy)
        self.last_flows = ()
        self.last_lifecycle = ()
        self.last_warmups = ()
        self.last_solver_diagnostics = {}

    def due(self, current_ns: int) -> bool:
        return int(current_ns) >= self.next_tick_ns

    def build_plan(self, current_ns: int, profiler, schedulers) -> AffinityPlan:
        snapshot = profiler.snapshot(current_ns, schedulers)
        all_prefill = [s for s in schedulers if s.pd_type == "prefill"]
        self.last_lifecycle = self.lifecycle.update(current_ns, snapshot["prefix_states"], all_prefill)
        prefill = [s for s in all_prefill if s.accepts_new_requests]
        decode = [s for s in schedulers if s.pd_type == "decode" and s.accepts_new_requests]
        # A colocated deployment has neither role.  Treat its instances as both
        # ends so CASR observability remains useful without changing semantics.
        if not prefill:
            prefill = [s for s in schedulers if s.accepts_new_requests]
        if not decode:
            decode = [s for s in schedulers if s.accepts_new_requests]

        p_weights = {}
        d_weights = {}
        fallbacks = {}

        proposed = self.policy.solve(snapshot, prefill, decode, self.solver)
        self.last_flows = tuple(self._validate_flows(proposed, snapshot, prefill, decode))
        self.last_solver_diagnostics = dict(self.solver.diagnostics)
        self.last_solver_diagnostics["policy"] = self.policy_spec
        by_id = {scheduler.instance_id: scheduler for scheduler in prefill}
        warmups = []
        class_demand = {}
        p_class_flow = {}
        for flow in self.last_flows:
            class_demand[flow.class_id] = class_demand.get(flow.class_id, 0.0) + flow.flow
            p_class_flow[flow.prefill_id, flow.class_id] = (
                p_class_flow.get((flow.prefill_id, flow.class_id), 0.0) + flow.flow)
        for flow in self.last_flows:
            p_weights.setdefault(flow.class_id, {})[flow.prefill_id] = (
                p_class_flow[flow.prefill_id, flow.class_id] / class_demand[flow.class_id])
            key = (flow.prefill_id, flow.class_id)
            d_weights.setdefault(key, {})[flow.decode_id] = (
                d_weights.get(key, {}).get(flow.decode_id, 0.0) +
                flow.flow / p_class_flow[key])
            candidate = profiler.warm_candidate(flow.class_id)
            if candidate is not None and flow.prefill_id in by_id:
                warmed_bytes = by_id[flow.prefill_id].warm_prefix(*candidate)
                if warmed_bytes:
                    warmups.append({"class_id": flow.class_id,
                                    "prefill_id": flow.prefill_id,
                                    "bytes": warmed_bytes})
        for (prefill_id, class_id), weights in d_weights.items():
            fallbacks[prefill_id, class_id] = tuple(s.instance_id for s in decode
                                                    if s.instance_id not in weights)
        self.last_warmups = tuple(warmups)

        self.version += 1
        self.next_tick_ns = int(current_ns) + self.interval_ns
        return AffinityPlan(
            version=self.version,
            expires_at_ns=int(current_ns) + self.plan_ttl_ns,
            prefill_weights=p_weights,
            decode_weights=d_weights,
            fallback_decode_ids=fallbacks,
        )

    @staticmethod
    def _validate_flows(flows, snapshot, prefill, decode):
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
        # the last row overwrite the earlier ones.
        expected = {}
        for row in snapshot["prefix_states"]:
            class_id = row["class_id"]
            expected[class_id] = (expected.get(class_id, 0.0) +
                                  max(float(row["arrival_rate_ewma"]), 1.0))
        assigned = {class_id: 0.0 for class_id in expected}
        for item in normalized:
            assigned[item.class_id] += item.flow
        for class_id, demand in expected.items():
            if abs(assigned[class_id] - demand) > 1e-6 * max(1.0, demand):
                raise PolicyError(
                    f"CASR policy violates flow conservation for {class_id}: "
                    f"assigned={assigned[class_id]}, demand={demand}")
        return normalized

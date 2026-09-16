"""Turn flow-solver output into an ``AffinityPlan``.

Both the simulator control loop and the real-system router need the same
mapping from ``f_ijk`` to the fast router's weight tables.  Keeping it in one
place is what makes the real deployment and the simulator run the *same*
algorithm rather than two implementations that merely look alike.
"""

from __future__ import annotations

from .affinity import AffinityPlan


def build_affinity_plan(flows, decode, version: int, expires_at_ns: int,
                        warm_classes=()):
    """Convert ``FlowAssignment`` rows into normalized P/D weight tables.

    ``prefill_weights[class_id][p]`` is the share of a class served by prefill
    ``p``; ``decode_weights[(p, class_id)][d]`` is the share of that share which
    decode ``d`` receives.  Decode instances that the solver left at zero are
    still recorded as ordered fallbacks so the fast router always has a target.
    """
    flows = tuple(flows)
    class_demand = {}
    p_class_flow = {}
    for flow in flows:
        class_demand[flow.class_id] = class_demand.get(flow.class_id, 0.0) + flow.flow
        key = (flow.prefill_id, flow.class_id)
        p_class_flow[key] = p_class_flow.get(key, 0.0) + flow.flow

    prefill_weights = {}
    decode_weights = {}
    for flow in flows:
        prefill_weights.setdefault(flow.class_id, {})[flow.prefill_id] = (
            p_class_flow[flow.prefill_id, flow.class_id] / class_demand[flow.class_id])
        key = (flow.prefill_id, flow.class_id)
        decode_weights.setdefault(key, {})[flow.decode_id] = (
            decode_weights.get(key, {}).get(flow.decode_id, 0.0) +
            flow.flow / p_class_flow[key])

    decode_ids = [scheduler.instance_id for scheduler in decode]
    fallbacks = {}
    for (prefill_id, class_id), weights in decode_weights.items():
        fallbacks[prefill_id, class_id] = tuple(
            instance_id for instance_id in decode_ids if instance_id not in weights)

    return AffinityPlan(
        version=version,
        expires_at_ns=int(expires_at_ns),
        prefill_weights=prefill_weights,
        decode_weights=decode_weights,
        fallback_decode_ids=fallbacks,
    )

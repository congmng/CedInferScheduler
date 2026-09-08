"""Small reference policy demonstrating the public CASR extension API."""

from .flow_solver import FlowAssignment


class RoundRobinPolicy:
    """Assign each observed class to a deterministic P/D pair."""

    def __init__(self, config=None):
        self.config = config or {}

    def solve(self, snapshot, prefill, decode, solver):
        assignments = []
        for index, row in enumerate(sorted(snapshot["prefix_states"],
                                           key=lambda value: value["class_id"])):
            flow = max(float(row["arrival_rate_ewma"]), 1.0)
            assignments.append(FlowAssignment(
                row["class_id"], prefill[index % len(prefill)].instance_id,
                decode[index % len(decode)].instance_id, flow, 0.0))
        return assignments

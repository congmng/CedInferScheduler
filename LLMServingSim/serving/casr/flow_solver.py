"""Dependency-free baseline allocator for CASR f[class, P, D] flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class FlowAssignment:
    class_id: str
    prefill_id: int
    decode_id: int
    flow: float
    cost: float
    link_ids: tuple[str, ...] = ()
    prefill_overflow: float = 0.0
    decode_overflow: float = 0.0
    link_overflow: float = 0.0


@dataclass(frozen=True)
class SharedLink:
    link_id: str
    capacity: float
    pairs: frozenset[tuple[int, int]] = frozenset()

    def carries(self, prefill_id: int, decode_id: int) -> bool:
        return not self.pairs or (prefill_id, decode_id) in self.pairs


@dataclass(frozen=True)
class FlowSolverConfig:
    prefill_capacity: Mapping[int, float] = field(default_factory=dict)
    decode_capacity: Mapping[int, float] = field(default_factory=dict)
    shared_links: tuple[SharedLink, ...] = ()
    overflow_penalty: float = 10.0
    solver: str = "greedy"

    @classmethod
    def from_dict(cls, raw):
        raw = raw or {}
        links = []
        for index, item in enumerate(raw.get("shared_links", ())):
            pairs = frozenset((int(pair[0]), int(pair[1])) for pair in item.get("pairs", ()))
            links.append(SharedLink(str(item.get("id", f"link-{index}")),
                                    float(item["capacity"]), pairs))
        return cls(
            prefill_capacity={int(key): float(value)
                              for key, value in raw.get("prefill_capacity", {}).items()},
            decode_capacity={int(key): float(value)
                             for key, value in raw.get("decode_capacity", {}).items()},
            shared_links=tuple(links),
            overflow_penalty=float(raw.get("overflow_penalty", 10.0)),
            solver=str(raw.get("solver", "greedy")).lower(),
        )


class CapacityAwareFlowSolver:
    """Greedy min-cost baseline with explicit P and D capacity accounting.

    It is intentionally deterministic and dependency-free.  Its output has
    the same f_ijk surface as the later LP backend, so replacing this policy
    does not affect router, scheduler, or trace integration.
    """

    def __init__(self, config: FlowSolverConfig | None = None):
        self.config = config or FlowSolverConfig()
        self.backend = "greedy"
        self.diagnostics = {}
        self._fallback = None

    def solve(self, rows, prefill, decode):
        if self.config.solver == "lp":
            try:
                return self._solve_lp(rows, prefill, decode)
            except ImportError:
                self._fallback = "ortools unavailable"
        return self._solve_greedy(rows, prefill, decode)

    def _solve_greedy(self, rows, prefill, decode):
        self.backend = "greedy"
        p_load = {sched.instance_id: 0.0 for sched in prefill}
        d_load = {sched.instance_id: 0.0 for sched in decode}
        link_load = {link.link_id: 0.0 for link in self.config.shared_links}
        assignments = []
        for row in sorted(rows, key=lambda value: (-value["arrival_rate_ewma"],
                                                    -value["request_count"], value["class_id"])):
            flow = max(float(row["arrival_rate_ewma"]), 1.0)
            hit = min(0.95, row["hit_tokens_ewma"] / max(1, row["requested_tokens"]))
            p_work = flow * max(0.05, 1.0 - hit)
            best = None
            for p_sched in prefill:
                p_capacity = max(1.0, self.config.prefill_capacity.get(
                    p_sched.instance_id, float(p_sched.max_num_seqs)))
                for d_sched in decode:
                    d_capacity = max(1.0, self.config.decode_capacity.get(
                        d_sched.instance_id, float(d_sched.max_num_seqs)))
                    link_cost = abs(p_sched.start_npu - d_sched.start_npu) * 0.001
                    p_overflow = max(0.0, p_load[p_sched.instance_id] + p_work - p_capacity)
                    d_overflow = max(0.0, d_load[d_sched.instance_id] + flow - d_capacity)
                    carried = tuple(link for link in self.config.shared_links
                                    if link.carries(p_sched.instance_id, d_sched.instance_id))
                    link_overflow = sum(max(0.0, link_load[link.link_id] + flow - link.capacity)
                                        for link in carried)
                    cost = ((p_load[p_sched.instance_id] + p_work) / p_capacity +
                            (d_load[d_sched.instance_id] + flow) / d_capacity + link_cost +
                            self.config.overflow_penalty * (p_overflow + d_overflow + link_overflow))
                    candidate = (cost, p_sched.instance_id, d_sched.instance_id,
                                 p_sched, d_sched, carried, p_overflow, d_overflow, link_overflow)
                    if best is None or candidate[:3] < best[:3]:
                        best = candidate
            cost, _, _, p_sched, d_sched, carried, p_overflow, d_overflow, link_overflow = best
            p_load[p_sched.instance_id] += p_work
            d_load[d_sched.instance_id] += flow
            for link in carried:
                link_load[link.link_id] += flow
            assignments.append(FlowAssignment(row["class_id"], p_sched.instance_id,
                                              d_sched.instance_id, flow, cost,
                                              tuple(link.link_id for link in carried),
                                              p_overflow, d_overflow, link_overflow))
        self.diagnostics = {"backend": self.backend, "overflow_penalty": self.config.overflow_penalty}
        if self._fallback is not None:
            self.diagnostics["fallback"] = self._fallback
            self._fallback = None
        return assignments

    def _solve_lp(self, rows, prefill, decode):
        """Solve continuous f_ijk with capacity and overflow slack variables."""
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            raise RuntimeError("OR-Tools GLOP backend is unavailable")
        self.backend = "ortools-glop"
        # Prefix observations are keyed by (prefill instance, class), so a
        # class may occur multiple times.  LP variables are keyed by class and
        # P/D pair; aggregate those rows before constructing variables instead
        # of silently letting the last row overwrite the demand.
        grouped = {}
        for row in rows:
            class_id = row["class_id"]
            entry = grouped.setdefault(class_id, {
                "class_id": class_id, "arrival_rate_ewma": 0.0,
                "hit_tokens_ewma": 0.0, "requested_tokens": 0.0,
            })
            entry["arrival_rate_ewma"] += max(float(row["arrival_rate_ewma"]), 1.0)
            entry["hit_tokens_ewma"] += float(row.get("hit_tokens_ewma", 0.0))
            entry["requested_tokens"] += float(row.get("requested_tokens", 0.0))
        active_rows = list(grouped.values())
        flows = {}
        work = {}
        for row in active_rows:
            class_id = row["class_id"]
            demand = max(float(row["arrival_rate_ewma"]), 1.0)
            hit = min(0.95, row["hit_tokens_ewma"] / max(1, row["requested_tokens"]))
            work[class_id] = (demand, max(0.05, 1.0 - hit))
            for p_sched in prefill:
                for d_sched in decode:
                    flows[class_id, p_sched.instance_id, d_sched.instance_id] = solver.NumVar(
                        0.0, solver.infinity(), f"f_{len(flows)}")
        for class_id, (demand, _) in work.items():
            solver.Add(sum(flows[class_id, p.instance_id, d.instance_id]
                           for p in prefill for d in decode) == demand)
        p_slack = {}
        for p_sched in prefill:
            p_slack[p_sched.instance_id] = solver.NumVar(0.0, solver.infinity(),
                                                          f"overflow_p_{p_sched.instance_id}")
            cap = self.config.prefill_capacity.get(p_sched.instance_id, float(p_sched.max_num_seqs))
            solver.Add(sum(flows[class_id, p_sched.instance_id, d.instance_id] * work[class_id][1]
                           for class_id in work for d in decode) <= cap + p_slack[p_sched.instance_id])
        d_slack = {}
        for d_sched in decode:
            d_slack[d_sched.instance_id] = solver.NumVar(0.0, solver.infinity(),
                                                          f"overflow_d_{d_sched.instance_id}")
            cap = self.config.decode_capacity.get(d_sched.instance_id, float(d_sched.max_num_seqs))
            solver.Add(sum(flows[class_id, p.instance_id, d_sched.instance_id]
                           for class_id in work for p in prefill) <= cap + d_slack[d_sched.instance_id])
        link_slack = {}
        for link in self.config.shared_links:
            link_slack[link.link_id] = solver.NumVar(0.0, solver.infinity(),
                                                      f"overflow_link_{link.link_id}")
            solver.Add(sum(flows[class_id, p.instance_id, d.instance_id]
                           for class_id in work for p in prefill for d in decode
                           if link.carries(p.instance_id, d.instance_id)) <= link.capacity + link_slack[link.link_id])
        objective = solver.Objective()
        for (class_id, p_id, d_id), variable in flows.items():
            p_sched = next(item for item in prefill if item.instance_id == p_id)
            d_sched = next(item for item in decode if item.instance_id == d_id)
            objective.SetCoefficient(variable, abs(p_sched.start_npu - d_sched.start_npu) * 0.001)
        for variable in (*p_slack.values(), *d_slack.values(), *link_slack.values()):
            objective.SetCoefficient(variable, self.config.overflow_penalty)
        objective.SetMinimization()
        if solver.Solve() != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("CASR LP did not find an optimal solution")
        assignments = []
        for (class_id, p_id, d_id), variable in flows.items():
            value = variable.solution_value()
            if value <= 1e-9:
                continue
            carried = tuple(link.link_id for link in self.config.shared_links if link.carries(p_id, d_id))
            assignments.append(FlowAssignment(class_id, p_id, d_id, value,
                                              objective.Value() / max(1.0, work[class_id][0]), carried))
        self.diagnostics = {
            "backend": self.backend,
            "objective": objective.Value(),
            "prefill_overflow": {key: value.solution_value() for key, value in p_slack.items()},
            "decode_overflow": {key: value.solution_value() for key, value in d_slack.items()},
            "link_overflow": {key: value.solution_value() for key, value in link_slack.items()},
        }
        return assignments

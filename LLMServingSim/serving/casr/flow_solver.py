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
    capacity_bytes_per_s: float = 0.0

    def carries(self, prefill_id: int, decode_id: int) -> bool:
        return not self.pairs or (prefill_id, decode_id) in self.pairs


@dataclass(frozen=True)
class FlowSolverConfig:
    prefill_capacity: Mapping[int, float] = field(default_factory=dict)
    decode_capacity: Mapping[int, float] = field(default_factory=dict)
    shared_links: tuple[SharedLink, ...] = ()
    overflow_penalty: float = 10.0
    solver: str = "greedy"
    pair_costs: Mapping[tuple[int, int], Mapping[str, float]] = field(default_factory=dict)
    class_kv_bytes: Mapping[str, float] = field(default_factory=dict)
    default_kv_bytes: float = 0.0
    queue_weight: float = 0.0

    @classmethod
    def from_dict(cls, raw):
        raw = raw or {}
        links = []
        for index, item in enumerate(raw.get("shared_links", ())):
            pairs = frozenset((int(pair[0]), int(pair[1])) for pair in item.get("pairs", ()))
            links.append(SharedLink(str(item.get("id", f"link-{index}")),
                                    float(item.get("capacity", float("inf"))), pairs,
                                    float(item.get("capacity_bytes_per_s", 0.0))))
        pair_costs = {}
        for key, value in (raw.get("pair_costs") or {}).items():
            if isinstance(key, str):
                p_id, d_id = (int(part) for part in key.replace("/", ",").split(",", 1))
            else:
                p_id, d_id = (int(key[0]), int(key[1]))
            pair_costs[p_id, d_id] = {str(name): float(amount)
                                      for name, amount in (value or {}).items()}
        return cls(
            prefill_capacity={int(key): float(value)
                              for key, value in raw.get("prefill_capacity", {}).items()},
            decode_capacity={int(key): float(value)
                             for key, value in raw.get("decode_capacity", {}).items()},
            shared_links=tuple(links),
            overflow_penalty=float(raw.get("overflow_penalty", 10.0)),
            solver=str(raw.get("solver", "greedy")).lower(),
            pair_costs=pair_costs,
            class_kv_bytes={str(key): float(value)
                            for key, value in (raw.get("class_kv_bytes") or {}).items()},
            default_kv_bytes=float(raw.get("default_kv_bytes", 0.0)),
            queue_weight=float(raw.get("queue_weight", 0.0)),
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
        self._runtime_queue = {}

    def set_telemetry(self, telemetry):
        """Install best-effort exporter queue samples for the next solve."""
        self._runtime_queue = {}
        for worker_id, metrics in (telemetry or {}).get("workers", {}).items():
            try:
                instance_id = int(worker_id)
            except (TypeError, ValueError):
                continue
            for name in ("vllm_num_requests_waiting", "queue", "queue_depth"):
                if name in metrics:
                    self._runtime_queue[instance_id] = max(0.0, float(metrics[name]))
                    break

    def solve(self, rows, prefill, decode, work_overrides=None):
        if self.config.solver == "lp":
            try:
                return self._solve_lp(rows, prefill, decode, work_overrides)
            except ImportError:
                self._fallback = "ortools unavailable"
        return self._solve_greedy(rows, prefill, decode, work_overrides)

    @staticmethod
    def _aggregate_rows(rows, prefill, work_overrides=None):
        """Aggregate demand once while retaining cache state per P and class."""
        by_id = {int(scheduler.instance_id): scheduler for scheduler in prefill}
        grouped = {}
        for row in rows:
            class_id = row["class_id"]
            entry = grouped.setdefault(class_id, {
                "class_id": class_id, "arrival_rate_ewma": 0.0,
                "hit_tokens_ewma": {}, "requested_tokens": {},
                "kv_bytes_per_request": 0.0,
            })
            entry["arrival_rate_ewma"] += max(float(row.get("arrival_rate_ewma", 0.0)), 1.0)
            p_id = int(row.get("prefill_instance_id", next(iter(by_id), -1)))
            entry["hit_tokens_ewma"][p_id] = entry["hit_tokens_ewma"].get(p_id, 0.0) + float(row.get("hit_tokens_ewma", 0.0))
            entry["requested_tokens"][p_id] = entry["requested_tokens"].get(p_id, 0.0) + float(row.get("requested_tokens", 0.0))
            entry["kv_bytes_per_request"] = max(
                entry["kv_bytes_per_request"], float(row.get("kv_bytes_per_request", 0.0)))
        work = {}
        for class_id, entry in grouped.items():
            for scheduler in prefill:
                p_id = int(scheduler.instance_id)
                requested = max(1.0, entry["requested_tokens"].get(p_id, 0.0))
                hit = min(0.95, entry["hit_tokens_ewma"].get(p_id, 0.0) / requested)
                value = max(0.05, 1.0 - hit)
                if work_overrides and (p_id, class_id) in work_overrides:
                    value = max(0.05, min(1.0, float(work_overrides[p_id, class_id])))
                work[p_id, class_id] = value
        return grouped, work

    def _class_kv_bytes(self, class_id, entry):
        return max(0.0, self.config.class_kv_bytes.get(
            class_id, entry.get("kv_bytes_per_request", 0.0) or self.config.default_kv_bytes))

    def _pair_cost(self, prefill, decode, class_id, entry):
        config = self.config.pair_costs.get((prefill.instance_id, decode.instance_id), {})
        distance = abs(prefill.start_npu - decode.start_npu) * 0.001
        rtt = config.get("rtt_ms", 0.0) / 1000.0
        bandwidth = config.get("bandwidth_bytes_per_s", 0.0)
        kv_bytes = self._class_kv_bytes(class_id, entry)
        transfer = kv_bytes / bandwidth if bandwidth > 0 else 0.0
        waiting = max(float(len(getattr(decode, "waiting", ()))),
                      self._runtime_queue.get(int(decode.instance_id), 0.0))
        running = len(getattr(decode, "running", ()))
        max_num_seqs = getattr(decode, "max_num_seqs", 1)
        queue = self.config.queue_weight * ((waiting * 4 + running) /
                                            max(1, max_num_seqs))
        return distance + rtt + transfer + queue

    def _solve_greedy(self, rows, prefill, decode, work_overrides=None):
        self.backend = "greedy"
        p_load = {sched.instance_id: 0.0 for sched in prefill}
        d_load = {sched.instance_id: 0.0 for sched in decode}
        link_load = {link.link_id: 0.0 for link in self.config.shared_links}
        assignments = []
        grouped, work = self._aggregate_rows(rows, prefill, work_overrides)
        for class_id, entry in sorted(grouped.items(), key=lambda item: (
                -item[1]["arrival_rate_ewma"], item[0])):
            flow = max(float(entry["arrival_rate_ewma"]), 1.0)
            best = None
            for p_sched in prefill:
                p_work = flow * work[p_sched.instance_id, class_id]
                p_capacity = max(1.0, self.config.prefill_capacity.get(p_sched.instance_id, float(p_sched.max_num_seqs)))
                for d_sched in decode:
                    d_capacity = max(1.0, self.config.decode_capacity.get(
                        d_sched.instance_id, float(d_sched.max_num_seqs)))
                    link_cost = self._pair_cost(p_sched, d_sched, class_id, entry)
                    p_overflow = max(0.0, p_load[p_sched.instance_id] + p_work - p_capacity)
                    d_overflow = max(0.0, d_load[d_sched.instance_id] + flow - d_capacity)
                    carried = tuple(link for link in self.config.shared_links
                                    if link.carries(p_sched.instance_id, d_sched.instance_id))
                    kv_bytes = self._class_kv_bytes(class_id, entry)
                    link_overflow = sum(max(
                        0.0, link_load[link.link_id] + flow * (kv_bytes if link.capacity_bytes_per_s else 1.0) -
                        (link.capacity_bytes_per_s if link.capacity_bytes_per_s else link.capacity))
                        for link in carried)
                    cost = ((p_load[p_sched.instance_id] + p_work) / p_capacity +
                            (d_load[d_sched.instance_id] + flow) / d_capacity + link_cost +
                            self.config.overflow_penalty * (p_overflow + d_overflow + link_overflow))
                    candidate = (cost, p_sched.instance_id, d_sched.instance_id,
                                 p_sched, d_sched, carried, p_overflow, d_overflow, link_overflow)
                    if best is None or candidate[:3] < best[:3]:
                        best = candidate
            cost, _, _, p_sched, d_sched, carried, p_overflow, d_overflow, link_overflow = best
            selected_p_work = flow * work[p_sched.instance_id, class_id]
            p_load[p_sched.instance_id] += selected_p_work
            d_load[d_sched.instance_id] += flow
            for link in carried:
                link_load[link.link_id] += flow * (self._class_kv_bytes(class_id, entry)
                                                   if link.capacity_bytes_per_s else 1.0)
            assignments.append(FlowAssignment(class_id, p_sched.instance_id,
                                              d_sched.instance_id, flow, cost,
                                              tuple(link.link_id for link in carried),
                                              p_overflow, d_overflow, link_overflow))
        self.diagnostics = {"backend": self.backend, "overflow_penalty": self.config.overflow_penalty,
                            "objective": sum(item.flow * item.cost for item in assignments)}
        if self._fallback is not None:
            self.diagnostics["fallback"] = self._fallback
            self._fallback = None
        return assignments

    def _solve_lp(self, rows, prefill, decode, work_overrides=None):
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
        grouped, p_work = self._aggregate_rows(rows, prefill, work_overrides)
        flows = {}
        work = {}
        for class_id, row in grouped.items():
            demand = max(float(row["arrival_rate_ewma"]), 1.0)
            work[class_id] = demand
            for p_sched in prefill:
                for d_sched in decode:
                    flows[class_id, p_sched.instance_id, d_sched.instance_id] = solver.NumVar(
                        0.0, solver.infinity(), f"f_{len(flows)}")
        for class_id, demand in work.items():
            solver.Add(sum(flows[class_id, p.instance_id, d.instance_id]
                           for p in prefill for d in decode) == demand)
        p_slack = {}
        for p_sched in prefill:
            p_slack[p_sched.instance_id] = solver.NumVar(0.0, solver.infinity(),
                                                          f"overflow_p_{p_sched.instance_id}")
            cap = self.config.prefill_capacity.get(p_sched.instance_id, float(p_sched.max_num_seqs))
            solver.Add(sum(flows[class_id, p_sched.instance_id, d.instance_id] * p_work[p_sched.instance_id, class_id]
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
            solver.Add(sum(flows[class_id, p.instance_id, d.instance_id] *
                           (self._class_kv_bytes(class_id, grouped[class_id])
                            if link.capacity_bytes_per_s else 1.0)
                           for class_id in work for p in prefill for d in decode
                           if link.carries(p.instance_id, d.instance_id)) <=
                       (link.capacity_bytes_per_s if link.capacity_bytes_per_s else link.capacity) +
                       link_slack[link.link_id])
        objective = solver.Objective()
        for (class_id, p_id, d_id), variable in flows.items():
            p_sched = next(item for item in prefill if item.instance_id == p_id)
            d_sched = next(item for item in decode if item.instance_id == d_id)
            objective.SetCoefficient(variable, self._pair_cost(
                p_sched, d_sched, class_id, grouped[class_id]))
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
                                              objective.Value() / max(1.0, work[class_id]), carried))
        self.diagnostics = {
            "backend": self.backend,
            "objective": objective.Value(),
            "prefill_overflow": {key: value.solution_value() for key, value in p_slack.items()},
            "decode_overflow": {key: value.solution_value() for key, value in d_slack.items()},
            "link_overflow": {key: value.solution_value() for key, value in link_slack.items()},
        }
        return assignments

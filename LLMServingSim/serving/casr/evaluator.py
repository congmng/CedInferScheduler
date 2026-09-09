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
    """Compare one-step P row edits using the same inner flow solver."""

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

    @staticmethod
    def _hit_work(rows, class_id):
        ratios = []
        for row in rows:
            if row["class_id"] != class_id:
                continue
            requested = max(1.0, float(row.get("requested_tokens", 0.0)))
            ratios.append(min(0.95, float(row.get("hit_tokens_ewma", 0.0)) / requested))
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

    def evaluate(self, snapshot, active_prefill, all_prefill, decode, solver,
                 current_ns, last_action_ns=-1, min_active=1):
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

        inactive = [s for s in all_prefill if s not in active_prefill and
                    s.admission_state == "INACTIVE"]
        for candidate in sorted(inactive, key=lambda item: item.instance_id)[:1]:
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
                gain = (self.window_ns / 1_000_000_000.0) * (base_objective - candidate_objective)
                gain -= default_cost
                candidates.append(StructuralDecision(
                    "+P", mode, gain, base_objective, candidate_objective,
                    tuple(item.instance_id for item in candidate_prefill),
                    f"counterfactual add Prefill {candidate.instance_id}",
                    warm_classes if mode == "warm" else ()))

        if len(active_prefill) > min_active:
            for candidate in sorted(active_prefill, key=lambda item: item.instance_id, reverse=True)[:1]:
                candidate_prefill = [item for item in active_prefill if item is not candidate]
                solver.solve(rows, candidate_prefill, decode)
                candidate_objective = float(solver.diagnostics.get("objective", 0.0))
                gain = (self.window_ns / 1_000_000_000.0) * (base_objective - candidate_objective)
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

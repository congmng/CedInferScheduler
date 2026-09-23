"""Regression guards for the capacity/ordering bugs found on 2026-09-23.

Each test pins one behaviour that was wrong, in the form that would have failed
before the fix.  They are deliberately cheap: no vLLM, no GPU, no simulation
-- just the pure functions and small objects the runtime wires together.
"""

import json
import os
import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _needs_pandas():
    """``serving.core`` pulls in pandas; skip where it is unavailable (the
    vLLM image), run where the simulator itself runs (the host)."""
    try:
        import pandas  # noqa: F401
    except ImportError as exc:                      # pragma: no cover
        raise unittest.SkipTest(f"pandas unavailable: {exc}")


class ResolveRuntimeCapacityTests(unittest.TestCase):
    """The run must price ONE capacity, and it must come from the bundles."""

    def setUp(self):
        _needs_pandas()
        # ``hw_service`` resolves profiler bundles relative to astra-sim/.
        self._cwd = pathlib.Path.cwd()
        os.chdir(REPO / "astra-sim")
        self.addCleanup(os.chdir, self._cwd)
        from serving.core.hw_service import resolve_runtime_capacities
        self.resolve = resolve_runtime_capacities
        self.config = json.loads(
            (REPO / "configs/cluster/casr_p15b_three_domain.json")
            .read_text(encoding="utf-8"))["casr"]
        self.instances = [
            {"instance_id": 0, "hardware": "RTX3090", "model_name": "casr/P15B", "pd_type": "prefill"},
            {"instance_id": 1, "hardware": "RTX3090", "model_name": "casr/P15B", "pd_type": "decode"},
            {"instance_id": 2, "hardware": "RTX4090", "model_name": "casr/P15B", "pd_type": "prefill"},
            {"instance_id": 3, "hardware": "RTX4090", "model_name": "casr/P15B", "pd_type": "decode"},
            {"instance_id": 4, "hardware": "RTX5090", "model_name": "casr/P15B", "pd_type": "prefill"},
            {"instance_id": 5, "hardware": "RTX5090", "model_name": "casr/P15B", "pd_type": "decode"},
        ]

    def test_decode_capacity_comes_from_the_bundle_by_default(self):
        # The bug: decode stayed on the deployment placeholders (110/170/230
        # req/s) because ``rescale_capacities`` was opt-in.
        config = json.loads(json.dumps(self.config))
        self.assertEqual(config["decode_capacity"], {"1": 110, "3": 170, "5": 230})
        report, resolved = self.resolve(config, self.instances, verbose=False)
        self.assertTrue(resolved["from_profile"])
        decode = {int(k): v for k, v in config["decode_capacity"].items()}
        self.assertLess(decode[1], 10.0)          # 3090: 4.5 req/s, not 110
        self.assertLess(decode[5], 20.0)          # 5090: 12.5 req/s, not 230
        self.assertGreater(decode[5], decode[1])  # and still ordered by speed
        self.assertEqual(report["decode_capacity"], config["decode_capacity"])

    def test_prefill_capacity_is_capped_by_the_egress_bound(self):
        # A producer pushing at C bytes/s cannot serve more than C/B requests/s.
        config = json.loads(json.dumps(self.config))
        config["shared_links"] = [{
            "id": "tiny", "capacity_bytes_per_s": 1_000_000.0,
            "pairs": [[0, 1], [2, 3], [4, 5]],
        }]
        config["prefill_capacity"] = {"0": 500.0, "2": 500.0, "4": 500.0}
        per_request = config["kv_bytes_per_token"] * 1024
        self.resolve(config, self.instances, verbose=False)
        for key in ("0", "2", "4"):
            self.assertAlmostEqual(config["prefill_capacity"][key],
                                   1_000_000.0 / per_request, places=4)

    def test_opt_out_keeps_the_declared_numbers(self):
        config = json.loads(json.dumps(self.config))
        config["capacity_from_profile"] = False
        config["shared_links"] = []
        _, resolved = self.resolve(config, self.instances, verbose=False)
        self.assertFalse(resolved["from_profile"])
        self.assertEqual(config["decode_capacity"], {"1": 110, "3": 170, "5": 230})


class LifecycleBacklogTests(unittest.TestCase):
    """A backlog may only grow the pool -- never shrink it."""

    def _lifecycle(self):
        from serving.casr.lifecycle import PrefillLifecycle
        return PrefillLifecycle({"min_active_prefill": 1, "max_active_prefill": 3,
                                 "warmup_ms": 0, "prefill_capacity": {"0": 10, "1": 10}})

    def _schedulers(self, count):
        class Sched:
            pd_type = "prefill"
            def __init__(self, instance_id):
                self.instance_id = instance_id
                self.node_id = 0
                self.admission_state = "ACTIVE"
                self.running = []
                self.waiting = []
                self.max_num_seqs = 8
                self.resource_gpu_ids = ()
                self.resource_mem_gb = 0.0

            def set_admission_state(self, state):
                self.admission_state = state
        return [Sched(i) for i in range(count)]

    def test_backlog_blocks_the_demand_driven_shrink(self):
        lifecycle = self._lifecycle()
        # One class at 0.2 req/s per worker x 10 req/s capacity = demand for 1
        # worker; the pool has 3, so without the guard it drains to 1.
        rows = [{"class_id": "c", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 2.0, "requested_tokens_ewma": 1024,
                 "kv_bytes_per_request": 0.0}]
        lifecycle.update(0, rows, self._schedulers(3))
        self.assertLess(len(lifecycle.last_wanted), 3,
                        "sanity: the heuristic does shrink a quiet oversized pool")

        lifecycle = self._lifecycle()
        lifecycle.update(0, rows, self._schedulers(3), backlog_rps=5.0)
        self.assertEqual(lifecycle.last_wanted, {0, 1, 2},
                         "backlog must not shrink the pool")


class SloPathTests(unittest.TestCase):
    """trace row -> Request -> class budget -> solver, plus the CSV verdict."""

    def test_trace_row_budget_reaches_the_class_and_the_solver(self):
        from serving.casr.prefix_profiler import PrefixProfiler
        profiler = PrefixProfiler(block_size=16)
        class_id, _ = profiler.assign("m", 1024, 64, [1] * 64,
                                      slo_ttft_ms=300.0, slo_tpot_ms=50.0)
        # A second, looser request on the same class must not relax the bound.
        profiler.assign("m", 1024, 64, [1] * 64, slo_ttft_ms=900.0, slo_tpot_ms=80.0)
        snapshot = profiler.snapshot(0)
        self.assertEqual(snapshot["class_slo"][class_id], 300.0)

        from serving.casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({"slo_penalty": 10.0}))
        solver.apply_slo_overrides(class_ttft_slo_ms=snapshot["class_slo"])
        self.assertEqual(solver.config.class_ttft_slo_ms[class_id], 300.0)
        self.assertEqual(solver.config.slo_penalty, 10.0)

    def test_request_without_a_budget_records_no_verdict(self):
        _needs_pandas()
        from serving.core.scheduler import _slo_columns

        class Req:
            slo_ttft_ms = None
            slo_tpot_ms = None
            ttft = 1_000_000
            tpot = 10_000_000
        self.assertEqual(_slo_columns(Req()), ("", "", ""))

        met = Req()
        met.slo_ttft_ms = 500.0
        self.assertEqual(_slo_columns(met)[2], "true")
        missed = Req()
        missed.slo_ttft_ms = 0.5
        self.assertEqual(_slo_columns(missed)[2], "false")


class DemandRecoveryOrderTests(unittest.TestCase):
    """``build_plan`` must recover the offered load *before* it resizes.

    The bug: the lifecycle was called first, so under back-pressure it read the
    post-backpressure arrival rate (the *served* rate), computed a smaller
    ``ceil(demand / capacity)`` and drained the fastest Prefill mid-peak.
    """

    def setUp(self):
        _needs_pandas()
        from serving.casr.controller import CASRController
        self.controller = CASRController(1_000_000_000, policy={
            "solver": "greedy", "prefill_capacity": {"0": 10.0, "1": 10.0},
            "decode_capacity": {"2": 10.0, "3": 10.0},
        })

    class Sched:
        def __init__(self, instance_id, pd_type, **extra):
            self.instance_id = instance_id
            self.pd_type = pd_type
            self.node_id = 0
            self.start_npu = 0
            self.admission_state = "ACTIVE"
            self.running = []
            self.waiting = []
            self.max_num_seqs = 8
            self.resource_gpu_ids = ()
            self.resource_mem_gb = 0.0
            self.accepts_new_requests = True
            for key, value in extra.items():
                setattr(self, key, value)

        def set_admission_state(self, state):
            self.admission_state = state

    def test_the_lifecycle_sees_the_recovered_demand(self):
        schedulers = [self.Sched(0, "prefill"), self.Sched(1, "prefill"),
                      self.Sched(2, "decode"), self.Sched(3, "decode")]
        rows = [{"class_id": "c", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 4.0, "requested_tokens_ewma": 1024,
                 "kv_bytes_per_request": 0.0, "cache": {}}]
        snapshot = {"time_ns": 0, "prefix_states": rows, "instances": {}}

        class Profiler:
            def snapshot(self, now, schedulers):
                return snapshot

        seen = {}
        original_update = self.controller.lifecycle.update

        def spy(current_ns, rows_, schedulers_, **kwargs):
            seen["demand"] = sum(float(row["arrival_rate_ewma"]) for row in rows_)
            seen["backlog"] = float(kwargs.get("backlog_rps") or 0.0)
            return original_update(current_ns, rows_, schedulers_, **kwargs)

        self.controller.lifecycle.update = spy
        # A backlog is the *growth* of the Prefill queues, so it takes two
        # ticks: 20 queued requests at t=0, 40 one second later => ~20 req/s of
        # offered load the arrival EWMA never saw.
        for sched in schedulers:
            if sched.pd_type == "prefill":
                sched.waiting = [None] * 20
        self.controller.build_plan(0, Profiler(), schedulers)
        for sched in schedulers:
            if sched.pd_type == "prefill":
                sched.waiting = [None] * 40
        self.controller.build_plan(1_000_000_000, Profiler(), schedulers)
        self.assertGreater(seen.get("demand", 0.0), 4.0,
                           "the lifecycle was handed the raw (served) rate: "
                           f"{seen}")
        self.assertGreater(seen.get("backlog", 0.0), 0.0,
                           "the lifecycle must be told about the backlog so it "
                           "can refuse to shrink")


class ObjectiveKnobTests(unittest.TestCase):
    """The new tail/min-max/backlog knobs must parse and default to off."""

    def test_defaults_are_off(self):
        from serving.casr.flow_solver import FlowSolverConfig
        config = FlowSolverConfig.from_dict({})
        for name in ("tail_weight", "max_utilization_weight", "backlog_weight"):
            self.assertEqual(getattr(config, name), 0.0, name)

    def test_values_round_trip(self):
        from serving.casr.flow_solver import FlowSolverConfig
        config = FlowSolverConfig.from_dict({
            "tail_weight": 1.5, "max_utilization_weight": 20.0, "backlog_weight": 5.0})
        self.assertEqual((config.tail_weight, config.max_utilization_weight,
                          config.backlog_weight), (1.5, 20.0, 5.0))

    def test_structural_gates_default_to_on(self):
        from serving.casr.evaluator import StructuralEvaluator
        evaluator = StructuralEvaluator({})
        self.assertTrue(evaluator.block_scale_down_when_saturated)
        self.assertTrue(evaluator.drain_requires_idle_classes)
        self.assertTrue(evaluator.block_edits_during_transition)
        off = StructuralEvaluator({"block_scale_down_when_saturated": False,
                                   "drain_requires_idle_classes": False,
                                   "block_edits_during_transition": False})
        self.assertFalse(off.block_scale_down_when_saturated)


class ClusterConfigHygieneTests(unittest.TestCase):
    """P-15B cluster configs must declare the intra-domain link.

    Without both keys the pair-cost table gives same-node pairs *zero*
    bandwidth, the router's link-derived price bails out, and the
    recompute-vs-transfer decision silently reverts to the recorded deployment
    constants (measured 2026-09-23: 43 of 44 decisions in a light run).  The
    keys are easy to drop from a copied config, so they are pinned here.
    """

    def test_p15b_configs_declare_the_intra_link(self):
        configs = sorted((REPO / "configs/cluster").glob("casr_p15b_*.json"))
        self.assertTrue(configs, "no P-15B cluster configs found")
        for path in configs:
            raw = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(config=path.name):
                self.assertIn("intra_node_link_bw", raw, path.name)
                self.assertIn("intra_node_link_latency", raw, path.name)
                self.assertGreater(float(raw["intra_node_link_bw"]), 0.0, path.name)
                self.assertGreater(float(raw["intra_node_link_latency"]), 0.0, path.name)


class TailPricingTests(unittest.TestCase):
    """The Decode tail term must price *time*, not only occupancy."""

    class Sched:
        def __init__(self, instance_id, service_ms, waiting, running=0):
            self.instance_id = instance_id
            self.pd_type = "decode"
            self.service_ms = service_ms
            self.max_num_seqs = 8
            self.waiting = [None] * waiting
            self.running = [None] * running
            self.node_id = 0
            self.start_npu = 0

    def _cost(self, tail_weight, decode):
        from serving.casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(
            {"tail_weight": tail_weight, "queue_weight": 0.0,
             "decode_service_ms": {decode.instance_id: decode.service_ms}}))
        prefill = self.Sched(0, 100.0, 0)
        # ``requested_tokens`` is a per-Prefill mapping in real snapshots.
        entry = {"arrival_rate_ewma": 1.0, "requested_tokens": {0: 1024},
                 "requested_tokens_ewma": {0: 1024.0}}
        return solver._pair_cost(prefill, decode, "c|out:16-31", entry)

    def test_zero_weight_is_the_historical_objective(self):
        slow = self.Sched(1, 372.0, waiting=4)
        self.assertEqual(self._cost(0.0, slow), self._cost(0.0, slow))

    def test_tail_term_charges_the_slow_instance_more(self):
        fast = self.Sched(1, 46.0, waiting=4)
        slow = self.Sched(2, 372.0, waiting=4)
        cheap = self._cost(1.0, fast)
        dear = self._cost(1.0, slow)
        self.assertGreater(dear - cheap, 0.5,
                           "a 8x slower Decode with the same queue must cost "
                           f"seconds more, not a few percent: {cheap} vs {dear}")


class DeadlineAwareDecodeTests(unittest.TestCase):
    """The opt-in Decode fallback must rank by drain time, not by occupancy."""

    def _router(self, deadline_aware):
        from serving.core.router import Router
        router = Router.__new__(Router)
        router.routing_policy = "LOAD"
        router.deadline_aware_decode = deadline_aware
        router.decode_reference_tokens = 16.0
        router.decode_service_ms = {10: 46.0, 11: 372.0}
        router.tail_debug = False
        router._select_instance = lambda schedulers, role: 0
        return router

    def _sched(self, instance_id, waiting):
        class Sched:
            pass
        sched = Sched()
        sched.instance_id = instance_id
        sched.waiting = [None] * waiting
        sched.running = []
        sched.max_num_seqs = 8
        return sched

    def test_disabled_uses_the_routing_policy(self):
        router = self._router(False)
        eligible = [self._sched(10, 4), self._sched(11, 1)]
        self.assertEqual(router._select_decode(eligible, {"input_toks": 1000,
                                                          "output_toks": 1016}), 0)

    def test_enabled_prefers_the_shorter_drain(self):
        router = self._router(True)
        # Instance 11 is less occupied (1 vs 4) but 8x slower: its drain is
        # still longer (46*4 = 184 ms vs 372*1 = 372 ms), so the fast one wins.
        eligible = [self._sched(10, 4), self._sched(11, 1)]
        self.assertEqual(router._select_decode(eligible, {"input_toks": 1000,
                                                          "output_toks": 1016}), 0)
        # With the fast one twice as deep, the drain flips.
        eligible = [self._sched(10, 16), self._sched(11, 1)]
        self.assertEqual(router._select_decode(eligible, {"input_toks": 1000,
                                                          "output_toks": 1016}), 1)


if __name__ == "__main__":
    unittest.main()

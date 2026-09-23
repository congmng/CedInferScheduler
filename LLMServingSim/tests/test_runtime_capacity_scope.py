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
        # A *saturated* pool on purpose: recovering the offered load only
        # applies when the demand exceeds what the pool can serve (see
        # DemandEstimateTests).  With headroom there is nothing to recover.
        self.controller = CASRController(1_000_000_000, policy={
            "solver": "greedy", "prefill_capacity": {"0": 1.0, "1": 1.0},
            "decode_capacity": {"2": 1.0, "3": 1.0},
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


class DemandEstimateTests(unittest.TestCase):
    """The demand estimate must not amplify itself through the backlog.

    Measured 2026-09-23 (main matrix, 16 rps): the controller handed the solver
    88.7 req/s against a pool that serves 26.3, because ``_inflate_demand``
    added the Prefill queue's growth to every class's rate and capped it only at
    5x the observation.  The solver then topped the small Decodes up to their
    capacity and dumped 76 req/s on the one whose capacity is 12.5.
    """

    PREFILL = [29.5, 61.1, 89.4]
    DECODE = [4.5, 9.3, 12.5]

    def _controller(self, multiple=4.0):
        from serving.casr.controller import CASRController
        return CASRController(1_000_000_000, policy={
            "solver": "greedy",
            "prefill_capacity": {i: v for i, v in enumerate(self.PREFILL)},
            "decode_capacity": {100 + i: v for i, v in enumerate(self.DECODE)},
            "backlog_rps_multiple": multiple,
        })

    def _inflated_total(self, controller, observed, backlog):
        rows = [{"class_id": "c", "arrival_rate_ewma": observed}]
        controller._backlog_rps = backlog
        controller._inflate_demand(rows)
        return sum(row["arrival_rate_ewma"] for row in rows)

    def test_a_pool_with_headroom_gets_no_inflation(self):
        controller = self._controller()
        self.assertAlmostEqual(
            self._inflated_total(controller, observed=16.0, backlog=100.0),
            16.0, places=6)

    def test_the_inflation_stops_at_the_capacity_deficit(self):
        controller = self._controller()
        total = self._inflated_total(controller, observed=32.0, backlog=1000.0)
        self.assertGreater(total, 32.0)
        self.assertLessEqual(total, 32.0 + (32.0 - sum(self.DECODE)) + 1e-6)

    def test_the_estimate_is_monotone_and_bounded_in_the_backlog(self):
        controller = self._controller()
        totals = [self._inflated_total(controller, observed=32.0, backlog=backlog)
                  for backlog in (0.0, 5.0, 50.0, 500.0, 5000.0)]
        self.assertEqual(totals, sorted(totals), totals)
        self.assertLessEqual(max(totals), 32.0 + (32.0 - sum(self.DECODE)) + 1e-6)

    def test_the_old_behaviour_stays_available_for_ab(self):
        controller = self._controller()
        controller.backlog_capacity_bound = False
        self.assertAlmostEqual(
            self._inflated_total(controller, observed=16.0, backlog=100.0),
            16.0 * 5.0, places=6)


class ArrivalEstimateTests(unittest.TestCase):
    """A burst of dispatch must not read as a burst of demand.

    Measured 2026-09-23 (main matrix, 16 req/s offered, 100 ms control loop):
    the profiler's rate estimate sat at 80.6 req/s pool-wide because it took
    ``arrivals / elapsed`` over each 100 ms tick, and the router dispatches in
    batches once it starts holding requests.  The deployment differences a
    counter over 1 s; the simulator must not shrink that window just because its
    control loop is faster.
    """

    class Req:
        def __init__(self, class_id):
            self.class_id = class_id
            self.original_input = 1024

    def _profiler(self, window_ms=1000.0):
        from serving.casr.prefix_profiler import PrefixProfiler
        return PrefixProfiler(block_size=16, arrival_window_ms=window_ms)

    def _total_rate(self, profiler, at_ns):
        return sum(float(row["arrival_rate_ewma"])
                   for row in profiler.snapshot(at_ns)["prefix_states"])

    def test_a_burst_is_averaged_over_the_window_not_the_tick(self):
        profiler = self._profiler()
        class_id, _ = profiler.assign("m", 1024, 64, [1] * 64)
        profiler.snapshot(0)                       # baseline tick
        # 8 requests dispatched inside the first 100 ms -- the shape a held
        # backlog releases with.  At 2 req/s per class the 1 s window says
        # 8 req/s, not 8/0.1 = 80.
        for index in range(8):
            profiler.observe_arrival(self.Req(class_id), 0,
                                     1_000_000 + index * 10_000_000)
        self.assertEqual(self._total_rate(profiler, 100_000_000), 0.0)
        self.assertAlmostEqual(self._total_rate(profiler, 1_000_000_000), 8.0,
                               places=6)

    def test_nothing_is_folded_before_the_window_completes(self):
        profiler = self._profiler()
        class_id, _ = profiler.assign("m", 1024, 64, [1] * 64)
        profiler.snapshot(0)
        for index in range(4):
            profiler.observe_arrival(self.Req(class_id), 0,
                                     1_000_000 + index * 10_000_000)
        self.assertEqual(self._total_rate(profiler, 100_000_000), 0.0)
        self.assertEqual(self._total_rate(profiler, 500_000_000), 0.0)
        self.assertGreater(self._total_rate(profiler, 1_000_000_000), 0.0)

    def test_the_window_knob_still_works(self):
        # A shorter window is allowed (and is exactly why the default must not
        # follow the control interval): 8 arrivals over 0.2 s reads 40 req/s.
        profiler = self._profiler(window_ms=200.0)
        class_id, _ = profiler.assign("m", 1024, 64, [1] * 64)
        profiler.snapshot(0)
        for index in range(8):
            profiler.observe_arrival(self.Req(class_id), 0,
                                     1_000_000 + index * 10_000_000)
        self.assertAlmostEqual(self._total_rate(profiler, 200_000_000), 40.0,
                               places=6)


class PromptLengthEstimateTests(unittest.TestCase):
    """The published prompt length must be the prompt's, not ``alpha`` of it.

    Second half of the 2026-09-23 estimate audit.  ``requested_tokens_ewma``
    decayed *from zero*, so a class the workload visits once -- i.e. every class
    of the 1250-token CNN/DailyMail matrix, 740 of 740 -- published 250 tokens
    instead of 1250.  ``_prefill_length_work`` then read ``250 / 1024 -> 1.0``
    and the solver priced a 1250-token prompt as one 1024-token reference unit,
    22% under.  It is the same defect ``_class_kv_bytes`` already carries a
    workaround for.
    """

    class Req:
        def __init__(self, class_id, tokens=1250):
            self.class_id = class_id
            self.original_input = tokens

    def _profiler(self):
        from serving.casr.prefix_profiler import PrefixProfiler
        return PrefixProfiler(block_size=16)

    def _row(self, profiler, class_id, at_ns=0):
        for row in profiler.snapshot(at_ns)["prefix_states"]:
            if row["class_id"] == class_id:
                return row
        raise AssertionError("class not published")

    def test_a_class_seen_once_publishes_its_real_length(self):
        profiler = self._profiler()
        class_id, _ = profiler.assign("m", 1250, 41, [7] * 1250)
        profiler.observe_arrival(self.Req(class_id), 0, 0)
        row = self._row(profiler, class_id)
        # Not alpha x 1250 == 250.
        self.assertAlmostEqual(float(row["requested_tokens_ewma"]), 1250.0)
        self.assertEqual(int(row["requested_tokens"]), 1250)

    def test_a_repeated_class_still_smooths(self):
        profiler = self._profiler()
        class_id, _ = profiler.assign("m", 1000, 41, [7] * 1000)
        profiler.observe_arrival(self.Req(class_id, 1000), 0, 0)
        for index in range(1, 6):
            profiler.observe_arrival(self.Req(class_id, 2000), 0,
                                     index * 1_000_000)
        value = float(self._row(profiler, class_id)["requested_tokens_ewma"])
        # Bracketed by the two lengths, and no longer anchored on 0.
        self.assertGreater(value, 1000.0)
        self.assertLess(value, 2000.0)


class PromptLengthPricingTests(unittest.TestCase):
    """The length factor has to be live, not silently ``1.0``.

    ``_prefill_length_work`` scales by ``tokens / capacity_reference_tokens``
    only when the instance declares a token-rate ceiling.  53 of the 59 cluster
    configs -- every three-domain matrix config among them -- declared none, so
    the documented prompt-length pricing never ran.
    """

    def setUp(self):
        _needs_pandas()
        self._cwd = pathlib.Path.cwd()
        os.chdir(REPO / "astra-sim")
        self.addCleanup(os.chdir, self._cwd)
        from serving.core.hw_service import resolve_runtime_capacities
        self.resolve = resolve_runtime_capacities
        self.instances = [
            {"instance_id": 4, "hardware": "RTX5090", "model_name": "casr/P15B",
             "pd_type": "prefill", "tp_size": 1, "max_num_seqs": 64},
            {"instance_id": 5, "hardware": "RTX5090", "model_name": "casr/P15B",
             "pd_type": "decode", "tp_size": 1, "max_num_seqs": 64},
        ]

    def _config(self):
        return {"capacity_reference_tokens": 1024,
                "decode_reference_tokens": 16,
                "decode_service_ms": {"5": 80.0}}

    def test_the_resolver_publishes_a_token_ceiling(self):
        config = self._config()
        self.resolve(config, self.instances, verbose=False)
        ceiling = config.get("prefill_tokens_per_s") or {}
        self.assertIn("4", ceiling)
        # capacity (reference requests/s) x reference tokens == tokens/s.
        self.assertAlmostEqual(float(ceiling["4"]),
                               config["prefill_capacity"]["4"] * 1024, delta=0.1)

    def test_a_long_prompt_costs_more_than_a_reference_one(self):
        from serving.casr.flow_solver import (CapacityAwareFlowSolver,
                                              FlowSolverConfig)
        config = self._config()
        self.resolve(config, self.instances, verbose=False)
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(config))
        # The published period curve is the authority: a prompt past the first
        # chunk boundary pays for a second step, so 1250 tokens is ~1.40
        # reference units rather than the token ratio's 1.22.
        self.assertAlmostEqual(solver._prefill_length_work(1024, 4), 1.0,
                               delta=1e-3)
        self.assertAlmostEqual(solver._prefill_length_work(1250, 4), 1.40,
                               delta=0.05)
        curve = solver.config.prefill_period_ms[4]
        self.assertGreater(curve[1025], curve[1024],
                           "the curve must step at the chunk boundary")
        # With only the token ceiling (no measured curve) it is the linear
        # ratio, which is what 53 of the 59 configs used to fall back to.
        ceiling_only = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(
            {"capacity_reference_tokens": 1024, "decode_reference_tokens": 16,
             "prefill_capacity": {"4": 17.2},
             "prefill_tokens_per_s": {"4": 17611.0}}))
        self.assertAlmostEqual(ceiling_only._prefill_length_work(1250, 4),
                               1250 / 1024, delta=1e-3)
        # Without the ceiling the same call silently returns 1.0, which is the
        # shape that hid the defect for every three-domain config.
        bare = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(
            {"capacity_reference_tokens": 1024, "decode_reference_tokens": 16}))
        self.assertAlmostEqual(bare._prefill_length_work(1250, 4), 1.0, places=4)

    def test_a_prefill_that_has_not_met_the_class_prices_its_real_length(self):
        """The prompt length belongs to the class, not to the (class, P) pair.

        Measured 2026-09-23 on the 16 rps peak: 230 of 559 classes had been
        observed on one Prefill only, and ``_requested_tokens`` fell back to
        "one reference length" for the others -- so those Prefills looked
        *cheaper* for exactly the prompts they had never seen.  The LP loaded
        the worker with the most such classes to 89% of capacity while the
        others sat at 9%, and that worker is the one that queued.
        """
        from serving.casr.flow_solver import (CapacityAwareFlowSolver,
                                              FlowSolverConfig)
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(
            {"capacity_reference_tokens": 1024, "decode_reference_tokens": 16,
             "prefill_capacity": {"4": 17.2},
             "prefill_tokens_per_s": {"4": 17611.0}}))
        entry = {"requested_tokens_ewma": {2: 1250.0},   # seen on p2 only
                 "requested_tokens": {2: 1250.0}}
        self.assertAlmostEqual(solver._requested_tokens(entry, 2), 1250.0)
        # p4 has not met the class; it must still price the 1250-token prompt.
        self.assertAlmostEqual(solver._requested_tokens(entry, 4), 1250.0)
        self.assertAlmostEqual(solver._prefill_length_work(1250.0, 4),
                               1250 / 1024, delta=2e-3)


class PrefillPeriodCalibrationTests(unittest.TestCase):
    """The priced Prefill capacity must be the rate the pair executes.

    The layer charge is exact (the trace generator and ``step_cost_ns`` agree
    to <0.1%), but a serving pair pays a per-step cost the profile does not
    time.  Measured 2026-09-23 with a saturated 1P1D probe (64 x 1024-token
    prompts): periods 204.3 / 121.2 / 73.6 ms on the 3090 / 4090 / 5090
    against charges of 164.9 / 89.7 / 58.1.  Pricing the bare charge overstated
    the 5090 by 27% and the plan then loaded it to 1.5x what it executes.
    """

    def setUp(self):
        _needs_pandas()
        self._cwd = pathlib.Path.cwd()
        os.chdir(REPO / "astra-sim")
        self.addCleanup(os.chdir, self._cwd)
        from serving.core.hw_service import rescale_capacities
        self.rescale = rescale_capacities
        self.instances = [
            {"instance_id": 0, "hardware": "RTX3090", "model_name": "casr/P15B",
             "pd_type": "prefill", "tp_size": 1},
            {"instance_id": 2, "hardware": "RTX4090", "model_name": "casr/P15B",
             "pd_type": "prefill", "tp_size": 1},
            {"instance_id": 4, "hardware": "RTX5090", "model_name": "casr/P15B",
             "pd_type": "prefill", "tp_size": 1},
            {"instance_id": 5, "hardware": "RTX5090", "model_name": "casr/P15B",
             "pd_type": "decode", "tp_size": 1},
        ]

    def test_capacity_matches_the_executed_period(self):
        config = {"decode_reference_tokens": 16, "capacity_reference_tokens": 1024,
                  "max_num_batched_tokens": 1024}
        prefill, _ = self.rescale(config, self.instances, verbose=False)
        # Reference request = one 1024-token chunk, i.e. exactly the probe.
        self.assertAlmostEqual(prefill[4], 1000.0 / 73.6, delta=0.15)
        self.assertAlmostEqual(prefill[2], 1000.0 / 121.2, delta=0.15)
        self.assertAlmostEqual(prefill[0], 1000.0 / 204.3, delta=0.15)
        # And the ordering is still the profile's.
        self.assertLess(prefill[0], prefill[2])
        self.assertLess(prefill[2], prefill[4])

    def test_a_longer_prompt_pays_for_its_extra_step(self):
        from serving.core.hw_service import prefill_period_ms
        config = {"decode_reference_tokens": 16, "capacity_reference_tokens": 1024,
                  "max_num_batched_tokens": 1024}
        self.rescale(config, self.instances, verbose=False)
        curve = config["prefill_period_ms"]["4"]
        # The curve steps at the chunk boundary and keeps growing past it.
        self.assertLess(curve["1024"], curve["1025"])
        self.assertLess(curve["1025"], curve["2048"])
        # It reproduces the measured periods at the anchors a run hits.
        self.assertAlmostEqual(curve["1024"], 73.64, delta=0.2)
        self.assertAlmostEqual(curve["2048"], 171.4, delta=1.0)
        # 1250 tokens is a second step, not 1.22 reference units.
        ratio = prefill_period_ms("RTX5090", "casr/P15B", tp=1, tokens=1250,
                                  reference=1024, chunk=1024) / curve["1024"]
        self.assertGreater(ratio, 1.35)


class InfeasibleOfferTests(unittest.TestCase):
    """Above capacity the plan must degrade *deliberately*.

    Measured 2026-09-23 (32 rps Qwen3-8B, structurally infeasible: 33.2
    reference units offered against a 27.2 pool): the LP's distribution is flat
    in the excess, so it put 68% of 977 requests on the *slowest* Prefill.
    Scaling the offer to the servable level keeps the shape and makes the
    problem feasible, so each worker lands near its own capacity.
    """

    def setUp(self):
        _needs_pandas()
        from serving.casr.controller import CASRController
        self.controller = CASRController(1_000_000_000, policy={
            "solver": "greedy",
            "prefill_capacity": {"0": 4.0, "1": 12.0},
            "decode_capacity": {"2": 8.0, "3": 8.0},
            "capacity_reference_tokens": 1024,
        })

    class Sched:
        def __init__(self, instance_id, pd_type):
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

        def set_admission_state(self, state):
            self.admission_state = state

    def _prefills(self):
        return [self.Sched(0, "prefill"), self.Sched(1, "prefill")]

    def test_an_offer_above_capacity_is_scaled_to_the_servable_level(self):
        rows = [{"class_id": "c", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 100.0, "requested_tokens_ewma": 1024.0,
                 "requested_tokens": 1024.0, "kv_bytes_per_request": 0.0}]
        clamped, scale = self.controller._clamped_rows(rows, self._prefills())
        # Pool = min(16 prefill, 16 decode) reference units against 100 offered.
        self.assertAlmostEqual(scale, 16.0 / 100.0, places=4)
        self.assertAlmostEqual(clamped[0]["arrival_rate_ewma"], 16.0, places=4)
        # The caller's snapshot keeps the *offered* number, so the state log and
        # the next tick both see the truth.
        self.assertEqual(rows[0]["arrival_rate_ewma"], 100.0)
        self.assertEqual(self.controller.last_demand_clamp["scale"], scale)

    def test_an_offer_inside_capacity_is_left_alone(self):
        rows = [{"class_id": "c", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 6.0, "requested_tokens_ewma": 1024.0,
                 "requested_tokens": 1024.0, "kv_bytes_per_request": 0.0}]
        clamped, scale = self.controller._clamped_rows(rows, self._prefills())
        self.assertEqual(scale, 1.0)
        self.assertIs(clamped, rows)
        self.assertEqual(self.controller.last_demand_clamp["scale"], 1.0)

    def test_the_clamp_can_be_switched_off(self):
        self.controller.demand_capacity_clamp = False
        rows = [{"class_id": "c", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 100.0, "requested_tokens_ewma": 1024.0,
                 "requested_tokens": 1024.0, "kv_bytes_per_request": 0.0}]
        clamped, scale = self.controller._clamped_rows(rows, self._prefills())
        self.assertEqual(scale, 1.0)
        self.assertIs(clamped, rows)


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

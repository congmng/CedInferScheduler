import unittest
from types import SimpleNamespace

from serving.casr.affinity import AffinityPlan
from serving.casr.flow_solver import (CapacityAwareFlowSolver, FlowAssignment,
                                      FlowSolverConfig)
from serving.casr.prefix_profiler import PrefixProfiler
from serving.casr.resources import ResourceOrchestrator
from serving.casr.evaluator import StructuralEvaluator
from serving.casr.state import parse_prometheus_text
from serving.casr.executor import ReconfigExecutor


class _Scheduler:
    def __init__(self, instance_id, start_npu):
        self.instance_id = instance_id
        self.start_npu = start_npu
        self.max_num_seqs = 8


class _Memory:
    def __init__(self, gb):
        self.npu_mem = gb * 1024 ** 3


class _QueuedScheduler(_Scheduler):
    """Scheduler with observable queue depth, for the TTFT SLO model."""

    def __init__(self, instance_id, start_npu, running=0, waiting=0):
        super().__init__(instance_id, start_npu)
        self.running = [None] * running
        self.waiting = [None] * waiting


class _ResourceScheduler:
    def __init__(self, instance_id, state="ACTIVE", node_id=0):
        self.instance_id = instance_id
        self.node_id = node_id
        self.num_npus = 1
        self.memory = _Memory(24)
        self.admission_state = state
        self.running = []
        self.waiting = []

    def set_admission_state(self, state):
        self.admission_state = state


class CasrTests(unittest.TestCase):
    def test_zero_flow_assignments_are_placeholders_not_errors(self):
        """A zero-flow row must not fail validation.

        The builtin solver emits one row per (class, prefill, decode) and a
        class whose EWMA arrival rate has decayed to zero gets flow 0 --
        ``class_demand_floor_rps`` defaults to 0.0.  Rejecting that made every
        builtin-solver run die on the first such class, which is what
        ``tests/run_casr_comparison.sh`` hit before this was fixed.
        """
        from serving.casr.controller import CASRController
        from serving.casr.flow_solver import FlowAssignment

        controller = CASRController(100_000_000)
        snapshot = {"prefix_states": [
            {"class_id": "m|p16:aa", "arrival_rate_ewma": 4.0},
            {"class_id": "m|p16:bb", "arrival_rate_ewma": 0.0},
        ]}
        prefill = [SimpleNamespace(instance_id=0)]
        decode = [SimpleNamespace(instance_id=1)]
        kept = controller._validate_flows(
            [FlowAssignment("m|p16:aa", 0, 1, 4.0, 1.0),
             FlowAssignment("m|p16:bb", 0, 1, 0.0, 0.0)],
            snapshot, prefill, decode)
        self.assertEqual([item.class_id for item in kept], ["m|p16:aa"])

    def test_arrival_rate_counts_requests_over_the_window(self):
        """The simulator must estimate demand the same way the real router does.

        Arrivals are counted over the control window instead of inverting the
        gap between them, so a burst dispatched in the same millisecond cannot
        look like thousands of requests per second.
        """
        profiler = PrefixProfiler(block_size=16, ewma_alpha=0.5)
        class_id, _ = profiler.assign("m", 32, 8, range(32))
        request = SimpleNamespace(class_id=class_id, original_input=32)
        profiler.snapshot(0)
        for _ in range(4):
            profiler.observe_arrival(request, 0, at_ns=10_000_000)
        rows = profiler.snapshot(1_000_000_000)["prefix_states"]
        rate = next(row["arrival_rate_ewma"] for row in rows if row["class_id"] == class_id)
        self.assertAlmostEqual(rate, 4.0)

        for _ in range(100):
            profiler.observe_arrival(request, 0, at_ns=1_000_000_000)
        rows = profiler.snapshot(2_000_000_000)["prefix_states"]
        rate = next(row["arrival_rate_ewma"] for row in rows if row["class_id"] == class_id)
        self.assertAlmostEqual(rate, 52.0)  # 0.5 * 100 + 0.5 * 4

    def test_prefix_class_is_stable_and_block_aligned(self):
        profiler = PrefixProfiler(block_size=16)
        first = profiler.assign("model", 33, 8, range(33))
        second = profiler.assign("model", 33, 8, range(33))
        self.assertEqual(first, second)
        self.assertTrue(first[0].startswith("model|p32:"))

    def test_affinity_weights_are_normalized_and_expire(self):
        plan = AffinityPlan(3, 100, {"c": {0: 2, 1: 1}})
        self.assertEqual(plan.prefill_for("c"), {0: 2 / 3, 1: 1 / 3})
        self.assertFalse(plan.is_expired(99))
        self.assertTrue(plan.is_expired(100))

    def test_solver_conserves_flow_and_accounts_link_overflow(self):
        rows = [{"class_id": "c", "arrival_rate_ewma": 3.0,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 10}]
        prefill = [_Scheduler(0, 0), _Scheduler(1, 1)]
        decode = [_Scheduler(2, 2), _Scheduler(3, 3)]
        config = FlowSolverConfig.from_dict({
            "prefill_capacity": {"0": 8, "1": 8},
            "decode_capacity": {"2": 1, "3": 8},
            "shared_links": [{"id": "l", "capacity": 1, "pairs": [[0, 2], [1, 2]]}],
        })
        flows = CapacityAwareFlowSolver(config).solve(rows, prefill, decode)
        self.assertAlmostEqual(sum(flow.flow for flow in flows), 3.0)
        self.assertTrue(any(flow.decode_id == 3 for flow in flows))

    def test_lp_aggregates_duplicate_class_observations(self):
        rows = [
            {"class_id": "c", "arrival_rate_ewma": 2.0,
             "request_count": 1, "hit_tokens_ewma": 0.0,
             "requested_tokens": 10},
            {"class_id": "c", "arrival_rate_ewma": 3.0,
             "request_count": 1, "hit_tokens_ewma": 0.0,
             "requested_tokens": 10},
        ]
        prefill = [_Scheduler(0, 0), _Scheduler(1, 1)]
        decode = [_Scheduler(2, 2), _Scheduler(3, 3)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({"solver": "lp"}))
        try:
            flows = solver.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        self.assertAlmostEqual(sum(flow.flow for flow in flows), 5.0)

    def test_low_rate_observations_are_not_inflated_to_one_request_per_row(self):
        """Demand is the *sum of the estimated rates*, not a row count.

        Every row used to contribute ``max(rate, 1.0)``.  The real router
        emits one row per (prefill, class), so a few hundred lightly loaded
        classes produced several hundred req/s of phantom demand against ~100
        req/s of capacity: the LP then solved a structurally overloaded problem
        (measured prefill overflow ~7x, KV link overflow ~25x) and replayed
        those fractions onto real traffic.  See
        ``docs/实验结果汇总.md`` §5.5.
        """
        rows = [{"class_id": f"c{i}", "arrival_rate_ewma": 0.1,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 10,
                 "prefill_instance_id": i % 2}
                for i in range(12)]
        prefill = [_Scheduler(0, 0), _Scheduler(1, 1)]
        decode = [_Scheduler(2, 2)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({"solver": "lp"}))
        try:
            flows = solver.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        self.assertAlmostEqual(solver.diagnostics["total_demand_rps"], 1.2, places=6)
        self.assertAlmostEqual(sum(flow.flow for flow in flows), 1.2, places=6)

    def test_compute_cost_steers_flow_off_the_slow_instance(self):
        """A slow same-domain pair must lose to a fast one once compute is priced.

        Capacity alone is not enough: with wide capacity the LP is free to put
        overflow on the slowest instance because that pair has zero link cost.
        """
        rows = [{"class_id": "c", "arrival_rate_ewma": 3.0,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 10}]
        prefill = [_Scheduler(0, 0), _Scheduler(1, 0)]
        decode = [_Scheduler(2, 0), _Scheduler(3, 0)]
        base = {
            "solver": "lp",
            "prefill_capacity": {"0": 500, "1": 500},
            "decode_capacity": {"2": 500, "3": 500},
        }
        free = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(base))
        priced = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(dict(
            base,
            compute_weight=1.0,
            prefill_service_ms={"0": 40.0, "1": 400.0},
            decode_service_ms={"2": 150.0, "3": 400.0},
        )))
        try:
            free_flows = free.solve(rows, prefill, decode)
            priced_flows = priced.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        self.assertAlmostEqual(sum(f.flow for f in free_flows), 3.0)
        self.assertAlmostEqual(sum(f.flow for f in priced_flows), 3.0)
        # index 0/2 is the fast pair; the priced solve must never use 1 or 3.
        self.assertEqual({(f.prefill_id, f.decode_id) for f in priced_flows},
                         {(0, 2)})
        self.assertEqual(FlowSolverConfig.from_dict({
            "compute_weight": 2.5,
            "prefill_service_ms": {"4": 39.2},
            "decode_service_ms": {"11": 144.7},
        }).prefill_service_ms[4], 39.2)

    def test_slo_penalty_moves_flow_off_an_over_slo_pair(self):
        """``p_slo`` must trade a cheaper pair for an SLO-compliant one.

        Decode 2 is faster (50 ms) but fully queued, so its predicted TTFT is
        ``prefill 40 + 50 x 1.0 = 90 ms``; Decode 3 is slower (100 ms) but idle,
        so it predicts ``40 + 0 = 40 ms``.  Without the SLO term the LP takes
        the cheaper compute; with an 80 ms bound it must take the compliant
        pair, and the penalised pair has to be visible in the diagnostics.
        """
        rows = [{"class_id": "c", "arrival_rate_ewma": 3.0,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 10}]
        prefill = [_QueuedScheduler(0, 0)]
        decode = [_QueuedScheduler(2, 0, running=8),
                  _QueuedScheduler(3, 0, running=0)]
        base = {
            "solver": "lp",
            "prefill_capacity": {"0": 500},
            "decode_capacity": {"2": 500, "3": 500},
            "compute_weight": 1.0,
            "prefill_service_ms": {"0": 40.0},
            "decode_service_ms": {"2": 50.0, "3": 100.0},
            "queue_weight": 0.0,
        }
        cheap = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(base))
        guarded = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(dict(
            base, ttft_slo_ms=80.0, slo_penalty=100.0)))
        try:
            cheap_flows = cheap.solve(rows, prefill, decode)
            guarded_flows = guarded.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        self.assertEqual({(f.prefill_id, f.decode_id) for f in cheap_flows},
                         {(0, 2)})
        self.assertEqual({(f.prefill_id, f.decode_id) for f in guarded_flows},
                         {(0, 3)})
        self.assertEqual(guarded.diagnostics["ttft_slo_ms"], 80.0)
        self.assertIn((0, 2), guarded.diagnostics["slo_violating_pairs"])

    def test_slo_term_is_inert_until_it_is_configured(self):
        rows = [{"class_id": "c", "arrival_rate_ewma": 2.0,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 10}]
        prefill = [_QueuedScheduler(0, 0)]
        decode = [_QueuedScheduler(2, 0, running=8)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp", "prefill_capacity": {"0": 500},
            "decode_capacity": {"2": 500}, "compute_weight": 1.0,
            "prefill_service_ms": {"0": 40.0}, "decode_service_ms": {"2": 50.0},
        }))
        try:
            solver.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        # Default ``ttft_slo_ms=0`` keeps the previous objective exactly.
        self.assertEqual(solver.diagnostics["slo_violating_pairs"], [])
        self.assertEqual(FlowSolverConfig.from_dict(
            {"ttft_slo_ms": 250.0, "slo_penalty": 1e6}).slo_penalty, 1e6)
        self.assertEqual(FlowSolverConfig.from_dict(
            {"class_ttft_slo_ms": {"c": 120.0}}).class_ttft_slo_ms["c"], 120.0)

    def test_each_cost_term_decides_only_where_it_dominates(self):
        """A controlled "fast-but-far vs slow-but-near" conflict.

        The real cluster cannot currently produce this conflict: with ``p3090a``
        down, the slow Decode ``d3090a`` also has no same-domain Prefill, so the
        slow option is *also* the far one -- the three cost terms all point the
        same way and an ablation cannot separate them (measured 2026-09-12).
        This probe builds the conflict explicitly on the shared solver:

        * ``d_fast``  : service 100 ms, cross-domain link.
        * ``d_near``  : service 400 ms, same domain (transfer 0).

        Two link regimes are tried.  Whichever term dominates decides, and the
        ablation of *that* term is what flips the choice -- i.e. no term is
        individually redundant, but none of them is individually sufficient
        either.
        """
        rows = [{"class_id": "c", "arrival_rate_ewma": 3.0,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 2048, "kv_bytes_per_request": 3.0e8}]
        prefill = [_Scheduler(0, 0)]

        def solve(transfer_bandwidth, **overrides):
            decode = [_Scheduler(1, 1), _Scheduler(2, 0)]
            options = {
                "solver": "lp",
                "prefill_capacity": {"0": 500},
                "decode_capacity": {"1": 500, "2": 500},
                "compute_weight": 1.0,
                "prefill_service_ms": {"0": 50.0},
                "decode_service_ms": {"1": 100.0, "2": 400.0},
                "pair_costs": {
                    (0, 1): {"rtt_ms": 5.0,
                             "bandwidth_bytes_per_s": transfer_bandwidth},
                    (0, 2): {"rtt_ms": 0.0, "bandwidth_bytes_per_s": 0.0},
                },
                "queue_weight": 0.0,
                "network_weight": 1.0,
            }
            options.update(overrides)
            solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(options))
            flows = solver.solve(rows, prefill, decode)
            return {(f.prefill_id, f.decode_id) for f in flows}

        try:
            # 300 MB over 6 GB/s -> ~50 ms transfer: service dominates, so the
            # fast-but-far Decode wins until the service term is removed.
            fast_link = 6.0e9
            self.assertEqual(solve(fast_link), {(0, 1)})
            self.assertEqual(solve(fast_link, network_weight=0.0), {(0, 1)})
            self.assertEqual(solve(fast_link, compute_weight=0.0), {(0, 2)})
            # 300 MB over 0.6 GB/s -> ~500 ms transfer: the link dominates, so
            # the near-but-slow Decode wins until the network term is removed.
            slow_link = 0.6e9
            self.assertEqual(solve(slow_link), {(0, 2)})
            self.assertEqual(solve(slow_link, network_weight=0.0), {(0, 1)})
            self.assertEqual(solve(slow_link, compute_weight=0.0), {(0, 2)})
        except ImportError:
            self.skipTest("OR-Tools is optional")

    def test_prefill_class_limit_spreads_the_working_set(self):
        """A Prefill cannot hold an unbounded number of distinct prefixes.

        The LP prices cache *hits* per (class, Prefill) but has no notion of
        the working set, so with one clearly cheaper Prefill it piles every
        class there and pays in evictions -- measured 2026-09-13: ``casr_lp``
        put 95% of the wide trace's 537 classes on ``p5090`` (``prefill_ms``
        P50 80.7 -> 105.4 ms while ``decode_ms`` did not move) and lost ~6%.
        """
        rows = [{"class_id": f"c{i}", "arrival_rate_ewma": 1.0,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 10} for i in range(6)]
        prefill = [_QueuedScheduler(0, 0), _QueuedScheduler(1, 0)]
        decode = [_QueuedScheduler(2, 0)]
        base = {"solver": "lp", "prefill_capacity": {"0": 500, "1": 500},
                "decode_capacity": {"2": 500}, "compute_weight": 1.0,
                "prefill_service_ms": {"0": 40.0, "1": 400.0}}
        loose = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(base))
        capped = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(
            dict(base, prefill_class_limit=3)))
        try:
            loose_flows = loose.solve(rows, prefill, decode)
            capped_flows = capped.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")

        def held(flows):
            per = {}
            for flow in flows:
                per.setdefault(flow.prefill_id, set()).add(flow.class_id)
            return {p: len(classes) for p, classes in per.items()}

        # Without the cap the cheaper Prefill takes everything...
        self.assertGreater(max(held(loose_flows).values()), 3)
        # ...and with it the working set is spread and flow is conserved.
        self.assertLessEqual(max(held(capped_flows).values()), 3)
        self.assertAlmostEqual(sum(f.flow for f in capped_flows),
                               sum(f.flow for f in loose_flows))
        self.assertEqual(capped.diagnostics["capped_classes"], 3)

    def test_plan_objective_charges_uncovered_classes(self):
        """A stale plan must not look free for the classes it never named.

        The controller keeps the incumbent unless the candidate is clearly
        better.  If a class missing from the incumbent scored zero, the
        incumbent would win forever, new classes would never enter the plan,
        and the router would serve them from its least-loaded fallback -- which
        is exactly what frozen ``version=1`` plans did in the live run.
        """
        rows = [
            {"class_id": "a", "arrival_rate_ewma": 1.0, "request_count": 1,
             "hit_tokens_ewma": 0.0, "requested_tokens": 10},
            {"class_id": "b", "arrival_rate_ewma": 1.0, "request_count": 1,
             "hit_tokens_ewma": 0.0, "requested_tokens": 10},
        ]
        prefill = [_Scheduler(0, 0)]
        decode = [_Scheduler(1, 0)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp", "prefill_capacity": {"0": 500},
            "decode_capacity": {"1": 500}}))
        covering = AffinityPlan(version=1, expires_at_ns=0,
                                prefill_weights={"a": {0: 1.0}, "b": {0: 1.0}},
                                decode_weights={(0, "a"): {1: 1.0},
                                                (0, "b"): {1: 1.0}},
                                fallback_decode_ids={})
        partial = AffinityPlan(version=2, expires_at_ns=0,
                               prefill_weights={"a": {0: 1.0}},
                               decode_weights={(0, "a"): {1: 1.0}},
                               fallback_decode_ids={})
        self.assertLess(solver.plan_objective(covering, rows, prefill, decode),
                        solver.plan_objective(partial, rows, prefill, decode))
        # The knob is configurable, and an explicit zero restores the old
        # (broken) scoring for anyone who needs the previous behaviour.
        lax = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp", "prefill_capacity": {"0": 500},
            "decode_capacity": {"1": 500}, "plan_uncovered_penalty": 0.0}))
        # With the charge disabled the partial plan is scored as free.
        self.assertEqual(lax.plan_objective(partial, rows, prefill, decode), 0.0)
        self.assertEqual(FlowSolverConfig.from_dict(
            {"plan_uncovered_penalty": 3.0}).plan_uncovered_penalty, 3.0)

    def test_ttft_overheads_make_an_idle_slow_pair_violate(self):
        """A queue-only decode term predicts ~0 ms idle, so nothing ever binds.

        Measured on the real cluster: the router-observed ``prefill_ms`` is
        ~101 ms where the compute service is 48.5 ms, and TTFT keeps ~56 ms on
        the decode side even with an empty queue.  With those two calibrated
        terms an idle 3090a pair (184 + 68 + 56 = 308 ms) is over a 250 ms
        interactive bound, while the 5090 pair (52 + 48.5 + 56 = 156 ms) is not.
        """
        rows = [{"class_id": "c", "arrival_rate_ewma": 3.0,
                 "request_count": 1, "hit_tokens_ewma": 0.0,
                 "requested_tokens": 10}]
        prefill = [_QueuedScheduler(0, 0), _QueuedScheduler(1, 1)]
        decode = [_QueuedScheduler(2, 0), _QueuedScheduler(3, 1)]
        base = {
            "solver": "lp",
            "prefill_capacity": {"0": 500, "1": 500},
            "decode_capacity": {"2": 500, "3": 500},
            "compute_weight": 1.0,
            "prefill_service_ms": {"0": 48.5, "1": 68.0},
            "decode_service_ms": {"2": 144.7, "3": 421.6},
            "queue_weight": 0.0,
            "pair_costs": {},
        }
        queue_only = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(dict(
            base, class_ttft_slo_ms={"c": 250.0}, slo_penalty=10.0)))
        calibrated = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(dict(
            base, class_ttft_slo_ms={"c": 250.0}, slo_penalty=10.0,
            prefill_overhead_ms={"0": 52.0, "1": 184.0},
            decode_ttft_ms={"2": 56.0, "3": 163.0})))
        try:
            queue_only.solve(rows, prefill, decode)
            calibrated.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        # Queue-only model: every pair looks SLO-clean while idle.
        self.assertEqual(queue_only.diagnostics["slo_violating_pairs"], [])
        # Calibrated model: the slow domain's pairs are flagged.
        self.assertTrue(calibrated.diagnostics["slo_violating_pairs"])

    def test_plan_objective_prices_congestion_and_link_overflow(self):
        """Hysteresis must score a plan with the objective the LP minimised.

        The incumbent is usually the cheaper-but-concentrated plan, so a purely
        linear ``plan_objective`` always rates it better than the balanced
        optimum and plan hysteresis pins the overloaded plan in place.  The
        capacity, congestion and link-overflow terms have to be priced here too,
        otherwise cold-start routing never spreads.
        """
        rows = [{"class_id": "c", "arrival_rate_ewma": 4.0,
                 "prefill_instance_id": 0,
                 "hit_tokens_ewma": 0.0, "requested_tokens": 2048}]
        prefill = [_Scheduler(0, 0), _Scheduler(1, 0)]
        decode = [_Scheduler(2, 0)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp",
            "compute_weight": 1.0,
            "utilization_weight": 1.0,
            "prefill_service_ms": {"0": 10.0, "1": 60.0},
            "decode_service_ms": {"2": 50.0},
            "prefill_capacity": {"0": 500, "1": 500},
            "decode_capacity": {"2": 500},
            "default_kv_bytes": 100000000.0,
            "shared_links": [
                {"id": "l0", "capacity_bytes_per_s": 200000000.0, "pairs": [[0, 2]]},
                {"id": "l1", "capacity_bytes_per_s": 200000000.0, "pairs": [[1, 2]]},
            ],
        }))
        concentrated = AffinityPlan(1, 0, {"c": {0: 1.0}}, {(0, "c"): {2: 1.0}})
        balanced = AffinityPlan(2, 0, {"c": {0: 0.5, 1: 0.5}},
                                {(0, "c"): {2: 1.0}, (1, "c"): {2: 1.0}})
        concentrated_cost = solver.plan_objective(concentrated, rows, prefill, decode)
        balanced_cost = solver.plan_objective(balanced, rows, prefill, decode)
        # Concentrated is genuinely cheaper per unit of flow (Prefill 0 is fast)
        # and must still lose once its KV push budget is exceeded.
        self.assertGreater(concentrated_cost, balanced_cost)

    def test_kv_bytes_derive_from_tokens_for_byte_denominated_links(self):
        """The simulator must charge a shared link in bytes, like the real router.

        Real LMCache producer links are measured in bytes/s, so the shared-link
        budget only matches reality when the solver knows how many KV bytes a
        class moves.  The trace path reports tokens, not bytes, so the solver
        has to derive them from the per-request token count.

        It uses the *exact* observation when there is one.  The token EWMA is
        only a fallback: a unique-prompt workload observes every class exactly
        once, so its EWMA never converges past ``alpha x tokens`` -- 250 for a
        1250-token prompt -- which under-priced the handoff five-fold and hid
        the producer-egress constraint that actually binds this fabric
        (measured 2026-09-16).
        """
        config = FlowSolverConfig.from_dict({
            "kv_bytes_per_token": 147456.0,
            "default_kv_bytes": 1.0,
        })
        solver = CapacityAwareFlowSolver(config)
        entry = {"requested_tokens_ewma": {0: 2048.0}, "requested_tokens": {0: 999999.0}}
        self.assertAlmostEqual(solver._class_kv_bytes("c", entry), 999999.0 * 147456.0)
        # Without an exact observation the EWMA is still used.
        self.assertAlmostEqual(
            solver._class_kv_bytes("c", {"requested_tokens_ewma": {0: 2048.0}}),
            2048.0 * 147456.0)
        # An explicit per-class size still wins over the derived one.
        config = FlowSolverConfig.from_dict({
            "kv_bytes_per_token": 147456.0,
            "class_kv_bytes": {"c": 42.0},
        })
        self.assertAlmostEqual(CapacityAwareFlowSolver(config)._class_kv_bytes("c", entry), 42.0)
        # Producers that already report bytes (the real controller) are unchanged.
        self.assertAlmostEqual(
            solver._class_kv_bytes("c", {"kv_bytes_per_request": 7.0}), 7.0)

    def test_low_demand_classes_are_single_homed(self):
        """A class with two requests must not pay two cold prefills.

        The LP is often degenerate for a low-demand class and splits it across
        Prefills; each extra Prefill costs one cold prefill for that class.
        """
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp",
            "single_home_below_rps": 5.0,
            "prefill_capacity": {"0": 100, "1": 100},
            "decode_capacity": {"2": 100},
        }))
        grouped = {"c": {"arrival_rate_ewma": 1.0}}
        prefill = [_Scheduler(0, 0), _Scheduler(1, 0)]
        decode = [_Scheduler(2, 0)]
        assignments = [FlowAssignment("c", 0, 2, 0.6, 0.0),
                       FlowAssignment("c", 1, 2, 0.4, 0.0)]
        out = solver._collapse_low_demand(assignments, grouped, prefill, decode)
        self.assertEqual({row.prefill_id for row in out}, {0})
        self.assertAlmostEqual(sum(row.flow for row in out), 1.0)
        self.assertEqual(solver._last_single_homed, ["c"])

    def test_hot_classes_keep_the_lp_split(self):
        """A class whose demand needs two edges is left alone."""
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp", "single_home_below_rps": 5.0,
            "prefill_capacity": {"0": 100, "1": 100},
            "decode_capacity": {"2": 100},
        }))
        grouped = {"c": {"arrival_rate_ewma": 50.0}}
        prefill = [_Scheduler(0, 0), _Scheduler(1, 0)]
        decode = [_Scheduler(2, 0)]
        assignments = [FlowAssignment("c", 0, 2, 25.0, 0.0),
                       FlowAssignment("c", 1, 2, 25.0, 0.0)]
        out = solver._collapse_low_demand(assignments, grouped, prefill, decode)
        self.assertEqual({row.prefill_id for row in out}, {0, 1})

    def test_single_home_is_skipped_when_the_target_is_full(self):
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp", "single_home_below_rps": 5.0,
            "prefill_capacity": {"0": 1.0, "1": 100},
            "decode_capacity": {"2": 100},
        }))
        grouped = {"c": {"arrival_rate_ewma": 4.0}}
        prefill = [_Scheduler(0, 0), _Scheduler(1, 0)]
        decode = [_Scheduler(2, 0)]
        assignments = [FlowAssignment("c", 0, 2, 3.0, 0.0),
                       FlowAssignment("c", 1, 2, 1.0, 0.0)]
        out = solver._collapse_low_demand(assignments, grouped, prefill, decode)
        self.assertEqual({row.prefill_id for row in out}, {0, 1})

    def test_cached_prefix_keeps_flow_on_the_slower_prefill(self):
        """A warm prefix must beat a faster but cold Prefill.

        The prefill compute term has to scale with the tokens that still need
        computing; otherwise the LP treats a cached class and a cold one as
        equally expensive and migrates it away from its cache.
        """
        row = {"class_id": "c", "arrival_rate_ewma": 3.0,
               "prefill_instance_id": 1,
               "hit_tokens_ewma": 2048.0, "requested_tokens": 2048.0 * 60,
               "requested_tokens_ewma": 2048.0}
        cold = dict(row, hit_tokens_ewma=0.0)
        prefill = [_Scheduler(0, 0), _Scheduler(1, 0)]
        decode = [_Scheduler(2, 0), _Scheduler(3, 0)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "lp",
            "compute_weight": 1.0,
            "prefill_capacity": {"0": 500, "1": 500},
            "decode_capacity": {"2": 500, "3": 500},
            # instance 1 is the slow Prefill, instance 2 the fast Decode.
            "prefill_service_ms": {"0": 40.0, "1": 100.0},
            "decode_service_ms": {"2": 100.0, "3": 300.0},
        }))
        try:
            flows = solver.solve([row], prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        self.assertEqual({(f.prefill_id, f.decode_id) for f in flows}, {(1, 2)})
        # The same class without a warm prefix belongs on the fast Prefill.
        try:
            flows = solver.solve([cold], prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        self.assertEqual({(f.prefill_id, f.decode_id) for f in flows}, {(0, 2)})

    def test_congestion_weight_spreads_flow_across_identical_instances(self):
        """A linear objective stacks everything on one pair; a convex one spreads.

        Every instance here is identical, so the only thing that can keep the
        LP from putting all demand on one of them is a marginal price that grows
        with load -- the term the greedy baseline gets for free by pricing
        ``load / capacity`` as it sweeps classes.
        """
        rows = [{"class_id": "c", "arrival_rate_ewma": 12.0,
                 "hit_tokens_ewma": 0.0, "requested_tokens": 100,
                 "requested_tokens_ewma": 100.0}]
        prefill = [_Scheduler(0, 0), _Scheduler(1, 0)]
        decode = [_Scheduler(2, 0), _Scheduler(3, 0)]
        base = {
            "solver": "lp", "compute_weight": 1.0,
            "prefill_capacity": {"0": 10, "1": 10},
            "decode_capacity": {"2": 10, "3": 10},
            "prefill_service_ms": {"0": 100.0, "1": 100.0},
            "decode_service_ms": {"2": 100.0, "3": 100.0},
        }
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict(
            dict(base, utilization_weight=1.0)))
        try:
            flows = solver.solve(rows, prefill, decode)
        except ImportError:
            self.skipTest("OR-Tools is optional")
        loads = {}
        for flow in flows:
            loads[flow.prefill_id] = loads.get(flow.prefill_id, 0.0) + flow.flow
        self.assertEqual(set(loads), {0, 1})
        self.assertAlmostEqual(sum(loads.values()), 12.0)
        # No instance may be pushed to its capacity: the convex price is what
        # keeps the split balanced (the weightless LP gave 2/10).
        self.assertLess(max(loads.values()), 10.0)

    def test_cumulative_counter_does_not_fake_a_cache_hit(self):
        """``hit_tokens_ewma`` is per request, so is its denominator."""
        row = {"class_id": "c", "arrival_rate_ewma": 3.0,
               "hit_tokens_ewma": 2048.0, "requested_tokens": 2048.0 * 60}
        prefill = [_Scheduler(0, 0)]
        decode = [_Scheduler(2, 0)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig())
        work = solver._aggregate_rows([row], prefill, decode)[1]
        # Without the per-request EWMA the only honest answer is "unknown", so
        # the solver must fall back to "everything still has to be computed".
        self.assertGreater(work[0, "c"], 0.9)
        row["requested_tokens_ewma"] = 2048.0
        work = solver._aggregate_rows([row], prefill, decode)[1]
        self.assertAlmostEqual(work[0, "c"], 0.05)

    def test_resource_pool_releases_before_scale_out(self):
        pool = ResourceOrchestrator({
            "startup_ms": 10,
            "reclaim_ms": 5,
            "nodes": {"0": {"gpu_count": 1, "gpu_mem_gb": 24}},
        })
        first = _ResourceScheduler(0)
        second = _ResourceScheduler(1)
        first.pd_type = second.pd_type = "prefill"
        events = pool.bootstrap([first, second], 0)
        self.assertEqual(first.admission_state, "ACTIVE")
        self.assertEqual(second.admission_state, "INACTIVE")
        self.assertTrue(any(e.action == "resource_reject" for e in events))

        pool.reconcile([first], {1}, 0)
        self.assertIn(0, pool._pending_release)
        events = pool.reconcile([second], {1}, 5_000_000)
        self.assertEqual(second.admission_state, "WARMING")
        self.assertEqual(pool.snapshot()["allocations"]["1"]["gpu_ids"], [0])
        self.assertTrue(any(e.action == "resource_release" for e in events))

    def test_lifecycle_completes_reused_worker_warmup(self):
        from serving.casr.lifecycle import PrefillLifecycle

        lifecycle = PrefillLifecycle({
            "min_active_prefill": 1,
            "warmup_ms": 10,
            "resources": {
                "startup_ms": 10,
                "reclaim_ms": 5,
                "nodes": {"0": {"gpu_count": 2, "gpu_mem_gb": [24, 24]}},
            },
        })
        first = _ResourceScheduler(0)
        second = _ResourceScheduler(1)
        first.pd_type = second.pd_type = "prefill"
        first.max_num_seqs = second.max_num_seqs = 8
        lifecycle.update(0, [], [first, second])
        demand = [{"arrival_rate_ewma": 100.0}]
        lifecycle.update(4_000_000, demand, [first, second])
        lifecycle.update(15_000_000, demand, [first, second])
        self.assertEqual(second.admission_state, "ACTIVE")

    def test_resource_pool_matches_per_gpu_memory(self):
        pool = ResourceOrchestrator({
            "nodes": {"0": {"gpu_count": 2, "gpu_mem_gb": [24, 48]}},
        })
        large = _ResourceScheduler(0)
        large.memory = _Memory(96)
        events = pool.bootstrap([large], 0)
        self.assertEqual(large.admission_state, "INACTIVE")
        self.assertEqual(events[0].action, "resource_reject")

    def test_structural_evaluator_selects_capacity_repair(self):
        active = _Scheduler(0, 0)
        candidate = _Scheduler(1, 1)
        active.admission_state = "ACTIVE"
        candidate.admission_state = "INACTIVE"
        decode = _Scheduler(2, 2)
        rows = [{"class_id": "hot", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 4.0, "request_count": 4,
                 "hit_tokens_ewma": 0.0, "requested_tokens": 64}]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "solver": "greedy",
            "prefill_capacity": {"0": 1, "1": 4},
            "decode_capacity": {"2": 8},
        }))
        decision = StructuralEvaluator({
            "evaluation_window_ms": 1000,
            "gain_threshold_abs": 0.01,
            "gain_threshold_rel": 0.0,
            "warm_cost": 2.0,
        }).evaluate({"prefix_states": rows}, [active],
                    [active, candidate], [decode], solver, 0)
        self.assertEqual(decision.action, "+P")
        self.assertEqual(decision.mode, "cold")
        self.assertIn(1, decision.wanted_ids)
        self.assertGreater(decision.gain, 0.01)

    def test_pair_network_cost_changes_assignment(self):
        rows = [{"class_id": "c", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 1.0, "request_count": 1,
                 "hit_tokens_ewma": 0.0, "requested_tokens": 64}]
        prefill = [_Scheduler(0, 0), _Scheduler(1, 1)]
        decode = [_Scheduler(2, 2), _Scheduler(3, 3)]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "pair_costs": {
                "0,2": {"rtt_ms": 1000},
                "0,3": {"rtt_ms": 1000},
                "1,2": {"rtt_ms": 1000},
                "1,3": {"rtt_ms": 0},
            },
        }))
        flows = solver.solve(rows, prefill, decode)
        self.assertEqual(len(flows), 1)
        self.assertEqual((flows[0].prefill_id, flows[0].decode_id), (1, 3))

    def test_prometheus_parser_handles_comments_labels_and_invalid_values(self):
        metrics = parse_prometheus_text("""
        # HELP queue queue depth
        queue{worker=\"p0\"} 4
        kv_read_bytes_total 1.5e3
        broken not-a-number
        """)
        self.assertEqual(metrics["queue"], 4.0)
        self.assertEqual(metrics["kv_read_bytes_total"], 1500.0)
        self.assertNotIn("broken", metrics)

    def test_external_queue_telemetry_is_used_by_pair_cost(self):
        decode = _Scheduler(2, 2)
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "queue_weight": 1.0,
        }))
        solver.set_telemetry({"workers": {"2": {"queue": 8}}})
        cost = solver._pair_cost(_Scheduler(0, 0), decode, "c", {})
        self.assertGreaterEqual(cost, 8.0 / decode.max_num_seqs)

    def test_external_link_telemetry_overrides_static_capacity(self):
        config = FlowSolverConfig.from_dict({
            "shared_links": [{"id": "wan", "capacity": 100,
                              "capacity_bytes_per_s": 1000}],
        })
        solver = CapacityAwareFlowSolver(config)
        solver.set_telemetry({"links": {"wan": {"bandwidth_bytes_per_s": 250}}})
        self.assertEqual(solver._link_capacity(config.shared_links[0]), 250.0)
        self.assertTrue(solver._link_uses_bytes(config.shared_links[0]))

    def test_cache_capacity_ablation_forces_full_prefill_work(self):
        rows = [{"class_id": "hot", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 1.0, "hit_tokens_ewma": 95,
                 "requested_tokens": 100}]
        scheduler = _Scheduler(0, 0)
        cache_solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({}))
        no_cache_solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "use_cache_capacity": False,
        }))
        _, cache_work = cache_solver._aggregate_rows(rows, [scheduler])
        _, no_cache_work = no_cache_solver._aggregate_rows(rows, [scheduler])
        self.assertAlmostEqual(cache_work[0, "hot"], 0.05)
        self.assertEqual(no_cache_work[0, "hot"], 1.0)

    def test_structural_gain_can_be_disabled(self):
        active = _Scheduler(0, 0)
        candidate = _Scheduler(1, 1)
        active.admission_state = "ACTIVE"
        candidate.admission_state = "INACTIVE"
        rows = [{"class_id": "hot", "prefill_instance_id": 0,
                 "arrival_rate_ewma": 4.0, "request_count": 4,
                 "hit_tokens_ewma": 0.0, "requested_tokens": 64}]
        solver = CapacityAwareFlowSolver(FlowSolverConfig.from_dict({
            "prefill_capacity": {"0": 1, "1": 4},
        }))
        decision = StructuralEvaluator({"enabled": False}).evaluate(
            {"prefix_states": rows}, [active], [active, candidate],
            [_Scheduler(2, 2)], solver, 0)
        self.assertEqual(decision.action, "keep")
        self.assertIn("disabled", decision.reason)

    def test_reconfig_executor_formats_scale_command_without_shell(self):
        decision = type("Decision", (), {
            "action": "+P", "mode": "warm", "wanted_ids": (0, 1),
        })()
        executor = ReconfigExecutor({
            "backend": "command",
            "scale_out": ["workerctl", "add", "{instance_id}", "{mode}"],
        })
        result = executor.apply(decision, [0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].instance_id, 1)
        self.assertFalse(result[0].ok)
        self.assertEqual(executor.apply(decision, [0], blocked_ids=[1]), ())


if __name__ == "__main__":
    unittest.main()

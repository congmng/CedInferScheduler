import unittest

from serving.casr.affinity import AffinityPlan
from serving.casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
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

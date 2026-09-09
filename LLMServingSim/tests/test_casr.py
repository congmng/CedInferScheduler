import unittest

from serving.casr.affinity import AffinityPlan
from serving.casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
from serving.casr.prefix_profiler import PrefixProfiler
from serving.casr.resources import ResourceOrchestrator


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

    def test_resource_pool_matches_per_gpu_memory(self):
        pool = ResourceOrchestrator({
            "nodes": {"0": {"gpu_count": 2, "gpu_mem_gb": [24, 48]}},
        })
        large = _ResourceScheduler(0)
        large.memory = _Memory(96)
        events = pool.bootstrap([large], 0)
        self.assertEqual(large.admission_state, "INACTIVE")
        self.assertEqual(events[0].action, "resource_reject")


if __name__ == "__main__":
    unittest.main()

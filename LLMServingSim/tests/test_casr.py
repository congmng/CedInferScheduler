import unittest

from serving.casr.affinity import AffinityPlan
from serving.casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
from serving.casr.prefix_profiler import PrefixProfiler


class _Scheduler:
    def __init__(self, instance_id, start_npu):
        self.instance_id = instance_id
        self.start_npu = start_npu
        self.max_num_seqs = 8


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


if __name__ == "__main__":
    unittest.main()

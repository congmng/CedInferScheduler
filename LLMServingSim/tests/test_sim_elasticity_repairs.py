"""Regression tests for the 2026-09-15 simulator elasticity repairs.

Five defects were found while trying to reproduce the real cluster's structural
elasticity in the simulator, and each one had the same shape: a decision was
made correctly and then thrown away by the layer that consumed it.

1. ``PrefillLifecycle`` sized the pool from declared requests/s only, so a
   KV-egress-bound run looked like "one worker is enough".
2. An evaluator decision was recomputed away by the next demand tick instead of
   being held for at least the container boot.
3. The router placed classes the plan could not name by least-loaded, ignoring
   the plan's intended *rate* split -- so a freshly scaled-out worker stayed
   idle (measured: run 0 / wait 0 while two others queued 82 and 74).
4. A Decode whose KV pool filled made the scheduler raise, killing long-prompt
   runs at the first backlog instead of applying back-pressure.
5. The KV pool had no calibration knob, so a card's pool could not be scaled
   without editing the memory model.
"""

import collections
import unittest

from serving.core.memory_model import MemoryModel
from serving.core.router import Router
from serving.core.scheduler import Scheduler
from serving.casr.affinity import AffinityPlan
from serving.casr.controller import CASRController
from serving.casr.evaluator import StructuralEvaluator
from serving.casr.flow_solver import FlowAssignment
from serving.casr.lifecycle import PrefillLifecycle


class _Memory:
    """Minimal memory surface the resource orchestrator reads (bytes)."""

    def __init__(self, gb=24):
        self.npu_mem = gb * 1024 ** 3


class _Sched:
    def __init__(self, instance_id, state="ACTIVE", pd_type="prefill", node_id=0):
        self.instance_id = instance_id
        self.pd_type = pd_type
        self.max_num_seqs = 16
        self.admission_state = state
        self.running = []
        self.waiting = []
        self.node_id = node_id
        self.num_npus = 1
        self.memory = _Memory()
        self.instance_id = instance_id

    def set_admission_state(self, state):
        self.admission_state = str(state).upper()

    @property
    def accepts_new_requests(self):
        return self.admission_state == "ACTIVE"


class LifecycleCapacityTests(unittest.TestCase):
    def test_egress_budget_binds_before_compute_for_long_prompts(self):
        life = PrefillLifecycle({
            "min_active_prefill": 1, "max_active_prefill": 4,
            "prefill_capacity": {0: 63},
            "prefill_tokens_per_s": {0: 14000},
            "kv_egress_gbps": 0.26,
        })
        rows = [{"requested_tokens_ewma": 1250.0, "kv_bytes_per_request": 0.0,
                 "arrival_rate_ewma": 1.0}]
        # compute limit 14000/1250 = 11.2 req/s; egress limit 0.26 GB/s / 184 MB
        # = 1.41 req/s -- the wire-adjacent producer is the binding one.
        self.assertAlmostEqual(life._effective_capacity(_Sched(0), rows), 1.41,
                               places=1)

    def test_short_prompts_are_not_limited_by_the_egress_budget(self):
        life = PrefillLifecycle({
            "min_active_prefill": 1, "max_active_prefill": 4,
            "prefill_capacity": {0: 63},
            "prefill_tokens_per_s": {0: 14000},
            "kv_egress_gbps": 2.0,
        })
        rows = [{"requested_tokens_ewma": 254.0, "kv_bytes_per_request": 0.0}]
        # 2 GB/s / 37 MB = 53 req/s, above the 63-req/s declared ceiling's
        # length-scaled 55 req/s, so compute stays in charge.
        self.assertGreater(life._effective_capacity(_Sched(0), rows), 50.0)

    def test_an_evaluator_decision_survives_the_next_demand_tick(self):
        life = PrefillLifecycle({
            "min_active_prefill": 1, "max_active_prefill": 4,
            "warmup_ms": 45000, "override_hold_ms": 1000,
            "prefill_capacity": {0: 63, 1: 63},
            "resources": {"startup_ms": 45000,
                          "nodes": {"0": {"gpu_count": 2, "gpu_mem_gb": [24, 24]}}},
        })
        prefills = [_Sched(0), _Sched(1, state="INACTIVE")]
        rows = [{"requested_tokens_ewma": 1250.0, "kv_bytes_per_request": 0.0,
                 "arrival_rate_ewma": 1.0}]
        life.update(10 ** 9, rows, prefills, wanted_override={0, 1})
        # The hold is the boot time (45 s), not the 1 s configured hold: a
        # worker that is still warming must not be drained by the next tick.
        life.update(2 * 10 ** 9, rows, prefills)          # +1 s, still inside
        self.assertEqual(life.last_wanted, {0, 1})


class RouterPlanAggregateTests(unittest.TestCase):
    def router(self):
        schedulers = [_Sched(0), _Sched(1), _Sched(10, pd_type="decode")]
        return Router(len(schedulers), schedulers, req_num=0,
                      routing_policy="LOAD", seed=1)

    def plan(self):
        return AffinityPlan(version=1, expires_at_ns=10 ** 18,
                            prefill_weights={"c": {0: 0.8, 1: 0.2}},
                            decode_weights={(0, "c"): {10: 1.0},
                                            (1, "c"): {10: 1.0}})

    def test_aggregate_uses_flow_rates_not_per_class_shares(self):
        """Per-class shares are single-homed; only the rates carry the intent.

        The measured bug: the plan asked for 1.81 of 8.86 req/s on the fresh
        worker, but summing per-class shares put ~2% there and the worker stayed
        idle for the whole run.
        """
        router = self.router()
        flows = [FlowAssignment("c", 0, 10, 7.05, 0.1),
                 FlowAssignment("c", 1, 10, 1.81, 0.1)]
        router.install_affinity_plan(self.plan(), flows)
        self.assertAlmostEqual(router._plan_prefill_totals[0], 7.05 / 8.86, places=3)
        self.assertAlmostEqual(router._plan_prefill_totals[1], 1.81 / 8.86, places=3)

    def test_unplanned_classes_follow_that_aggregate(self):
        router = self.router()
        flows = [FlowAssignment("c", 0, 10, 7.05, 0.1),
                 FlowAssignment("c", 1, 10, 1.81, 0.1)]
        router.install_affinity_plan(self.plan(), flows)
        candidates = router.prefill_schedulers
        picked = collections.Counter()
        for _ in range(200):
            sched = router._select_weighted(candidates,
                                            router._plan_prefill_totals,
                                            ("prefill_aggregate",))
            picked[sched.instance_id] += 1
        # ~20% must reach the fresh worker instead of 0%.
        self.assertGreater(picked[1], 20)
        self.assertLess(picked[1], 60)

    def test_the_split_survives_a_plan_reinstall(self):
        """The control loop re-installs the plan every tick; the split must hold.

        The measured bug (2026-09-23, 16 rps P-15B peak): installing a plan also
        reset the deficit counters, and at ~1.6 arrivals per 100 ms tick the
        balance never recovered -- a 91 / 9 plan dispatched 737 of 740 requests
        to the 91 % worker, which then queued ~250 requests while the other
        worker sat at zero.  Every pick here is preceded by an install, exactly
        as the event loop does it.
        """
        router = self.router()
        flows = [FlowAssignment("c", 0, 10, 9.1, 0.1),
                 FlowAssignment("c", 1, 10, 0.9, 0.1)]
        candidates = router.prefill_schedulers
        picked = collections.Counter()
        for _ in range(740):
            router.install_affinity_plan(self.plan(), flows)
            sched = router._select_weighted(candidates,
                                            router._plan_prefill_totals,
                                            ("prefill_aggregate",))
            picked[sched.instance_id] += 1
        share = picked[1] / sum(picked.values())
        # ~1/10 of the plan's flow, not zero: the band is 5 % of the score.
        self.assertGreater(share, 0.03)
        self.assertLess(share, 0.20)
        self.assertGreater(picked[0], picked[1])


class DecodeBackpressureTests(unittest.TestCase):
    class _KV:
        def __init__(self, admit):
            self.admit = admit

        def get_computed_blocks(self, req):
            return [], 0, 0

        def allocate_slots(self, *args, **kwargs):
            return [1] if self.admit else None

        def take_traffic(self):
            return None

    class _Memory:
        block_size = 16
        npu_pool = type("P", (), {"num_blocks": 10, "bytes_per_block": 2 ** 20})()

        def pd_kv_bytes(self, tokens):
            return float(tokens) * 1024.0

    def stub(self, admit):
        stub = type("Stub", (), {})()
        stub.instance_id = 7
        stub.kv = self._KV(admit)
        stub.memory = self._Memory()
        stub.pd_buffer_bytes = None
        stub.pd_staging = {}
        stub.running = []
        stub.backpressure_events = 0
        return stub

    def test_a_full_pool_refuses_instead_of_raising(self):
        request = type("Req", (), {"id": 1, "num_tokens_reached": 1251,
                                   "num_computed_tokens": 0,
                                   "original_input": 1251,
                                   "instance_id": 0, "status": None})()
        stub = self.stub(admit=False)
        self.assertFalse(Scheduler.add_decode(stub, request))
        self.assertEqual(stub.backpressure_events, 1)
        self.assertEqual(stub.running, [])

    def test_an_admitted_request_is_taken_normally(self):
        request = type("Req", (), {"id": 2, "num_tokens_reached": 1251,
                                   "num_computed_tokens": 0,
                                   "original_input": 1251,
                                   "instance_id": 0, "status": None})()
        stub = self.stub(admit=True)
        self.assertTrue(Scheduler.add_decode(stub, request))
        self.assertEqual(len(stub.running), 1)


class KvPoolCalibrationTests(unittest.TestCase):
    def blocks(self, kv_scale):
        model = MemoryModel("Qwen/Qwen3-8B", instance_id=0, node_id=0, num_npus=1,
                            tp_size=1, npu_mem=24, cpu_mem=256, block_size=16, fp=16,
                            enable_prefix_caching=False, enable_prefix_sharing=False,
                            prefix_pool=None, prefix_storage=None,
                            kv_cache_dtype="auto", npu_memory_utilization=0.9,
                            kv_scale=kv_scale)
        return model.npu_pool.num_blocks

    def test_scale_multiplies_the_kv_part_of_the_card(self):
        base = self.blocks(1.0)
        self.assertGreater(base, 1000)          # ~2887 on a 24 GB card
        self.assertGreater(self.blocks(1.5), base * 1.4)


class _FakeSolver:
    """Solver stand-in whose objective is a function of the active set."""

    def __init__(self, objective):
        self.diagnostics = {}
        self.config = type("C", (), {"prefill_capacity": {}, "prefill_service_ms": {}})()
        self._objective = objective

    def solve(self, rows, prefill, decode, overrides=None):
        key = tuple(sorted(s.instance_id for s in prefill))
        self.diagnostics = {"objective": float(self._objective(key))}


class StructuralPoolCeilingTests(unittest.TestCase):
    """``max_active_prefill`` has to bind every layer that can start a worker.

    Measured 2026-09-15: the elasticity replay's "static one worker" arm
    declared ``max_active_prefill=1`` and still ran three Prefills.  Two holes
    let that through -- the ceiling is written in ``casr.lifecycle`` while the
    structural evaluator read only ``casr.structural``, and the lifecycle's
    ``wanted_override`` path skipped the clamp its own demand heuristic
    applies.
    """

    def policy(self, cap):
        return {"lifecycle": {"min_active_prefill": 1, "max_active_prefill": cap},
                "structural": {"enabled": True, "evaluation_window_ms": 60000.0,
                               "startup_s": 45.0}}

    def test_the_controller_hands_the_lifecycle_ceiling_to_the_evaluator(self):
        controller = CASRController(100_000_000, policy=self.policy(1))
        self.assertEqual(controller.evaluator.max_active_prefill, 1)

    def test_an_explicit_structural_ceiling_wins_over_the_lifecycle_one(self):
        policy = self.policy(4)
        policy["structural"]["max_active_prefill"] = 1
        controller = CASRController(100_000_000, policy=policy)
        self.assertEqual(controller.evaluator.max_active_prefill, 1)

    def test_an_uncapped_pool_stays_uncapped(self):
        policy = self.policy(None)
        del policy["lifecycle"]["max_active_prefill"]
        controller = CASRController(100_000_000, policy=policy)
        self.assertIsNone(controller.evaluator.max_active_prefill)

    def test_the_override_path_cannot_exceed_the_ceiling(self):
        life = PrefillLifecycle({
            "min_active_prefill": 1, "max_active_prefill": 1,
            "resources": {"startup_ms": 45000,
                          "nodes": {"0": {"gpu_count": 2, "gpu_mem_gb": [24, 24]}}},
        })
        prefills = [_Sched(0), _Sched(1, state="INACTIVE"), _Sched(2, state="INACTIVE")]
        rows = [{"requested_tokens_ewma": 1250.0, "kv_bytes_per_request": 0.0,
                 "arrival_rate_ewma": 1.8}]
        life.update(10 ** 9, rows, prefills, wanted_override={0, 1, 2})
        self.assertEqual(life.last_wanted, {0})

    def test_a_warming_worker_counts_against_the_ceiling(self):
        evaluator = StructuralEvaluator({"enabled": True, "evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0, "max_active_prefill": 2})
        active = [_Sched(0)]
        warming = [_Sched(1, state="WARMING"), _Sched(2, state="INACTIVE")]
        rows = [{"class_id": "c", "arrival_rate_ewma": 1.0}]
        # Adding Prefill 2 is worth 10 units/s, but the booting Prefill 1
        # already occupies the second slot of the two-worker pool.
        decision = evaluator.evaluate({"prefix_states": rows}, active, active + warming,
                                      [_Sched(10, pd_type="decode")],
                                      _FakeSolver(lambda ids: 10.0 if len(ids) == 1 else 2.0),
                                      current_ns=10 ** 18, last_action_ns=-1, min_active=1)
        self.assertEqual(decision.action, "keep")
        self.assertIn("no eligible", decision.reason)

    def test_a_serving_only_pool_still_grows_below_the_ceiling(self):
        evaluator = StructuralEvaluator({"enabled": True, "evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0, "max_active_prefill": 2})
        active = [_Sched(0)]
        inactive = [_Sched(1, state="INACTIVE")]
        rows = [{"class_id": "c", "arrival_rate_ewma": 1.0}]
        decision = evaluator.evaluate({"prefix_states": rows}, active, active + inactive,
                                      [_Sched(10, pd_type="decode")],
                                      _FakeSolver(lambda ids: 10.0 if len(ids) == 1 else 2.0),
                                      current_ns=10 ** 18, last_action_ns=-1, min_active=1)
        self.assertEqual(decision.action, "+P")


if __name__ == "__main__":
    unittest.main()


class PredictiveScaleOutTests(unittest.TestCase):
    """Start a spare when the offer approaches the pool's *push* capacity.

    An evaluator that has to observe the gain cannot pay for a 45 s boot inside
    a 90 s peak: measured in the six-domain arena, the reactive ``+P`` arm
    settled at 14572 ms where an egress-sized pool reached 1120 ms.  The signal
    used here is the offer the profiler already reports against the capacity the
    solver prices (which the controller fills with the producer's 0.26 GB/s
    egress, ~1.4 req/s for a 1250-token prompt).
    """

    def _evaluator(self, **config):
        policy = {"enabled": True, "startup_s": 0.0, "max_active_prefill": 3}
        policy.update(config)
        return StructuralEvaluator(policy)

    def _pref(self, instance_id, capacity=0.0, active=True):
        class _Sched:
            pass
        sched = _Sched()
        sched.instance_id = instance_id
        sched.pd_type = "prefill"
        sched.admission_state = "ACTIVE" if active else "INACTIVE"
        sched.running = []
        sched.waiting = []
        sched.max_num_seqs = 16
        sched.memory = _Memory()
        return sched

    def test_an_offer_above_the_push_capacity_starts_a_spare(self):
        class _Solver:
            class config:
                prefill_capacity = {0: 1.4, 2: 1.4, 4: 1.4}
                prefill_service_ms = {}
        evaluator = self._evaluator()
        decision = evaluator._predictive_scale_decision(
            [{"arrival_rate_ewma": 2.0, "class_id": "c"}],
            [self._pref(0, 1.4)], [self._pref(0, 1.4), self._pref(2, 1.4, active=False)],
            _Solver(), [self._pref(1)], 0, 1.0, ())
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "+P")
        self.assertIn(2, decision.wanted_ids)

    def test_an_offer_inside_the_push_capacity_leaves_the_pool_alone(self):
        class _Solver:
            class config:
                prefill_capacity = {0: 2.8, 2: 2.8}
                prefill_service_ms = {}
        evaluator = self._evaluator()
        decision = evaluator._predictive_scale_decision(
            [{"arrival_rate_ewma": 1.0, "class_id": "c"}],
            [self._pref(0, 2.8)], [self._pref(0, 2.8), self._pref(2, 2.8, active=False)],
            _Solver(), [self._pref(1)], 0, 1.0, ())
        self.assertIsNone(decision)

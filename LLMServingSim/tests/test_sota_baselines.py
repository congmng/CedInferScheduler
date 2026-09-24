#!/usr/bin/env python3
"""The SOTA baseline arms run in the simulator.

``docs/SOTA覆盖与模拟器基线映射.md`` lists the mechanisms the comparison still
lacks.  Each one is added as a *runnable arm* with its own semantics, not as a
re-label of CASR, so the tests here pin the semantics that make it that
system's mechanism (a threshold rule for DOPD, a transfer-cost gate for
PrfaaS, an enumeration for DistServe's P:D ratio, migration for Llumnix).
"""

from __future__ import annotations

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class _Scheduler:
    """Duck type the controller/lifecycle pass to the structural rule."""

    def __init__(self, instance_id, role="prefill", *, capacity=10.0,
                 waiting=0, running=0, max_num_seqs=8, state="ACTIVE"):
        self.instance_id = instance_id
        self.pd_type = role
        self.capacity = capacity
        self.waiting = [None] * waiting
        self.running = [None] * running
        self.max_num_seqs = max_num_seqs
        self.admission_state = state
        self.service_ms = 100.0
        self.start_npu = 0


class ThresholdScalerTests(unittest.TestCase):
    """DOPD-style autoscaling: pool size follows utilisation, not a plan."""

    def _solver(self, capacities):
        from serving.casr.flow_solver import FlowSolverConfig

        class Solver:
            def __init__(self, config):
                self.config = config

        return Solver(FlowSolverConfig.from_dict({
            "prefill_capacity": {str(key): value
                                 for key, value in capacities.items()}}))

    def _rows(self, rate):
        return {"prefix_states": [{"class_id": "c", "prefill_instance_id": 0,
                                   "arrival_rate_ewma": rate,
                                   "requested_tokens": 1024,
                                   "hit_tokens_ewma": 0.0}]}

    def _scaler(self, **config):
        from serving.casr.autoscalers import ThresholdScaler
        return ThresholdScaler({"scale_ticks": 1, "enabled": True, **config})

    def test_scales_out_on_sustained_high_utilisation(self):
        active = [_Scheduler(0, capacity=10.0)]
        spare = _Scheduler(4, capacity=20.0, state="INACTIVE")
        solver = self._solver({0: 10.0, 4: 20.0})
        scaler = self._scaler(scale_ticks=3)
        decisions = [scaler.evaluate(self._rows(10.0), active, active + [spare],
                                     [_Scheduler(1, "decode")], solver,
                                     1_000_000 * tick) for tick in range(3)]
        self.assertEqual([item.action for item in decisions], ["keep", "keep", "+P"])
        # The spare that removes the most utilisation wins, not the first id.
        self.assertEqual(decisions[-1].wanted_ids, (0, 4))
        self.assertIn("utilisation 1.00", decisions[-1].reason)

    def test_a_pool_inside_the_band_is_left_alone(self):
        active = [_Scheduler(0, capacity=10.0)]
        spare = _Scheduler(4, capacity=20.0, state="INACTIVE")
        solver = self._solver({0: 10.0, 4: 20.0})
        scaler = self._scaler()
        decision = scaler.evaluate(self._rows(5.0), active, active + [spare],
                                   [_Scheduler(1, "decode")], solver, 0)
        self.assertEqual(decision.action, "keep")
        self.assertIn("hysteresis band", decision.reason)
        self.assertAlmostEqual(scaler.last_metrics["utilization"], 0.5)

    def test_the_queue_depth_can_trigger_scale_out(self):
        active = [_Scheduler(0, capacity=100.0, waiting=5, running=4)]
        spare = _Scheduler(4, capacity=20.0, state="INACTIVE")
        solver = self._solver({0: 100.0, 4: 20.0})
        scaler = self._scaler()
        decision = scaler.evaluate(self._rows(1.0), active, active + [spare],
                                   [_Scheduler(1, "decode")], solver, 0)
        self.assertEqual(decision.action, "+P")
        self.assertIn("queue 3.00", decision.reason)

    def test_a_backlog_counts_as_demand(self):
        # The observed arrival rate collapses under back-pressure, so a rule
        # that ignored the controller's backlog estimate would see an idle pool.
        active = [_Scheduler(0, capacity=10.0)]
        spare = _Scheduler(4, capacity=20.0, state="INACTIVE")
        solver = self._solver({0: 10.0, 4: 20.0})
        scaler = self._scaler(scale_out_utilization=0.8)
        decision = scaler.evaluate(self._rows(1.0), active, active + [spare],
                                   [_Scheduler(1, "decode")], solver, 0,
                                   backlog_rps=9.0)
        self.assertEqual(decision.action, "+P")

    def test_it_will_not_drain_a_worker_that_has_work(self):
        active = [_Scheduler(0, capacity=10.0),
                  _Scheduler(4, capacity=20.0, waiting=1)]
        solver = self._solver({0: 10.0, 4: 20.0})
        scaler = self._scaler(scale_in_utilization=0.9)
        decision = scaler.evaluate(self._rows(0.5), active, active,
                                   [_Scheduler(1, "decode")], solver, 0,
                                   min_active=1)
        self.assertEqual(decision.action, "keep")

    def test_scale_in_takes_the_smallest_worker_and_respects_min_active(self):
        active = [_Scheduler(0, capacity=10.0), _Scheduler(4, capacity=20.0)]
        solver = self._solver({0: 10.0, 4: 20.0})
        scaler = self._scaler(scale_in_utilization=0.5)
        decision = scaler.evaluate(self._rows(1.0), active, active,
                                   [_Scheduler(1, "decode")], solver, 0,
                                   min_active=1)
        self.assertEqual(decision.action, "-P")
        self.assertEqual(decision.wanted_ids, (4,))
        # At the floor the rule stops, even with the same cold reading.
        scaler = self._scaler(scale_in_utilization=0.5)
        single = [_Scheduler(0, capacity=10.0)]
        decision = scaler.evaluate(self._rows(1.0), single, single,
                                   [_Scheduler(1, "decode")],
                                   self._solver({0: 10.0}), 0, min_active=1)
        self.assertEqual(decision.action, "keep")
        self.assertIn("min_active_prefill", decision.reason)

    def test_max_active_prefill_caps_scale_out(self):
        active = [_Scheduler(0, capacity=10.0)]
        warming = _Scheduler(4, capacity=20.0, state="WARMING")
        spare = _Scheduler(2, capacity=15.0, state="INACTIVE")
        solver = self._solver({0: 10.0, 4: 20.0, 2: 15.0})
        scaler = self._scaler(max_active_prefill=2)
        decision = scaler.evaluate(self._rows(30.0), active,
                                   [active[0], warming, spare],
                                   [_Scheduler(1, "decode")], solver, 0)
        self.assertEqual(decision.action, "keep")
        self.assertIn("max_active_prefill", decision.reason)

    def test_a_fresh_scale_out_is_held_before_it_can_be_undone(self):
        active = [_Scheduler(0, capacity=10.0)]
        spare = _Scheduler(4, capacity=20.0, state="INACTIVE")
        solver = self._solver({0: 10.0, 4: 20.0})
        scaler = self._scaler(scale_in_utilization=0.9, threshold_hold_ms=10000)
        grown = scaler.evaluate(self._rows(30.0), active, active + [spare],
                                [_Scheduler(1, "decode")], solver, 0)
        self.assertEqual(grown.action, "+P")   # this arms the hold window
        both = active + [spare]
        decision = scaler.evaluate(self._rows(0.5), both, both,
                                   [_Scheduler(1, "decode")], solver,
                                   1_000_000, min_active=1)
        self.assertEqual(decision.action, "keep")
        self.assertIn("hold", decision.reason)

    def test_the_rule_is_selected_by_config(self):
        from serving.casr.autoscalers import (ThresholdScaler,
                                              build_structural_evaluator)
        from serving.casr.evaluator import StructuralEvaluator
        self.assertIsInstance(build_structural_evaluator({}), StructuralEvaluator)
        self.assertIsInstance(build_structural_evaluator({"rule": "dopd"}),
                              ThresholdScaler)
        self.assertIsInstance(build_structural_evaluator({"rule": "threshold"}),
                              ThresholdScaler)
        with self.assertRaises(ValueError):
            build_structural_evaluator({"rule": "nope"})


class PrfaasOffloadTests(unittest.TestCase):
    """Cross-DC Prefill offload: local first, then a hard transfer budget."""

    class _Node:
        """Scheduler duck type the router scores (see ``_instance_load_score``)."""

        def __init__(self, instance_id, role, node_id, capacity=100.0):
            self.instance_id = instance_id
            self.pd_type = role
            self.node_id = node_id
            self.max_num_seqs = 64
            self.waiting = []
            self.running = []
            self.capacity = capacity
            self.model = "P-15B"
            self.accepts_new_requests = True

    def _router(self, **options):
        from serving.core.router import Router
        prefills = [self._Node(0, "prefill", 0), self._Node(2, "prefill", 1)]
        decodes = [self._Node(1, "decode", 0)]
        config = {
            "prefill_capacity": {"0": 100.0, "2": 100.0},
            "decode_capacity": {"1": 100.0},
            "kv_bytes_per_token": 16960.0,
            # 1250-token request => 21.2 MB of KV.
            "pair_costs": {
                "0,1": {"rtt_ms": 0.0, "bandwidth_bytes_per_s": 257e6},
                "2,1": {"rtt_ms": 48.0, "bandwidth_bytes_per_s": 110e6},
            },
            "prfaas_local_load_threshold": 0.5,
            "prfaas_max_offload_ms": 300.0,
            **options,
        }
        router = Router(3, prefills + decodes, 0, "PRFAAS",
                        policy_options=config)
        router.prefill_service_ms = {0: 825.0, 2: 449.0}
        return router, prefills, decodes

    @staticmethod
    def _request(tokens=1250):
        return {"input_hash_ids": list(range(tokens)),
                "input_tok_ids": list(range(tokens))}

    def test_the_local_dc_is_preferred_while_it_has_room(self):
        router, prefills, _ = self._router()
        index = router._prfaas_select(prefills, "prefill", self._request())
        self.assertEqual(prefills[index].instance_id, 0)
        self.assertEqual(router._counters["prfaas_local"], 1)

    def test_a_loaded_local_dc_offloads_under_the_budget(self):
        router, prefills, _ = self._router()
        # The local producer is at 61% of its slots once this request is added.
        router._assigned[0] = 60
        index = router._prfaas_select(prefills, "prefill", self._request())
        self.assertEqual(prefills[index].instance_id, 2)
        self.assertEqual(router._counters["prfaas_offload"], 1)
        self.assertEqual(router._counters.get("prfaas_local", 0), 0)

    def test_the_transfer_budget_refuses_an_expensive_offload(self):
        router, prefills, _ = self._router(prfaas_max_offload_ms=100.0)
        router._assigned[0] = 60
        index = router._prfaas_select(prefills, "prefill", self._request())
        # 1250 tokens x 16960 B over 110 MB/s + 48 ms = 241 ms > 100 ms, so the
        # remote producer is refused and the request stays in its own DC.
        self.assertEqual(prefills[index].instance_id, 0)
        self.assertEqual(router._counters["prfaas_gate_blocked"], 1)
        self.assertEqual(router._counters.get("prfaas_offload", 0), 0)

    def test_a_zero_budget_disables_the_gate(self):
        router, prefills, _ = self._router(prfaas_max_offload_ms=0.0)
        router._assigned[0] = 60
        index = router._prfaas_select(prefills, "prefill", self._request())
        self.assertEqual(prefills[index].instance_id, 2)

    def test_without_a_decode_target_it_falls_back_to_load(self):
        router, prefills, _ = self._router()
        router.decode_schedulers = []
        index = router._prfaas_select(prefills, "prefill", self._request())
        self.assertEqual(prefills[index].instance_id, 0)


class BootModelTests(unittest.TestCase):
    """ServerlessLLM-style loading: startup derived from what it loads."""

    def test_the_model_is_off_unless_asked_for(self):
        from serving.core.boot_model import resolve_startup_ms
        self.assertEqual(resolve_startup_ms({}, 20000.0), 20000.0)
        self.assertEqual(resolve_startup_ms(None, 45000.0), 45000.0)

    def test_the_measured_warm_restart_is_reproduced(self):
        # 15.26 GiB at the deployment's measured 780 MB/s is the 21.7 s cold
        # weight load; engine init (17.2 s) plus the *warm* 2.8 s page-cache
        # read is the 20 s median the 46-restart sweep measured.
        from serving.core.boot_model import (P15B_WEIGHTS_BYTES,
                                             resolve_startup_ms)
        cold = resolve_startup_ms({"model": "weight_load",
                                   "load_bandwidth_bytes_per_s": 780e6,
                                   "overlap": False}, 0.0)
        self.assertGreater(cold, 37000.0)
        self.assertLess(cold, 40000.0)
        warm_bandwidth = P15B_WEIGHTS_BYTES / 2.8      # the measured warm read
        warm = resolve_startup_ms({"model": "weight_load",
                                   "load_bandwidth_bytes_per_s": warm_bandwidth,
                                   "overlap": False}, 0.0)
        self.assertAlmostEqual(warm, 20000.0, delta=300.0)

    def test_a_faster_loader_is_engine_bound_when_it_overlaps(self):
        from serving.core.boot_model import resolve_startup_ms
        fast = resolve_startup_ms({"model": "weight_load",
                                   "load_bandwidth_bytes_per_s": 3.0e9,
                                   "overlap": True}, 45000.0)
        self.assertAlmostEqual(fast, 17200.0, places=3)
        serial = resolve_startup_ms({"model": "weight_load",
                                     "load_bandwidth_bytes_per_s": 3.0e9,
                                     "overlap": False}, 0.0)
        self.assertGreater(serial, fast)
        self.assertLess(serial, 24000.0)


class LlumnixMigrationTests(unittest.TestCase):
    """Llumnix-style migration moves queued Decode work and charges the copy."""

    def _router(self, **options):
        from serving.core.router import Router
        prefills = [PrfaasOffloadTests._Node(0, "prefill", 0),
                    PrfaasOffloadTests._Node(2, "prefill", 1)]
        decodes = [PrfaasOffloadTests._Node(1, "decode", 0),
                   PrfaasOffloadTests._Node(3, "decode", 1)]
        config = {
            "prefill_capacity": {"0": 100.0, "2": 100.0},
            "decode_capacity": {"1": 10.0, "3": 10.0},
            "kv_bytes_per_token": 16960.0,
            "pair_costs": {
                "0,1": {"rtt_ms": 0.0, "bandwidth_bytes_per_s": 257e6},
                "0,3": {"rtt_ms": 48.0, "bandwidth_bytes_per_s": 110e6},
                "2,1": {"rtt_ms": 48.0, "bandwidth_bytes_per_s": 110e6},
                "2,3": {"rtt_ms": 0.0, "bandwidth_bytes_per_s": 257e6},
            },
            "llumnix_interval_ms": 1000.0,
            "llumnix_batch": 4,
            **options,
        }
        router = Router(4, prefills + decodes, 0, "LOAD", policy_options=config)
        return router, prefills, decodes

    @staticmethod
    def _request(index, tokens=1250, instance_id=1):
        from serving.core.request import Request
        return Request(index, "P-15B", tokens, tokens + 16,
                       arrival=index * 1_000_000, instance_id=instance_id)

    def test_it_moves_queued_requests_off_a_loaded_decode(self):
        from serving.core.migration import LlumnixMigrator
        router, prefills, decodes = self._router()
        hot, cold = decodes
        hot.waiting = [self._request(index) for index in range(6)]
        router._assigned[1] = 20          # 2.1 slots' worth of work
        migrator = LlumnixMigrator(router.policy_options)
        moved = migrator.migrate(router, 1_000_000)
        self.assertEqual(moved, 4)        # the per-tick batch
        self.assertEqual(len(hot.waiting), 2)
        self.assertEqual(len(cold.waiting), 4)
        # The newest arrivals leave first, so the head of the queue is intact.
        self.assertEqual([req.id for req in hot.waiting], [0, 1])
        # 1250 tokens x 16960 B over 110 MB/s + 48 ms = 240.7 ms, charged once.
        self.assertAlmostEqual(cold.waiting[0].migration_ns / 1e6, 240.7,
                               places=1)
        self.assertEqual(cold.waiting[0].decode_instance_id, 3)
        self.assertEqual(router._counters["llumnix_migrated"], 4)

    def test_a_balanced_pool_is_left_alone(self):
        from serving.core.migration import LlumnixMigrator
        router, _, decodes = self._router()
        decodes[0].waiting = [self._request(0)]
        migrator = LlumnixMigrator(router.policy_options)
        self.assertEqual(migrator.migrate(router, 0), 0)
        self.assertIn("no eligible pair", migrator.last_reason)

    def test_a_running_request_is_not_moved(self):
        from serving.core.migration import LlumnixMigrator
        router, _, decodes = self._router()
        hot = decodes[0]
        running = self._request(99)
        running.status = "RUNNING"
        hot.running = [running]
        router._assigned[1] = 20
        migrator = LlumnixMigrator(router.policy_options)
        self.assertEqual(migrator.migrate(router, 0), 0)
        self.assertEqual(hot.running, [running])
        self.assertEqual(hot.running[0].migration_ns, 0)

    def test_a_refused_handoff_is_re_targeted(self):
        from serving.core.migration import LlumnixMigrator
        router, _, decodes = self._router()
        pending = self._request(7)
        pending.decode_instance_id = 1
        router._pending_handoffs = [pending]
        router._assigned[1] = 20
        migrator = LlumnixMigrator(router.policy_options)
        self.assertEqual(migrator.migrate(router, 0), 1)
        self.assertEqual(pending.decode_instance_id, 3)
        self.assertGreater(pending.migration_ns, 0)

    def test_the_copy_cost_uses_the_pair_tables(self):
        router, prefills, decodes = self._router()
        # Prefill 0 and Decode 3 are on different nodes: 110 MB/s + 48 ms.
        cost = router.node_link_cost_ms(decodes[0], decodes[1], 1250)
        self.assertAlmostEqual(cost, 240.7, places=1)
        same = router.node_link_cost_ms(decodes[1], decodes[1], 1250)
        # Prefill 2 and Decode 3 share a node: 257 MB/s and no hop.
        self.assertAlmostEqual(same, 82.5, places=1)

    def test_the_charged_copy_reaches_the_measured_latency(self):
        from serving.core.request import Request
        req = Request(1, "P-15B", 1250, 1266, arrival=0, instance_id=1)
        req.set_ttft(100_000_000)
        self.assertEqual(req.ttft, 100_000_000)
        req.migration_ns = 5_000_000
        req.set_ttft(100_000_000)
        req.add_latency(1_000_000_000)
        self.assertEqual(req.ttft, 105_000_000)
        self.assertEqual(req.latency, 1_005_000_000)
        # The per-token rate is unchanged: a copy delays the first token, it
        # does not slow the Decode down.
        self.assertEqual(req.tpot, (1_005_000_000 - 105_000_000) // 15)

    def test_an_interval_of_zero_disables_the_arm(self):
        from serving.core.migration import LlumnixMigrator
        migrator = LlumnixMigrator({"llumnix_interval_ms": 0.0})
        self.assertFalse(migrator.enabled)
        self.assertFalse(migrator.due(10_000_000_000))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the 2026-09-14 modelling fixes.

Three defects were measured on the real fabric and are pinned here:

1. the LP scaled nothing by prompt/output *length*, so ``prefill_overflow``
   stayed 0 while a Prefill's queue was 48 deep;
2. the router's ``chars_per_token`` was a constant that under-stated every
   token count (and therefore every KV byte the link budget is denominated in);
3. the structural evaluator had no economics: ``startup_cost = 0`` with a 1 s
   horizon meant "scale out whenever a stopped spare exists", and no holding
   cost meant ``-P`` could never pay for itself.
"""

import json
import os
import pathlib
import sys
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO / "serving", REPO / "deploy" / "real_lmcache_pd"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import casr_control                                          # noqa: E402
from casr.evaluator import StructuralEvaluator               # noqa: E402
from casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig  # noqa: E402
from disagg_router import Router                             # noqa: E402


class Sched:
    """Duck type the solver reads (mirrors ``casr_control.RealInstance``)."""

    def __init__(self, instance_id, role, *, max_num_seqs=16, capacity=1.0,
                 service_ms=0.0, admission_state="ACTIVE", enabled=True):
        self.instance_id = instance_id
        self.pd_type = role
        self.max_num_seqs = max_num_seqs
        self.running = []
        self.waiting = []
        self.capacity = capacity
        self.solver_capacity = capacity
        self.service_ms = service_ms
        self.start_npu = 0
        self.admission_state = admission_state
        self.enabled = enabled

    @property
    def accepts_new_requests(self):
        return self.admission_state == "ACTIVE"


def row(class_id, rate, *, tokens, prefill_id=0, kv_per_token=147456):
    return {"class_id": class_id, "arrival_rate_ewma": rate,
            "prefill_instance_id": prefill_id, "hit_tokens_ewma": 0.0,
            "requested_tokens": tokens, "requested_tokens_ewma": tokens,
            "kv_bytes_per_request": tokens * kv_per_token}


class PromptLengthWorkTests(unittest.TestCase):
    """``prefill_capacity`` is a reference-length number; length must scale it."""

    def solve(self, tokens, rate):
        config = FlowSolverConfig.from_dict({
            "solver": "lp",
            "prefill_capacity": {"0": 63.0},
            "decode_capacity": {"11": 1000.0},
            # Measured 2026-09-14: p5090 sustains 14.1 ktok/s on unique
            # prefixes, i.e. 11.3 req/s at 1250 tokens.
            "prefill_tokens_per_s": {"0": 14000.0},
            "overflow_penalty": 10.0,
        })
        solver = CapacityAwareFlowSolver(config)
        prefill = [Sched(0, "prefill", capacity=63.0, service_ms=48.5)]
        decode = [Sched(11, "decode", capacity=1000.0, service_ms=144.7)]
        solver.solve([row("c", rate, tokens=tokens)], prefill, decode)
        return solver.diagnostics

    def test_long_prompt_consumes_proportionally_more_capacity(self):
        diag = self.solve(1250, rate=1.0)
        work = diag["work"]["c"][0]
        self.assertAlmostEqual(work, 1250 * 63.0 / 14000.0, places=3)

    def test_prompt_below_the_ceiling_pays_one_unit(self):
        # 180 tokens is below the knee (14000 / 63 = 222 tokens), so the
        # declared capacity is the binding constraint and the factor is 1.
        diag = self.solve(180, rate=1.0)
        self.assertEqual(diag["work"]["c"][0], 1.0)

    def test_work_increases_with_prompt_length(self):
        short = self.solve(254, rate=1.0)["work"]["c"][0]
        long = self.solve(1250, rate=1.0)["work"]["c"][0]
        self.assertGreater(long, short)

    def test_overflow_appears_only_for_the_length_that_exceeds_capacity(self):
        # 20 req/s (the offered rate) against a nominal 63 req/s Prefill:
        # 1250-token requests cannot be served, 254-token ones can.
        long_diag = self.solve(1250, rate=20.0)
        short_diag = self.solve(254, rate=20.0)
        self.assertGreater(long_diag["prefill_overflow"][0], 0.0)
        self.assertEqual(short_diag["prefill_overflow"][0], 0.0)


class OutputLengthWorkTests(unittest.TestCase):
    """The Decode leg scales with the class's output-token bucket."""

    def solve(self, class_id, rate):
        config = FlowSolverConfig.from_dict({
            "solver": "lp",
            "prefill_capacity": {"0": 1000.0},
            "decode_capacity": {"11": 4.0},
            "decode_reference_tokens": 16,
            "decode_fixed_ms": 20.0,
            "decode_ms_per_1k_tokens": 14000.0,
        })
        solver = CapacityAwareFlowSolver(config)
        prefill = [Sched(0, "prefill", capacity=1000.0)]
        decode = [Sched(11, "decode", capacity=4.0)]
        solver.solve([row(class_id, rate, tokens=200)], prefill, decode)
        return solver.diagnostics

    def test_double_length_output_costs_about_double(self):
        # 3 req/s of a 16-token class fits a 4 req/s Decode; the same rate of a
        # 32-token class needs ~1.9x the occupancy and must overflow.
        short = self.solve("m|p|in:128-255|out:16-31", rate=3.0)
        long = self.solve("m|p|in:128-255|out:32-63", rate=3.0)
        self.assertEqual(short["work"]["m|p|in:128-255|out:16-31"][0], 1.0)
        self.assertEqual(short["decode_overflow"][11], 0.0)
        self.assertGreater(long["decode_overflow"][11], 0.0)


class ConfigParsingTests(unittest.TestCase):
    def test_documentation_keys_are_ignored(self):
        config = FlowSolverConfig.from_dict({
            "prefill_capacity": {"0": 63.0, "_comment": "measured"},
            "prefill_tokens_per_s": {"0": 14000.0, "_note": "tokens/s"},
            "decode_capacity": {"11": 65.0, "_comment": 1.0},
        })
        self.assertEqual(config.prefill_capacity, {0: 63.0})
        self.assertEqual(config.prefill_tokens_per_s, {0: 14000.0})
        self.assertEqual(config.decode_capacity, {11: 65.0})


class BacklogDemandTests(unittest.TestCase):
    """Post-backpressure arrivals must be corrected by the queue growth."""

    def controller(self):
        config = {
            "hosts": {},
            "prefills": [{"id": "p0", "host": "127.0.0.1", "domain": "a",
                          "port": 8100, "instance_id": 0, "solver_capacity": 10,
                          "max_num_seqs": 16, "service_ms": 50.0}],
            "decodes": [{"id": "d0", "host": "127.0.0.1", "domain": "a",
                         "port": 8200, "instance_id": 11, "solver_capacity": 10,
                         "max_num_seqs": 16, "service_ms": 150.0}],
            "links": [],
        }
        return casr_control.build_controller(
            config, {"solver": "lp", "model_name": "m", "container_prefix": "t-"})

    def test_queue_growth_is_added_to_every_class_proportionally(self):
        controller = self.controller()
        prefill = controller.prefill_by_id(0)
        base = 10 ** 18
        controller.sample_arrivals(base)
        controller.observe("c1", prefill, 200, 200 * 147456, base)
        controller.observe("c2", prefill, 200, 200 * 147456, base)
        controller.sample_arrivals(base + 10 ** 9)
        baseline = {(r["class_id"]): r["arrival_rate_ewma"] for r in controller.rows()}
        self.assertGreater(baseline["c1"], 0.0)

        # The queue grows by 4 requests over the next second: the offered rate
        # is higher than the served rate the EWMA saw, and the LP must be told.
        prefill.waiting = [None] * 4
        controller.sample_arrivals(base + 2 * 10 ** 9)
        inflated = {(r["class_id"]): r["arrival_rate_ewma"] for r in controller.rows()}
        self.assertGreater(inflated["c1"], baseline["c1"])
        self.assertGreater(inflated["c2"], baseline["c2"])
        self.assertGreater(controller.as_dict()["demand"]["backlog_rps"], 0.0)

    def test_a_draining_queue_does_not_subtract_demand(self):
        controller = self.controller()
        prefill = controller.prefill_by_id(0)
        base = 10 ** 18
        prefill.waiting = [None] * 8
        controller.sample_arrivals(base)
        controller.observe("c1", prefill, 200, 200 * 147456, base)
        controller.sample_arrivals(base + 10 ** 9)
        prefill.waiting = []
        controller.sample_arrivals(base + 2 * 10 ** 9)
        self.assertGreaterEqual(controller.as_dict()["demand"]["backlog_rps"], 0.0)


class _FakeSolver:
    """Solver stand-in whose objective is a function of the active Prefill set."""

    def __init__(self, objective):
        self.diagnostics = {}
        self.config = type("C", (), {"prefill_capacity": {}, "prefill_service_ms": {}})()
        self._objective = objective

    def solve(self, rows, prefill, decode, overrides=None):
        key = tuple(sorted(s.instance_id for s in prefill))
        self.diagnostics = {"objective": float(self._objective(key))}


class StructuralEconomicsTests(unittest.TestCase):
    """A structural edit must repay its start-up inside the evaluation horizon."""

    def evaluate(self, objective, *, active_ids=(0,), inactive_ids=(1,),
                 config=None, min_active=1):
        structurally = {"enabled": True, "gain_threshold_abs": 0.001,
                        "gain_threshold_rel": 0.01}
        structurally.update(config or {})
        evaluator = StructuralEvaluator(structurally)
        active = [Sched(i, "prefill") for i in active_ids]
        inactive = [Sched(i, "prefill", admission_state="INACTIVE")
                    for i in inactive_ids]
        rows = [{"class_id": "c", "arrival_rate_ewma": 1.0}]
        decode = [Sched(11, "decode")]
        return evaluator.evaluate({"prefix_states": rows}, active,
                                  active + inactive, decode,
                                  _FakeSolver(objective), current_ns=10 ** 18,
                                  last_action_ns=-1, min_active=min_active)

    def test_an_idle_window_cannot_amortise_a_container_boot(self):
        # 1 s horizon (the old default) and a 45 s boot: nothing can be repaid.
        decision = self.evaluate(lambda ids: 0.04 if len(ids) == 1 else 0.0,
                                 config={"evaluation_window_ms": 1000.0,
                                         "startup_s": 45.0})
        self.assertEqual(decision.action, "keep")

    def test_a_sustained_overload_pays_for_the_boot(self):
        # 8 objective units per second of relief for 60 s, minus 45 s of boot.
        decision = self.evaluate(lambda ids: 10.0 if len(ids) == 1 else 2.0,
                                 config={"evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0})
        self.assertEqual(decision.action, "+P")
        self.assertAlmostEqual(decision.gain, 15.0 * 8.0, places=6)

    def test_a_boot_longer_than_the_horizon_is_never_worth_starting(self):
        decision = self.evaluate(lambda ids: 10.0 if len(ids) == 1 else 2.0,
                                 config={"evaluation_window_ms": 30000.0,
                                         "startup_s": 45.0})
        self.assertEqual(decision.action, "keep")

    def test_a_holding_cost_lets_scale_in_pay_off(self):
        # Removing an almost-idle Prefill costs 0.1 objective units per second
        # of congestion but saves the holding cost of the reserved instance.
        decision = self.evaluate(lambda ids: 0.1 if len(ids) == 1 else 0.0,
                                 active_ids=(0, 1), inactive_ids=(),
                                 min_active=1,
                                 config={"evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0,
                                         "holding_cost": 0.5})
        self.assertEqual(decision.action, "-P")
        self.assertIn("remove Prefill", decision.reason)

    def test_without_a_holding_cost_scale_in_stays_negative(self):
        decision = self.evaluate(lambda ids: 0.1 if len(ids) == 1 else 0.0,
                                 active_ids=(0, 1), inactive_ids=(),
                                 min_active=1,
                                 config={"evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0})
        self.assertEqual(decision.action, "keep")

    def test_a_busy_pool_is_never_shrunk(self):
        """No removal while the pool has work, and none for a busy candidate.

        The counterfactual prices the *pool*, not the queue sitting on one
        instance, so a busy worker looks exactly like an idle one.  Measured
        2026-09-16 on the small cluster: a ``-P`` stopped the Prefill that had
        served 291 of 300 requests and the pool collapsed; even a light load
        (1.05 req/s over two Prefills) reproduced it, because every instance is
        briefly idle between requests and a per-candidate check missed it.
        """
        evaluator = StructuralEvaluator({"enabled": True,
                                         "evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0, "holding_cost": 0.5})
        busy = Sched(0, "prefill")
        busy.running = [None] * 4
        idle = Sched(1, "prefill")
        rows = [{"class_id": "c", "arrival_rate_ewma": 1.0}]
        decode = [Sched(11, "decode")]
        decision = evaluator.evaluate(
            {"prefix_states": rows}, [busy, idle], [busy, idle], decode,
            _FakeSolver(lambda ids: 0.1 if len(ids) == 2 else 0.0),
            current_ns=10 ** 18, last_action_ns=-1, min_active=1)
        self.assertEqual(decision.action, "keep")
        self.assertIn("no eligible", decision.reason)

    def test_an_idle_pool_can_still_scale_in(self):
        """The holding cost has to keep working when nothing is in flight."""
        evaluator = StructuralEvaluator({"enabled": True,
                                         "evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0, "holding_cost": 0.5})
        first, second = Sched(0, "prefill"), Sched(1, "prefill")
        decision = evaluator.evaluate(
            {"prefix_states": [{"class_id": "c", "arrival_rate_ewma": 1.0}]},
            [first, second], [first, second], [Sched(11, "decode")],
            _FakeSolver(lambda ids: 0.1 if len(ids) == 2 else 0.0),
            current_ns=10 ** 18, last_action_ns=-1, min_active=1)
        self.assertEqual(decision.action, "-P")
        self.assertEqual(len(decision.wanted_ids), 1)

    def test_the_deployment_inflight_counter_also_blocks_removal(self):
        """``inflight`` spans the whole request lifetime on the real router."""
        evaluator = StructuralEvaluator({"enabled": True,
                                         "evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0, "holding_cost": 0.5})
        first, second = Sched(0, "prefill"), Sched(1, "prefill")
        first.inflight = 2          # dispatched, not finished, no queue visible
        decision = evaluator.evaluate(
            {"prefix_states": [{"class_id": "c", "arrival_rate_ewma": 1.0}]},
            [first, second], [first, second], [Sched(11, "decode")],
            _FakeSolver(lambda ids: 0.1 if len(ids) == 2 else 0.0),
            current_ns=10 ** 18, last_action_ns=-1, min_active=1)
        self.assertEqual(decision.action, "-P")
        self.assertEqual(decision.wanted_ids, (0,),
                         "only the instance with no in-flight work may be removed")

    def test_an_all_busy_pool_has_no_removal_candidate(self):
        evaluator = StructuralEvaluator({"enabled": True,
                                         "evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0, "holding_cost": 0.5})
        first, second = Sched(0, "prefill"), Sched(1, "prefill")
        first.waiting = [None] * 3
        second.running = [None] * 2
        decision = evaluator.evaluate(
            {"prefix_states": [{"class_id": "c", "arrival_rate_ewma": 1.0}]},
            [first, second], [first, second], [Sched(11, "decode")],
            _FakeSolver(lambda ids: 0.1 if len(ids) == 2 else 0.0),
            current_ns=10 ** 18, last_action_ns=-1, min_active=1)
        self.assertEqual(decision.action, "keep")
        self.assertIn("no eligible", decision.reason)

    def test_max_active_prefill_caps_scale_out(self):
        decision = self.evaluate(lambda ids: 10.0 if len(ids) == 1 else 0.0,
                                 active_ids=(0,), inactive_ids=(1, 2),
                                 config={"evaluation_window_ms": 60000.0,
                                         "startup_s": 45.0,
                                         "max_active_prefill": 1})
        self.assertEqual(decision.action, "keep")
        self.assertIn("no eligible", decision.reason)

    def test_the_best_candidate_wins_not_the_lowest_id(self):
        # Adding instance 3 is worth far more than adding instance 1.
        def objective(ids):
            if 3 in ids:
                return 0.0
            return 10.0

        decision = self.evaluate(objective, active_ids=(0,), inactive_ids=(1, 3),
                                 config={"evaluation_window_ms": 60000.0,
                                         "startup_s": 0.0})
        self.assertEqual(decision.action, "+P")
        self.assertIn("Prefill 3", decision.reason)
        self.assertIn(3, decision.wanted_ids)

    def test_the_cooldown_is_never_shorter_than_the_boot(self):
        evaluator = StructuralEvaluator({"dwell_time_ms": 3000.0,
                                         "startup_s": 45.0})
        self.assertEqual(evaluator.dwell_ns, 45 * 10 ** 9)


class LocalTransferSymmetryTests(unittest.TestCase):
    """Every policy may prefill locally, and both paths pay their own queue."""

    def config(self):
        return {
            "weights": {"chars_per_token": 4.0, "kv_bytes_per_token": 147456},
            "prefills": [{"id": "p0", "host": "127.0.0.1", "port": 8100,
                          "capacity": 63, "max_num_seqs": 16, "instance_id": 0,
                          "service_ms": 48.5}],
            "decodes": [{"id": "d0", "host": "127.0.0.1", "port": 8200,
                         "capacity": 65, "max_num_seqs": 16, "instance_id": 11,
                         "service_ms": 144.7}],
            "links": [],
        }

    def router(self, policy="casr_lp"):
        return Router(self.config(), policy, "")

    def decide(self, router, kv_bytes=1024 * 147456):
        return router.kv_exchange_decision(router.prefills[0], router.decodes[0],
                                           kv_bytes)

    def test_baselines_are_not_pinned_to_transfer(self):
        self.assertEqual(self.router("load").local_prefill_mode, "auto")
        self.assertEqual(self.router("cache_aware").local_prefill_mode, "auto")

    def test_the_old_asymmetry_remains_available_for_an_ab(self):
        with mock.patch.dict(os.environ, {"LOCAL_PREFILL_BASELINES": "never"}):
            router = self.router("cache_aware")
        self.assertEqual(router.local_prefill_mode, "never")

    def test_local_prefill_wins_on_an_idle_decode(self):
        router = self.router()
        use_local, local_ms, transfer_ms = self.decide(router)
        self.assertTrue(use_local)
        self.assertLess(local_ms, transfer_ms)

    def test_the_decode_externality_grows_with_its_batch(self):
        idle = self.router()
        _, idle_local, _ = self.decide(idle)
        busy = self.router()
        busy.decodes[0].inflight = busy.decodes[0].max_inflight
        _, busy_local, busy_transfer = self.decide(busy)
        # rho = 1 -> the M/G/1 term is capped at LOCAL_PREFILL_QUEUE_CAP (3x).
        self.assertAlmostEqual(busy_local, idle_local * 4.0, delta=1.0)
        # On this fabric a local recompute stays far cheaper even then, which is
        # what elastic-a1 measured end to end (346 ms vs 1394 ms).
        self.assertLess(busy_local, busy_transfer)

    def test_a_cheap_handoff_wins_once_the_decode_is_saturated(self):
        with mock.patch.dict(os.environ, {"TRANSFER_MS_PER_1K_LOCAL": "150"}):
            router = self.router()
        router.decodes[0].inflight = router.decodes[0].max_inflight
        use_local, local_ms, transfer_ms = self.decide(router)
        self.assertFalse(use_local)
        self.assertGreater(local_ms, transfer_ms)

    def test_a_backed_up_prefill_is_charged_to_the_transfer_path(self):
        router = self.router()
        router.prefills[0].prefill_ms_ewma = 5000.0
        use_local, local_ms, transfer_ms = self.decide(router)
        self.assertTrue(use_local)
        self.assertGreater(transfer_ms, local_ms)


class PromptTokenCalibrationTests(unittest.TestCase):
    """``chars_per_token`` is fitted from the engine, not trusted as a constant."""

    class _Response:
        def __init__(self, prompt_tokens):
            self._tokens = prompt_tokens

        def json(self):
            return {"usage": {"prompt_tokens": self._tokens}}

    def router(self):
        config = {
            "weights": {"chars_per_token": 6.7},
            "prefills": [{"id": "p0", "host": "127.0.0.1", "port": 8100,
                          "capacity": 10, "max_num_seqs": 10, "instance_id": 0}],
            "decodes": [{"id": "d0", "host": "127.0.0.1", "port": 8200,
                         "capacity": 10, "instance_id": 10}],
        }
        return Router(config, "load", "")

    def test_ratio_converges_on_the_engines_own_token_count(self):
        router = self.router()
        payload = {"messages": [{"role": "user", "content": "x" * 4500}]}
        self.assertEqual(router.prompt_tokens(payload), int(4500 / 6.7))
        for _ in range(router.prompt_token_calibration_reqs):
            router.calibrate_prompt_tokens(payload, self._Response(1000))
        self.assertAlmostEqual(router.chars_per_token, 4.5, places=6)
        self.assertEqual(router.prompt_tokens(payload), 1000)

    def test_a_missing_usage_block_is_harmless(self):
        router = self.router()
        before = router.chars_per_token
        router.calibrate_prompt_tokens({"prompt": "hello"}, self._Response(0))
        self.assertEqual(router.chars_per_token, before)


if __name__ == "__main__":
    unittest.main()

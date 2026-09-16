"""Real-system CASR control-plane tests.

These exercise the observation/actuation adapter that lets the simulator's
CASR package drive the real multi-domain P/D router.  They run without any
live container: Docker and the Prometheus scrape are replaced by fakes.
"""

import asyncio
import json
import os
import pathlib
import sys
import time
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
for extra in (REPO / "serving", REPO / "deploy" / "real_lmcache_pd"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from casr.affinity import AffinityPlan                      # noqa: E402
from casr.evaluator import StructuralDecision               # noqa: E402
import casr_control                                         # noqa: E402
from disagg_router import Router, build_instances, parse_sse_line  # noqa: E402

CONFIG_PATH = REPO / "deploy" / "real_lmcache_pd" / "router_config.json"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


class FakeDocker:
    """Minimal Docker stand-in keyed by container name."""

    def __init__(self, running=None):
        self.running = dict(running or {})
        self.started = []
        self.stopped = []
        self.queries = []

    def state(self, instance):
        self.queries.append(instance.id)
        running = self.running.get(instance.id, True)
        return {"running": running, "status": "running" if running else "exited"}

    def start(self, instance):
        self.running[instance.id] = True
        self.started.append(instance.id)
        return True, ""

    def stop(self, instance):
        self.running[instance.id] = False
        self.stopped.append(instance.id)
        return True, ""


class RealControlTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.options = dict(self.config["casr"])

    def test_instance_ids_match_between_router_and_controller(self):
        controller = casr_control.build_controller(self.config, self.options)
        router_prefills = build_instances(self.config["prefills"], "prefill")
        router_decodes = build_instances(self.config["decodes"], "decode")

        enabled_prefills = [s for s in controller.prefill if s.enabled]
        enabled_decodes = [s for s in controller.decode if s.enabled]
        self.assertEqual([i.instance_id for i in router_prefills],
                         [s.instance_id for s in enabled_prefills])
        self.assertEqual([i.instance_id for i in router_decodes],
                         [s.instance_id for s in enabled_decodes])
        # Disabled hosts keep their slot so enabling the shared node later
        # cannot silently renumber live instances.
        self.assertEqual([i.instance_id for i in controller.prefill][-1], 4)
        self.assertIsNone(controller.real_by_id(99))

    def test_equalized_capacity_leaves_service_time_as_the_only_gap(self):
        """The ablation knob must remove the capacity gap, not the service one.

        On Dolly, ``casr_noservice`` still avoided the slow Decode, which means
        the capacity map (d3090a 26 vs d5090 65) was doing the work.  To
        attribute the gain we have to be able to equalise capacities and see
        what is left.
        """
        with mock.patch.dict(os.environ,
                             {"EQUALIZE_DECODE_CAPACITY": "max"}):
            controller = casr_control.build_controller(self.config, self.options)
        capacities = controller.solver.config.decode_capacity
        self.assertEqual(len(set(capacities.values())), 1)
        # ``max`` of the configured decode capacities.
        self.assertEqual(capacities[11], max(capacities.values()))
        # The heterogeneity itself is untouched: service times still differ.
        by_id = {inst.instance_id: inst for inst in controller.decode}
        self.assertNotEqual(by_id[11].service_ms, by_id[12].service_ms)
        with mock.patch.dict(os.environ,
                             {"EQUALIZE_DECODE_CAPACITY": "42"}):
            fixed = casr_control.build_controller(self.config, self.options)
        self.assertEqual(set(fixed.solver.config.decode_capacity.values()), {42.0})

    def test_router_capacity_equalization_flattens_the_heuristic_signal(self):
        """The fast router needs its own capacity flattening, not just the LP's.

        ``_pair_cost`` ranks Decode by ``inflight / capacity``, and a slow
        instance also has a smaller fast-router capacity (110 vs 230), so a
        service-term ablation that leaves capacity alone cannot isolate the
        service term at all.
        """
        baseline = Router(self.config, "casr_noservice", None)
        self.assertGreater(len({i.capacity for i in baseline.decodes}), 1)
        with mock.patch.dict(os.environ,
                             {"EQUALIZE_DECODE_CAPACITY": "max"}):
            flat = Router(self.config, "casr_noservice", None)
        self.assertEqual(len({i.capacity for i in flat.decodes}), 1)
        # The service times are untouched: only the capacity signal is gone.
        self.assertNotEqual(flat.decode_by_id(11).service_ms,
                            flat.decode_by_id(12).service_ms)

    def test_least_loaded_is_weighted_by_capacity_not_by_string_id(self):
        """The ``load`` baseline must balance, not pin one instance.

        ``min(..., key=(inflight / capacity, inst.id))`` used a *string*
        tie-break, and in-flight counts are small integers that are usually
        tied, so the policy repeatedly chose the lexicographically smallest id
        (``d3090a`` for Decode, ``p4090`` for Prefill).  Measured on the real
        cluster 2026-09-13: 60 requests gave ``p4090=50 p5090=7 p_a100=3``
        even though ``p_a100`` has the largest capacity; after the fix the
        same run gave ``p_a100=41 p4090=14 p5090=5``.
        """
        router = Router(self.config, "load", None)
        picks = {}
        for inst in router.prefills:
            inst.inflight = 0
        for _ in range(30):
            chosen = router._pick_load(router.prefills)
            picks[chosen.id] = picks.get(chosen.id, 0) + 1
            chosen.inflight += 1
        busiest = max(picks, key=lambda name: picks[name])
        largest = max(router.prefills, key=lambda inst: inst.capacity).id
        self.assertEqual(busiest, largest)
        # And the balance must not be pure id order: the second-largest
        # capacity must not be starved by the smallest-id instance.
        self.assertGreater(picks[largest],
                           picks[min(inst.id for inst in router.prefills)])

    def test_unseen_class_falls_back_to_cost_not_least_loaded(self):
        """Most real traffic is first-seen classes, so the fallback matters.

        The ShareGPT replay forms ~537 classes for 600 requests, so the plan
        cannot name most requests' classes and ``class_missing`` dominates.
        Falling back to least-loaded then sends them to whichever Prefill is
        idle -- here the 3090a that is ~2.5x slower -- which is exactly the
        behaviour the plan exists to avoid.
        """
        router = Router(self.config, "casr_lp", None)
        for inst in router.prefills:
            inst.inflight = 0          # nothing is busy: least-loaded is blind
        router.controller.last_plan = None
        chosen = router._pick_prefill_planned(
            "qwen3-8b|unseen|in:512-1023|out:16-31", "bucket-0", 0)
        self.assertEqual(chosen.id, "p5090")
        self.assertEqual(router._plan_stats["expired"], 1)
        # Cost ranking also beats load when *every* Prefill looks equally idle
        # but their measured service+overhead differs (100.5 / 252 / 112.2 ms).
        ranked = sorted(
            router.prefills,
            key=lambda i: i.service_ms + self.config["casr"]
            ["prefill_overhead_ms"].get(str(i.instance_id), 0.0))
        self.assertEqual(ranked[0].id, "p5090")
        self.assertEqual(ranked[-1].id, "p3090a")

    def test_unseen_class_keeps_prefix_affinity_after_the_first_pick(self):
        """The first pick is cost-ranked, every later one reuses it.

        A class the plan cannot name still has a prefix: re-ranking all
        Prefills on each request scatters its requests and re-prefills the same
        tokens cold every time.  The heuristic ``casr`` pins the bucket, and
        the plan policies have to do the same on their fallback path.
        """
        router = Router(self.config, "casr_lp", None)
        for inst in router.prefills:
            inst.inflight = 0
        router.controller.last_plan = None
        class_id, bucket = "qwen3-8b|unseen|in:512-1023|out:16-31", "bucket-affinity"
        first = router._pick_prefill_planned(class_id, bucket, 0)
        self.assertEqual(first.id, "p5090")
        self.assertEqual(router.affinity.get(bucket), "p5090")
        # Make p5090 the worst choice on paper; the pin must still hold.
        for inst in router.prefills:
            if inst.id == "p5090":
                inst.service_ms = 10_000.0
        second = router._pick_prefill_planned(class_id, bucket, 0)
        self.assertEqual(second.id, "p5090")

    def test_prefix_pin_is_dropped_when_it_becomes_the_bottleneck(self):
        """Affinity is kept only while it is not the more loaded choice.

        Measured 2026-09-13 on the strong-reuse (deep) tier: unconditional
        pinning put 73% of the traffic on one pair and made ``casr``'s trimmed
        mean *worse* than its own no-affinity ablation -- the cache reuse did
        not pay for the queue it created.  With the guard the same run improves
        trimmed mean 512.1 -> 380.1 ms and stays best on p50/p95/TTFT-p95.
        """
        router = Router(self.config, "casr", None)
        warm = next(i for i in router.prefills if i.id == "p5090")
        for inst in router.prefills:
            inst.prefill_inflight = 0
        router.affinity["bucket-x"] = warm.id
        # Idle everywhere: the pin holds (it is what buys the cache hit).
        self.assertEqual(router._pick_prefill("bucket-x").id, warm.id)
        # The pinned instance now carries a clearly deeper Prefill queue.
        warm.prefill_inflight = 20
        self.assertNotEqual(router._pick_prefill("bucket-x").id, warm.id)

    def test_decode_inflight_is_released_when_the_stream_finishes(self):
        """The Decode accounting must survive until the request really ends.

        The streaming handler returns its ``StreamingResponse`` before the
        generator body runs, so releasing the Decode counters in the handler's
        own ``finally`` fired at *dispatch* time.  Measured 2026-09-13: the cost
        model never saw more than 4 in-flight Decodes against a 32-request
        client, so it kept choosing the fastest one (186/200 requests) and the
        decode p50 doubled.  ``_release_decode`` is now called from wherever the
        request actually finishes, so the accounting has to be idempotent per
        request and give both the counter and the link bytes back.
        """
        router = Router(self.config, "casr", None)
        prefill = router.prefill_by_id(0)
        decode = next(i for i in router.decodes if i.id == "d4090")
        kv_bytes = 37_000_000.0
        prefill.inflight += 1
        decode.inflight += 1
        router.link_inflight_bytes[(prefill.domain, decode.domain)] = kv_bytes
        router._release_decode(prefill, decode, kv_bytes)
        self.assertEqual(decode.inflight, 0)
        self.assertEqual(prefill.inflight, 1)
        self.assertNotIn((prefill.domain, decode.domain), router.link_inflight_bytes)

    def test_cross_domain_cost_tracks_link_occupancy_not_serialisation(self):
        """A cross-domain hop is priced by what is already on the link.

        Measured 2026-09-13 (docs/实验结果汇总.md §5.10): a 156 MB cross-domain
        handoff costs +0.4 ms when the link is idle -- LMCache overlaps the push
        with the Prefill -- and the very same decision costs +31.8% once the
        offered cross-domain traffic (3.8 GB/s at the 3800-token tier) exceeds
        the ~2 GB/s link.  Pricing ``kv_bytes / bandwidth`` instead made the
        router refuse a profitable cross-domain pairing on short prompts and
        under-priced the saturated case.
        """
        router = Router(self.config, "casr", None)
        prefill = next(i for i in router.prefills if i.id == "p4090")
        same = next(i for i in router.decodes if i.id == "d4090")
        cross = next(i for i in router.decodes if i.id == "d5090")
        kv_bytes = 156 * 1000 * 1000

        # Idle link: the only cross-domain cost left is the measured RTT.
        idle_same = router._pair_cost(prefill, same, kv_bytes)
        idle_cross = router._pair_cost(prefill, cross, kv_bytes)
        rtt_s = router.links[(prefill.domain, cross.domain)][1] / 1000.0
        expected_idle_gap = (router.w_network * rtt_s
                             + router.w_service
                             * (cross.service_ms - same.service_ms) / 1000.0)
        self.assertAlmostEqual(idle_cross - idle_same, expected_idle_gap, places=6)

        # Occupied link: the same transfer now waits for the bytes in flight.
        router.link_inflight_bytes[(prefill.domain, cross.domain)] = 4 * kv_bytes
        busy_cross = router._pair_cost(prefill, cross, kv_bytes)
        self.assertAlmostEqual(busy_cross - idle_cross,
                               router.w_network * 4 * kv_bytes /
                               (router.links[(prefill.domain, cross.domain)][0] * 1e9 / 8.0),
                               places=6)

    def test_runtime_disabled_instances_are_not_structural_candidates(self):
        """``DISABLED_INSTANCES`` must disable the slow control loop too.

        The fast router already skipped runtime-disabled slots; the controller
        only honoured the config's ``enabled`` flag, so ``casr_full`` kept
        proposing ``+P`` on a container that was never launched (observed live
        as a failed ``scale_out`` of ``p3090b`` on 2026-09-11).
        """
        disabled = "p3090b,p4090,d4090"
        with mock.patch.dict(os.environ, {"DISABLED_INSTANCES": disabled}):
            controller = casr_control.build_controller(self.config, self.options)
            router_prefills = build_instances(self.config["prefills"], "prefill")
            router_decodes = build_instances(self.config["decodes"], "decode")

        enabled_prefills = [s for s in controller.prefill if s.enabled]
        enabled_decodes = [s for s in controller.decode if s.enabled]
        # Both halves of the control plane agree on the live topology...
        self.assertEqual([i.instance_id for i in router_prefills],
                         [s.instance_id for s in enabled_prefills])
        self.assertEqual([i.instance_id for i in router_decodes],
                         [s.instance_id for s in enabled_decodes])
        # ...and the disabled slots are INACTIVE, so the evaluator's candidate
        # list (``[s for s in self.prefill if s.enabled]``) cannot name them.
        by_id = {s.id: s for s in controller.prefill}
        self.assertFalse(by_id["p3090b"].enabled)
        self.assertFalse(by_id["p4090"].enabled)
        self.assertEqual(by_id["p3090b"].admission_state, "INACTIVE")
        self.assertEqual(by_id["p4090"].admission_state, "INACTIVE")
        # Same derivation as the probe test: whatever the config enables and
        # the run did not disable must be exactly what both halves agree on.
        disabled_ids = {token for token in disabled.split(",") if token}
        expected = [spec["id"] for spec in self.config["prefills"]
                    if spec.get("enabled", True) and spec["id"] not in disabled_ids]
        self.assertEqual([s.id for s in enabled_prefills], expected)

    def test_solver_config_is_parsed_from_the_shared_casr_block(self):
        controller = casr_control.build_controller(self.config, self.options)
        solver = controller.solver
        self.assertEqual(solver.config.solver, "lp")
        # Prefill capacity comes from the config's measured saturation point
        # (tests/calibrate_pd_throughput.py, 2026-09-13).  Assert the mapping
        # rather than the number so recalibration does not break the test, but
        # do require that the declared value is NOT the linear
        # ``max_num_seqs * 1000 / service_ms`` extrapolation, which overstated
        # the knee 5-7x (408 vs 59 req/s for p4090).
        self.assertEqual(solver.config.prefill_capacity[4],
                         float(self.config["casr"]["prefill_capacity"]["4"]))
        self.assertLess(solver.config.prefill_capacity[4],
                        round(16 * 1000.0 / 39.2))
        # Decode capacity is an explicit *measured saturation throughput*
        # (tests/calibrate_pd_throughput.py) and must win over the derived
        # ``max_num_seqs * 1000 / service_ms`` estimate, which overstates the
        # knee by ~2x.
        self.assertEqual(solver.config.decode_capacity[11],
                         float(self.config["casr"]["decode_capacity"]["11"]))
        self.assertLess(solver.config.decode_capacity[11],
                        round(16 * 1000.0 / 144.7))
        self.assertAlmostEqual(solver.config.queue_weight, 1.0)
        # Every instance knows which container host to scale through.
        self.assertEqual(controller.prefill_by_id(2).ssh, "buaa@10.212.70.38")
        self.assertTrue(controller.prefill_by_id(2).ssh)
        # Cross-domain pairs carry the measured link cost; same-domain is free.
        prefills = {instance.domain: instance for instance in controller.prefill}
        self.assertNotEqual(prefills["4090"].start_npu, prefills["5090"].start_npu)
        self.assertEqual(prefills["4090"].start_npu, 4)

    def test_lp_backend_emits_a_normalized_affinity_plan(self):
        controller = casr_control.build_controller(self.config, self.options)
        prefill = controller.prefill_by_id(4)
        # ``rows()`` reports the *sampled* rate, not the raw arrival counter.
        # The first ``sample_arrivals`` only opens the measurement window (the
        # controller has no previous tick to difference against and drops the
        # counter), so the arrivals have to be bracketed by two samples -- which
        # is exactly the cadence the real control loop uses.
        controller.sample_arrivals(1_000_000)
        controller.observe("qwen3-8b|cls|in:1024-2047|out:1-1", prefill,
                           2048, 2048 * 147456, now_ns=1_000)
        controller.observe("qwen3-8b|cls|in:1024-2047|out:1-1", prefill,
                           2048, 2048 * 147456, now_ns=2_000_000_000)
        controller.sample_arrivals(3_000_000_000)
        flows = controller.solver.solve(controller.rows(), controller.prefill,
                                        controller.decode)
        self.assertEqual(controller.solver.backend, "ortools-glop")
        plan = casr_control.build_affinity_plan(
            flows, controller.decode, version=1, expires_at_ns=10 ** 18)
        weights = plan.prefill_for("qwen3-8b|cls|in:1024-2047|out:1-1")
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertTrue(all(instance_id in {s.instance_id for s in controller.prefill}
                            for instance_id in weights))

    def test_stopped_container_is_demoted_and_scaled_out(self):
        options = dict(self.options, scale_backend="docker")
        controller = casr_control.build_controller(self.config, options)
        docker = FakeDocker({"p4090": False})
        controller.docker = docker

        controller._sync_container_states(now_ns=0)
        recovered = controller.prefill_by_id(4)
        self.assertEqual(recovered.admission_state, "INACTIVE")

        decision = StructuralDecision("+P", "cold", 1.0, 0.0, 0.0,
                                      (0, 1, 2, 4), "counterfactual add Prefill 4")
        results = controller.apply_structural(decision, now_ns=0)
        # The topology config decides which other instances are already
        # active, so assert the stopped container is restarted rather than
        # pinning the exact set.
        self.assertIn("p4090", docker.started)
        self.assertEqual(recovered.admission_state, "WARMING")
        self.assertTrue(results[0]["ok"])

        recovered.healthy = True
        controller._sync_container_states(now_ns=int(controller.warmup_s * 1e9))
        self.assertEqual(recovered.admission_state, "ACTIVE")

    def test_active_containers_are_probed_once_per_refresh_window(self):
        """A remote ``docker inspect`` is an SSH round-trip (~0.3 s).

        Probing every ``ACTIVE`` Prefill on every 1 s tick stretched the
        control loop and moved the arrival-rate sampling window, which is what
        made ``casr_full``'s first (and, under 3 % hysteresis, only) plan
        differ from ``casr_lp``'s.  Probes must therefore be rate limited.
        """
        options = dict(self.options, scale_backend="docker")
        controller = casr_control.build_controller(self.config, options)
        docker = FakeDocker()
        controller.docker = docker

        controller._sync_container_states(now_ns=0)
        first = list(docker.queries)
        # Derive the expected probe set from the topology file instead of
        # hardcoding it: the config gains and loses domains as shared nodes
        # become available (the A100 was added on 2026-09-13).
        expected = [spec["id"] for spec in self.config["prefills"]
                    if spec.get("enabled", True)]
        self.assertEqual(first, expected)

        controller._sync_container_states(now_ns=1_000_000_000)
        self.assertEqual(docker.queries, first)

        controller._sync_container_states(
            now_ns=int(controller.state_refresh_s * 1e9) + 1)
        self.assertEqual(len(docker.queries), 2 * len(first))

    def test_observed_per_class_budget_reaches_the_solver(self):
        """The workload's own per-class TTFT budget must drive the objective.

        ``class_ttft_slo_ms`` existed in the solver but nothing could populate
        it.  The router now forwards the SLO each request carried, the
        controller keeps the tightest bound seen per class, and every tick
        republishes it into the shared solver config.
        """
        controller = casr_control.build_controller(self.config, self.options)
        class_id = "qwen3-8b|cls|in:512-1023|out:16-31"
        prefill = controller.prefill_by_id(0)
        controller.observe(class_id, prefill, 1024, 1024 * 147456,
                           now_ns=1_000, slo_ttft_ms=300.0)
        controller.observe(class_id, prefill, 1024, 1024 * 147456,
                           now_ns=2_000, slo_ttft_ms=800.0)
        # The binding (tightest) budget is what the router must honour.
        self.assertEqual(controller._class_slo[class_id], 300.0)

        # A tick republishes it without needing a live metrics scrape.
        controller.refresh_metrics = lambda: None
        controller.tick(3_000)
        self.assertEqual(controller.solver.config.class_ttft_slo_ms,
                         {class_id: 300.0})
        self.assertEqual(controller.as_dict()["class_slo"],
                         {"classes": 1, "tightest_ms": 300.0})

    def test_scale_in_drains_then_stops_the_container(self):
        options = dict(self.options, scale_backend="docker")
        controller = casr_control.build_controller(self.config, options)
        docker = FakeDocker()
        controller.docker = docker
        victim = controller.prefill_by_id(2)
        victim.set_admission_state("ACTIVE")
        victim.running = [None]
        victim.waiting = []

        decision = StructuralDecision("-P", "none", 1.0, 0.0, 0.0, (0, 1, 4), "remove p3090b")
        controller.apply_structural(decision, now_ns=0)
        self.assertEqual(victim.admission_state, "DRAINING")
        self.assertEqual(docker.stopped, [])

        victim.running = []
        controller._sync_container_states(now_ns=1)
        self.assertEqual(victim.admission_state, "INACTIVE")
        # The decision's ``wanted_ids`` says which prefills stay; every other
        # active one drains, so the exact stop list depends on the topology.
        self.assertIn("p3090b", docker.stopped)


class SseParseTests(unittest.TestCase):
    """Router-side SSE folding, used to put TTFT/TPOT into the metrics file."""

    def test_ttft_is_the_first_chunk_that_carries_text(self):
        state = {"first_token_ms": None, "tokens": 0, "usage": {}}
        state["now_ms"] = 42.0
        parse_sse_line('data: {"choices": [{"text": ""}]}', state)
        self.assertIsNone(state["first_token_ms"])
        state["now_ms"] = 123.5
        parse_sse_line('data: {"choices": [{"text": "a"}]}', state)
        self.assertEqual(state["first_token_ms"], 123.5)
        state["now_ms"] = 200.0
        parse_sse_line('data: {"choices": [{"text": "b"}]}', state)
        self.assertEqual(state["first_token_ms"], 123.5)
        self.assertEqual(state["tokens"], 2)

    def test_usage_chunk_and_junk_lines(self):
        state = {"first_token_ms": None, "tokens": 0, "usage": {}}
        parse_sse_line(": keep-alive", state)
        parse_sse_line("data: [DONE]", state)
        parse_sse_line("data: not-json", state)
        self.assertEqual(state["tokens"], 0)
        parse_sse_line('data: {"choices": [], "usage": {"completion_tokens": 16}}',
                       state)
        self.assertEqual(state["usage"]["completion_tokens"], 16)


class WarmupTests(unittest.IsolatedAsyncioTestCase):
    """New plan edges are warmed in the background, off the client's path."""

    def setUp(self):
        config = load_config()
        config = dict(config)
        config["casr"] = dict(config["casr"], warmup_enabled=True,
                              warmup_concurrency=2, warmup_min_share=0.1)
        self.router = Router(config, "casr_lp", metrics_path="")

    def plan(self, weights):
        return AffinityPlan(version=1, expires_at_ns=10 ** 18,
                            prefill_weights={"c": weights},
                            decode_weights={(pid, "c"): {11: 1.0} for pid in weights})

    def test_warm_targets_skip_warmed_edges_and_small_shares(self):
        weights = {0: 0.55, 4: 0.35, 1: 0.1, 2: 0.05}
        targets = self.router.warm_targets("c", self.plan(weights))
        self.assertEqual(sorted(prefill.id for _, prefill, _ in targets),
                         ["p4090", "p5090"])
        self.router._warmed_edges.add(("c", 0))
        self.assertEqual([prefill.id for _, prefill, _ in
                          self.router.warm_targets("c", self.plan(weights))],
                         ["p4090"])

    async def test_maybe_warm_schedules_each_edge_once(self):
        self.router.controller.last_plan = self.plan({0: 0.5, 4: 0.5})
        self.router.remember_prompt("c", "/chat/completions",
                                    {"model": "m", "messages": [], "max_tokens": 4})
        scheduled = []

        async def fake_edge(key, prefill, decode, payload, endpoint):
            scheduled.append((key, prefill.id, decode.id, endpoint))
            self.router._warmed_edges.add(key)
            self.router._warm_inflight.discard(key)

        self.router._warm_edge = fake_edge
        self.assertEqual(self.router.maybe_warm("c", time.monotonic_ns()), 2)
        await asyncio.sleep(0)
        self.assertEqual(sorted((key, prefill) for key, prefill, _, _ in scheduled),
                         [(("c", 0), "p5090"), (("c", 4), "p4090")])
        # Nothing is warmed twice, and warming stays off once disabled.
        self.assertEqual(self.router.maybe_warm("c", time.monotonic_ns()), 0)
        self.router.warmup_enabled = False
        self.assertEqual(self.router.maybe_warm("c", time.monotonic_ns()), 0)

    async def test_serving_prefill_is_not_rewarmed(self):
        # The Prefill serving the triggering request warms its own prefix, so
        # warming it here too would recompute the same prefix and push the same
        # KV twice.
        self.router.controller.last_plan = self.plan({0: 0.5, 4: 0.5})
        self.router.remember_prompt("c", "/chat/completions",
                                    {"model": "m", "messages": [], "max_tokens": 4})
        scheduled = []

        async def fake_edge(key, prefill, decode, payload, endpoint):
            scheduled.append((key, prefill.id))

        self.router._warm_edge = fake_edge
        self.assertEqual(self.router.maybe_warm("c", time.monotonic_ns(),
                                                serving_prefill_id=0), 1)
        await asyncio.sleep(0)
        self.assertEqual(scheduled, [(("c", 4), "p4090")])

    async def test_warmup_is_opt_in(self):
        config = load_config()
        router = Router(config, "casr_lp", metrics_path="")
        self.assertFalse(router.warmup_enabled)
        router.remember_prompt("c", "/chat/completions", {"model": "m"})
        self.assertEqual(router._prompt_cache, {})


class ArrivalRateTests(unittest.TestCase):
    """The demand fed to the LP must not be inflated by inter-arrival noise."""

    def setUp(self):
        config = load_config()
        self.controller = casr_control.build_controller(
            config, dict(config["casr"], solver="lp"))

    def test_simultaneous_arrivals_do_not_inflate_the_rate(self):
        prefill = self.controller.prefill_by_id(4)
        base = 10 ** 18
        self.controller.sample_arrivals(base)  # establish the window baseline
        for index in range(20):
            # 20 requests inside one millisecond: the old 1/gap estimator read
            # this as ~200k req/s and the LP then over-spread every class.
            self.controller.observe("cls", prefill, 2048, 2048 * 147456,
                                    base + index * 1000)
        self.controller.sample_arrivals(base + 10 ** 9)
        rate = self.controller.states[(4, "cls")].arrival_rate_ewma
        self.assertAlmostEqual(rate, 20.0, delta=0.5)

    def test_rate_tracks_the_offered_load(self):
        prefill = self.controller.prefill_by_id(4)
        base = 10 ** 18
        self.controller.sample_arrivals(base)
        # 5 req/s in every one-second window, including an idle one.
        for window in (0, 1, 2):
            for index in range(5 if window != 1 else 0):
                self.controller.observe("cls", prefill, 2048, 2048 * 147456,
                                        base + (window * 5 + index) * 200_000_000)
            self.controller.sample_arrivals(base + (window + 1) * 10 ** 9)
        # 5 -> idle (0.2*0 + 0.8*5 = 4.0) -> 5 (0.2*5 + 0.8*4.0 = 4.2).
        rate = self.controller.states[(4, "cls")].arrival_rate_ewma
        self.assertAlmostEqual(rate, 4.2, delta=0.1)


class PlanHysteresisTests(unittest.TestCase):
    """The plan layer must not churn class->Prefill assignments every tick."""

    def setUp(self):
        self.config = {
            "hosts": {"a": {"ssh": "user@host-a"}, "b": {"ssh": "user@host-b"}},
            "prefills": [{"id": "pA", "host": "10.0.0.1", "domain": "a", "port": 8100,
                          "instance_id": 0, "service_ms": 50.0,
                          "solver_capacity": 320, "max_num_seqs": 16},
                         {"id": "pB", "host": "10.0.0.2", "domain": "b", "port": 8100,
                          "instance_id": 1, "service_ms": 50.0,
                          "solver_capacity": 320, "max_num_seqs": 16}],
            "decodes": [{"id": "dA", "host": "10.0.0.1", "domain": "a", "port": 8200,
                         "instance_id": 11, "service_ms": 150.0,
                         "solver_capacity": 106, "max_num_seqs": 16},
                        {"id": "dB", "host": "10.0.0.2", "domain": "b", "port": 8200,
                         "instance_id": 12, "service_ms": 150.0,
                         "solver_capacity": 106, "max_num_seqs": 16}],
            "links": [],
        }
        self.options = {"solver": "lp", "model_name": "m", "container_prefix": "t-",
                        "plan_ttl_s": 4.0, "plan_gain_threshold_rel": 0.03,
                        "compute_weight": 1.0}
        self.controller = casr_control.build_controller(self.config, self.options)

    def plan(self, prefill_id, decode_id, version=1):
        return casr_control.AffinityPlan(
            version=version, expires_at_ns=10 ** 18,
            prefill_weights={"c": {prefill_id: 1.0}},
            decode_weights={(prefill_id, "c"): {decode_id: 1.0}})

    def row(self, class_id="c", demand=5.0):
        return {"class_id": class_id, "arrival_rate_ewma": demand,
                "prefill_instance_id": 0, "hit_tokens_ewma": 2048.0,
                "requested_tokens": 2048.0 * 40, "requested_tokens_ewma": 2048.0,
                "kv_bytes_per_request": 2048 * 147456}

    def publish(self, candidate, rows=None, active=None):
        return self.controller._publish_plan(
            candidate, rows if rows is not None else [self.row()],
            active if active is not None else self.controller.prefill,
            self.controller.decode, 10 ** 18)

    def test_equivalent_plan_is_kept_without_bumping_the_version(self):
        self.controller.last_plan = self.plan(0, 11, version=7)
        published = self.publish(self.plan(1, 12, version=8))
        self.assertEqual(published.version, 7)
        self.assertEqual(published.prefill_for("c"), {0: 1.0})

    def test_a_clearly_better_plan_is_adopted(self):
        self.controller.last_plan = self.plan(0, 11, version=7)
        # Make dA much slower than dB, so moving to pB/dB is far more than 3%.
        self.controller.solver.config.decode_service_ms[11] = 400.0
        published = self.publish(self.plan(1, 12, version=8))
        self.assertEqual(published.version, 8)
        self.assertEqual(published.prefill_for("c"), {1: 1.0})

    def test_plan_naming_an_inactive_instance_is_replaced(self):
        self.controller.last_plan = self.plan(0, 11, version=7)
        published = self.publish(self.plan(1, 12, version=8),
                                 active=[self.controller.prefill[1]])
        self.assertEqual(published.version, 8)

    def test_hysteresis_rejects_an_incumbent_that_overruns_a_link_budget(self):
        """Cold start must not freeze a plan that saturates a Prefill's KV push.

        The first plan is solved while demand is still ramping, so it stacks
        every class on the cheapest pair.  If the incumbent is scored with a
        purely linear cost it always looks cheaper than the spread candidate and
        hysteresis never releases it -- the observed first-third collapse.
        """
        options = dict(self.options, utilization_weight=1.0,
                       shared_links=[{"id": "l0", "capacity_bytes_per_s": 1e8,
                                      "pairs": [[0, 11], [0, 12]]}])
        controller = casr_control.build_controller(self.config, options)
        controller.last_plan = self.plan(0, 11, version=7)
        candidate = casr_control.AffinityPlan(
            version=8, expires_at_ns=10 ** 18,
            prefill_weights={"c": {0: 0.5, 1: 0.5}},
            decode_weights={(0, "c"): {11: 1.0}, (1, "c"): {11: 1.0}})
        published = controller._publish_plan(
            candidate, [self.row(demand=20.0)], controller.prefill,
            controller.decode, 10 ** 18)
        self.assertEqual(published.version, 8)

    def test_hysteresis_is_off_without_a_threshold(self):
        self.options.pop("plan_gain_threshold_rel")
        controller = casr_control.build_controller(self.config, self.options)
        controller.last_plan = self.plan(0, 11, version=7)
        published = controller._publish_plan(self.plan(1, 12, version=8),
                                             [self.row()], controller.prefill,
                                             controller.decode, 10 ** 18)
        self.assertEqual(published.version, 8)


class PlanReplayTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.router = Router(self.config, "casr_lp", metrics_path="")

    def install(self, prefill_weights, decode_weights, fallbacks=None):
        self.router.controller.last_plan = AffinityPlan(
            version=1, expires_at_ns=10 ** 18,
            prefill_weights=prefill_weights,
            decode_weights=decode_weights,
            fallback_decode_ids=fallbacks or {})

    def test_weighted_replay_spreads_across_the_plan(self):
        self.install({"c": {0: 0.5, 4: 0.5}}, {(0, "c"): {11: 1.0}})
        chosen = [self.router.pick_prefill("c", "c", 0).instance_id for _ in range(4)]
        self.assertEqual(sorted(set(chosen)), [0, 4])
        self.assertEqual(chosen.count(0), chosen.count(4))
        decode = self.router.pick_decode(self.router.prefill_by_id(0), "c", "c",
                                         kv_bytes=1, now_ns=0)
        self.assertEqual(decode.instance_id, 11)

    def test_unplanned_class_falls_back_to_least_loaded(self):
        self.install({}, {})
        picked = self.router.pick_prefill("unknown", "unknown", 0)
        self.assertIn(picked.instance_id, {i.instance_id for i in self.router.prefills})
        decode = self.router.pick_decode(picked, "unknown", "unknown", 1, 0)
        self.assertIn(decode.instance_id, {i.instance_id for i in self.router.decodes})

    def test_decode_fallback_is_used_when_weights_are_empty(self):
        self.install({"c": {0: 1.0}}, {(0, "c"): {}},
                     fallbacks={(0, "c"): (11, 14)})
        decode = self.router.pick_decode(self.router.prefill_by_id(0), "c", "c", 1, 0)
        self.assertIn(decode.instance_id, (11, 14))

    def test_decode_fallback_uses_pair_cost_not_least_loaded(self):
        """The fallback must not undo the solver's decode choice.

        Fallbacks are the Decodes the plan left at zero.  Choosing the least
        loaded one sends traffic to whatever the cost model avoided (the slow
        3090a decode stays idle and therefore always looks free).
        """
        self.install({"c": {0: 1.0}}, {(0, "c"): {}},
                     fallbacks={(0, "c"): (11, 12, 14)})
        for instance in self.router.decodes:
            instance.inflight = 5
        slow = self.router.decode_by_id(12)
        slow.inflight = 0
        picked = self.router.pick_decode(self.router.prefill_by_id(0), "c", "c", 1, 0)
        self.assertNotEqual(picked.instance_id, 12)

    def test_plan_fallback_does_not_force_the_avoided_decode(self):
        """A plan that only lists the avoided Decode as fallback must not pin it.

        ``build_affinity_plan`` records fallbacks as the Decodes the solver left
        at zero -- which can be exactly the slow instance the cost model just
        rejected.  The router has to re-decide with the cost model instead.
        """
        self.install({"c": {0: 1.0}}, {(0, "c"): {}},
                     fallbacks={(0, "c"): (12,)})
        picked = self.router.pick_decode(self.router.prefill_by_id(0), "c", "c", 1, 0)
        self.assertEqual(picked.instance_id, 11)


class WarmBeforeSwitchTests(unittest.TestCase):
    """A plan move waits for the new edge's prefix instead of paying a cold prefill."""

    def setUp(self):
        config = load_config()
        config["casr"] = dict(config.get("casr", {}))
        config["casr"]["warmup_enabled"] = True
        config["casr"]["warm_before_switch"] = True
        config["casr"]["warm_switch_timeout_s"] = 3.0
        self.router = Router(config, "casr_lp", metrics_path="")

    def install(self, version, prefill_weights):
        self.router.controller.last_plan = AffinityPlan(
            version=version, expires_at_ns=10 ** 18,
            prefill_weights=prefill_weights, decode_weights={})

    def _pick(self, now_ns):
        return self.router.pick_prefill("c", "c", now_ns).instance_id

    def test_plan_move_waits_until_the_new_edge_is_warm(self):
        self.install(1, {"c": {0: 1.0}})
        first = self._pick(0)
        # The first served request warms that edge in LMCache.
        self.router._warmed_edges.add(("c", first))
        # The LP now prefers a different Prefill, but its prefix is cold.
        self.install(2, {"c": {4: 1.0}})
        held = self._pick(1_000_000)
        self.assertEqual(held, first)
        # Once the warmer has seeded the new edge, the move goes through.
        self.router._warmed_edges.add(("c", 4))
        self.assertEqual(self._pick(2_000_000), 4)

    def test_a_cold_incumbent_does_not_block_the_move_forever(self):
        self.install(1, {"c": {0: 1.0}})
        self._pick(0)
        self.router._warmed_edges.clear()
        self.install(2, {"c": {4: 1.0}})
        self.assertEqual(self._pick(1_000_000), 4)

    def test_switch_deadline_releases_a_stuck_warm(self):
        self.install(1, {"c": {0: 1.0}})
        first = self._pick(0)
        self.router._warmed_edges.add(("c", first))
        self.install(2, {"c": {4: 1.0}})
        self.assertEqual(self._pick(1_000_000), first)
        # Warming never succeeds; after the deadline the plan wins anyway.
        self.assertEqual(self._pick(5_000_000_000), 4)


class CacheAwareRouterTests(unittest.TestCase):
    """SGLang-style cache-aware routing: longest prefix first, load spills."""

    def setUp(self):
        config = {
            "prefills": [
                {"id": "p0", "host": "127.0.0.1", "port": 8100,
                 "capacity": 10, "max_num_seqs": 10, "instance_id": 0},
                {"id": "p1", "host": "127.0.0.1", "port": 8101,
                 "capacity": 10, "max_num_seqs": 10, "instance_id": 1},
            ],
            "decodes": [
                {"id": "d0", "host": "127.0.0.1", "port": 8200,
                 "capacity": 10, "instance_id": 10},
            ],
        }
        self.router = Router(config, "cache_aware", "")

    def _pick(self, prompt):
        bucket = self.router.classify({"prompt": prompt})
        return self.router.pick_prefill(bucket, bucket, 0)

    def test_repeat_prompt_sticks_to_the_prefill_that_holds_it(self):
        prompt = "shared prefix " * 400
        first = self._pick(prompt)
        second = self._pick(prompt)
        third = self._pick(prompt)
        self.assertEqual(first.id, second.id)
        self.assertEqual(second.id, third.id)

    def test_busy_match_spills_to_least_loaded(self):
        prompt = "hot prefix " * 400
        first = self._pick(prompt)
        # Saturate the matched Prefill past the 0.5 load threshold.
        first.inflight = int(first.max_inflight * 0.5)
        spilled = self._pick(prompt)
        self.assertNotEqual(spilled.id, first.id)

    def test_unseen_prompt_starts_on_the_least_loaded_prefill(self):
        self.router.prefills[0].inflight = 8
        self.router.prefills[1].inflight = 1
        picked = self._pick("a completely fresh prompt " * 200)
        self.assertEqual(picked.id, "p1")


if __name__ == "__main__":
    unittest.main()

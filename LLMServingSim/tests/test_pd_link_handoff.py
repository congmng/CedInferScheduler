"""The P/D KV handoff is a link resource, not an NPU workload.

Regression tests for the 2026-09-16 alignment work (see
``docs/模拟器与真机一致性核查.md`` 附十之十三).  Three defects were behind the
27x Decode TPOT the cluster's own placement reproduced in the simulator:

1. the handoff's RECV graph ran *on* the selected Decode NPU, so every decode
   step queued behind a 0.7-1.7 s KV receive -- while the cluster's d5090 held
   14.9 ms p50 TPOT through 300 handoffs;
2. the Prefill paid the same bytes again as exposed communication on its own
   timeline (10.97 s of an 11.77 s run), so the worker the cluster had running
   looked permanently busy;
3. the request's TTFT was stamped when the *Prefill* finished, which left the
   whole handoff inside the request's first inter-token gap and inflated TPOT
   by it.

``PdHandoffLink`` is what replaces the graph: bytes occupy the producer's
egress, the request becomes decodable one link latency later, and both engines
keep running.  The end-to-end test is gated because it drives a full ASTRA-Sim
run:

    SIM_PD_LINK_E2E=1 python3 -m pytest tests/test_pd_link_handoff.py -q
"""

import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.casr.lifecycle import PrefillLifecycle      # noqa: E402
from serving.core.pd_link import PdHandoffLink           # noqa: E402


GB = 1024 ** 3


class PdHandoffLinkTests(unittest.TestCase):
    def setUp(self):
        # 0.11 GB/s cross-domain, 0.3 GB/s inside a node; bandwidth is GB/s,
        # which is bytes/ns, so 110 MB takes 1 s on the slow pair.
        self.link = PdHandoffLink(
            bandwidth_gbps=lambda src, dst: 0.3 if src == dst else 0.11,
            latency_ns=lambda src, dst: 1_000 if src == dst else 48_000_000)

    def test_a_handoff_lands_after_its_bytes_and_the_link_latency(self):
        due = self.link.enqueue(0, 0, 0, 300_000_000, "p0", now_ns=5)
        self.assertEqual(due, 5 + 1_000_000_000 + 1_000)

    def test_one_producer_serialises_its_pushes(self):
        first = self.link.enqueue(0, 0, 0, 300_000_000, "first", now_ns=0)
        second = self.link.enqueue(0, 0, 0, 300_000_000, "second", now_ns=0)
        self.assertEqual(second - first, 1_000_000_000)

    def test_two_producers_push_in_parallel(self):
        first = self.link.enqueue(0, 0, 0, 300_000_000, "p0", now_ns=0)
        second = self.link.enqueue(2, 1, 1, 300_000_000, "p2", now_ns=0)
        self.assertEqual(first, second)

    def test_pop_due_returns_landed_payloads_in_order(self):
        link = PdHandoffLink(bandwidth_gbps=lambda src, dst: 0.3,
                             latency_ns=lambda src, dst: 0)
        link.enqueue(0, 0, 0, 300_000_000, "first", now_ns=0)
        link.enqueue(0, 0, 0, 300_000_000, "second", now_ns=0)
        self.assertEqual(link.next_due(), 1_000_000_000)
        self.assertEqual(link.pop_due(999_999_999), [])
        self.assertEqual(link.pop_due(1_000_000_000), ["first"])
        self.assertEqual(link.pop_due(10 ** 12), ["second"])
        self.assertIsNone(link.next_due())
        self.assertEqual(link.in_flight(), 0)

    def test_queue_wait_is_reported(self):
        self.link.enqueue(0, 0, 0, 300_000_000, "first", now_ns=0)
        self.link.enqueue(0, 0, 0, 300_000_000, "second", now_ns=0)
        self.assertAlmostEqual(self.link.stats()["mean_egress_wait_ms"], 500.0)
        self.assertEqual(self.link.stats()["bytes_shipped"], 600_000_000)


class _Scheduler:
    """Minimal scheduler surface ``PrefillLifecycle`` reads."""

    def __init__(self, instance_id, pd_type="prefill", node_id=0, waiting=0):
        self.instance_id = instance_id
        self.pd_type = pd_type
        self.node_id = node_id
        self.running = []
        self.waiting = [None] * waiting
        self.admission_state = "ACTIVE"
        self.max_num_seqs = 16
        self.num_npus = 1
        self.resource_gpu_ids = ()
        self.resource_mem_gb = 0.0
        self.memory = type("M", (), {"npu_mem": 24 * GB})()

    def set_admission_state(self, state):
        self.admission_state = str(state).upper()

    @property
    def accepts_new_requests(self):
        return self.admission_state == "ACTIVE"


def _rows(rate=1.0):
    return [{"class_id": "c", "arrival_rate_ewma": rate,
             "requested_tokens_ewma": 1250.0, "kv_bytes_per_request": 184e6,
             "prefix_tokens": 0}]


class LifecyclePoolStabilityTests(unittest.TestCase):
    """A pool that is not being resized must keep the workers it has.

    The demand heuristic ranks by *instantaneous* load, so re-picking the whole
    set every tick drained the worker that had just been given a request --
    measured 2026-09-16: p5090 was drained on the tick it received request 0,
    and the arm then replayed 247 of 300 handoffs across domains.
    """

    def _lifecycle(self, **extra):
        config = {"min_active_prefill": 2, "max_active_prefill": 3,
                  "scale_on_demand": False,
                  "prefill_capacity": {"0": 63, "2": 45, "4": 59},
                  # A pool can only be drained when the orchestrator owns the
                  # GPUs (that is what turns "not wanted" into INACTIVE).
                  "resources": {"startup_ms": 0, "reclaim_ms": 0,
                                "nodes": {str(node): {"gpu_count": 2,
                                                      "gpu_mem_gb": [24, 24]}
                                          for node in range(3)}}}
        config.update(extra)
        return PrefillLifecycle(config)

    def test_the_pool_does_not_churn_when_one_worker_takes_the_first_request(self):
        schedulers = [_Scheduler(0, node_id=0),
                      _Scheduler(2, node_id=1, waiting=1),
                      _Scheduler(4, node_id=2)]
        lifecycle = self._lifecycle()
        lifecycle.update(1_000_000, _rows(), schedulers)
        first = [s.instance_id for s in schedulers if s.accepts_new_requests]
        self.assertIn(2, first, "the worker holding the request must survive")
        for _ in range(3):
            lifecycle.update(1_000_000, _rows(), schedulers)
        self.assertEqual(
            [s.instance_id for s in schedulers if s.accepts_new_requests], first)

    def test_a_recorded_pool_seeds_the_wanted_set(self):
        schedulers = [_Scheduler(0, node_id=0), _Scheduler(2, node_id=1),
                      _Scheduler(4, node_id=2)]
        lifecycle = self._lifecycle(initial_active_prefill=[2, 4])
        for _ in range(3):
            lifecycle.update(1_000_000, _rows(), schedulers)
        self.assertEqual([s.instance_id for s in schedulers
                          if s.accepts_new_requests], [2, 4])


@unittest.skipUnless(os.environ.get("SIM_PD_LINK_E2E") == "1",
                     "set SIM_PD_LINK_E2E=1 to run the forced-handoff replay")
class ForcedHandoffEndToEndTests(unittest.TestCase):
    """A Decode must keep stepping while another request's KV is inbound."""

    def test_decode_tpot_is_the_local_step_and_ttft_carries_the_handoff(self):
        base = REPO / "configs" / "cluster" / "casr_real_small3_generated.json"
        config = json.loads(base.read_text(encoding="utf-8"))
        config["casr"]["local_prefill"] = "never"
        with tempfile.TemporaryDirectory(prefix="pd-link-e2e.") as tmp:
            arm = pathlib.Path(tmp) / "arm.json"
            arm.write_text(json.dumps(config), encoding="utf-8")
            csv_path = pathlib.Path(tmp) / "run.csv"
            command = [sys.executable, "-m", "serving",
                       "--cluster-config", str(arm),
                       "--dataset", "workloads/cnndm-long-pool-qwen3-8b.jsonl",
                       "--num-reqs", "6", "--dtype", "bfloat16",
                       "--block-size", "16", "--max-output-tokens", "16",
                       "--max-num-seqs", "16", "--client-concurrency", "4",
                       "--log-level", "WARNING",
                       "--output", str(csv_path),
                       "--enable-casr", "--casr-solver", "lp",
                       "--casr-control-interval-ms", "1000",
                       "--inputs-root", str(pathlib.Path(tmp) / "inputs")]
            proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-2000:])
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

        handoffs = [row for row in rows if row["exchange"] == "transfer"]
        self.assertTrue(handoffs, "the arm must force the handoff path")
        for row in handoffs:
            tpots = [float(value) / 1e6
                     for value in row["ITL"].strip("[]").split(",")[:1]]
            # The local Decode step is ~15 ms; a handoff is 0.6-1.7 s of link
            # time, and none of it may land in an inter-token gap.
            self.assertLess(float(row["TPOT"]) / 1e6, 30.0, row)
            self.assertGreater(float(row["TTFT"]) / 1e6, 400.0, row)
            self.assertEqual(len(tpots), 1)


if __name__ == "__main__":
    unittest.main()

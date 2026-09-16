"""The simulator has to make the same "hand the KV over or recompute" choice.

Measured 2026-09-16 on the 3-domain cluster: the real router answered 200/200
requests of the Dolly trace by letting the Decode recompute the prompt
(``exchange=local``, ``prefill_ms`` p50 = 0), while the simulator shipped every
KV and ended up bound by the 0.11 GB/s egress budget -- 6934 ms E2E p50 against
the cluster's 281.8 ms.  These tests pin the decision and the capacity-weighted
``load`` placement that goes with it.
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.core.router import Router                              # noqa: E402


class _Sched:
    def __init__(self, instance_id, node_id=0, max_num_seqs=16, capacity=0.0,
                 waiting=0, running=0):
        self.instance_id = instance_id
        self.node_id = node_id
        self.max_num_seqs = max_num_seqs
        self.capacity = capacity
        self.waiting = [None] * waiting
        self.running = [None] * running

    @property
    def accepts_new_requests(self):
        return True


def router(**options):
    # The decision helpers only touch the policy options, so a bare instance is
    # enough here (the full constructor needs a live scheduler list).
    instance = Router.__new__(Router)
    instance.local_prefill_mode = str(options.get("local_prefill", "auto")).lower()
    instance.local_prefill_ms_per_1k = float(options.get("local_prefill_ms_per_1k", 93.0))
    instance.transfer_ms_per_1k_local = float(options.get("transfer_ms_per_1k_local", 585.0))
    instance.transfer_ms_per_1k_cross = float(options.get("transfer_ms_per_1k_cross", 1351.0))
    instance.transfer_fixed_ms_cross = float(options.get("transfer_fixed_ms_cross", 48.0))
    instance.local_prefill_queue_weight = float(options.get("local_prefill_queue_weight", 1.0))
    instance.local_prefill_queue_cap = float(options.get("local_prefill_queue_cap", 3.0))
    instance.capacity_tables = {
        "prefill": {int(k): float(v) for k, v in (options.get("prefill_capacity") or {}).items()},
        "decode": {int(k): float(v) for k, v in (options.get("decode_capacity") or {}).items()},
    }
    instance.prefill_rr_counter = 0
    instance.decode_rr_counter = 0
    return instance


class LocalPrefillDecisionTests(unittest.TestCase):
    def test_recompute_wins_for_short_prompts_on_an_idle_decode(self):
        # 93 ms/1k tokens of recompute against 585 ms/1k of same-host handoff.
        use_local, local_ms, transfer_ms = router().kv_exchange_decision(
            _Sched(0, node_id=0), _Sched(1, node_id=0), tokens=300)
        self.assertTrue(use_local)
        self.assertLess(local_ms, transfer_ms)
        self.assertAlmostEqual(transfer_ms, 585.0 * 0.3, places=6)

    def test_cross_domain_transfer_pays_its_fixed_cost(self):
        _, _, transfer_ms = router().kv_exchange_decision(
            _Sched(0, node_id=0), _Sched(1, node_id=9), tokens=1000)
        self.assertAlmostEqual(transfer_ms, 48.0 + 1351.0, places=6)

    def test_a_saturated_decode_can_prefer_the_handoff(self):
        busy = _Sched(1, node_id=0, max_num_seqs=8, waiting=8)
        use_local, local_ms, transfer_ms = router().kv_exchange_decision(
            _Sched(0, node_id=0), busy, tokens=5000)
        # The M/G/1 externality is capped at 3x the recompute cost, so a long
        # prompt against a saturated Decode stops looking free.
        self.assertLessEqual(local_ms, 93.0 * 5.0 * (1 + 3.0) + 1e-6)
        self.assertEqual(use_local, local_ms < transfer_ms)

    def test_the_tables_and_defaults_are_permissive(self):
        # No decode capacity table: the decision still works off max_num_seqs.
        use_local, _, _ = router().kv_exchange_decision(
            _Sched(0), _Sched(1), tokens=64)
        self.assertTrue(use_local)


class CapacityWeightedLoadTests(unittest.TestCase):
    def test_an_idle_tie_resolves_toward_the_larger_capacity(self):
        instance = router(prefill_capacity={0: 63, 2: 45}, decode_capacity={})
        instance.routing_policy = "LOAD"
        schedulers = [_Sched(0, capacity=63), _Sched(2, capacity=45)]
        self.assertEqual(instance._least_load_select(schedulers, "prefill"), 0)

    def test_the_busiest_candidate_loses_regardless_of_capacity(self):
        instance = router(prefill_capacity={0: 63, 2: 45})
        schedulers = [_Sched(0, capacity=63, running=4), _Sched(2, capacity=45)]
        self.assertEqual(instance._least_load_select(schedulers, "prefill"), 1)

    def test_without_a_table_it_falls_back_to_max_num_seqs(self):
        instance = router()
        schedulers = [_Sched(0, max_num_seqs=8), _Sched(2, max_num_seqs=64)]
        # Equal queue, so the larger max_num_seqs wins the tie.
        self.assertEqual(instance._least_load_select(schedulers, "prefill"), 1)


if __name__ == "__main__":
    unittest.main()

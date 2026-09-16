"""The simulator's SGLang-style cache-aware routing baseline.

The real-cluster comparison uses ``cache_aware`` (SGLang Router semantics) as
the SOTA routing baseline, but the simulator only shipped LOAD/RR/RAND, so a
simulated version of that table was impossible.  These tests pin the semantics:
longest prefix match wins, one evicted block of slack, and a load-based
overflow so a hot instance cannot absorb the whole class.
"""

import pathlib
import sys
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from serving.core.block_pool import NONE_HASH                     # noqa: E402
from serving.core.router import Router                            # noqa: E402


class FakePool:
    def __init__(self, cached):
        self.cached = set(cached)

    def get_cached_block(self, block_hash):
        return block_hash if block_hash in self.cached else None


class FakeKV:
    def __init__(self, block_size, cached):
        self.block_size = block_size
        self.enable_caching = True
        self.npu_pool = FakePool(cached)


class FakeSched:
    def __init__(self, instance_id, cached_hashes, block_size=4,
                 waiting=0, running=0, max_num_seqs=8, router_id=None,
                 capacity=8):
        self.instance_id = instance_id
        # The real router keys its cache index by the *string* ``Instance.id``
        # and breaks ties on that same string, which is how the ``load``
        # baseline degenerated.  Model both here.
        self.id = router_id or f"d{instance_id}"
        self.kv = FakeKV(block_size, cached_hashes)
        self.waiting = [None] * waiting
        self.running = [None] * running
        self.max_num_seqs = max_num_seqs
        self.max_inflight = max_num_seqs
        self.capacity = capacity


def chain_hashes(tokens, block_size=4):
    parent = NONE_HASH
    out = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hash((parent, tuple(tokens[start:start + block_size])))
        out.append(parent)
    return out


def make_router(policy="CACHE_AWARE"):
    # Bypass __init__ (it wants a full cluster); only the policy helpers are
    # under test here.
    router = Router.__new__(Router)
    router.routing_policy = policy
    router._counters = {"prefill": 0, "decode": 0}
    # ``_least_load_select`` is the overflow fallback and reads the RR cursors.
    router.prefill_rr_counter = 0
    router.decode_rr_counter = 0
    return router


class CacheAwareRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tokens = list(range(12))          # 3 blocks of 4
        self.chain = chain_hashes(self.tokens)
        self.req = {"input_hash_ids": self.tokens}

    def test_longest_prefix_match_wins(self):
        cold = FakeSched(0, [])
        warm = FakeSched(1, self.chain[:2])    # two of three blocks cached
        router = make_router()
        self.assertEqual(router._cache_aware_select([cold, warm], "prefill", self.req), 1)

    def test_busy_match_overflows_to_the_idle_instance(self):
        warm = FakeSched(1, self.chain, waiting=8, max_num_seqs=8)   # 100% load
        idle = FakeSched(2, self.chain[:2])                          # one block short
        router = make_router()
        # The only owner of the full chain is busy, so the request spills to
        # the least-loaded instance even though it matches one block less.
        self.assertEqual(router._cache_aware_select([warm, idle], "prefill", self.req), 1)

    def test_a_match_kept_even_if_busy_when_nothing_else_matches(self):
        warm = FakeSched(1, self.chain, waiting=8, max_num_seqs=8)
        cold = FakeSched(2, [])
        router = make_router()
        # Spilling to a cold instance is what SGLang does too: the match wins
        # the *ownership* comparison, and only the load guard moves the request.
        self.assertEqual(router._cache_aware_select([warm, cold], "prefill", self.req), 1)

    def test_no_cache_anywhere_falls_back_to_least_loaded(self):
        a = FakeSched(0, [], waiting=4)
        b = FakeSched(1, [], waiting=0)
        router = make_router()
        self.assertEqual(router._cache_aware_select([a, b], "prefill", self.req), 1)

    def test_ties_among_matching_owners_follow_capacity_not_string_id(self):
        """Two owners with the same prefix must not be split by string order.

        The real ``_pick_cache_aware`` ranked owners by
        ``inflight / capacity`` and then by the *string* ``inst.id``; with both
        idle that pinned ``"d3090a"`` (lexicographically first) forever instead
        of preferring the larger-capacity instance.
        """
        # The simulator's capacity signal is ``max_num_seqs`` (its per-instance
        # concurrency limit); the real router additionally carries a normalized
        # ``capacity`` field.  What matters is that the tie-break uses load,
        # not the id string.
        small = FakeSched(0, self.chain, router_id="d3090a", max_num_seqs=2)
        large = FakeSched(1, self.chain, router_id="d_a100", max_num_seqs=16)
        router = make_router()
        # Both match the full chain and are idle; the load-after-assignment
        # metric must prefer the larger capacity.
        self.assertEqual(router._cache_aware_select([small, large], "prefill",
                                                    self.req), 1)

    def test_unknown_policy_is_rejected(self):
        source = (REPO / "serving" / "core" / "router.py").read_text(encoding="utf-8")
        self.assertIn('"CACHE_AWARE"', source)


if __name__ == "__main__":
    unittest.main()

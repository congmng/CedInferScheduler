"""``--shard i/N`` has to split a grid exactly, or the bundle is silently wrong.

The per-shot cost of a profile run is a fixed ~3.5 s of ``torch.profiler``
bookkeeping, so the only way to shorten a 4-hour sweep is to fire it on several
cards at once.  ``--shard`` is that mechanism: shard ``i`` of ``N`` keeps the
shots at positions ``≡ i (mod N)`` of the composed grid.  Two ways this can go
wrong and neither shows up as an error at run time:

* the shards overlap -- one shape is measured twice and another never;
* the shards miss -- the same shape is never fired.

So the property worth pinning is exactly "disjoint and exhaustive", for every N
from 1 to the grid size.
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from profiler.core.shard import apply_shard, parse_shard          # noqa: E402


class ParseShardTests(unittest.TestCase):
    def test_none_means_no_sharding(self):
        self.assertIsNone(parse_shard(None))
        # A single shard is the whole grid, so it is normalized away.
        self.assertIsNone(parse_shard("0/1"))

    def test_it_reads_index_over_total(self):
        self.assertEqual(parse_shard("0/3"), (0, 3))
        self.assertEqual(parse_shard("2/3"), (2, 3))

    def test_bad_shares_are_rejected(self):
        for raw in ("3/3", "1", "a/b", "0/0", "-1/2"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_shard(raw)


class ApplyShardTests(unittest.TestCase):
    def _split(self, shots, total):
        return [apply_shard(list(shots), (index, total))
                for index in range(total)]

    def test_no_shard_returns_the_whole_grid(self):
        self.assertEqual(apply_shard([1, 2, 3], None), [1, 2, 3])

    def test_shards_are_disjoint_and_exhaustive(self):
        shots = list(range(37))
        for total in (1, 2, 3, 4, 5, 8, 37, 64):
            with self.subTest(total=total):
                pieces = self._split(shots, total)
                flat = [shot for piece in pieces for shot in piece]
                self.assertEqual(sorted(flat), shots,
                                 "every shot must be fired exactly once")
                self.assertEqual(len(flat), len(set(flat)),
                                 "no shot may be fired twice")

    def test_a_shard_keeps_its_own_positions(self):
        shots = list(range(10))
        self.assertEqual(apply_shard(list(shots), (0, 3)), [0, 3, 6, 9])
        self.assertEqual(apply_shard(list(shots), (1, 3)), [1, 4, 7])
        self.assertEqual(apply_shard(list(shots), (2, 3)), [2, 5, 8])

    def test_an_empty_grid_stays_empty(self):
        for index in range(3):
            self.assertEqual(apply_shard([], (index, 3)), [])


if __name__ == "__main__":
    unittest.main()

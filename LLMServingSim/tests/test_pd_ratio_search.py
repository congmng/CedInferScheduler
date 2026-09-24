#!/usr/bin/env python3
"""The DistServe/Splitwise-style P:D ratio search (``pd_ratio_search.py``)."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


class TraceReadingTests(unittest.TestCase):
    def _write(self, rows, meta=None):
        directory = tempfile.mkdtemp()
        path = pathlib.Path(directory) / "trace.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                        encoding="utf-8")
        if meta is not None:
            pathlib.Path(str(path) + ".meta.json").write_text(
                json.dumps(meta), encoding="utf-8")
        return path

    def test_output_tokens_are_the_generated_count(self):
        # The trace file's ``output_toks`` is what the model *generates*; the
        # simulator's request carries input + output.  Reading it as a total
        # made a 72-token trace look like a 1-token one.
        from pd_ratio_search import read_trace
        path = self._write([
            {"input_toks": 1000, "output_toks": 64, "arrival_time_ns": 0},
            {"input_toks": 1000, "output_toks": 96, "arrival_time_ns": 1_000_000_000},
        ])
        trace = read_trace(path)
        self.assertAlmostEqual(trace["output_tokens"], 80.0)
        self.assertAlmostEqual(trace["prompt_tokens"], 1000.0)

    def test_the_peak_phase_is_what_a_ratio_is_sized_for(self):
        from pd_ratio_search import read_trace
        path = self._write(
            [{"input_toks": 1024, "output_toks": 32, "arrival_time_ns": index * 1_000_000_000}
             for index in range(10)],
            meta={"phase_rates_rps": [2, 16, 2]})
        trace = read_trace(path)
        self.assertEqual(trace["offered_rps"], 16.0)
        self.assertAlmostEqual(trace["mean_rps"], 10.0 / 9.0, places=3)


class TopologyAndPruningTests(unittest.TestCase):
    def _cluster(self, uniform=True):
        if uniform:
            nodes = [
                {"num_instances": 2, "instances": [
                    {"instance_id": 0, "pd_type": "prefill"},
                    {"instance_id": 1, "pd_type": "decode"}]},
                {"num_instances": 2, "instances": [
                    {"instance_id": 2, "pd_type": "prefill"},
                    {"instance_id": 3, "pd_type": "decode"}]},
            ]
            return {"nodes": nodes, "intra_node_link_bw": 0.257,
                    "casr": {"shared_links": [
                        {"id": "l", "pairs": [[0, 1], [0, 3], [2, 1]]}]}}
        return {"nodes": [{"num_instances": 4, "instances": [
            {"instance_id": 0, "pd_type": "prefill"},
            {"instance_id": 1, "pd_type": "prefill"},
            {"instance_id": 2, "pd_type": "decode"},
            {"instance_id": 3, "pd_type": "decode"}]}],
            "casr": {}}

    def test_a_domain_cluster_only_expresses_n_to_n(self):
        from pd_ratio_search import candidate_ratios, is_node_uniform
        cluster = self._cluster(uniform=True)
        self.assertTrue(is_node_uniform(cluster))
        candidates = candidate_ratios(["1:1", "2:1", "1:2", "2:2"], True, 2, 2, 2)
        self.assertEqual(candidates, [(1, 1), (2, 2)])
        # Without intra-node keys (a single-node arena) any split is allowed.
        arena = self._cluster(uniform=False)
        self.assertFalse(is_node_uniform(arena))
        candidates = candidate_ratios(["1:1", "2:1", "1:2", "2:2"], False, 2, 2, 1)
        self.assertEqual(candidates, [(1, 1), (1, 2), (2, 1), (2, 2)])

    def test_pruning_keeps_the_node_counts_consistent(self):
        from pd_ratio_search import prune_config
        pruned = prune_config(self._cluster(uniform=True), [0], [1])
        self.assertEqual(len(pruned["nodes"]), 1)
        self.assertEqual(pruned["nodes"][0]["num_instances"], 2)
        self.assertEqual(pruned["num_nodes"], 1)
        # A link whose every pair lost an endpoint disappears with them.
        pairs = [pair for link in pruned["casr"]["shared_links"]
                 for pair in link["pairs"]]
        self.assertEqual(pairs, [[0, 1]])


class CandidateScoringTests(unittest.TestCase):
    def _nodes(self):
        return [
            (0, [(0, 10.0)], [(1, 8.0)]),
            (1, [(2, 4.0)], [(3, 2.0)]),
        ]

    def _trace(self, offered=6.0, prompt=1024.0, output=16.0):
        return {"prompt_tokens": prompt, "output_tokens": output,
                "offered_rps": offered}

    def test_more_prefills_raise_a_prefill_bound_goodput(self):
        from pd_ratio_search import evaluate
        # A 4096-token prompt makes the Prefill leg bind (prompt work 4x).
        trace = self._trace(offered=100.0, prompt=4096.0)
        small = evaluate(1, 1, trace, 16.0,
                         self._nodes(), uniform=False)
        large = evaluate(2, 1, trace, 16.0,
                         self._nodes(), uniform=False)
        self.assertGreater(large[0], small[0])
        self.assertEqual(large[4], [(1, 8.0)])

    def test_the_decode_batch_factor_scales_the_decode_leg(self):
        from pd_ratio_search import evaluate
        # 48 output tokens cost 3 reference units of Decode, so Decode binds.
        trace = self._trace(offered=100.0, output=48.0)
        raw = evaluate(1, 1, trace, 16.0,
                       self._nodes(), uniform=False, decode_batch_factor=1.0)
        batched = evaluate(1, 1, trace, 16.0,
                           self._nodes(), uniform=False, decode_batch_factor=4.8)
        self.assertAlmostEqual(batched[2], raw[2] * 4.8, places=6)
        self.assertGreater(batched[0], raw[0])

    def test_a_node_uniform_pool_keeps_whole_nodes(self):
        from pd_ratio_search import evaluate
        goodput, p_cap, d_cap, best_p, best_d = evaluate(
            1, 1, self._trace(offered=6.0), 16.0, self._nodes(), uniform=True)
        # The first node is the stronger one on both legs, so it is the one
        # that stays up.
        self.assertEqual([instance for instance, _ in best_p], [0])
        self.assertEqual([instance for instance, _ in best_d], [1])
        self.assertAlmostEqual(p_cap, 10.0)
        self.assertAlmostEqual(d_cap, 8.0)
        self.assertAlmostEqual(goodput, 6.0)   # the offered rate binds


if __name__ == "__main__":
    unittest.main()

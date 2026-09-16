"""The P/D handoff must cost one RTT per batch, not one per transformer layer.

Measured 2026-09-16 against the live five-domain cluster: a single 32-token
request took 1898 ms of simulated TTFT (real: 50 ms) because the trace put the
KV bytes on *every* ``qkv_proj`` row, the Chakra converter turns each of those
rows into its own SEND/RECV pair, and ASTRA-Sim charges the full link latency to
each one -- 37 x 48 ms = 1.78 s of exposed communication.  The same trace also
handed the decode peer the prefill's whole *logits* tensor (9.7 MB) instead of
the sampled token.
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.core.trace_generator import (                       # noqa: E402
    _aggregate_kv_handoff, _finalize_prefill_handoff)


def row(name, comm_size=0, output_loc="LOCAL", output_size=0, tag="NONE"):
    return (name, "100", "LOCAL", "4096", "LOCAL", "8192", output_loc,
            str(output_size), "NONE", str(comm_size), tag)


class AggregateKvHandoffTests(unittest.TestCase):
    def test_layerwise_kv_becomes_one_send(self):
        rows = [row("layernorm_1"), row("qkv_proj_2", 131072), row("attention_3"),
                row("qkv_proj_4", 131072), row("qkv_proj_5", 131072)]
        out = _aggregate_kv_handoff(rows)
        sizes = [int(r[9]) for r in out]
        self.assertEqual(sum(sizes), 3 * 131072, "the byte total must not change")
        self.assertEqual(sizes.count(393216), 1, "one row carries the whole handoff")
        self.assertEqual(sizes[1], 0)
        self.assertEqual(sizes[-1], 393216, "the handoff stays on the last block")

    def test_a_single_row_is_left_alone(self):
        rows = [row("qkv_proj_2", 131072)]
        self.assertEqual(_aggregate_kv_handoff(rows), rows)

    def test_non_kv_rows_are_untouched(self):
        rows = [row("qkv_proj_2", 100), row("o_proj_3", 999), row("qkv_proj_4", 200)]
        out = _aggregate_kv_handoff(rows)
        self.assertEqual(out[1][9], "999")
        self.assertEqual(int(out[0][9]) + int(out[2][9]), 300)

    def test_sub_batches_keep_their_own_handoff(self):
        rows = [row("qkv_proj_2", 100, tag="BATCH_1"),
                row("qkv_proj_3", 200, tag="BATCH_1"),
                row("qkv_proj_4", 400, tag="BATCH_2"),
                row("qkv_proj_5", 800, tag="BATCH_2")]
        out = _aggregate_kv_handoff(rows)
        by_tag = {}
        for item in out:
            by_tag.setdefault(item[10], []).append(int(item[9]))
        self.assertEqual(sum(by_tag["BATCH_1"]), 300)
        self.assertEqual(sum(by_tag["BATCH_2"]), 1200)
        self.assertEqual(by_tag["BATCH_1"].count(300), 1)
        self.assertEqual(by_tag["BATCH_2"].count(1200), 1)


class PrefillOutputHandoffTests(unittest.TestCase):
    class _Bctx:
        class _Batch:
            requests = (object(), object(), object())

        batch = _Batch()

    def test_logits_are_not_shipped_to_the_decode(self):
        # lm_head routes its output to REMOTE; the converter sizes the
        # stage-boundary SEND from that row's output_size.
        rows = [row("qkv_proj_1", 100), row("qkv_proj_2", 100),
                row("lm_head_3", output_loc="REMOTE:0", output_size=9723904)]
        out = _finalize_prefill_handoff(rows, self._Bctx())
        self.assertEqual(int(out[-1][7]), 4 * 3,
                         "only the sampled tokens cross the wire")
        self.assertEqual(int(out[-1][9]), 0, "lm_head carries no KV")
        self.assertEqual(sum(int(r[9]) for r in out), 200, "KV still aggregated")

    def test_a_batch_without_a_remote_row_is_unchanged(self):
        rows = [row("qkv_proj_1", 100), row("qkv_proj_2", 100)]
        out = _finalize_prefill_handoff(rows, self._Bctx())
        self.assertEqual(sum(int(r[9]) for r in out), 200)


if __name__ == "__main__":
    unittest.main()

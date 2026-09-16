import unittest
import contextlib
import io
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

from tests.aggregate_real_comparison import (row_slo_satisfied, slo_satisfied,
                                             trimmed_mean)
from tests.real_dataset_client import row_slo, slo_ok, sse_update
from tests.real_multidomain_client import prefix_index_for


class SseUpdateTests(unittest.TestCase):
    """The streaming replay path must yield a real TTFT, not a connect time."""

    def test_first_token_wins_even_after_empty_chunks(self):
        state = {"first_token_ms": None, "tokens": 0, "usage": {}}
        state["now_ms"] = 10.0
        sse_update("data: {\"choices\": [{\"text\": \"\"}]}", state)
        self.assertIsNone(state["first_token_ms"])
        state["now_ms"] = 120.5
        sse_update("data: {\"choices\": [{\"text\": \"Hel\"}]}", state)
        self.assertEqual(state["first_token_ms"], 120.5)
        state["now_ms"] = 200.0
        sse_update("data: {\"choices\": [{\"text\": \"lo\"}]}", state)
        # TTFT is the *first* token; later chunks only move the token count.
        self.assertEqual(state["first_token_ms"], 120.5)
        self.assertEqual(state["tokens"], 2)

    def test_usage_and_non_data_lines_are_ignored_or_captured(self):
        state = {"first_token_ms": None, "tokens": 0, "usage": {}}
        sse_update(": keep-alive", state)
        sse_update("", state)
        sse_update("data: [DONE]", state)
        sse_update("data: not-json", state)
        self.assertEqual(state["tokens"], 0)
        sse_update("data: {\"choices\": [], \"usage\": {\"completion_tokens\": 16}}",
                   state)
        self.assertEqual(state["usage"]["completion_tokens"], 16)


class SloTests(unittest.TestCase):
    def test_unmeasured_request_cannot_satisfy_an_slo(self):
        # A non-streaming round has no ttft_ms; counting it as a hit would
        # silently inflate attainment.
        self.assertFalse(slo_satisfied({"total_ms": 100.0}, 500.0, None))
        self.assertTrue(slo_satisfied({"ttft_ms": 100.0}, 500.0, None))
        self.assertFalse(slo_satisfied({"ttft_ms": 600.0}, 500.0, None))

    def test_every_configured_slo_must_hold(self):
        row = {"ttft_ms": 100.0, "tpot_ms": 80.0}
        self.assertTrue(slo_satisfied(row, 500.0, 100.0))
        self.assertFalse(slo_satisfied(row, 500.0, 50.0))

    def test_trimmed_mean_drops_the_transfer_glitches(self):
        values = [100.0] * 99 + [50000.0]
        self.assertLess(trimmed_mean(values, 0.01), 200.0)


class TraceCarriedSloTests(unittest.TestCase):
    """The SLO bound must travel with the request, not be a global constant."""

    def test_trace_row_overrides_the_client_fallback(self):
        args = SimpleNamespace(slo_ttft_ms=500.0, slo_tpot_ms=50.0)
        self.assertEqual(row_slo({"slo_ttft_ms": 300.0}, args), (300.0, 50.0))
        self.assertEqual(row_slo({}, args), (500.0, 50.0))

    def test_no_bound_means_no_constraint(self):
        args = SimpleNamespace(slo_ttft_ms=0.0, slo_tpot_ms=0.0)
        self.assertEqual(row_slo({}, args), (None, None))
        self.assertTrue(slo_ok(None, None, None, None))

    def test_unmeasured_metric_is_unknown_not_a_pass(self):
        self.assertIsNone(slo_ok(None, 10.0, 500.0, None))
        self.assertFalse(slo_ok(600.0, 10.0, 500.0, None))
        self.assertTrue(slo_ok(100.0, 10.0, 500.0, 50.0))

    def test_aggregator_prefers_the_per_request_verdict(self):
        # The router judged the request against its own bound: trust that.
        self.assertTrue(row_slo_satisfied({"slo_ok": True}, 500.0, 50.0))
        self.assertFalse(row_slo_satisfied({"slo_ok": False}, 500.0, 50.0))
        # No recorded verdict -> fall back to the CLI thresholds.
        self.assertTrue(row_slo_satisfied({"ttft_ms": 100.0}, 500.0, None))


class AggregateSloTests(unittest.TestCase):
    """Rounds without a recorded verdict must not be scored as free passes."""

    def _aggregate(self, rounds, *extra):
        from tests import aggregate_real_comparison as agg
        argv = ["aggregate", *map(str, rounds), "--baseline", "rr", *extra]
        buffer = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buffer):
            agg.main()
        output = buffer.getvalue()
        return json.loads(output[output.rindex("\n{\n"):])

    def _write_round(self, directory, rows):
        path = pathlib.Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "metrics-rr.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")

    def test_trace_bound_applies_to_rounds_without_a_recorded_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            new = pathlib.Path(tmp) / "new"
            old = pathlib.Path(tmp) / "old"
            # Round 1: the router recorded a per-request verdict.
            self._write_round(new, [
                {"status": 200, "total_ms": 100.0, "ttft_ms": 100.0, "ts": 0.0,
                 "slo_ttft_ms": 500.0, "slo_ok": True},
                {"status": 200, "total_ms": 900.0, "ttft_ms": 900.0, "ts": 1.0,
                 "slo_ttft_ms": 500.0, "slo_ok": False},
            ])
            # Round 2: measured, but no verdict and no bound on the row.
            self._write_round(old, [
                {"status": 200, "total_ms": 100.0, "ttft_ms": 100.0, "ts": 0.0},
                {"status": 200, "total_ms": 900.0, "ttft_ms": 900.0, "ts": 1.0},
            ])
            summary = self._aggregate([new, old])
        slo = summary["rr"]["slo"]
        self.assertEqual(slo["source"], "per-request")
        self.assertEqual(slo["hits"], 2)          # not 3: the 900 ms row misses
        self.assertEqual(slo["attainment"], 0.5)

    def test_goodput_window_is_summed_per_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = pathlib.Path(tmp) / "r1"
            second = pathlib.Path(tmp) / "r2"
            # Two rounds 1000 s apart: the gap must not become "measurement time".
            self._write_round(first, [
                {"status": 200, "total_ms": 100.0, "ttft_ms": 100.0, "ts": 0.0,
                 "slo_ttft_ms": 500.0, "slo_ok": True},
                {"status": 200, "total_ms": 100.0, "ttft_ms": 100.0, "ts": 10.0,
                 "slo_ttft_ms": 500.0, "slo_ok": True},
            ])
            self._write_round(second, [
                {"status": 200, "total_ms": 100.0, "ttft_ms": 100.0, "ts": 1000.0,
                 "slo_ttft_ms": 500.0, "slo_ok": True},
                {"status": 200, "total_ms": 100.0, "ttft_ms": 100.0, "ts": 1010.0,
                 "slo_ttft_ms": 500.0, "slo_ok": True},
            ])
            summary = self._aggregate([first, second])
        slo = summary["rr"]["slo"]
        self.assertEqual(slo["window_s"], 20.0)
        self.assertAlmostEqual(slo["goodput_rps"], 0.2, places=4)


class PrefixIndexTests(unittest.TestCase):
    def test_clients_share_hot_prefixes_and_drift_to_fresh_ones(self):
        # Two hot prefixes shared by four clients: clients 0/2 use prefix 0 and
        # clients 1/3 use prefix 1, so each hot class carries twice the load of
        # a single client and saturates one Prefill's KV-egress budget.
        self.assertEqual(prefix_index_for(0, 4, 2, 240), 0)
        self.assertEqual(prefix_index_for(1, 4, 2, 240), 1)
        self.assertEqual(prefix_index_for(2, 4, 2, 240), 0)
        self.assertEqual(prefix_index_for(3, 4, 2, 240), 1)
        # After the drift the same clients move to a brand-new slice (2 and 3)
        # that no Prefill has cached yet.
        self.assertEqual(prefix_index_for(240, 4, 2, 240), 2)
        self.assertEqual(prefix_index_for(241, 4, 2, 240), 3)
        self.assertEqual(prefix_index_for(244, 4, 2, 240), 2)

    def test_no_drift_keeps_every_client_on_its_own_prefix(self):
        for index in range(4):
            self.assertEqual(prefix_index_for(index, 4, 4, 10**9), index)


if __name__ == "__main__":
    unittest.main()

"""The replay tool has to compare the two stacks on the same terms.

It reads a recorded real run, runs the matching simulator arms, and prints the
side-by-side table.  These tests pin the parts that make that comparison fair:
the correctness gate's probe requests are excluded, a locally-recomputed
simulator row is attributed to the instance that actually served it, and the
simulator's renumbered instance ids map back to the deployment's names.
"""

import csv
import json
import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import replay_real_run as replay                                   # noqa: E402


class RealSummaryTests(unittest.TestCase):
    def test_probe_requests_are_excluded_and_percentiles_are_reported(self):
        rows = [
            {"request_id": "kvcheck-1", "status": 200, "total_ms": 9999.0,
             "ttft_ms": 9999.0, "tpot_ms": 9999.0, "exchange": "local",
             "decode": "d5090"},
            {"request_id": "1", "status": 200, "total_ms": 100.0, "ttft_ms": 10.0,
             "tpot_ms": 5.0, "exchange": "local", "decode": "d5090"},
            {"request_id": "2", "status": 200, "total_ms": 300.0, "ttft_ms": 30.0,
             "tpot_ms": 15.0, "exchange": "local", "decode": "d4090"},
        ]
        summary = replay.summarise_real(rows)
        self.assertEqual(summary["n"], 2, "the correctness probe is not workload")
        # Upper-middle for an even count, matching the repo's other summarisers.
        self.assertEqual(summary["e2e_p50"], 300.0)
        self.assertEqual(summary["ttft_p50"], 30.0)
        self.assertEqual(summary["exchange"], {"local": 2})
        self.assertEqual(summary["served"], {"d5090": 1, "d4090": 1})


class SimSummaryTests(unittest.TestCase):
    def rows(self):
        # Times are nanoseconds in the simulator's CSV.
        return [
            {"latency": str(300e6), "TTFT": str(40e6), "TPOT": str(15e6),
             "exchange": "local", "prefill_instance_id": "1",
             "decode_instance_id": "5"},
            {"latency": str(500e6), "TTFT": str(60e6), "TPOT": str(20e6),
             "exchange": "transfer", "prefill_instance_id": "4",
             "decode_instance_id": "5"},
        ]

    def test_local_rows_are_attributed_to_the_instance_that_ran_them(self):
        summary = replay.summarise_sim(self.rows(),
                                       {1: "d5090", 4: "p4090", 5: "d4090"})
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["e2e_p50"], 500.0)
        # The locally-computed request names the Decode that served it, not the
        # pair the router recorded; a transferring one names its Decode.
        self.assertEqual(summary["served"], {"d5090": 1, "d4090": 1})
        self.assertEqual(summary["exchange"], {"local": 1, "transfer": 1})


class InstanceNameTests(unittest.TestCase):
    def test_generated_ids_map_back_to_deployment_names(self):
        names = replay.instance_names(
            REPO / "configs" / "cluster" / "casr_real_small3_generated.json")
        self.assertEqual(names, {0: "p5090", 2: "p3090a", 4: "p4090",
                                 1: "d5090", 3: "d3090a", 5: "d4090"})

    def test_policy_flags_mirror_the_real_arms(self):
        self.assertEqual(replay.sim_args("load", 1.0), ["--request-routing-policy", "LOAD"])
        self.assertEqual(replay.sim_args("cache_aware", 1.0),
                         ["--request-routing-policy", "CACHE_AWARE"])
        flags = replay.sim_args("casr_lp", 1.0)
        self.assertIn("--enable-casr", flags)
        self.assertEqual(flags[flags.index("--casr-solver") + 1], "lp")
        self.assertEqual(flags[flags.index("--casr-control-interval-ms") + 1], "1000")


if __name__ == "__main__":
    unittest.main()

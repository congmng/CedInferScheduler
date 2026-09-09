import argparse
import json
import tempfile
import unittest
from pathlib import Path

from workloads.generators.casr import (
    _hot_choice,
    _make_prefix_tokens,
    _next_arrival_ns,
    run,
)


def _namespace(**overrides):
    args = argparse.Namespace(
        output=str(Path(tempfile.mkdtemp()) / "trace.jsonl"),
        num_reqs=12,
        seed=42,
        hotspot_mode="zipf",
        num_prefixes=3,
        reuse_rate=0.8,
        prefix_len=16,
        input_len=32,
        output_len=16,
        vocab_size=1000,
        zipf_exp=1.0,
        drift_from=0,
        drift_to=1,
        sps=10.0,
        arrival_model="poisson",
        first_arrival_sec=0.0,
        burst_size=4,
        burst_gap_sec=10.0,
        burst_poisson=False,
        phase_rates=None,
        phase_durations_sec=None,
        edge_fraction=0.5,
        link_degrade_at_sec=-1.0,
        decode_tier_mode="homogeneous",
        slow_fraction=0.3,
        write_manifest=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class WorkloadGeneratorTests(unittest.TestCase):
    def test_prefix_tokens_are_deterministic_and_distinct(self):
        first = _make_prefix_tokens(0, 16, 1000, 42)
        again = _make_prefix_tokens(0, 16, 1000, 42)
        second = _make_prefix_tokens(1, 16, 1000, 42)
        self.assertEqual(first, again)
        self.assertNotEqual(first, second)

    def test_next_arrival_is_non_decreasing_for_poisson(self):
        rng = __import__("random").Random(42)
        previous = None
        times = []
        args = _namespace(num_reqs=10)
        for index in range(args.num_reqs):
            current = _next_arrival_ns(previous, index, args, rng)
            times.append(current)
            previous = current
        self.assertEqual(times, sorted(times))

    def test_run_emits_required_fields_and_ground_truth_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _namespace(output=str(Path(tmp) / "trace.jsonl"),
                              num_reqs=10, reuse_rate=0.8)
            self.assertEqual(run(args), 0)
            with Path(args.output).open() as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 10)
            for row in rows:
                self.assertEqual(len(row["input_tok_ids"]), args.input_len)
                self.assertEqual(len(row["output_tok_ids"]), args.output_len)
                self.assertIn(row["hotspot_id"], {None, "hotspot-0", "hotspot-1", "hotspot-2"})
            arrivals = [row["arrival_time_ns"] for row in rows]
            self.assertEqual(arrivals, sorted(arrivals))

            manifest_path = Path(args.output + ".meta.json")
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["num_requests"], 10)
            self.assertEqual(len(manifest["prefixes"]), args.num_prefixes)
            self.assertEqual(len(manifest["requests"]), 10)

    def test_drift_mode_moves_from_first_to_second_hotspot(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _namespace(output=str(Path(tmp) / "trace.jsonl"),
                              num_reqs=40, reuse_rate=1.0,
                              hotspot_mode="drift", num_prefixes=2,
                              drift_from=0, drift_to=1)
            self.assertEqual(run(args), 0)
            with Path(args.output + ".meta.json").open() as handle:
                manifest = json.load(handle)
            labels = [item["hotspot_id"] for item in manifest["requests"]]
            self.assertEqual(labels[0], "hotspot-0")
            self.assertEqual(labels[-1], "hotspot-1")

    def test_hot_choice_respects_reuse_rate_zero_and_one(self):
        rng_zero = __import__("random").Random(7)
        rng_one = __import__("random").Random(7)
        args_zero = _namespace(reuse_rate=0.0, hotspot_mode="stable")
        args_one = _namespace(reuse_rate=1.0, hotspot_mode="stable")
        for index in range(20):
            self.assertIsNone(_hot_choice(index, 20, args_zero, rng_zero))
            self.assertEqual(_hot_choice(index, 20, args_one, rng_one), 0)

    def test_phase_schedule_emits_low_high_low_arrivals(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _namespace(
                output=str(Path(tmp) / "phases.jsonl"),
                num_reqs=7,
                phase_rates="2,4,2",
                phase_durations_sec="1,1,1",
                arrival_model="uniform",
            )
            self.assertEqual(run(args), 0)
            with Path(args.output).open() as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([row["phase"] for row in rows[:2]], ["low"] * 2)
            self.assertIn("high", [row["phase"] for row in rows])
            self.assertEqual(rows[-1]["phase"], "low")
            self.assertEqual(rows[0]["arrival_time_ns"], 0)
            self.assertEqual(rows[1]["arrival_time_ns"], 500_000_000)
            self.assertEqual(rows[2]["arrival_time_ns"], 1_000_000_000)
            self.assertEqual(rows[3]["arrival_time_ns"], 1_250_000_000)


if __name__ == "__main__":
    unittest.main()

"""The plan must price the engine it is planning for.

CASR's per-instance service times come from the deployment's ``service_ms``
(measured per request, 16 output tokens, concurrency 8).  The simulator
executes from the profiler bundles, and the two disagree about how much slower
a second card is: the deployment puts ``d4090`` at 1.09x ``d5090`` while the
profiler says 1.74x -- and the cluster's own measured TPOT at 1250-token
prompts (24.6 ms against 14.5 ms, i.e. 1.70x) agrees with the profiler.

Priced with the flat numbers, the LP sent 82 of 452 requests of the
small-cluster elasticity arm to ``d4090`` and its mean latency rose from
1525 ms to 1788 ms.
"""

import json
import os
import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.core.hw_service import rescale_service_times, step_cost_ns  # noqa: E402


class _UnderAstraSim(unittest.TestCase):
    """Profile tables resolve relative to the simulator's cwd (``astra-sim``)."""

    def setUp(self):
        self._cwd = os.getcwd()
        os.chdir(REPO / "astra-sim")

    def tearDown(self):
        os.chdir(self._cwd)


class StepCostTests(_UnderAstraSim):
    def test_a_decode_step_is_a_weight_read(self):
        """One card's step cost ranks the cards the way the cluster does.

        Measured TPOT at 1250-token prompts: d5090 14.5-15.0 ms, d4090
        24.6 ms, d3090a 43-47 ms.
        """
        costs = {hw: step_cost_ns(hw, "Qwen/Qwen3-8B", tp=1, tokens=1)
                 for hw in ("RTX5090", "RTX4090", "RTX3090")}
        self.assertLess(costs["RTX5090"], costs["RTX4090"])
        self.assertLess(costs["RTX4090"], costs["RTX3090"])
        # 5090 -> 4090 measured 1.70x on the cluster, profiled ~1.70x.
        self.assertGreater(costs["RTX4090"] / costs["RTX5090"], 1.5)
        self.assertLess(costs["RTX4090"] / costs["RTX5090"], 1.95)


class RescaleServiceTimesTests(_UnderAstraSim):
    def _instances(self):
        return [
            {"instance_id": 0, "hardware": "RTX5090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 2, "pd_type": "prefill"},
            {"instance_id": 1, "hardware": "RTX5090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "decode"},
            {"instance_id": 3, "hardware": "RTX3090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "decode"},
            {"instance_id": 5, "hardware": "RTX4090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "decode"},
        ]

    def test_the_anchor_keeps_its_measured_value_and_the_rest_follow_the_profiler(self):
        config = {"decode_service_ms": {"1": 144.7, "3": 421.6, "5": 157.1}}
        out = rescale_service_times(config, self._instances(), "decode_service_ms",
                                    tokens=1, verbose=False)
        self.assertEqual(out[1], 144.7)
        self.assertGreater(out[5], 157.1 * 1.4)
        self.assertLess(out[5], 157.1 * 2.0)
        # Ordering now matches the cluster's TPOT: 5090 < 4090 < 3090a.
        self.assertLess(out[1], out[5])
        self.assertLess(out[5], out[3])
        self.assertEqual(sorted(config["decode_service_ms"]), ["1", "3", "5"])

    def test_instances_without_a_profile_keep_their_measured_value(self):
        config = {"decode_service_ms": {"1": 144.7, "9": 200.0}}
        out = rescale_service_times(config, self._instances(), "decode_service_ms",
                                    tokens=1, verbose=False)
        self.assertEqual(out[9], 200.0)

    def test_no_configured_values_is_a_no_op(self):
        config = {}
        self.assertEqual(
            rescale_service_times(config, self._instances(), "decode_service_ms",
                                  verbose=False),
            {})


if __name__ == "__main__":
    unittest.main()

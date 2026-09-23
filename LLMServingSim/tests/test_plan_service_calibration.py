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

from serving.core.hw_service import (attention_step_ns, prefill_charge_ns,
                                     prefill_period_ms,
                                     prefill_pipeline_overhead_ms,
                                     rescale_capacities, rescale_service_times,
                                     step_cost_ns)  # noqa: E402
from serving.core.trace_generator import (DECODE_STEP_SCALE,  # noqa: E402
                                          _load_perf_db, decode_step_scale,
                                          set_timing_calibration)
from serving.core.utils import get_config  # noqa: E402


class _UnderAstraSim(unittest.TestCase):
    """Profile tables resolve relative to the simulator's cwd (``astra-sim``)."""

    def setUp(self):
        self._cwd = os.getcwd()
        os.chdir(REPO / "astra-sim")

    def tearDown(self):
        os.chdir(self._cwd)

    def _instances(self):
        return [
            {"instance_id": 0, "hardware": "RTX5090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "prefill"},
            {"instance_id": 1, "hardware": "RTX5090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "decode"},
            {"instance_id": 2, "hardware": "RTX3090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "prefill"},
            {"instance_id": 3, "hardware": "RTX3090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "decode"},
            {"instance_id": 4, "hardware": "RTX4090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "prefill"},
            {"instance_id": 5, "hardware": "RTX4090", "model_name": "Qwen/Qwen3-8B",
             "tp_size": 1, "pd_type": "decode"},
        ]


class StepCostTests(_UnderAstraSim):
    def test_a_step_cost_matches_the_measured_cluster(self):
        """One card's step cost ranks the cards the way the cluster does.

        Measured TPOT at 1250-token prompts: d5090 14.5-15.0 ms, d4090
        24.6 ms, d3090a 43-47 ms.  ``decode=True`` prices the engine the
        deployment runs: the 0.29.0 bundles give 9.68 / 16.63 / 31.07 ms of
        kernel time, and the Decode-side calibration carries the rest
        (attention, sampling, scheduler, host) at ~1.5x on all three.
        """
        costs = {hw: step_cost_ns(hw, "Qwen/Qwen3-8B", tp=1, tokens=1,
                                  decode=True)
                 for hw in ("RTX5090", "RTX4090", "RTX3090")}
        self.assertLess(costs["RTX5090"], costs["RTX4090"])
        self.assertLess(costs["RTX4090"], costs["RTX3090"])
        # The absolute level is the engine's, not a proxy: each card's
        # calibrated Decode step has to land on the cluster's own TPOT.
        self.assertAlmostEqual(costs["RTX5090"] / 1e6, 14.8, delta=1.0)
        self.assertAlmostEqual(costs["RTX4090"] / 1e6, 24.6, delta=2.0)
        self.assertAlmostEqual(costs["RTX3090"] / 1e6, 45.0, delta=3.0)
        # 5090 -> 4090 measured 1.70x on the cluster, priced ~1.72x.
        self.assertGreater(costs["RTX4090"] / costs["RTX5090"], 1.5)
        self.assertLess(costs["RTX4090"] / costs["RTX5090"], 1.95)

    def test_the_decode_calibration_is_explicit_and_reversible(self):
        """The raw profile is still readable; the factor is one named knob."""
        raw = step_cost_ns("RTX5090", "Qwen/Qwen3-8B", tp=1, tokens=1)
        priced = step_cost_ns("RTX5090", "Qwen/Qwen3-8B", tp=1, tokens=1,
                              decode=True)
        self.assertAlmostEqual(priced / raw, decode_step_scale("RTX5090"),
                               delta=0.02)
        # Cards with no anchor still get a defined (default) factor.
        self.assertEqual(decode_step_scale("H100"), DECODE_STEP_SCALE)

    def test_prefill_cost_scales_with_the_prompt(self):
        short = step_cost_ns("RTX5090", "Qwen/Qwen3-8B", tp=1, tokens=1,
                             decode=True)
        long = step_cost_ns("RTX5090", "Qwen/Qwen3-8B", tp=1, tokens=1024)
        self.assertGreater(long / short, 4.0)
        # Raw profiled prefill on the 0.29.0 bundle (was 107.1 ms on 0.27.1).
        self.assertAlmostEqual(long / 1e6, 68.4, delta=6.0)


class AttentionIsPricedTests(_UnderAstraSim):
    """The capacity model must charge the op that actually fills the step.

    Regression guard for the 2026-09-23 audit (docs/实验数据集与对比基线说明.md
    6.19): ``step_cost_ns`` summed dense + MoE only, on the assumption that
    attention is ~0.01 ms at 1k context.  That is true for a dense model and
    false for a compressed-KV one, and the plan priced one 5090 at 89 req/s of
    prefill against a measured ~13.5 -- so it single-homed all 740 requests of
    the 16 rps peak on that worker and the router stalled behind it.
    """

    def test_attention_dominates_a_compressed_kv_prefill(self):
        """P-15B's attention is most of the step; Qwen3-8B's is a rounding error."""
        p15b = step_cost_ns("RTX5090", "casr/P15B", tp=1, tokens=1024) / 1e6
        qwen = step_cost_ns("RTX5090", "Qwen/Qwen3-8B", tp=1,
                            tokens=1024) / 1e6
        # P-15B: 11.2 ms of dense + MoE + 47.0 ms of attention == 58.1 ms
        # (the dense-only figure the old code produced was 11.2 ms).
        self.assertAlmostEqual(p15b, 58.1, delta=4.0)
        # Qwen3-8B: 68.4 ms dense + 2.9 ms attention == 71.2 ms, i.e. extra
        # attention is under 5% of the step -- why the omission never showed.
        self.assertAlmostEqual(qwen, 71.2, delta=4.0)
        # A compressed model's prefill is no longer "cheap" next to a dense
        # one just because its weights are smaller.
        self.assertLess(p15b, qwen)

    def test_the_block_type_dispatch_is_used_for_every_layer(self):
        """Each layer must be priced against its own attention table.

        Charging the single full-attention table to all 28 layers is what the
        flat walk does when it ignores ``layers_block_type``; the windowed
        (r4/r128) layers are much cheaper, so the dispatched total has to come
        out lower.  This is the same dispatch the timeline uses.
        """
        config = get_config("casr/P15B")
        db = _load_perf_db("RTX5090", "casr/P15B", "bf16", {1},
                           config["model_type"])
        architecture = db["architecture"]
        dispatched = attention_step_ns(db, 1, config, architecture, 1024)
        full_only = attention_step_ns(
            db, 1, {**config, "layers_block_type": None}, architecture, 1024)
        self.assertGreater(dispatched, 0)
        self.assertGreater(full_only, dispatched)
        # All-full is ~222 ms against ~47 ms dispatched on this bundle.
        self.assertGreater(full_only / dispatched, 2.0)

    def test_capacity_and_step_are_one_statement(self):
        """The priced capacity is the reciprocal of the priced *period*.

        The plan's constraint and its cost have to describe the same machine,
        and the number has to land on what the timeline executes: the P-15B
        peak ran p4 (RTX5090) at ~13.5 req/s of 1024-token prompts.
        """
        config = {"decode_reference_tokens": 16, "capacity_reference_tokens": 1024}
        prefill, decode = rescale_capacities(config, self._instances(),
                                             verbose=False)
        period_ms = prefill_period_ms("RTX5090", "Qwen/Qwen3-8B", tp=1,
                                      tokens=1024, reference=1024) 
        self.assertAlmostEqual(prefill[0], 1000.0 / period_ms, delta=0.05)
        # The period is the charge plus the measured per-step overhead, and the
        # charge is what the layer profile says.
        charge_ms = prefill_charge_ns("RTX5090", "Qwen/Qwen3-8B", tp=1,
                                      tokens=1024) / 1e6
        self.assertAlmostEqual(period_ms - charge_ms,
                               prefill_pipeline_overhead_ms("RTX5090"), delta=0.01)
        # Decode is untouched by the attention term: the calibrated 1-token
        # step already carries it, so its capacity must stay at what the TPOT
        # anchor implies.
        decode_ms = step_cost_ns("RTX5090", "Qwen/Qwen3-8B", tp=1, tokens=1,
                                 decode=True) / 1e6
        self.assertAlmostEqual(decode[1], 1000.0 / (16 * decode_ms), delta=0.05)

    def test_the_compressed_kv_capacity_lands_on_the_measured_pool(self):
        """p4's priced prefill capacity is within 30% of the measured 13.5."""
        config = {"decode_reference_tokens": 16, "capacity_reference_tokens": 1024}
        prefill, _ = rescale_capacities(config, self._p15b_instances(), verbose=False)
        self.assertGreater(prefill[4], 13.5 * 0.7)
        self.assertLess(prefill[4], 13.5 * 1.3)
        # The dense-only figure (89.4) is what made the plan single-home.
        self.assertLess(prefill[4], 30.0)

    def _p15b_instances(self):
        return [
            {"instance_id": 4, "hardware": "RTX5090", "model_name": "casr/P15B",
             "tp_size": 1, "pd_type": "prefill"},
            {"instance_id": 5, "hardware": "RTX5090", "model_name": "casr/P15B",
             "tp_size": 1, "pd_type": "decode"},
        ]


class RescaleServiceTimesTests(_UnderAstraSim):

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


class CapacityTests(_UnderAstraSim):
    """Capacities are requests/s of reference-length work, at engine speed."""

    def test_capacity_comes_from_the_profiled_step(self):
        config = {"decode_reference_tokens": 16, "capacity_reference_tokens": 1024}
        prefill, decode = rescale_capacities(config, self._instances(), verbose=False)
        # One 5090: 16 reference output tokens x ~14.8 ms per step.
        self.assertAlmostEqual(decode[1], 1000.0 / (16 * 14.84), delta=0.3)
        # A 1024-token prefill charges ~68 ms of layer time on the 0.29.0
        # bundle and the pair executes it in ~84 ms (charge + one step of
        # measured overhead), so ~11.9 reference requests/s.
        self.assertAlmostEqual(prefill[0], 1000.0 / 84.0, delta=0.6)
        self.assertLess(decode[5], decode[1])
        # Slower card, lower capacity: 3090a (2) < 4090 (4) < 5090 (0).
        self.assertLess(prefill[2], prefill[4])
        self.assertLess(prefill[4], prefill[0])
        self.assertEqual(sorted(config["prefill_capacity"]), ["0", "2", "4"])
        self.assertEqual(sorted(config["decode_capacity"]), ["1", "3", "5"])


if __name__ == "__main__":
    unittest.main()

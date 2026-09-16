"""Hybrid-attention KV geometry: fewer full layers, windowed ones capped.

A hybrid model (sliding-window / linear-attention layers mixed with full
attention) keeps a much smaller KV footprint on the same kernels.  The
simulator expresses that with ``kv_geometry`` in the model config: the first
``full_layers`` layers keep a normal cache, the rest hold at most
``window_tokens``.  ``profile_model`` keeps the kernels coming from the
profiled checkpoint, so this is a KV-geometry study rather than a re-profile.

Measured on the six-domain arena at a 1250-token prompt: the hybrid variant
(9 full layers + 27 x 256) moves 74 MB per handoff against 184 MB, which raised
the per-Prefill push capacity from 1.41 to 3.50 req/s and cut TTFT by ~30%.
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.core.memory_model import (MemoryModel,               # noqa: E402
                                       full_cluster_kv_bytes_per_token)

FULL = "Qwen/Qwen3-8B"
HYBRID = "Qwen/Qwen3-8B-hybrid"


def _model(name):
    return MemoryModel(name, instance_id=0, node_id=0, num_npus=1, tp_size=1,
                       npu_mem=32, cpu_mem=0, block_size=16, fp=16,
                       enable_prefix_caching=True, enable_prefix_sharing=False,
                       prefix_pool=None, prefix_storage=None)


class HybridKvGeometryTests(unittest.TestCase):
    def test_a_full_attention_model_is_unchanged(self):
        self.assertEqual(full_cluster_kv_bytes_per_token(FULL, 16, "auto"), 147456)
        self.assertEqual(
            full_cluster_kv_bytes_per_token(FULL, 16, "auto", tokens=1250), 147456)

    def test_windowed_layers_only_pay_for_their_window(self):
        # 9 full layers x 1250 tokens + 27 windowed x 256 = 18162 token-slots
        # against 36 x 1250 = 45000 for the full-attention model.
        average = full_cluster_kv_bytes_per_token(HYBRID, 16, "auto", tokens=1250)
        self.assertAlmostEqual(average, 147456 * 18162 / 45000, delta=64)
        # Below the window the marginal cost is the full-attention one, which is
        # what a block pool has to reserve for.
        self.assertEqual(
            full_cluster_kv_bytes_per_token(HYBRID, 16, "auto"), 147456)

    def test_the_memory_model_uses_the_same_arithmetic(self):
        hybrid = _model(HYBRID)
        self.assertEqual(hybrid.kv_tokens(1250), 9 * 1250 + 27 * 256)
        self.assertEqual(hybrid.kv_tokens(128), 36 * 128)   # below the window
        full = _model(FULL)
        self.assertAlmostEqual(
            hybrid.pd_kv_bytes(1250) / full.pd_kv_bytes(1250), 0.4036, delta=0.005)
        self.assertLess(hybrid.pd_kv_bytes(1250), full.pd_kv_bytes(1250))


if __name__ == "__main__":
    unittest.main()

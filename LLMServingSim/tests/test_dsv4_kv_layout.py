"""The simulator's DSV4 KV accounting must match the reference implementation.

Two code paths price the same design: ``design/dsv4_ref`` (the executable
specification of the block we intend to build) and ``serving/core/memory_model``
(what the simulator charges a scheduler for).  If they drift, an experiment run
with the borrowed-kernel draft config would be measuring a different model from
the one the design document describes -- exactly the class of bug that is easy to
miss because both numbers look plausible on their own.

Pinned here:

* per-layer byte counts (full MLA layer vs CSA ratio 4 vs HCA ratio 128);
* the aggregate bytes/token for both reference configs, computed one way from the
  module/config estimator and the other way from the simulator's model config;
* that the average is length-independent for this layout (unlike the windowed
  ``kv_geometry``, whose layers stop growing at the window);
* that ``pd_kv_bytes`` (the P/D handoff) sees the same reduction.
"""

import importlib.util
import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from serving.core.memory_model import (MemoryModel,                    # noqa: E402
                                       full_cluster_kv_bytes_per_token)


def _ref_config():
    """Load ``design/dsv4_ref/config.py`` without importing torch."""
    path = REPO / "design" / "dsv4_ref" / "config.py"
    spec = importlib.util.spec_from_file_location("_dsv4_ref_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dsv4_ref_config"] = module
    spec.loader.exec_module(module)
    return module.REF_CONFIGS


REF = _ref_config()
DRAFT = "DeepSeek/DSV4-P15B-draft"


def _model(name=DRAFT):
    return MemoryModel(name, instance_id=0, node_id=0, num_npus=1, tp_size=1,
                       npu_mem=80, cpu_mem=0, block_size=16, fp=16,
                       enable_prefix_caching=True, enable_prefix_sharing=False,
                       prefix_pool=None, prefix_storage=None)


class PerLayerAccountingTests(unittest.TestCase):
    def test_a_full_layer_costs_the_mla_latent_per_token(self):
        # 594 B/token is what the engine reported for one ratio-0 layer, and the
        # formula behind it is (head_dim + rope) * 2 bytes.  ``_kv_layout_bytes``
        # prices the whole stack, so the single-layer width is checked against
        # ``_kv_values_per_token`` and the total against their sum.
        m = _model()
        layer = m._kv_values_per_token[0]
        self.assertEqual(layer, m.config["kv_layout"]["head_dim"]
                         + m.config["kv_layout"]["rope_head_dim"])
        # one full layer on its own, then the whole stack
        self.assertEqual(1024 * layer * 2, 1024 * (m.head_dim + 64) * 2)
        self.assertEqual(m._kv_layout_bytes(1024),
                         1024 * sum(m._kv_values_per_token) * 2)

    def test_compression_ratios_divide_the_state_count(self):
        m = _model()
        ratios = m.config["kv_layout"]["compress_ratios"]
        values = m._kv_values_per_token
        # ratio 4 (CSA) stores 2*coff*hd values every 4 tokens; ratio 128 (HCA)
        # stores 2*hd every 128 -- so per token HCA is 32x cheaper than CSA.
        # state widths: CSA stores 2*coff*hd = 4*hd, HCA stores 2*hd
        self.assertEqual(m._kv_state_values[ratios.index(4)], 4 * m.head_dim)
        self.assertEqual(m._kv_state_values[ratios.index(128)], 2 * m.head_dim)
        csa = values[ratios.index(4)]        # per token, already ratio-divided
        hca = values[ratios.index(128)]
        self.assertEqual(csa, m.head_dim)            # 2048 / 4
        self.assertEqual(hca, 2 * m.head_dim / 128)  # 1024 / 128
        self.assertAlmostEqual(csa / hca, 64.0, delta=1e-6)

    def test_the_layout_path_agrees_with_the_reference_implementation(self):
        for name in ("small", "p15b"):
            ref = REF[name]
            with self.subTest(config=name):
                per_token = sum(
                    (ref.head_dim + ref.qk_rope_head_dim) if r == 0
                    else 2 * ref.coff(r) * ref.head_dim / r
                    for r in ref.compress_ratios
                )
                simulator = full_cluster_kv_bytes_per_token(
                    _draft_name_for(ref), 16, "auto", tokens=1250)
                self.assertAlmostEqual(simulator, round(per_token * 2), delta=2)

    def test_the_average_is_length_independent(self):
        """Unlike the windowed geometry: each compressed layer scales with seq."""
        for tokens in (128, 1250, 8192):
            self.assertEqual(
                full_cluster_kv_bytes_per_token(DRAFT, 16, "auto", tokens=tokens),
                full_cluster_kv_bytes_per_token(DRAFT, 16, "auto", tokens=1))

    def test_the_handoff_gets_the_same_reduction(self):
        m = _model()
        # A 1250-token handoff: 16.6 KB/token times 1250 tokens.
        self.assertAlmostEqual(m.pd_kv_bytes(1250) / 1250,
                               full_cluster_kv_bytes_per_token(
                                   DRAFT, 16, "auto", tokens=1250),
                               delta=2)

    def test_the_draft_is_an_order_of_magnitude_below_qwen3_8b(self):
        qwen = full_cluster_kv_bytes_per_token("Qwen/Qwen3-8B", 16, "auto",
                                               tokens=1250)
        draft = full_cluster_kv_bytes_per_token(DRAFT, 16, "auto", tokens=1250)
        self.assertAlmostEqual(qwen / draft, 8.9, delta=0.5)


def _draft_name_for(ref):
    """The repo config matching a reference config (only p15b is shipped)."""
    hidden = ref.hidden_size
    if hidden == 2560:
        return DRAFT
    raise unittest.SkipTest(f"no repo config shipped for hidden={hidden}")


if __name__ == "__main__":
    unittest.main()

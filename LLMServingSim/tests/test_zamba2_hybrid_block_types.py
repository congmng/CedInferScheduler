"""A hybrid with *different layer shapes*, not just different KV geometry.

The synthetic ``Qwen/Qwen3-8B-hybrid`` config exercises ``kv_geometry`` alone:
every layer runs the same Qwen3 kernel and only the KV footprint differs.  A
real hybrid is not like that -- Zamba2-1.2B instantiates 32 Mamba-only decoders
and 6 shared-attention+Mamba blocks, so the two kinds of layer do *not* cost the
same and the attention kernel runs on 6 layers, not 38.

This pins the three pieces that make that work:

* ``layers_block_type`` (HF's per-layer list) selects a pipeline from the
  architecture yaml's new ``layer_types`` table,
* ``kv_geometry.full_layer_indices`` names the KV-bearing layers, which are
  interleaved rather than first,
* the size/weight helpers know the Zamba2-only layer names (``mamba_norm``,
  ``mamba_mixer``, ``attention_norm``, ``shared_linear``) and the
  ``attention_head_dim`` / ``attention_hidden_size`` spellings.

Numbers come from the checkpoint header (``model.safetensors``,
1 215 064 704 parameters): a Mamba layer is 25.85 M parameters, the attention
pathway is 4096 wide on a 2048 hidden state, and only layer 5 stores the
transformer proper -- the other five hybrids reach it through adapters.
"""

import csv
import json
import pathlib
import sys
import tempfile
import unittest

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.core.memory_model import (MemoryModel, calculate_sizes,   # noqa: E402
                                       get_config)
from serving.core.trace_generator import (                            # noqa: E402
    BatchCtx, TraceCtx, _block_type_for, _build_transformer_block, _can_copy_blocks,
    _is_windowed_layer, _type_sequence)

MODEL = "Zyphra/Zamba2-1.2B"
ARCH_PATH = REPO / "profiler" / "models" / "zamba2.yaml"
CONFIG = get_config(MODEL)

DENSE_LAYERS = ["embedding", "mamba_norm", "mamba_mixer", "attention_norm",
                "qkv_proj", "rotary_emb", "shared_linear", "o_proj",
                "gate_up_proj", "act_fn", "down_proj", "final_layernorm"]


def _write_bundle(root: pathlib.Path) -> None:
    tp1 = root / "tp1"
    tp1.mkdir(parents=True, exist_ok=True)
    with (tp1 / "dense.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "tokens", "time_us"])
        for name in DENSE_LAYERS:
            for tokens, us in ((1, 1.0), (16, 4.0), (1024, 200.0), (4096, 800.0)):
                writer.writerow([name, tokens, us])
    with (tp1 / "per_sequence.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "sequences", "time_us"])
        for seqs, us in ((1, 145.0), (16, 200.0), (256, 260.0)):
            writer.writerow(["lm_head", seqs, us])
    with (tp1 / "attention.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["prefill_chunk", "kv_prefill", "n_decode",
                         "kv_decode", "time_us"])
        writer.writerow([0, 0, 1, 128, 8.0])
        writer.writerow([0, 0, 16, 128, 12.0])
        writer.writerow([0, 0, 16, 1024, 20.0])
        writer.writerow([1024, 0, 0, 0, 60.0])
        writer.writerow([1024, 128, 16, 128, 90.0])


def _perf_db(root: pathlib.Path) -> dict:
    arch = yaml.safe_load(ARCH_PATH.read_text(encoding="utf-8"))
    return {
        "meta": {}, "architecture": arch, "variant": "bf16",
        "hardware": "RTX4090", "model": MODEL, "root": str(root),
        "available_tps": [1], "tables": {},
    }


def _ctx(root: pathlib.Path) -> TraceCtx:
    placement = {"default": {"weights": "LOCAL", "kv_loc": "LOCAL",
                             "kv_evict_loc": "LOCAL"},
                 "block": {}, "layer": {}}
    return TraceCtx(
        hardware="RTX4090", model=MODEL, config=CONFIG, perf_db=_perf_db(root),
        node_id=0, fp=2, placement=placement, gate=None,
        enable_attn_offloading=False,
        power_model=None, pim_model=None, pim_channels=0,
        n_head=CONFIG["num_attention_heads"],
        kv_head=CONFIG["num_key_value_heads"],
        head_dim=CONFIG["attention_head_dim"], is_moe=False, kv_fp=2,
        pd_type="prefill", tp_size=1, pp_size=1, local_ep=1, ep_total=1,
        tp_dim=None, ep_dim=None, dp_sum_total_len=0,
    )


class _Batch:
    requests = (object(), object())
    num_prefill = 1
    num_decode = 1
    prefill_q_list = [64]
    prefill_k_list = [0]
    decode_k_list = [1024]
    batch_id = 7


def _bctx() -> BatchCtx:
    return BatchCtx(_Batch(), total_len=1088, prefill_chunk=64, kv_prefill=0,
                    n_decode=1, kv_decode_mean=1024, kv_decode_max=1024,
                    kv_decode_min=1024, lm_head_len=2, decode_lens=None,
                    channel_split=0)


def _model() -> MemoryModel:
    return MemoryModel(MODEL, instance_id=0, node_id=0, num_npus=1, tp_size=1,
                       npu_mem=32, cpu_mem=0, block_size=16, fp=16,
                       enable_prefix_caching=True, enable_prefix_sharing=False,
                       prefix_pool=None, prefix_storage=None)


class BlockTypeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        _write_bundle(self.root)
        self.ctx = _ctx(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_config_names_a_type_per_layer(self):
        names = CONFIG["layers_block_type"]
        self.assertEqual(len(names), CONFIG["num_hidden_layers"])
        self.assertEqual(names.count("hybrid"), 6)
        self.assertEqual(names.count("mamba"), 32)
        self.assertEqual([i for i, n in enumerate(names) if n == "hybrid"],
                         [5, 11, 17, 23, 29, 35])

    def test_one_block_may_not_stand_in_for_a_hybrids_layers(self):
        """The trace may only copy block 0 when the layers are interchangeable.

        ``_synthesize_trace`` builds one transformer block and repeats it
        ``num_hidden_layers`` times, which is what makes trace generation
        cheap.  For a model that declares ``layers_block_type`` that copy is
        wrong -- layer 0's pipeline is not the others' -- and it silently
        prices every layer as layer 0: the per-layer pipelines and the
        per-block-type attention tables both go unused.  The placement half of
        the same rule (``block_mode_on``) was already checked; this is the
        shape half.
        """
        self.assertFalse(_can_copy_blocks(self.ctx, False))
        self.assertFalse(_can_copy_blocks(self.ctx, True))

    def test_a_uniform_model_may_still_copy(self):
        """No ``layer_types`` in the yaml -> every layer is the same -> copy."""
        ctx = _ctx(self.root)
        ctx.perf_db["architecture"].pop("layer_types", None)
        self.assertTrue(_can_copy_blocks(ctx, False))

    def test_each_type_selects_its_own_pipeline(self):
        self.assertEqual(_block_type_for(self.ctx, 0), "mamba")
        self.assertEqual(_block_type_for(self.ctx, 5), "hybrid")
        self.assertEqual(_type_sequence(self.ctx, 0, "pre_attn", ["x"]),
                         ["mamba_norm", "mamba_mixer"])
        # A Mamba layer has no attention and no FFN of its own.
        self.assertEqual(_type_sequence(self.ctx, 0, "post_attn", ["x"]), [])
        self.assertEqual(_type_sequence(self.ctx, 0, "mlp_dense", ["x"]), [])
        # A hybrid layer carries the shared attention decoder plus the Mamba
        # decoder it is fused with.
        self.assertEqual(_type_sequence(self.ctx, 5, "pre_attn", ["x"]),
                         ["attention_norm", "qkv_proj", "rotary_emb",
                          "attention", "o_proj"])
        self.assertIn("mamba_mixer", _type_sequence(self.ctx, 5, "post_attn", []))
        self.assertEqual(_type_sequence(self.ctx, 5, "mlp_dense", []),
                         ["gate_up_proj", "act_fn", "down_proj"])

    def test_an_untyped_config_falls_back_to_the_flat_sequence(self):
        ctx = _ctx(self.root)
        ctx.config = dict(CONFIG)
        ctx.config.pop("layers_block_type")
        self.assertIsNone(_block_type_for(ctx, 5))
        self.assertEqual(_type_sequence(ctx, 5, "pre_attn", ["fallback"]),
                         ["fallback"])

    def test_mamba_layers_emit_two_ops_and_hybrid_layers_eleven(self):
        mamba_rows, _ = _build_transformer_block(self.ctx, _bctx(), 0, "NONE", "0")
        hybrid_rows, _ = _build_transformer_block(self.ctx, _bctx(), 5, "NONE", "0")
        self.assertEqual([r[0] for r in mamba_rows],
                         ["mamba_norm", "mamba_mixer"])
        self.assertEqual([r[0] for r in hybrid_rows],
                         ["attention_norm", "qkv_proj", "rotary_emb",
                          "attention", "o_proj", "shared_linear",
                          "mamba_norm", "mamba_mixer",
                          "gate_up_proj", "act_fn", "down_proj"])
        self.assertEqual(len(set(r[0] for r in hybrid_rows)), 11)

    def test_one_full_pass_runs_attention_six_times_not_38(self):
        attention_rows = 0
        for layer in range(CONFIG["num_hidden_layers"]):
            rows, _ = _build_transformer_block(self.ctx, _bctx(), layer, "NONE", "0")
            attention_rows += sum(1 for r in rows if r[0] == "attention")
        self.assertEqual(attention_rows, 6)

    def test_only_the_hybrid_layers_grow_their_kv(self):
        self.assertFalse(_is_windowed_layer(self.ctx, 5))
        self.assertFalse(_is_windowed_layer(self.ctx, 35))
        self.assertTrue(_is_windowed_layer(self.ctx, 4))
        self.assertTrue(_is_windowed_layer(self.ctx, 6))


class Zamba2SizingTests(unittest.TestCase):
    def test_head_dim_and_attention_width_come_from_the_zamba2_keys(self):
        # attention_head_dim=128 on 32 heads in a 2048-wide hidden state.
        _, qkv_w, qkv_out = calculate_sizes(MODEL, "qkv_proj", 1, parallel=1, fp=2)
        self.assertEqual(qkv_out, (4096 + 2 * 4096) * 2)
        self.assertEqual(qkv_w, 4096 * 12288 * 2)

    def test_the_mamba_mixer_is_priced_from_the_mamba_keys(self):
        _, mixer_w, _ = calculate_sizes(MODEL, "mamba_mixer", 1, parallel=1, fp=2)
        # in_proj (2048 x 8512) + conv1d (4 x 4352) + out_proj (4096 x 2048).
        expected = (2048 * 8512 + 4 * 4352 + 4096 * 2048) * 2
        self.assertEqual(mixer_w, expected)
        self.assertAlmostEqual(mixer_w / 1e6, 51.68, delta=0.05)

    def test_a_mamba_layer_matches_the_checkpoint_header(self):
        mamba = _model()
        # ``_get_weight_per_layer_type`` returns *bytes*, the header counts
        # parameters, so compare at 2 bytes per bf16 element.
        per_layer = mamba._get_weight_per_layer_type("mamba", 1, 1, 2) / 2
        self.assertAlmostEqual(per_layer / 1e6, 25.85, delta=0.05)
        hybrid = mamba._get_weight_per_layer_type("hybrid", 1, 1, 2) / 2
        self.assertGreater(hybrid, per_layer)
        # 32 Mamba + 6 hybrid is the checkpoint's 1.2B to within the documented
        # adapter-sharing approximation (the trace's total is an over-count).
        self.assertAlmostEqual(
            (32 * per_layer + 6 * hybrid) / 1e9, 1.66, delta=0.02)

    def test_the_kv_footprint_is_six_layers_plus_a_constant_state(self):
        mamba = _model()
        self.assertEqual(
            json.loads(json.dumps(CONFIG["kv_geometry"]))["full_layer_indices"],
            [5, 11, 17, 23, 29, 35])
        self.assertEqual(mamba.kv_tokens(1250), 6 * 1250 + 32 * 65)
        # Below the window the Mamba state is still one slot per token.
        self.assertEqual(mamba.kv_tokens(16), 38 * 16)
        # 6 x 4 KB/token (kv_dim 4096, 2 bytes, K+V) against Qwen3-8B's
        # 36 x 4 KB (kv_dim 1024): 0.67x, not the 0.17x the layer count implies.
        self.assertAlmostEqual(mamba.pd_kv_bytes(1250) / 184320000, 0.85, delta=0.02)


if __name__ == "__main__":
    unittest.main()

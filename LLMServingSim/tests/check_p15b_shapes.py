#!/usr/bin/env python3
"""P-15B step-1 gate: the module tree must equal the spec's parameter counts.

The plan (``docs/P15B-profile路径B实现计划.md``) makes this the main guard
against a mis-wired model, because a wrong shape does not crash anything --
it produces a profile bundle that looks plausible and is wrong.  Three
independent statements have to agree for each block type:

1. ``deploy/vllm_p15b`` built from the shipped config;
2. ``design/dsv4_ref`` built from the same single-ratio config;
3. the per-layer totals already reconciled against the 28-layer module tree in
   ``docs/DSV4-P15B设计.md`` section 4.

    docker run --rm --entrypoint python3 -v "$PWD":/work -w /work \
        vllm/vllm-openai:casr029 tests/check_p15b_shapes.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.vllm_p15b.config import P15BConfig                 # noqa: E402
from deploy.vllm_p15b.model import P15BForCausalLM             # noqa: E402
from design.dsv4_ref.config import DSV4RefConfig               # noqa: E402
from design.dsv4_ref.model import DSV4RefModel                 # noqa: E402

#: Per-layer totals, reconciled against the 28-layer module tree
#: (3x497,383,616 + 13x503,365,824 + 12x500,415,680 + embed + head + norm
#:  = 14,376,441,600).  Hardcoded on purpose: deriving them from the model
#: under test would make the gate agree with any bug.
EXPECTED = {
    "r0": 497_383_616,
    "r4": 503_365_824,
    "r128": 500_415_680,
}


def _decoder_params(model) -> int:
    """Everything except the embedding, the head and the final norm.

    Both spellings of the embedding are listed because the two models
    disagree: ``deploy/vllm_p15b`` follows vLLM (``embed_tokens``) while
    ``design/dsv4_ref`` uses ``embed``.  Missing one silently leaves the whole
    167,772,160-parameter table in the comparison.
    """
    total = sum(p.numel() for p in model.parameters())
    for name in ("embed_tokens", "embed", "lm_head", "norm"):
        module = getattr(model, name, None)
        if module is not None:
            total -= sum(p.numel() for p in module.parameters())
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default="configs/model/casr")
    parser.add_argument("--skip-parity", action="store_true",
                        help="skip the weight-transfer / logits comparison")
    args = parser.parse_args()

    failures = []
    for block, expected in EXPECTED.items():
        path = REPO / args.config_root / f"P15B-{block}.json"
        hf = json.loads(path.read_text(encoding="utf-8"))
        ratio = hf["compress_ratios"][0]

        ours = P15BForCausalLM(P15BConfig.from_hf(hf))
        our_params = _decoder_params(ours)

        ref_cfg = DSV4RefConfig(
            hidden_size=hf["hidden_size"], num_hidden_layers=1,
            num_attention_heads=hf["num_attention_heads"], head_dim=hf["head_dim"],
            qk_rope_head_dim=hf["qk_rope_head_dim"], q_lora_rank=hf["q_lora_rank"],
            o_lora_rank=hf["o_lora_rank"], o_groups=hf["o_groups"],
            index_head_dim=hf["index_head_dim"], index_topk=hf["index_topk"],
            n_routed_experts=hf["n_routed_experts"],
            num_experts_per_tok=hf["num_experts_per_tok"],
            moe_intermediate_size=hf["moe_intermediate_size"],
            vocab_size=hf["vocab_size"], compress_ratios=[int(ratio)])
        reference = _decoder_params(DSV4RefModel(ref_cfg))

        ok = our_params == reference == expected
        print(f"{block:<5} ratio={ratio:<4} ours={our_params:,} "
              f"reference={reference:,} expected={expected:,} "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(block)

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nall three block types match the spec")

    if not args.skip_parity:
        if not _parity(args.config_root):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Parity: same weights in, same logits out
# ---------------------------------------------------------------------------

#: reference ``state_dict`` key -> our key.  Mechanical, and deliberately
#: exhaustive: a key that is not listed shows up as "unmapped" below rather
#: than being silently left at its random initialisation.
_KEY_MAP = (
    ("embed.weight", "embed_tokens.weight"),
    ("norm.weight", "norm.weight"),
    ("lm_head.weight", "lm_head.weight"),
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("attn.qkv_down.weight", "self_attn.qkv_down.proj.weight"),
    ("attn.q_norm.weight", "self_attn.q_norm.weight"),
    ("attn.kv_norm.weight", "self_attn.kv_norm.weight"),
    ("attn.q_up.weight", "self_attn.q_up.proj.weight"),
    ("attn.o_lora_b.weight", "self_attn.o_lora_b.proj.weight"),
    ("attn.compressor.norm.weight", "self_attn.compressor.norm.weight"),
    ("attn.compressor.proj.weight", "self_attn.compressor.proj.weight"),
    ("attn.indexer.q_proj.weight", "self_attn.indexer.q_proj.weight"),
    ("attn.indexer.k_proj.weight", "self_attn.indexer.k_proj.weight"),
    ("attn.kv_state_proj.weight", "self_attn.kv_state_proj.proj.weight"),
    ("ffn.gate.weight", "mlp.gate.weight"),
    ("ffn.up", "mlp.up"),
    ("ffn.down", "mlp.down"),
)

#: Groups that carry an index inside the name.
_GROUP_MAP = (
    ("attn.o_lora_a.{i}.weight", "self_attn.o_lora_a.{i}.proj.weight"),
)


def _translate(reference_key: str) -> str | None:
    """Reference key -> our key, or None when nothing matches."""
    for suffix, ours in _KEY_MAP:
        if reference_key == suffix:
            return ours
    for suffix, ours in _GROUP_MAP:
        head, _, tail = suffix.partition(".{i}.")
        if reference_key.startswith(head + ".") and reference_key.endswith("." + tail):
            index = reference_key[len(head) + 1:-len(tail) - 1]
            if index.isdigit():
                return ours.format(i=index)
    if reference_key.startswith("layers."):
        layer, _, rest = reference_key[len("layers."):].partition(".")
        mapped = _translate(rest)
        return f"layers.{layer}.{mapped}" if mapped else None
    return None


def _build_pair(config_path, dtype, seed=0):
    """Our model and the reference model, same config, same weights."""
    import torch

    hf = json.loads(config_path.read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    ours = P15BForCausalLM(P15BConfig.from_hf(hf)).to(dtype).eval()

    ref_cfg = DSV4RefConfig(
        hidden_size=hf["hidden_size"],
        num_hidden_layers=hf["num_hidden_layers"],
        num_attention_heads=hf["num_attention_heads"], head_dim=hf["head_dim"],
        qk_rope_head_dim=hf["qk_rope_head_dim"], q_lora_rank=hf["q_lora_rank"],
        o_lora_rank=hf["o_lora_rank"], o_groups=hf["o_groups"],
        index_head_dim=hf["index_head_dim"], index_topk=hf["index_topk"],
        n_routed_experts=hf["n_routed_experts"],
        num_experts_per_tok=hf["num_experts_per_tok"],
        moe_intermediate_size=hf["moe_intermediate_size"],
        vocab_size=hf["vocab_size"],
        compress_ratios=[int(r) for r in hf["compress_ratios"]])
    torch.manual_seed(seed)
    reference = DSV4RefModel(ref_cfg).to(dtype).eval()
    return ours, reference


def _parity(config_root: str, dtype_name: str = "float32") -> bool:
    import torch

    dtype = getattr(torch, dtype_name)
    ok = True
    print()
    for block in EXPECTED:
        path = REPO / config_root / f"P15B-{block}.json"
        ours, reference = _build_pair(path, dtype)

        translated, unmapped = {}, []
        for key, value in reference.state_dict().items():
            ours_key = _translate(key)
            if ours_key is None:
                unmapped.append(key)
            else:
                translated[ours_key] = value
        if unmapped:
            print(f"{block:<5} UNMAPPED reference keys: {unmapped[:4]} ...")
            ok = False
            continue
        missing = ours.load_state_dict(translated, strict=False)
        if missing.unexpected_keys or missing.missing_keys:
            print(f"{block:<5} load_state_dict mismatched: "
                  f"missing={list(missing.missing_keys)[:4]} "
                  f"unexpected={list(missing.unexpected_keys)[:4]}")
            ok = False
            continue

        tokens = 256
        ids = torch.arange(tokens, dtype=torch.long)[None] % int(
            json.loads(path.read_text(encoding="utf-8"))["vocab_size"])
        with torch.no_grad():
            got = ours(ids).double()
            want, _ = reference(ids)
            want = want.double()
        diff = (got - want).abs().max().item()
        scale = want.abs().mean().item()
        print(f"{block:<5} parity ({dtype_name}, T={tokens}): "
              f"max|Δlogits|={diff:.3e}  (mean|logits|={scale:.3e})")
        if diff > 1e-4 * max(scale, 1e-6) + 1e-4:
            ok = False
    if not ok:
        print("\nPARITY FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())

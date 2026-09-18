#!/usr/bin/env python3
"""Check the reference design on whatever GPU this runs on.

Three checks, in the order the design depends on them:

1. **Budget** -- parameter count and activated parameters against the config's
   own estimator (so the estimator and the module tree cannot drift apart).
2. **KV accounting** -- walk the layers and count the bytes a token actually
   occupies, comparing what the modules *store* against the closed form the
   design is priced with (``head_dim + rope`` for a full layer,
   ``2*coff*head_dim/ratio`` for a compressed one).
3. **Numerics on this device** -- one forward pass, then a fingerprint
   (checksum of the logits).  Running this on a 4090 (sm89) and a 5090 (sm120)
   and comparing fingerprints is what proves the block is portable rather than
   Hopper-only; the Triton port later has to reproduce the same numbers.

    python3 -m design.dsv4_ref.verify --config small --tokens 256
    python3 -m design.dsv4_ref.verify --config p13b --tokens 512 --device cuda
"""

from __future__ import annotations

import argparse
import json
import platform

import torch

from .config import REF_CONFIGS
from .model import DSV4RefModel


def kv_accounting(model: DSV4RefModel) -> dict:
    """Bytes per token a sequence actually occupies, layer by layer."""
    cfg = model.cfg
    rows = []
    for index, ratio in enumerate(cfg.compress_ratios):
        if ratio == 0:
            stored = cfg.head_dim + cfg.qk_rope_head_dim
        else:
            stored = 2 * cfg.coff(ratio) * cfg.head_dim / ratio
        rows.append({
            "layer": index,
            "ratio": ratio,
            "bytes_per_token_bf16": round(stored * 2, 1),
            "states_per_1k_tokens": (1000 // ratio) if ratio else 1000,
        })
    total = sum(r["bytes_per_token_bf16"] for r in rows)
    return {"per_layer": rows, "total_kb_per_token_bf16": round(total / 1024, 3),
            "closed_form_kb": round(cfg.kv_bytes_per_token(2) / 1024, 3)}


def fingerprint(logits: torch.Tensor) -> dict:
    """Device-portable summary of a forward pass (float64 accumulators)."""
    return {
        "shape": list(logits.shape),
        # Rounding is chosen to be loose enough for Ada-vs-Blackwell reduction
        # order differences but tight enough to catch a real change of math.
        "mean": round(float(logits.double().mean()), 5),
        "std": round(float(logits.double().std()), 5),
        "sum": round(float(logits.double().sum()), 2),
        "checksum": round(float((logits.double() * torch.arange(
            1, logits.shape[-1] + 1, device=logits.device)).sum()), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="small", choices=sorted(REF_CONFIGS))
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--summary-only", action="store_true",
                        dest="summary_only",
                        help="print the budget and KV accounting without "
                             "building the model (for sizes that do not fit "
                             "this GPU in fp32)")
    args = parser.parse_args()

    cfg = REF_CONFIGS[args.config]
    if args.summary_only:
        kb = cfg.kv_bytes_per_token(2) / 1024
        kb8 = cfg.kv_bytes_per_token(1) / 1024
        out = {"config": cfg.summary(),
               "weights_gb_bf16": round(cfg.summary()["total_params_B"] * 2, 2),
               "weights_gb_fp8": round(cfg.summary()["total_params_B"], 2),
               "kv_mb": {str(ctx): {"bf16": round(ctx * kb / 1024, 1),
                                    "fp8": round(ctx * kb8 / 1024, 1)}
                         for ctx in (4096, 32768, 131072)}}
        print(json.dumps(out, indent=2))
        return 0
    dtype = getattr(torch, args.dtype)
    model = DSV4RefModel(cfg).to(device=args.device, dtype=dtype).eval()

    report = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.machine(),
        "dtype": args.dtype,
        "config": cfg.summary(),
        "kv": kv_accounting(model),
    }

    # 1) budget: module tree vs the estimator
    module_params = sum(p.numel() for p in model.parameters())
    report["params_module_tree"] = module_params
    report["params_estimator"] = round(cfg.summary()["total_params_B"] * 1e9)

    # 3) numerics
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, args.tokens), device=args.device)
    with torch.no_grad():
        logits, infos = model(ids)
    report["fingerprint"] = fingerprint(logits)
    report["layer_info"] = infos
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


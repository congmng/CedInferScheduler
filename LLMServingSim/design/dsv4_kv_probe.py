#!/usr/bin/env python3
"""Read back the KV cache specs vLLM actually builds for a DSV4 layer pattern.

Section 3.1 of docs/DSV4_10-30B跨卡设计.md measured a counter-intuitive thing:
a 12-layer DSV4 mix is charged ~98 KB per token of capacity although the
analytic figure is 4.3 KB.  Reading the source suggests why (the compressor's
``get_kv_cache_spec`` never passes ``tokens_per_state``, and its state cache is
fp32), but that is a reading, not a measurement.  This script measures it:
it wraps ``AttentionSpec.page_size_bytes`` so every distinct spec the engine
builds is printed with the fields that determine it, then boots a dummy-weight
model and prints the KV cache groups and the reported capacity.

Run inside the pinned image, on a machine with a free GPU:

    python3 -m design.dsv4_kv_probe --model /workspace/dsv4-small \
            --ratios 0,0,4,128,4,128,4,128,4,128,4,0 --max-model-len 4096

The ratios are passed to ``hf_overrides`` together with the matching
``num_hidden_layers``, so one config directory covers every layer count.
"""

from __future__ import annotations

import argparse
import json
import os

# V1 runs the engine core in a separate process by default, which would put the
# spec objects in a process this script cannot see.  Keep it in-process so the
# wrapper below actually observes the specs the scheduler builds.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def instrument() -> list[dict]:
    """Log every distinct KV cache spec the engine constructs."""
    from vllm.v1.kv_cache_interface import AttentionSpec

    seen: dict[tuple, dict] = {}
    order: list[dict] = []
    original = AttentionSpec.__dict__["page_size_bytes"]

    def patched(self):
        size = original.fget(self)
        num_states = self.get_num_kernel_states(self.block_size)
        try:
            content = self.state_content_size_bytes
        except Exception:  # pragma: no cover - defensive
            content = None
        key = (type(self).__name__, self.block_size, self.head_size,
               self.tokens_per_state, str(self.dtype), content,
               num_states, self.page_size_padded)
        if key not in seen:
            row = {
                "spec": type(self).__name__,
                "block_size": self.block_size,
                "head_size": self.head_size,
                "num_kv_heads": self.num_kv_heads,
                "tokens_per_state": self.tokens_per_state,
                "dtype": str(self.dtype),
                "state_content_bytes": content,
                "num_states": num_states,
                "page_size_bytes": size,
                "page_size_padded": self.page_size_padded,
                "sliding_window": getattr(self, "sliding_window", None),
                "b_per_token": round(size / self.block_size, 1),
            }
            seen[key] = row
            order.append(row)
            print("[spec] " + json.dumps(row, ensure_ascii=False), flush=True)
        return size

    AttentionSpec.page_size_bytes = property(patched)
    return order


def dump_groups(llm) -> None:
    """Best-effort walk to the KV cache groups the scheduler ended up with."""
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    for label, getter in [
        ("engine_core.scheduler.kv_cache_manager.kv_cache_config",
         lambda: core.scheduler.kv_cache_manager.kv_cache_config),
        ("engine_core.kv_cache_config", lambda: core.kv_cache_config),
    ]:
        try:
            config = getter()
        except Exception as exc:  # pragma: no cover - API drift
            print(f"[groups] {label} unavailable: {exc}", flush=True)
            continue
        groups = getattr(config, "kv_cache_groups", None) or []
        print(f"[groups] {len(groups)} KV cache group(s)", flush=True)
        for index, group in enumerate(groups):
            spec = getattr(group, "kv_cache_spec", None)
            names = getattr(group, "layer_names", None) or []
            print("[group] " + json.dumps({
                "index": index,
                "spec": type(spec).__name__ if spec is not None else None,
                "layers": len(names),
                "page_size_bytes": getattr(spec, "page_size_bytes", None),
                "block_size": getattr(spec, "block_size", None),
                "tokens_per_state": str(getattr(spec, "tokens_per_state", None)),
                "sliding_window": getattr(spec, "sliding_window", None),
                "layer_names": list(names)[:6],
            }, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="directory holding config.json (no weights needed)")
    parser.add_argument("--ratios", required=True,
                        help="comma separated compress_ratios, one per layer")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    parser.add_argument("--max-num-seqs", type=int, default=64)
    args = parser.parse_args()

    ratios = [int(r) for r in args.ratios.split(",") if r != ""]
    instrument()

    from vllm import LLM

    llm = LLM(
        model=args.model,
        # The config directory has no tokenizer, and this probe never feeds text
        # -- only the KV cache geometry matters here.
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        hf_overrides={
            "num_hidden_layers": len(ratios),
            "compress_ratios": ratios,
            "max_position_embeddings": args.max_model_len,
        },
    )
    print(f"[done] layers={len(ratios)} ratios={ratios}", flush=True)
    dump_groups(llm)
    print(f"[pid] {os.getpid()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

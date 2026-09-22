#!/usr/bin/env python3
"""Path B step 2: can this card actually load and run P-15B?

The whole reason for writing our own model is that the official DeepSeek-V4
path only instantiates on sm90+ (mHC kernel) and its o_proj is fp8-only.  This
is the check that says whether a given card got past that: build the model
from a shipped config with dummy weights, run one generate, and print the KV
capacity the engine allocated -- which must match the design's closed form
(1152 B/token for a full-MLA layer, 1024 for CSA, 16 for HCA).

    docker run --rm --gpus '"device=0"' \
        -e PYTHONPATH=/work/deploy/vllm_p15b \
        --entrypoint python3 -v "$PWD":/work -w /work \
        vllm/vllm-openai:casr029 tests/check_p15b_boot.py --block r4

PYTHONPATH must point at ``deploy/vllm_p15b`` so ``sitecustomize`` registers
the architecture before vLLM reads the config.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pathlib
import re
import contextlib
import shutil
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]

#: What the design says one layer of each type costs per token, in bytes.
EXPECTED_BYTES_PER_TOKEN = {"r0": 1152, "r4": 1024, "r128": 16}

#: The engine core runs in a separate process by default, which would put the
#: "GPU KV cache size" line in a stream this script cannot read.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _log_capture() -> list[str]:
    """Collect vLLM's log records; they bypass stdout redirection."""
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    logging.getLogger("vllm").addHandler(_Capture())
    return messages


def _kv_accounting(log: str) -> tuple[float, int] | None:
    memory = re.search(r"Available KV cache memory: ([\d.]+) GiB", log)
    capacity = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    if not (memory and capacity):
        return None
    return float(memory.group(1)), int(capacity.group(1).replace(",", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="r4", choices=sorted(EXPECTED_BYTES_PER_TOKEN))
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    config = REPO / "configs" / "model" / "casr" / f"P15B-{args.block}.json"
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="p15b_boot_"))
    shutil.copy(config, workdir / "config.json")

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None

    # No stdout redirection: vLLM (and torch's distributed init) call
    # ``fileno()`` on the streams, which a StringIO does not have
    # ("UnsupportedOperation: fileno").  The log handler above is enough to
    # read the KV accounting back, since it captures records directly.
    messages = _log_capture()
    try:
        llm = LLM(model=str(workdir), load_format="dummy", dtype="bfloat16",
                  enforce_eager=True, skip_tokenizer_init=True,
                  max_model_len=args.max_model_len, max_num_seqs=8,
                  gpu_memory_utilization=0.35)
        out = llm.generate([{"prompt_token_ids": list(range(args.tokens))}],
                           SamplingParams(max_tokens=2, temperature=0.0))
    except Exception as exc:
        print(f"BOOT FAILED on {name}"
              + (f" (sm{capability[0]}{capability[1]})" if capability else ""))
        print(f"  {type(exc).__name__}: {str(exc)[:400]}")
        for line in messages[-12:]:
            print(f"  | {line.strip()[:200]}")
        return 1

    produced = len(out[0].outputs[0].token_ids)
    print(f"BOOT OK  {name}" + (f" (sm{capability[0]}{capability[1]})" if capability else ""))
    print(f"  block={args.block}  generate -> {produced} tokens")
    accounting = _kv_accounting("\n".join(messages))
    if accounting:
        gib, tokens = accounting
        per_token = gib * 1024 ** 3 / tokens
        expected = EXPECTED_BYTES_PER_TOKEN[args.block]
        verdict = "ok" if abs(per_token - expected) <= max(2.0, expected * 0.02) else "MISMATCH"
        print(f"  KV: {gib:.2f} GiB / {tokens:,} tokens -> "
              f"{per_token:.0f} B/token (design {expected}) {verdict}")
    else:
        print("  KV: capacity line not found in the engine log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

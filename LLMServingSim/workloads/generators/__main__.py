"""CLI dispatch for workload generators.

Usage:
    python -m workloads.generators sharegpt --model <hf-id> --num-reqs 300 --sps 10 \
        --source <path-or-hf-id> --output workloads/sharegpt-<model>-<n>-sps<r>.jsonl
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="workloads.generators")
    sub = parser.add_subparsers(dest="generator", required=True)

    sg = sub.add_parser("sharegpt", help="ShareGPT -> LLMServingSim JSONL")
    from workloads.generators.sharegpt import register_args as sg_register
    sg_register(sg)

    casr = sub.add_parser("casr", help="Synthetic prefix-cache workload -> LLMServingSim JSONL")
    from workloads.generators.casr import register_args as casr_register
    casr_register(casr)

    dolly = sub.add_parser("dolly", help="Databricks Dolly-15k -> LLMServingSim JSONL")
    from workloads.generators.dolly import register_args as dolly_register
    dolly_register(dolly)

    cnndm = sub.add_parser("cnndm", help="CNN/DailyMail (long prompts) -> LLMServingSim JSONL")
    from workloads.generators.cnndm import register_args as cnndm_register
    cnndm_register(cnndm)

    args = parser.parse_args()

    if args.generator == "sharegpt":
        from workloads.generators.sharegpt import run
        return run(args)
    if args.generator == "casr":
        from workloads.generators.casr import run
        return run(args)
    if args.generator == "dolly":
        from workloads.generators.dolly import run
        return run(args)
    if args.generator == "cnndm":
        from workloads.generators.cnndm import run
        return run(args)

    parser.error(f"Unknown generator: {args.generator}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

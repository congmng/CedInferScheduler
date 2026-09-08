"""Synthetic CASR workload generator.

Generates JSONL traces for ``python -m serving`` that always include
``input_tok_ids``, so the simulator's prefix cache has real hashes to hit.

The generator is intentionally tokenizer-free: token IDs are drawn from a
small synthetic vocabulary and are deterministic for a given seed.  Requests
are labelled with their ground-truth hotspot/prefix, arrival region, link
state and decode tier.  A sidecar manifest (``<output>.meta.json``) records
the same labels plus the parameter set that produced the trace.

Usage:
    python -m workloads.generators casr \
        --output workloads/casr_zipf.jsonl \
        --num-reqs 1000 --sps 10 --seed 42 \
        --hotspot-mode zipf --num-prefixes 8 --reuse-rate 0.8 \
        --edge-fraction 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_VOCAB_SIZE = 32000
DEFAULT_PREFIX_LEN = 128
DEFAULT_INPUT_LEN = 128
DEFAULT_OUTPUT_LEN = 64


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output", required=True,
                   help="Output JSONL path.")
    p.add_argument("--num-reqs", type=int, required=True,
                   dest="num_reqs",
                   help="Number of requests to emit.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible traces.")

    # ---- Prefix / hotspot structure ---------------------------------------
    p.add_argument("--hotspot-mode",
                   choices=["stable", "zipf", "periodic", "drift"],
                   default="stable",
                   help="How requests select their reusable prefix: stable "
                        "(one hotspot), zipf (Zipf weights across hotspots), "
                        "periodic (round-robin), or drift (A->B transition "
                        "at constant arrival rate). Default stable.")
    p.add_argument("--num-prefixes", type=int, default=4,
                   dest="num_prefixes",
                   help="Number of distinct reusable prefixes. Default 4.")
    p.add_argument("--reuse-rate", type=float, default=0.8,
                   dest="reuse_rate",
                   help="Probability a request uses a hot prefix instead of "
                        "random tokens. Default 0.8.")
    p.add_argument("--prefix-len", type=int, default=DEFAULT_PREFIX_LEN,
                   dest="prefix_len",
                   help="Reusable prefix length in tokens. Default 128.")
    p.add_argument("--input-len", type=int, default=DEFAULT_INPUT_LEN,
                   dest="input_len",
                   help="Total input length in tokens; must be >= --prefix-len. "
                        "Default 128.")
    p.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN,
                   dest="output_len",
                   help="Output length in tokens. Default 64.")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE,
                   dest="vocab_size",
                   help="Synthetic vocabulary size. Default 32000.")
    p.add_argument("--zipf-exp", type=float, default=1.0,
                   dest="zipf_exp",
                   help="Zipf exponent used by --hotspot-mode zipf. Default 1.0.")
    p.add_argument("--drift-from", type=int, default=0,
                   dest="drift_from",
                   help="(--hotspot-mode drift) first hotspot index. Default 0.")
    p.add_argument("--drift-to", type=int, default=1,
                   dest="drift_to",
                   help="(--hotspot-mode drift) second hotspot index. Default 1.")

    # ---- Arrival model ----------------------------------------------------
    p.add_argument("--sps", type=float, default=10.0,
                   help="Mean arrival rate in requests per second for "
                        "poisson and uniform models. Default 10.")
    p.add_argument("--arrival-model", choices=["poisson", "uniform", "burst"],
                   default="poisson",
                   dest="arrival_model",
                   help="Arrival spacing. Default poisson.")
    p.add_argument("--first-arrival-sec", type=float, default=0.0,
                   dest="first_arrival_sec",
                   help="Offset (seconds) added to the first request arrival. "
                        "Default 0.")
    p.add_argument("--burst-size", type=int, default=8,
                   dest="burst_size",
                   help="(--arrival-model burst) requests per burst. Default 8.")
    p.add_argument("--burst-gap-sec", type=float, default=10.0,
                   dest="burst_gap_sec",
                   help="(--arrival-model burst) seconds between bursts. "
                        "Default 10.")
    p.add_argument("--burst-poisson", action="store_true", default=False,
                   dest="burst_poisson",
                   help="(--arrival-model burst) keep Poisson spacing inside "
                        "each burst instead of spacing them by --sps.")

    # ---- Experiment labels ------------------------------------------------
    p.add_argument("--edge-fraction", type=float, default=0.5,
                   dest="edge_fraction",
                   help="Fraction of requests tagged region=edge. Default 0.5.")
    p.add_argument("--link-degrade-at-sec", type=float, default=-1.0,
                   dest="link_degrade_at_sec",
                   help="Switch link_state from normal to degraded at this "
                        "offset in seconds; negative disables degradation. "
                        "Default -1.")
    p.add_argument("--decode-tier-mode", choices=["homogeneous", "skewed"],
                   default="homogeneous",
                   dest="decode_tier_mode",
                   help="Tag decode target speed. homogeneous marks every "
                        "request fast; skewed marks a controllable slow "
                        "fraction. Default homogeneous.")
    p.add_argument("--slow-fraction", type=float, default=0.3,
                   dest="slow_fraction",
                   help="(--decode-tier-mode skewed) fraction tagged slow. "
                        "Default 0.3.")
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction,
                   default=True,
                   dest="write_manifest",
                   help="Write a .meta.json sidecar with prefix definitions "
                        "and per-request ground-truth labels. Default enabled.")


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

def _bounded(value: float, lower: float, upper: float, name: str) -> float:
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}, got {value}")
    return value


def _make_prefix_tokens(index: int, length: int, vocab_size: int, seed: int) -> list[int]:
    """Deterministic synthetic prefix for ``index``.

    The first token is derived from the prefix index, making prefixes visibly
    distinct while leaving the remaining tokens deterministic and varied.
    """
    if length <= 0:
        return []
    state = random.Random((seed & 0xFFFFFFFF) ^ ((index + 1) * 0x9E3779B9))
    tokens = [int(1000 + index % max(1, vocab_size - 1000))]
    tokens.extend(state.randrange(0, vocab_size) for _ in range(length - 1))
    return tokens


def _random_tokens(length: int, vocab_size: int, rng: random.Random) -> list[int]:
    return [rng.randrange(0, vocab_size) for _ in range(length)]


def _select_hotspot(request_idx: int, num_reqs: int, mode: str,
                    num_prefixes: int, zipf_exp: float,
                    drift_from: int, drift_to: int,
                    rng: random.Random) -> int | None:
    """Return a hotspot index, or ``None`` for a non-hot random request."""
    if mode == "stable":
        return 0
    if mode == "periodic":
        return request_idx % num_prefixes
    if mode == "drift":
        phase = 0.0 if num_reqs <= 1 else request_idx / (num_reqs - 1)
        return drift_from if rng.random() > phase else drift_to
    if mode == "zipf":
        weights = [1.0 / (i + 1) ** zipf_exp for i in range(num_prefixes)]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        return rng.choices(range(num_prefixes), weights=probabilities, k=1)[0]
    raise ValueError(f"unknown hotspot mode: {mode}")


def _hot_choice(request_idx: int, num_reqs: int, args: argparse.Namespace,
                rng: random.Random) -> int | None:
    if rng.random() >= _bounded(args.reuse_rate, 0.0, 1.0, "--reuse-rate"):
        return None
    return _select_hotspot(request_idx, num_reqs, args.hotspot_mode,
                           args.num_prefixes, args.zipf_exp,
                           args.drift_from, args.drift_to, rng)


def _next_arrival_ns(previous_ns: int | None, index: int,
                     args: argparse.Namespace, rng: random.Random) -> int:
    """Return a monotonically non-decreasing arrival timestamp.

    ``index`` is zero-based and ``previous_ns`` is the previous request's
    timestamp, which keeps Poisson and burst inter-arrival gaps cumulative.
    """
    first_ns = int(args.first_arrival_sec * 1_000_000_000)
    if previous_ns is None:
        return first_ns

    if args.arrival_model == "burst":
        burst_index, offset = divmod(index, args.burst_size)
        if offset == 0:
            return previous_ns + int(args.burst_gap_sec * 1_000_000_000)
        if args.burst_poisson:
            spacing_ns = max(0.0, rng.expovariate(args.sps)) * 1_000_000_000
        else:
            spacing_ns = 1_000_000_000.0 / max(args.sps, 1e-9)
        return previous_ns + int(spacing_ns)
    if args.arrival_model == "uniform":
        return previous_ns + int(1_000_000_000.0 / max(args.sps, 1e-9))
    if args.arrival_model == "poisson":
        return previous_ns + int(max(0.0, rng.expovariate(args.sps)) * 1_000_000_000)
    raise ValueError(f"unknown arrival model: {args.arrival_model}")


def _build_tokens(prefix: list[int] | None, input_len: int, vocab_size: int,
                  rng: random.Random) -> list[int]:
    if prefix is None:
        return _random_tokens(input_len, vocab_size, rng)
    if input_len < len(prefix):
        raise ValueError("--input-len must be >= --prefix-len")
    suffix_len = input_len - len(prefix)
    return list(prefix) + _random_tokens(suffix_len, vocab_size, rng)


def _region(rng: random.Random, edge_fraction: float) -> str:
    return "edge" if rng.random() < _bounded(edge_fraction, 0.0, 1.0,
                                             "--edge-fraction") else "cloud"


def _decode_tier(rng: random.Random, args: argparse.Namespace) -> str:
    if args.decode_tier_mode == "homogeneous":
        return "fast"
    if args.decode_tier_mode == "skewed":
        slow_fraction = _bounded(args.slow_fraction, 0.0, 1.0,
                                 "--slow-fraction")
        return "slow" if rng.random() < slow_fraction else "fast"
    raise ValueError(f"unknown decode tier mode: {args.decode_tier_mode}")


def _link_state(arrival_ns: int, degrade_ns: int | None) -> str:
    if degrade_ns is not None and arrival_ns >= degrade_ns:
        return "degraded"
    return "normal"


def _prefix_id(prefix_tokens: Iterable[int]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for token in prefix_tokens:
        digest.update(int(token).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    if args.num_reqs < 0:
        raise ValueError("--num-reqs must be >= 0")
    if args.input_len < args.prefix_len:
        raise ValueError("--input-len must be >= --prefix-len")
    if not 0 <= args.drift_from < args.num_prefixes:
        raise ValueError("--drift-from is outside --num-prefixes")
    if not 0 <= args.drift_to < args.num_prefixes:
        raise ValueError("--drift-to is outside --num-prefixes")

    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefixes = [
        _make_prefix_tokens(index, args.prefix_len, args.vocab_size, args.seed)
        for index in range(args.num_prefixes)
    ]
    degrade_ns = (None if args.link_degrade_at_sec < 0
                  else int(args.link_degrade_at_sec * 1_000_000_000))

    manifest_requests: list[dict[str, object]] = []
    count_by_prefix = {index: 0 for index in range(args.num_prefixes)}
    non_hot_count = 0

    previous_ns = None
    with out_path.open("w", encoding="utf-8") as fout:
        for request_idx in range(args.num_reqs):
            arrival_ns = _next_arrival_ns(previous_ns, request_idx, args, rng)
            previous_ns = arrival_ns
            hotspot = _hot_choice(request_idx, args.num_reqs, args, rng)
            prefix = None if hotspot is None else prefixes[hotspot]
            input_ids = _build_tokens(prefix, args.input_len, args.vocab_size, rng)
            output_ids = _random_tokens(args.output_len, args.vocab_size, rng)
            region = _region(rng, args.edge_fraction)
            decode_tier = _decode_tier(rng, args)
            link_state = _link_state(arrival_ns, degrade_ns)

            row = {
                "input_toks": len(input_ids),
                "output_toks": len(output_ids),
                "arrival_time_ns": arrival_ns,
                "input_tok_ids": input_ids,
                "output_tok_ids": output_ids,
                "region": region,
                "link_state": link_state,
                "decode_tier": decode_tier,
                "hotspot_id": None if hotspot is None else f"hotspot-{hotspot}",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if hotspot is None:
                non_hot_count += 1
            else:
                count_by_prefix[hotspot] += 1
            if args.write_manifest:
                manifest_requests.append({
                    "request_id": request_idx,
                    "hotspot_id": row["hotspot_id"],
                    "arrival_time_ns": arrival_ns,
                    "input_toks": len(input_ids),
                    "output_toks": len(output_ids),
                    "region": region,
                    "link_state": link_state,
                    "decode_tier": decode_tier,
                })

    if args.write_manifest:
        manifest_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        manifest = {
            "generator": "casr",
            "seed": args.seed,
            "num_requests": args.num_reqs,
            "hotspot_mode": args.hotspot_mode,
            "reuse_rate": args.reuse_rate,
            "arrival_model": args.arrival_model,
            "sps": args.sps,
            "edge_fraction": args.edge_fraction,
            "prefixes": [
                {
                    "hotspot_id": f"hotspot-{index}",
                    "prefix_id": _prefix_id(prefixes[index]),
                    "prefix_len": args.prefix_len,
                    "prefix_tok_ids": prefixes[index],
                    "request_count": count_by_prefix[index],
                }
                for index in range(args.num_prefixes)
            ],
            "non_hot_request_count": non_hot_count,
            "requests": manifest_requests,
        }
        with manifest_path.open("w", encoding="utf-8") as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)
            fout.write("\n")
        print(f"Wrote {args.num_reqs} requests -> {out_path}")
        print(f"Wrote manifest -> {manifest_path}")
    else:
        print(f"Wrote {args.num_reqs} requests -> {out_path}")
    return 0

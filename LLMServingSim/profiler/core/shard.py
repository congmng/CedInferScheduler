"""Sharding a profile sweep across cards.

One profile shot costs a fixed ~3.5 s of ``torch.profiler`` bookkeeping
regardless of its shape (measured: 20 dense shots at 2 sequences take the same
3.5 s each as one attention shot at 256 decodes).  A grid therefore cannot be
shortened by measuring less per shot -- only by firing fewer of them, or by
firing them on several cards at once.

``--shard i/N`` is the second lever: keep the shots at positions ``≡ i (mod N)``
of the composed grid, run N processes on N GPUs with separate ``--out-root``
directories, then concatenate the CSVs (``tests/merge_profile_shards.py``).
``compose_shots`` walks fixed geometric axes, so the order is deterministic and
every shard agrees on the global position; the pieces are disjoint and cover
the grid exactly.

Kept free of torch/vLLM imports so the split can be unit-tested on a machine
that only has the simulator installed.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def parse_shard(raw: str | None) -> tuple[int, int] | None:
    """Parse ``--shard I/N`` into a 0-based ``(index, total)`` pair.

    ``None`` and ``0/1`` both mean "no sharding" and normalize to ``None``.
    Raises ``ValueError`` on anything malformed -- a typo here would silently
    measure a subset of the grid.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if "/" not in text:
        raise ValueError("--shard must look like I/N, e.g. 0/2")
    index_text, total_text = text.split("/", 1)
    try:
        index, total = int(index_text), int(total_text)
    except ValueError as exc:
        raise ValueError("--shard must look like I/N with integer parts") from exc
    if total < 1:
        raise ValueError("--shard total must be >= 1")
    if not 0 <= index < total:
        raise ValueError(f"--shard index must be in [0, {total}); got {index}")
    return (index, total) if total > 1 else None


def apply_shard(shots: Sequence[T], shard: tuple[int, int] | None) -> list[T]:
    """Keep only this process's share of ``shots`` (empty when no shard)."""
    if not shard:
        return list(shots)
    index, total = shard
    return [shot for position, shot in enumerate(shots)
            if position % total == index]

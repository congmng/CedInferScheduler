#!/usr/bin/env python3
"""Check that the Chakra ``.et`` graphs pair every send with a matching recv.

Why this exists: on 2026-09-15 a multi-domain simulator run produced **zero
tokens/s** for eleven minutes while the simulated clock raced to 4314 s.  The
cause was not the queueing model -- the Prefill's final output send still named
the legacy adjacent rank (4) while the Decode that actually posts the recv runs
on rank 5, so both sides waited forever.  The fix had been written in the tree
days earlier but never reached the *installed* chakra, so the run looked like a
modelling failure.

Running ASTRA-Sim to discover that costs minutes; parsing the ``.et`` files
costs milliseconds, so this is the cheap liveness gate for graph generation.

Invariant (see ``llm_converter.convert_prefill``): a recv node is written into
the file of the rank that executes it, and its ``comm_dst`` attribute must name
that same rank -- ASTRA-Sim pairs a send/recv on (tag, src, dst, size) *within
the receiver's local rank*.  A send to rank ``d`` must therefore have a
counterpart recv in ``<graph>.<d>.et`` with the identical key.

Usage:
    python3 tests/check_et_pairing.py --dir /tmp/graphs
    python3 tests/check_et_pairing.py --file a.0.et a.1.et
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CHAKRA_SRC = REPO / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "src"
if str(CHAKRA_SRC) not in sys.path:
    sys.path.insert(0, str(CHAKRA_SRC))


def rank_of(path: pathlib.Path) -> int:
    match = re.search(r"\.(\d+)\.et$", path.name)
    return int(match.group(1)) if match else -1


def read_graph(path):
    """Return ``(sends, recvs)`` as multisets of ``(tag, src, dst, size)``."""
    from chakra.src.converter.llm_converter import GlobalMetadata, Node, \
        COMM_SEND_NODE, COMM_RECV_NODE
    from chakra.src.third_party.utils.protolib import decodeMessage

    sends, recvs = collections.Counter(), collections.Counter()
    with open(path, "rb") as handle:
        decodeMessage(handle, GlobalMetadata())
        node = Node()
        while decodeMessage(handle, node):
            if node.type not in (COMM_SEND_NODE, COMM_RECV_NODE):
                node = Node()
                continue
            attrs = {item.name: item for item in node.attr}
            key = (
                attrs["comm_tag"].int32_val if "comm_tag" in attrs else -1,
                attrs["comm_src"].int32_val,
                attrs["comm_dst"].int32_val,
                attrs["comm_size"].int64_val if "comm_size" in attrs else -1,
            )
            (sends if node.type == COMM_SEND_NODE else recvs)[key] += 1
            node = Node()
    return sends, recvs


def check(paths):
    """Return a list of human-readable violations (empty means consistent)."""
    graphs = {}
    for path in paths:
        path = pathlib.Path(path)
        graphs[rank_of(path)] = (path, *read_graph(path))
    violations = []
    # A send runs on the file of its ``src`` rank and must find its recv in the
    # file of the ``dst`` rank; a recv runs on the file of its own rank, so its
    # ``comm_dst`` attribute has to name that rank.
    for rank, (path, sends, _recvs) in sorted(graphs.items()):
        for key, count in sends.items():
            tag, src, dst, size = key
            if src != rank:
                violations.append(
                    f"{path.name}: send key (tag={tag},src={src},dst={dst},size={size}) "
                    f"names src={src} but runs on rank {rank}")
            target = graphs.get(dst)
            if target is None:
                violations.append(f"{path.name}: send targets missing rank {dst}")
                continue
            if target[2].get(key, 0) < count:
                violations.append(
                    f"{path.name}: send key (tag={tag},src={src},dst={dst},size={size}) "
                    f"({count}x) has no matching recv in {target[0].name}")
    for rank, (path, _sends, recvs) in sorted(graphs.items()):
        for key, count in recvs.items():
            tag, src, dst, size = key
            if dst != rank:
                violations.append(
                    f"{path.name}: recv key (tag={tag},src={src},dst={dst},size={size}) "
                    f"is addressed to rank {dst} but runs on rank {rank} "
                    f"({count}x) -- the sender waits for a rank that never receives")
            source = graphs.get(src)
            if source is None:
                violations.append(f"{path.name}: recv from missing rank {src}")
                continue
            if source[1].get(key, 0) < count:
                violations.append(
                    f"{path.name}: recv key (tag={tag},src={src},dst={dst},size={size}) "
                    f"({count}x) has no matching send in {source[0].name}")
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="", help="directory holding *.et files")
    parser.add_argument("--file", nargs="*", default=[],
                        help="explicit .et files (overrides --dir)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.file:
        paths = [pathlib.Path(item) for item in args.file]
    else:
        if not args.dir:
            raise SystemExit("pass --dir or --file")
        paths = sorted(pathlib.Path(args.dir).glob("*.et"))
    if not paths:
        raise SystemExit("no .et files found")
    violations = check(paths)
    if violations:
        print(f"ET PAIRING FAILED ({len(paths)} files, {len(violations)} violations)")
        for item in violations[:20]:
            print(f"  - {item}")
        return 1
    if not args.quiet:
        print(f"ET PAIRING OK ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

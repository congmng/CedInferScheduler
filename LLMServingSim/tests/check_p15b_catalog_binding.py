#!/usr/bin/env python3
"""P-15B gate: every canonical layer name must name exactly one module.

The profiler binds a catalog entry to a module by
``class name == entry.vllm`` **and** ``entry.within in ancestor class names``
(``profiler/core/hooks/timings.py``).  Nothing in that rule stops one class
from being instantiated in two different places -- and when it is, *both*
nodes match the same canonical name, and ``DedupSink`` averages them into a
single CSV row (``profiler/core/writer.py``).

That is not hypothetical: the compressor reuses ``P15BKVNorm`` for its own
state norm, so on r4/r128 the ``kv_norm`` row is the mean of the attention's
576-wide norm and the compressor's 2048/1024-wide one.  The merge tool caught
it as a 17.3% cross-type disagreement on that one layer while every other
layer agreed to within 1%.

This gate reproduces the profiler's matching *before* a four-hour profile run:
it builds the module tree for each block type and reports, per canonical name,
how many modules match and what shapes they have.  CPU is enough -- no weights
are read.

    docker run --rm --entrypoint python3 -v "$PWD":/work -w /work \
        vllm/vllm-openai:casr029 tests/check_p15b_catalog_binding.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deploy.vllm_p15b.config import P15BConfig                  # noqa: E402
from deploy.vllm_p15b.model import P15BForCausalLM              # noqa: E402
from profiler.core.config import load_architecture              # noqa: E402
from profiler.core.hooks.timings import _match_slice            # noqa: E402

#: Catalog groups with their own CSV.  Checked separately because two groups
#: may legitimately bind the same class (the simulator reads them from
#: different tables); a collision *inside* one group is always a bug.
GROUPS = ("dense", "per_sequence", "attention", "moe")

#: Names whose modules vLLM supplies (they are not built by our package, so
#: the standalone tree never has them).  The *sequence* still lists them --
#: the simulator walks the yaml, not this tree.
VLLM_NATIVE = ("lm_head", "sampler")


def _slice_of(catalog, group: str) -> dict[str, dict]:
    """Catalog group in the shape ``_match_slice`` expects."""
    entries = getattr(catalog, group, {}) or {}
    return {name: {"vllm": entry.vllm, "within": entry.within}
            for name, entry in entries.items()}


def _tree(model) -> list[tuple[str, str, list[str]]]:
    """``(qualified path, class name, ancestor class names)`` per module."""
    found: list[tuple[str, str, list[str]]] = []

    def walk(module, path: str, ancestors: list[str]) -> None:
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            child_cls = type(child).__name__
            found.append((child_path, child_cls, ancestors))
            walk(child, child_path, ancestors + [child_cls])

    walk(model, "", [type(model).__name__])
    return found


def _shape(module) -> str:
    param = next(iter(module.parameters()), None)
    return "no params" if param is None else "x".join(str(d) for d in param.shape)


def _expected(arch, block: str) -> set[str]:
    """Canonical names this block type's own pipeline lists.

    ``layer_types`` is the same list the simulator walks, so it is the honest
    statement of what a bundle has to contain -- and it already encodes that
    r0 has no compressor and r128 has no CSA pieces.
    """
    pipeline = (arch.layer_types or {}).get(block)
    if pipeline is None:
        return set()
    names: set[str] = set()
    for phase in pipeline.model_dump().values():
        names.update(phase)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="profiler/models/p15b.yaml")
    parser.add_argument("--config-root", default="configs/model/casr")
    parser.add_argument("--blocks", default="r0,r4,r128")
    args = parser.parse_args()

    arch = load_architecture(REPO / args.arch)

    failures: list[str] = []
    absent: list[str] = []
    for block in [b for b in args.blocks.split(",") if b]:
        hf = json.loads((REPO / args.config_root / f"P15B-{block}.json")
                        .read_text(encoding="utf-8"))
        model = P15BForCausalLM(P15BConfig.from_hf(hf))
        modules = _tree(model)
        expected = _expected(arch, block)

        for group in GROUPS:
            slice_ = _slice_of(arch.catalog, group)
            if not slice_:
                continue
            for canonical, spec in slice_.items():
                matches = [(path, cls, _shape(model.get_submodule(path)))
                           for path, cls, ancestors in modules
                           if _match_slice(cls, ancestors, {canonical: spec})
                           == canonical]
                if len(matches) > 1:
                    failures.append(f"{block}:{group}:{canonical} AMBIGUOUS")
                    print(f"{block:<5}{group:<13}{canonical:<16}"
                          f"{matches[0][0]:<40}{'':>14}   AMBIGUOUS")
                    for path, _, shape in matches[1:]:
                        print(f"{'':<18}{'':<13}{'':<16}{path:<40}"
                              f"{shape:>14}   ^ also")
                    continue
                path, shape = (matches[0][0], matches[0][2]) if matches else ("-", "")
                wanted = canonical in expected and canonical not in VLLM_NATIVE
                if wanted and not matches:
                    failures.append(f"{block}:{group}:{canonical} MISSING")
                    print(f"{block:<5}{group:<13}{canonical:<16}"
                          f"{'-':<40}{'':>14}   MISSING (the pipeline lists it)")
                    continue
                if not matches:
                    absent.append(f"{block}:{group}:{canonical}")
                    print(f"{block:<5}{group:<13}{canonical:<16}"
                          f"{'-':<40}{'':>14}   n/a here")
                    continue
                print(f"{block:<5}{group:<13}{canonical:<16}"
                      f"{path:<40}{shape:>14}   ok")

    if failures:
        print("\nFAILED: one canonical name must bind exactly one module:")
        for item in failures:
            print(f"  {item}")
        print("\nAMBIGUOUS means two modules share a class name and both match;"
              "\nDedupSink averages them into one CSV row (see"
              "\nprofiler/core/writer.py).  Give the second one its own class --"
              "\nan unpriced helper class is fine.")
        return 1
    print(f"\nbinding ok; {len(absent)} name(s) absent by design, e.g. the other"
          "\nblock type's pieces and the vLLM-native head: "
          + ", ".join(sorted({n.split(":")[-1] for n in absent})))
    return 0


if __name__ == "__main__":
    sys.exit(main())

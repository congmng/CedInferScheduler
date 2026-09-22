#!/usr/bin/env python3
"""Assemble one hardware's P-15B bundle from a profile run + a dense re-run.

Why this exists: a block type is collected as one run per type (the profiler
always builds ``num_hidden_layers=1``), and afterwards one category may need
re-measuring -- on 2026-09-22 the catalog binding fix changed two ``dense.csv``
rows and nothing else.  Re-running the whole type would have cost 4-5 hours of
attention sweep to fix 2 rows, so the fix is collected separately with
``--categories dense`` and spliced in here.

    python3 tests/assemble_p15b_bundle.py --hardware RTX5090 \
        --profile-root /tmp/p15b-formal --dense-root /tmp/p15b-dense \
        --work-root /tmp/p15b-assembled --note "dense re-measured after the \
catalog binding fix"

It then runs ``merge_profile_types.py`` and both bundle checkers, so the
command either leaves a verified bundle in ``profiler/perf`` or fails loudly.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Files a full run produces, minus the one the re-run replaces.
FROM_PROFILE = ("attention.csv", "per_sequence.csv", "moe.csv")
MODEL = "casr/P15B"


def _type_dir(root: pathlib.Path, block: str, hardware: str) -> pathlib.Path:
    # Layout the profiler writes: <out-root>/<block>/<hardware>/<org>/<name>-<block>
    return (root / block / hardware / MODEL.split("/")[0]
            / f"P15B-{block}" / "bf16" / "tp1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--profile-root", required=True,
                        help="the full run's out-root (one dir per block type)")
    parser.add_argument("--dense-root", required=True,
                        help="the --categories dense run's out-root")
    parser.add_argument("--work-root", required=True,
                        help="scratch dir for the assembled tree")
    parser.add_argument("--blocks", default="r0,r4,r128")
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    blocks = [b for b in args.blocks.split(",") if b]
    profile_root = pathlib.Path(args.profile_root).expanduser()
    dense_root = pathlib.Path(args.dense_root).expanduser()
    work_root = pathlib.Path(args.work_root)

    problems: list[str] = []
    for block in blocks:
        src = _type_dir(profile_root, block, args.hardware)
        dense = _type_dir(dense_root, block, args.hardware) / "dense.csv"
        for name in FROM_PROFILE:
            if not (src / name).is_file():
                problems.append(f"{src / name} is missing -- that block type's "
                                f"collection is not finished")
        if not dense.is_file():
            problems.append(f"{dense} is missing -- run the dense re-measure "
                            f"first (CATEGORIES=dense)")
    if problems:
        print("cannot assemble yet:")
        for item in problems:
            print(f"  {item}")
        return 1

    for block in blocks:
        src = _type_dir(profile_root, block, args.hardware)
        dense = _type_dir(dense_root, block, args.hardware) / "dense.csv"
        dest = _type_dir(work_root, block, args.hardware)
        dest.mkdir(parents=True, exist_ok=True)
        for name in FROM_PROFILE:
            shutil.copy2(src / name, dest / name)
        shutil.copy2(dense, dest / "dense.csv")
        meta = src.parent / "meta.yaml"
        if meta.is_file():
            shutil.copy2(meta, dest.parent / "meta.yaml")
        print(f"{block}: {len(FROM_PROFILE)} files from the profile run + "
              f"dense.csv from the re-run -> {dest}")

    merge = [
        sys.executable, str(REPO / "tests" / "merge_profile_types.py"),
        "--out-root", "profiler/perf", "--hardware", args.hardware,
        "--model", MODEL, "--variant", args.variant, "--tp", str(args.tp),
        "--type-root", str(work_root),
    ]
    for note in args.note:
        merge += ["--note", note]
    print("\n$ " + " ".join(merge[-6:]) + " ...")
    if subprocess.run(merge, cwd=REPO).returncode:
        return 1

    if args.skip_checks:
        return 0
    # The two checkers take different flags: the first is per-hardware, the
    # second walks every bundle of a model and compares their provenance.
    checks = (
        ("check_profile_bundle.py",
         ["--hardware", args.hardware, "--tp", str(args.tp)]),
        ("check_profile_bundle_consistency.py", []),
    )
    for checker, extra in checks:
        cmd = [sys.executable, str(REPO / "tests" / checker),
               "--model", MODEL, "--variant", args.variant, *extra]
        print(f"\n$ {checker}")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print((result.stdout or result.stderr).strip())
        if result.returncode:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""One command that answers "may I trust this simulator run?".

Three gates, each of which independently caught a real failure on 2026-09-15:

1. **converter identity** -- the runtime must execute the checked-out chakra
   converter.  A three-day-old copy in ``site-packages`` silently kept the
   pre-fix handoff behaviour and deadlocked every multi-domain run.
2. **ET pairing** -- every send in the generated graphs must have a matching
   recv.  A mismatch is what "zero tokens/s for eleven minutes" looks like from
   the inside.
3. **liveness** -- a short replay must complete every request, with a bounded
   simulated span.  A backlogged-but-live run stretches; a dead one does not
   finish at all.

Nothing here measures *accuracy* (that is ``tests/calibrate_simulator.py``);
these are the preconditions for believing any number the simulator prints.

Usage:
    python3 tests/check_sim_ready.py
    python3 tests/check_sim_ready.py --config configs/cluster/<name>.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAKRA_SRC = REPO / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "src"
REPO_CONVERTER = CHAKRA_SRC / "converter" / "llm_converter.py"
DEFAULT_CONFIG = REPO / "configs" / "cluster" / "casr_real_qwen3_8b_three_domain_aligned.json"
DEFAULT_TRACE = "workloads/cnndm-short-real-qwen3-8b-600-sps14.jsonl"


def gate_converter_identity():
    sys.path.insert(0, str(CHAKRA_SRC))
    import chakra.src.converter.llm_converter as module
    loaded = pathlib.Path(module.__file__).resolve()
    if loaded.read_bytes() != REPO_CONVERTER.read_bytes():
        return False, (f"runtime chakra is {loaded}, which differs from the tree at "
                       f"{REPO_CONVERTER}; sync with: cp {REPO_CONVERTER} {loaded}")
    return True, f"{loaded}"


def gate_et_pairing():
    sys.path.insert(0, str(CHAKRA_SRC))
    sys.path.insert(0, str(REPO / "tests"))
    import chakra.src.converter.llm_converter as module
    import check_et_pairing
    header = ("PREFILL\t\tmodel_parallel_NPU_group: 1\t\tpp_stage_boundaries:"
              " 0\t\tpd_decode_npu_offset: 5")
    rows = [[name, "100", "LOCAL", "16", "LOCAL", "16", "LOCAL", "16", "NONE",
             "2048", "NONE"]
            for _ in range(2) for name in ("qkv_proj", "attn", "o_proj")]
    with tempfile.TemporaryDirectory() as tmp:
        module.LLMConverter("", str(pathlib.Path(tmp) / "llm"), num_npus=1,
                            npu_offset=3).convert_rows(header, rows)
        paths = sorted(pathlib.Path(tmp).glob("*.et"))
        violations = check_et_pairing.check(paths)
    if violations:
        return False, "; ".join(violations[:3])
    return True, f"{len(paths)} graphs, all send/recv pairs matched"


def gate_liveness(config, trace, num_reqs):
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "out.csv"
        # The builder resolves the cluster config relative to ``astra-sim/``, so
        # an absolute path becomes ``..//abs/path`` and fails to open.
        config_arg = (str(pathlib.Path(config).resolve().relative_to(REPO))
                      if pathlib.Path(config).resolve().is_relative_to(REPO)
                      else str(config))
        cmd = [sys.executable, "-m", "serving", "--cluster-config", config_arg,
               "--dataset", trace, "--num-reqs", str(num_reqs), "--dtype", "bfloat16",
               "--block-size", "16", "--log-level", "WARNING",
               "--request-routing-policy", "LOAD", "--output", str(out),
               "--inputs-root", str(pathlib.Path(tmp) / "inputs")]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              env={**__import__("os").environ, "SIM_ET_PAIRING_CHECK": "1"})
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            return False, "simulator exited non-zero: " + " | ".join(tail)
        check = subprocess.run(
            [sys.executable, str(REPO / "tests" / "check_sim_liveness.py"),
             "--csv", str(out), "--trace", str(REPO / trace)],
            cwd=REPO, capture_output=True, text=True)
        detail = (check.stdout or check.stderr).strip().splitlines()[-1:]
        return check.returncode == 0, " ".join(detail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--num-reqs", type=int, default=8, dest="num_reqs")
    args = parser.parse_args()
    gates = [
        ("converter identity", lambda: gate_converter_identity()),
        ("ET pairing", lambda: gate_et_pairing()),
        ("liveness", lambda: gate_liveness(args.config, args.trace, args.num_reqs)),
    ]
    failed = 0
    for name, run in gates:
        try:
            ok, detail = run()
        except Exception as exc:  # noqa: BLE001 - a gate must report, not crash
            ok, detail = False, repr(exc)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed += 0 if ok else 1
    print(json.dumps({"gates": len(gates), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

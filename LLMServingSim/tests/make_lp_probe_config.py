#!/usr/bin/env python3
"""Turn a simulator cluster config into a probe config for the LP decomposition.

``tests/diagnose_lp_terms.py`` reads the *deployment* shape -- ``prefills`` /
``decodes`` lists of named instances -- while the simulator describes the same
cluster as nodes holding ``pd_type`` instances.  This bridges the two, and
resolves the numbers the LP actually prices with:

* ``prefill_service_ms`` / ``decode_service_ms`` come from
  ``hw_service.rescale_service_times`` (fastest card keeps the configured
  anchor, the rest scale by the profiled step ratio);
* ``prefill_capacity`` / ``decode_capacity`` come from
  ``rescale_capacities`` (reference-length requests per second).

``hw_service`` resolves the profile bundles relative to the working directory,
so this script chdirs into ``astra-sim/`` itself rather than asking the caller
to remember.

    python3 tests/make_lp_probe_config.py \
        --cluster-config configs/cluster/casr_p15b_three_domain.json \
        --prompt-tokens 1250 --output /tmp/p15b_probe_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1250,
                        help="prompt length the Prefill service times are "
                             "rescaled for")
    parser.add_argument("--model", default=None,
                        help="override the model the probe prices "
                             "(default: the cluster config's own)")
    parser.add_argument("--output", default=None,
                        help="write the probe JSON here (default: stdout)")
    args = parser.parse_args()

    cluster = json.loads(pathlib.Path(args.cluster_config).read_text(encoding="utf-8"))
    casr = json.loads(json.dumps(cluster.get("casr") or {}))
    instances = [inst for node in cluster["nodes"] for inst in node["instances"]]
    if args.model:
        for inst in instances:
            inst["model_name"] = args.model

    # hw_service loads profiler/perf relative to the cwd and expects the
    # simulator's own layout below astra-sim/.
    astra = REPO / "astra-sim"
    cwd = pathlib.Path.cwd()
    os.chdir(astra)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(astra))
    try:
        from serving.core.hw_service import (rescale_capacities,
                                             rescale_service_times)

        rescale_service_times(casr, instances, key="prefill_service_ms",
                              tokens=args.prompt_tokens, verbose=False)
        rescale_service_times(casr, instances, key="decode_service_ms",
                              tokens=1, verbose=False)
        rescale_capacities(casr, instances, verbose=False)
    finally:
        os.chdir(cwd)

    def probe_instances(role: str):
        out = []
        for inst in instances:
            if str(inst.get("pd_type", "")).lower() != role:
                continue
            instance_id = int(inst["instance_id"])
            key = str(instance_id)
            service_key = f"{role}_service_ms"
            capacity_key = f"{role}_capacity"
            if key not in casr.get(service_key, {}):
                continue
            out.append({
                "id": f"{'p' if role == 'prefill' else 'd'}{instance_id}",
                "instance_id": instance_id,
                "capacity": casr[capacity_key][key],
                "service_ms": casr[service_key][key],
                "max_num_seqs": int(inst.get("max_num_seqs", 64) or 64),
            })
        return out

    probe = {
        "casr": casr,
        "weights": {"kv_bytes_per_token": casr.get("kv_bytes_per_token")},
        "prefills": probe_instances("prefill"),
        "decodes": probe_instances("decode"),
        "_probe_comment": (f"derived from {args.cluster_config} by "
                           "tests/make_lp_probe_config.py; feed it to "
                           "tests/diagnose_lp_terms.py"),
    }
    text = json.dumps(probe, ensure_ascii=False, indent=2)
    if args.output:
        pathlib.Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
        print("  prefill_capacity", casr.get("prefill_capacity"))
        print("  decode_capacity ", casr.get("decode_capacity"))
        print("  prefill ids     ", [i["id"] for i in probe["prefills"]])
        print("  decode ids      ", [i["id"] for i in probe["decodes"]])
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Ray-backed inventory and liveness check for the heterogeneous domains.

Ray is the control plane: every host joins the cluster with a
``domain:<name>`` and a ``gpu_type:<type>`` resource label.  This tool reads
those labels from the Ray dashboard REST API, cross-checks them against the
domains declared in ``router_config.json``, and probes the health endpoint of
every enabled Prefill/Decode instance.

It deliberately uses only the standard library: the Ray CLI on the head node
is bound to the system interpreter, while the experiment scripts run under a
different Python, and the dashboard API is reachable over plain HTTP.

Exit code is non-zero when ``--check`` is passed and either a configured
domain is not registered with Ray, or an enabled instance is unhealthy.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "real_lmcache_pd" / "router_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", default="http://127.0.0.1:8265",
                        help="Ray dashboard base URL")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="router config declaring domains and instances")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if a domain or instance is not ready")
    parser.add_argument("--skip-health", action="store_true",
                        help="only verify Ray domain labels; do not probe instances")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def _no_proxy_opener():
    # The control-plane proxy hijacks part of 10.212.* ; never use it here.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_nodes(opener, dashboard, timeout):
    url = f"{dashboard.rstrip('/')}/api/v0/nodes"
    with opener.open(url, timeout=timeout) as response:
        payload = json.load(response)
    data = payload.get("data", {})
    # The API nests the list once more as {"result": {"result": [...]}}.
    inner = data.get("result", data)
    if isinstance(inner, dict):
        inner = inner.get("result", [])
    return inner


def node_state(node):
    """Return ``(state, alive)`` for a dashboard node entry.

    The dashboard exposes ``state`` (ALIVE / DEAD); there is no ``alive``
    field.  A stale registration keeps its resource labels, so treating a DEAD
    entry as live would let a mislabelled or vanished worker pass the gate.
    """
    state = str(node.get("state", "ALIVE")).upper()
    return state, state == "ALIVE"


def node_labels(node):
    resources = node.get("resources_total") or {}
    domains = [key.split(":", 1)[1] for key in resources if key.startswith("domain:")]
    gpu_types = [key.split(":", 1)[1] for key in resources if key.startswith("gpu_type:")]
    return domains, gpu_types


def probe(url, timeout):
    request = urllib.request.Request(url, method="GET")
    try:
        with _no_proxy_opener().open(request, timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


def main() -> int:
    args = parse_args()
    opener = _no_proxy_opener()

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    instances = [item for item in (*config["prefills"], *config["decodes"])
                 if item.get("enabled", True)]
    configured_domains = {item["domain"] for item in instances}

    try:
        nodes = fetch_nodes(opener, args.dashboard, args.timeout)
    except Exception as exc:  # noqa: BLE001 - surface the exact failure
        print(f"failed to query Ray dashboard {args.dashboard}: {exc!r}", file=sys.stderr)
        return 2

    ray_domains = {}
    ray_by_ip = {}
    node_rows = []
    dead_nodes = []
    for node in nodes:
        state, alive = node_state(node)
        domains, gpu_types = node_labels(node)
        ip = node.get("node_ip")
        if not alive:
            dead_nodes.append({"ip": ip, "state": state,
                               "domains": domains, "gpu_types": gpu_types})
            continue
        for domain in domains:
            ray_domains[domain] = gpu_types
        entry = ray_by_ip.setdefault(ip, {"domains": set(), "gpu_types": set()})
        entry["domains"].update(domains)
        entry["gpu_types"].update(gpu_types)
        node_rows.append({
            "ip": ip,
            "state": state,
            "alive": alive,
            "domains": domains,
            "gpu_types": gpu_types,
        })

    instance_rows = []
    for item in instances:
        url = f"http://{item['host']}:{item['port']}/health"
        status = probe(url, args.timeout) if not args.skip_health else -1
        instance_rows.append({
            "id": item["id"], "domain": item["domain"], "role": item["role"]
            if "role" in item else ("prefill" if item in config["prefills"] else "decode"),
            "endpoint": url, "status": status, "ok": args.skip_health or status == 200,
        })

    # A configured domain is only truly registered if the Ray node sitting on
    # the same host IP advertises the same ``domain:<name>`` label.  Checking
    # names alone would hide a rename (e.g. config says 3090a, Ray says 3090).
    configured_by_domain = {}
    for item in instances:
        configured_by_domain.setdefault(item["domain"], set()).add(item["host"])
    domain_problems = []
    for domain in sorted(configured_by_domain):
        for host in sorted(configured_by_domain[domain]):
            entry = ray_by_ip.get(host)
            if entry is None:
                domain_problems.append({"domain": domain, "host": host,
                                        "issue": "host not registered with Ray"})
            elif domain not in entry["domains"]:
                domain_problems.append({
                    "domain": domain, "host": host,
                    "issue": f"Ray labels are {sorted(entry['domains'])}"})

    missing = sorted({row["domain"] for row in domain_problems})
    unavailable = [] if args.skip_health else [row for row in instance_rows if not row["ok"]]
    report = {
        "dashboard": args.dashboard,
        "ray_nodes": node_rows,
        "ray_dead_nodes": dead_nodes,
        "ray_domains": {domain: types for domain, types in sorted(ray_domains.items())},
        "configured_domains": sorted(configured_domains),
        "domain_problems": domain_problems,
        "domains_missing_from_ray": missing,
        "instances": instance_rows,
        "instances_unhealthy": [row["id"] for row in unavailable],
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Ray dashboard: {args.dashboard}")
        print(f"{'ray node':>16}  {'state':>5}  domains / gpu_types")
        for row in node_rows:
            print(f"{row['ip']:>16}  {row['state']:>5}  "
                  f"{','.join(row['domains']) or '-'} / {','.join(row['gpu_types']) or '-'}")
        for row in dead_nodes:
            print(f"{row['ip']:>16}  {row['state']:>5}  "
                  f"{','.join(row['domains']) or '-'} / {','.join(row['gpu_types']) or '-'}"
                  f"   (ignored)")
        print()
        print(f"{'instance':>12}  {'domain':>8}  {'role':>8}  {'health':>8}")
        for row in instance_rows:
            print(f"{row['id']:>12}  {row['domain']:>8}  {row['role']:>8}  {row['status']:>8}")
        if domain_problems:
            print("\nWARNING: configured domains do not match Ray labels:")
            for row in domain_problems:
                print(f"  domain {row['domain']} @ {row['host']}: {row['issue']}")
        if unavailable:
            print(f"WARNING: unhealthy instances: {[row['id'] for row in unavailable]}")
        if not domain_problems and not unavailable:
            if args.skip_health:
                print("\nall configured domains are registered with Ray")
            else:
                print("\nall configured domains are registered and all instances are healthy")

    # ``domain_problems`` covers label/host mismatches that ``missing`` only
    # summarises per domain; both must fail a --check run.
    if args.check and (domain_problems or unavailable):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

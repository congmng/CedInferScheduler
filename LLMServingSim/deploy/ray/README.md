# Ray control plane

`start_ray_cluster.sh` creates a lightweight Ray control plane over the
experiment hosts. The head node is the 4090 host by default; the other hosts
join with stable `domain:*` and `gpu_type:*` resource labels:

| 主机 | `domain:*` | `gpu_type:*` |
|---|---|---|
| `10.212.67.167` | `4090` | `RTX4090` |
| `10.212.67.68` | `3090a` | `RTX3090` |
| `10.212.70.196` | `5090` | `RTX5090` |
| `10.212.70.38` | `3090b` | `RTX3090` |
| `10.66.0.15` (`:2222`) | `a100` | `A100` |

The `domain:*` label is the name that `deploy/real_lmcache_pd/router_config.json`
uses for the same host, so the two must stay in sync.

The A100 host is registered under its overlay address `10.66.0.15`; `10.70.251.47`
is the same machine on an older route and must not be used. Its SSH port is
`2222`, so `WORKER_A100_PORT` (and every hand-run `ssh`) has to carry `-p 2222`.

## Inventory and health check

`domain_inventory.py` turns the Ray cluster into the source of truth for "which
domains exist". It reads node labels from the Ray dashboard REST API
(`:8265/api/v0/nodes`), cross-checks every domain declared in `router_config.json`
against the labels of the Ray node on the same host IP, and probes the health
endpoint of every enabled Prefill/Decode instance:

```bash
python3 deploy/ray/domain_inventory.py            # human readable table
python3 deploy/ray/domain_inventory.py --json     # machine readable report
python3 deploy/ray/domain_inventory.py --check    # non-zero if a domain or instance is not ready
```

It uses only the standard library, so it runs under any of the Python
interpreters on the head node (the `ray` CLI is bound to the system Python).

## Why vLLM P/D does not run inside Ray

Ray provides discovery, domain/resource labels and health collection. The vLLM
Prefill/Decode containers themselves are launched by
`deploy/real_lmcache_pd/start_multidomain_pd.sh` (host-networked `docker run`
over SSH). Running them as Ray tasks on GPU resources would let the Ray
scheduler and the inference containers race for the same accelerator, so the
two layers are kept separate on purpose.

The A100 host is not started by default because other users reserve its GPUs;
enable it with `WORKER_A100` and an explicit label during a maintenance window.

## Where Ray writes (data disk, not the system disk)

Ray puts its session directory, object spool and **all** of its logs under
`--temp-dir`, which defaults to `/tmp/ray` -- i.e. the system disk. The 3090a
root filesystem runs at 99%, so every host now points Ray at a data-disk
directory (the same paths are declared as `ray_temp_dir` in
`deploy/real_lmcache_pd/router_config.json`):

| 主机 | `--temp-dir` |
|---|---|
| `10.212.67.167` (head) | `/mnt/home/casr/ray` |
| `10.212.67.68` | `/data/sdb/model/casr/ray` |
| `10.212.70.196` | `/data/casr/ray` |
| `10.212.70.38` | `/data/casr/ray` |
| `10.66.0.15` | `/mnt/adminserver-nfsrdma/casr/ray` |

Override any of them with `HEAD_RAY_TEMP` / `WORKER_3090_TEMP` /
`WORKER_5090_TEMP` / `WORKER_3090B_TEMP` / `WORKER_A100_TEMP`. The script prints
the resolved locations and a `ray status` snapshot when it finishes, so a
misplaced temp dir is visible immediately.

#!/usr/bin/env bash
# Start a multi-domain LMCache P/D topology described by router_config.json.
#
# Each Prefill/Decode entry names its host, GPU, port and receiver ports.  The
# script renders per-instance LMCache configs, copies them to the target host,
# and launches host-networked vLLM containers.  The router
# (disagg_router.py) then steers individual requests across these instances.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/router_config.json}"
IMAGE="${IMAGE:-vllm/vllm-openai:casr029}"
MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
CONFIG_DIR="${CONFIG_DIR:-/home/buaa}"
TAG="${TAG:-casr-md}"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-/tmp/casr-vllm-cache}"
# Per-instance LMCache PD staging buffer.  It must hold every KV payload that is
# in flight at once; a long shared prefix under concurrency overruns 1 GiB and
# the receiver spins on "Failed to allocate memory object, retrying...".
PD_BUFFER_SIZE="${PD_BUFFER_SIZE:-1073741824}"
# Where the PD staging buffer lives.  ``cuda`` (LMCache's own default here) puts
# it in GPU memory, which competes with the model and exhausts after a few
# hundred unique prefixes -- the receiver then spins on
# ``pd_backend.py:939 Failed to allocate memory object`` and every later request
# hangs.  ``cpu`` keeps it in host RAM instead.
PD_BUFFER_DEVICE="${PD_BUFFER_DEVICE:-cuda}"
# ``pd_skip_proxy_notification`` lets the Prefill (sender) leg return to the
# client before its NIXL push has landed.  The LMCache receiver registers a
# chunk key at *allocation* time -- i.e. before the bytes arrive -- so a Decode
# that starts too early reads whatever the slot held before and generates
# garbage.  Measured 2026-09-13 on the A100 pair: with the flag on, the first
# 5 requests were correct and every later one returned 64 newlines or an
# immediate EOS; with the flag off, 20/20 requests (CNN/DailyMail articles,
# 1000-1800 token prompts) produced on-topic summaries.
# The flag was originally needed by LMCache's ``async`` PD backend, which
# demands a ZMQ proxy notification that an HTTP proxy cannot provide; we run
# ``pd_backend_mode: sync``, which does not need it.  Default is therefore
# "do not skip" (0).  Set to 1 only to reproduce the broken behaviour.
PD_SKIP_PROXY_NOTIFICATION="${PD_SKIP_PROXY_NOTIFICATION:-0}"
# Address of the ZMQ PULL endpoint that receives LMCache's "KV landed"
# notification.  Only used when PD_SKIP_PROXY_NOTIFICATION=0: the sender
# asserts on a missing proxy, and the orchestrator (the router) must hold the
# Decode leg back until the notification arrives.
# Empty means "no proxy endpoint": the sender then fails its
# ``pd_proxy_host is not None`` assertion, its LMCache engine dies, and the
# Decode recomputes the prompt locally.  That is the only combination measured
# on 2026-09-13 that answers *correctly* (see the warning below); set both
# variables to switch the notification path on instead.
PD_PROXY_HOST="${PD_PROXY_HOST:-}"
PD_PROXY_PORT="${PD_PROXY_PORT:-}"
# ``lmcache`` (default) uses LMCache's PD backend, which on 2026-09-13 shipped
# KV that the Decode could not use (see the deployment record).  ``native``
# switches both roles to vLLM's own ``NixlConnector``: the Prefill returns
# ``kv_transfer_params`` (remote_block_ids/engine id) and the router must hand
# them to the Decode leg -- that is the officially supported disaggregated
# path, and it never touches LMCache's staging buffer.
KV_TRANSFER_BACKEND="${KV_TRANSFER_BACKEND:-lmcache}"
# vLLM's NixlConnector refuses a handshake between engines whose compatibility
# hash differs -- i.e. between our vLLM 0.26 (A100 image) and 0.29 (the rest)
# domains.  Set to 1 to disable that check and see whether the transfer is
# still correct; the layouts are expected to match for the same model/dtype,
# but this must be verified per pair before any result is quoted.
KV_NATIVE_ALLOW_MIXED="${KV_NATIVE_ALLOW_MIXED:-0}"
# KV cache representation.  Empty keeps vLLM's default (bf16 for a bf16
# model); ``fp8`` halves the KV footprint per token (measured 2026-09-13 on
# A100: 12.14 GiB / 176,736 tokens = 73.7 KB/token vs 147.5 KB/token in bf16),
# which is the controlled axis for "how does KV size change the scheduling
# decision": same model, same hardware, half the bytes to move and to cache.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
# Restrict the launch to a subset of instance ids / domains (comma separated),
# so a single domain can be (re)started without touching the others.  Empty
# means "launch everything in the config".
ONLY="${ONLY:-}"
# Instance ids to skip for this run (comma separated).  Complements the config's
# ``enabled`` flag: transient node outages belong to the run, not the checked-in
# topology file.
DISABLED_INSTANCES="${DISABLED_INSTANCES:-}"
# Instance ids to *keep as they are* (comma separated): do not remove, do not
# relaunch, just require that the existing container is healthy.  Needed on a
# node whose NVIDIA userspace/kernel modules drifted apart under
# unattended-upgrade -- already-running containers keep working, but any new
# container fails with ``nvml error: driver/library version mismatch``.
# Safe to use for Decode instances: they hold no prefix cache, so preserving
# them cannot leak cache state between policy runs (Prefills are still cold).
PRESERVE_INSTANCES="${PRESERVE_INSTANCES:-}"
PLAN="${PLAN:-/tmp/casr-multidomain-plan.sh}"

export CONFIG IMAGE MODEL_NAME GPU_MEMORY_UTILIZATION MAX_MODEL_LEN MAX_NUM_SEQS CONFIG_DIR TAG VLLM_CACHE_DIR PD_BUFFER_SIZE PD_BUFFER_DEVICE PD_SKIP_PROXY_NOTIFICATION PD_PROXY_HOST PD_PROXY_PORT KV_TRANSFER_BACKEND KV_NATIVE_ALLOW_MIXED KV_CACHE_DTYPE ONLY DISABLED_INSTANCES PRESERVE_INSTANCES

python3 - > "$PLAN" <<'PY'
import json, os

config = json.load(open(os.environ["CONFIG"]))
config_dir = os.environ["CONFIG_DIR"]
tag = os.environ["TAG"]
image = os.environ["IMAGE"]
hosts = config["hosts"]

def ssh_prefix(host_name):
    spec = hosts[host_name]
    return f"ssh -o BatchMode=yes -p {spec.get('ssh_port', 22)} {spec['ssh']}"

def lmcache_yaml(role, item):
    lines = ["local_cpu: false", "max_local_cpu_size: 0", "max_local_disk_size: 0",
             "remote_serde: NULL", "", "enable_pd: true",
             f'pd_role: "{"sender" if role == "prefill" else "receiver"}"']
    if role == "decode":
        lines += [f'pd_peer_host: "{item["host"]}"',
                  f'pd_peer_init_port: [{item["init_port"]}]',
                  f'pd_peer_alloc_port: [{item["alloc_port"]}]',
                  f'pd_peer_query_port: [{item["query_port"]}]']
    lines += [f'pd_buffer_size: {os.environ["PD_BUFFER_SIZE"]}',
              f'pd_buffer_device: "{os.environ["PD_BUFFER_DEVICE"]}"',
              'pd_backend_mode: "sync"', 'save_unfull_chunk: true',
              'nixl_backends: ["UCX"]', 'transfer_channel: "nixl"']
    if role == "prefill" and os.environ.get("PD_SKIP_PROXY_NOTIFICATION", "0") == "1":
        lines.append("pd_skip_proxy_notification: true")
    if (role == "prefill" and os.environ.get("PD_SKIP_PROXY_NOTIFICATION", "0") != "1"
            and os.environ.get("PD_PROXY_PORT", "")):
        lines.append(f'pd_proxy_host: "{os.environ["PD_PROXY_HOST"]}"')
        lines.append(f'pd_proxy_port: {os.environ["PD_PROXY_PORT"]}')
    return "\n".join(lines) + "\n"

print("set -euo pipefail")
if not os.environ.get("PD_PROXY_PORT", "") and os.environ.get(
        "PD_SKIP_PROXY_NOTIFICATION", "0") != "1":
    print('# NOTE: PD_PROXY_PORT is unset, so the Prefill senders get no proxy\n'
          '# endpoint.  LMCache asserts on that and marks the engine as failed\n'
          '# init, which means no KV is cached or transferred and every Decode\n'
          '# recomputes the prompt itself.  Outputs are correct, the KV offload\n'
          '# is not there.  See docs/五台异构实验环境部署记录.md\n'
          '# "KV 搬运本身没有生效" for the three configurations and their\n'
          '# measured behaviour.')
only = {token.strip() for token in os.environ.get("ONLY", "").split(",") if token.strip()}
disabled = {token.strip() for token in os.environ.get("DISABLED_INSTANCES", "").split(",") if token.strip()}
preserve = {token.strip() for token in os.environ.get("PRESERVE_INSTANCES", "").split(",") if token.strip()}
# vLLM's NixlConnector binds a ZMQ handshake listener per engine, defaulting to
# a fixed port; two instances on the same host then collide
# ("Address already in use (addr='tcp://localhost:5600')").  Give every
# instance its own port unless the config names one.
instance_seq = 0
for role, key in (("prefill", "prefills"), ("decode", "decodes")):
    for item in config[key]:
        instance_seq += 1
        # A disabled instance stays in the file (ports, gpu, link budget are
        # still documented) but is not launched unless ONLY names it.
        if (not item.get("enabled", True) or item["id"] in disabled) and item["id"] not in only:
            continue
        if only and item["id"] not in only and item["domain"] not in only:
            continue
        spec = hosts[item["domain"]]
        ssh = ssh_prefix(item["domain"])
        # The local host shares the repo account but not the remote ``buaa``
        # home, so a host may declare its own staging directory.
        host_config_dir = spec.get("config_dir", config_dir)
        name = f"{tag}-{item['id']}"
        yaml_path = f"/tmp/{name}.yaml"
        open(yaml_path, "w").write(lmcache_yaml(role, item))
        remote_cfg = f"{host_config_dir}/casr-lmcache-pd/{name}.yaml"
        kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        rpc = "producer1" if role == "prefill" else "consumer1"
        if os.environ.get("KV_TRANSFER_BACKEND", "lmcache") == "native":
            # vLLM's own disaggregated path.  ``kv_both`` on both roles; the
            # Prefill answers with the remote block ids the Decode needs.
            kv_port = int(item.get("kv_port", 5600 + instance_seq))
            extra = ('\\"backends\\":[\\"UCX\\"]'
                     + (',\\"enforce_handshake_compat\\":false'
                        if os.environ.get("KV_NATIVE_ALLOW_MIXED", "0") == "1"
                        else ""))
            kv = ('{\\"kv_connector\\":\\"NixlConnector\\",'
                  '\\"kv_role\\":\\"kv_both\\",'
                  f'\\"kv_port\\":{kv_port},'
                  f'\\"kv_connector_extra_config\\":{{{extra}}}}}')
            lmcache_env = ""
            # The handshake listener port comes from this env var, not from the
            # config key (two instances on one host collide on the default
            # 5600), and the advertised host must be this instance's own
            # address for a remote Decode to reach it.
            nixl_env = (f'-e VLLM_NIXL_SIDE_CHANNEL_HOST={item["host"]} '
                        f'-e VLLM_NIXL_SIDE_CHANNEL_PORT={kv_port} ')
        else:
            kv = ('{\\"kv_connector\\":\\"LMCacheConnectorV1\\",\\"kv_role\\":\\"%s\\",'
                  '\\"kv_connector_extra_config\\":{\\"discard_partial_chunks\\":false,'
                  '\\"lmcache_rpc_port\\":\\"%s\\"}}') % (kv_role, rpc)
            lmcache_env = f'-e LMCACHE_CONFIG_FILE=/pd-config/{name}.yaml '
            nixl_env = ""
        model = item.get("model", spec["model"])
        ucx = spec["ucx_device"]
        # Shared machines (e.g. the A100 node also runs training jobs) need a
        # smaller footprint than the global default.  Allow host- and
        # instance-level overrides of the vLLM memory/length knobs.
        def knob(key, default):
            return item.get(key, spec.get(key, default))

        gpu_util = knob("gpu_memory_utilization", os.environ["GPU_MEMORY_UTILIZATION"])
        kv_cache_dtype_arg = (f"--kv-cache-dtype {os.environ['KV_CACHE_DTYPE']} "
                              if os.environ.get("KV_CACHE_DTYPE") else "")
        max_len = knob("max_model_len", os.environ["MAX_MODEL_LEN"])
        max_seqs = knob("max_num_seqs", os.environ["MAX_NUM_SEQS"])
        # Domains on a different vLLM/LMCache release (e.g. the A100 node ships
        # vLLM 0.26.0) can override the image per host or per instance.
        instance_image = item.get("image", spec.get("image", image))
        # Hosts whose dockerd predates CDI-aware ``--gpus`` (e.g. 29.1.3 with no
        # /etc/cdi spec at daemon start) must pin the device through the legacy
        # nvidia runtime instead; the flag is set per host in the config.
        runtime = spec.get("gpu_runtime")
        gpu_flags = (f"-e NVIDIA_VISIBLE_DEVICES={item['gpu']} --runtime={runtime}"
                     if runtime else f"--gpus 'device={item['gpu']}'")
        # ---- launch plumbing -------------------------------------------------
        # Every host writes its logs, its vLLM compile cache and its Ray state
        # to a *data* disk (``log_dir``/``cache_dir`` in the host block): the
        # 3090a root filesystem sits at 99%, and ``/var/lib/docker`` -- where
        # ``docker logs`` would otherwise accumulate -- is on the system disk of
        # every host.
        #
        # ``launcher`` picks how vLLM is started:
        #   docker (default) -- host-networked ``docker run``, as before;
        #   venv             -- ``<venv>/bin/vllm`` straight on the host, for
        #                       nodes where buaa has no usable docker access
        #                       (the A100 host today).
        launcher = item.get("launcher", spec.get("launcher", "docker"))
        log_dir = knob("log_dir", "/tmp/casr-logs")
        cache_dir = knob("cache_dir", os.environ["VLLM_CACHE_DIR"])
        log_path = f"{log_dir}/{name}.log"
        script_path = f"{log_dir}/{name}.sh"
        if launcher == "venv":
            venv_python = item.get("venv_python", spec.get("venv_python"))
            if not venv_python:
                raise SystemExit(f"{item['id']}: launcher=venv needs venv_python")
            vllm_bin = os.path.join(os.path.dirname(venv_python), "vllm")
            served_model = model
            lmcache_config_path = remote_cfg
            # The venv runs on the host, so host paths are the right ones.
            pid_path = f"{log_dir}/{name}.pid"
            runtime_cache_dir = cache_dir
            # docker pins the device with ``--gpus device=N``; the venv path
            # has to do it itself, otherwise vLLM grabs cuda:0 -- which on the
            # A100 host belongs to another user's job (measured 2026-09-15:
            # "Free memory on device cuda:0 (5.42/79.25 GiB) is less than
            # desired GPU memory utilization (0.35, 27.74 GiB)").
            gpu_env = [f"export CUDA_VISIBLE_DEVICES={item['gpu']}"]
        else:
            vllm_bin = "vllm"
            served_model = "/model"
            lmcache_config_path = f"/pd-config/{name}.yaml"
            # Inside the container only /casr-logs and /root/.cache/vllm exist:
            # the host ``log_dir``/``cache_dir`` are not visible at their own
            # paths, so the pid file and the cache root have to use the mount
            # points.  (Writing the host path here made the container exit at
            # ``set -e`` on the very first line.)
            pid_path = f"/casr-logs/{name}.pid"
            runtime_cache_dir = "/root/.cache/vllm"
            gpu_env = []
        # ``kv`` is escaped for the old inline ``ssh "... '{"k":"v"}' ..."`` form
        # (the outer double quotes ate one backslash).  The command now travels
        # in a script file, so the JSON has to be plain.
        kv_json = kv.replace('\\"', '"')
        serve = (f"{vllm_bin} serve {served_model} "
                 f"--served-model-name {os.environ['MODEL_NAME']} --host 0.0.0.0 "
                 f"--port {item['port']} --enforce-eager --dtype bfloat16 "
                 f"{kv_cache_dtype_arg}"
                 f"--gpu-memory-utilization {gpu_util} "
                 f"--max-model-len {max_len} --max-num-seqs {max_seqs} "
                 f"--kv-transfer-config '{kv_json}'")
        env_lines = gpu_env + ["export PYTHONHASHSEED=123",
                     "export UCX_TLS=cuda_copy,tcp",
                     f"export UCX_NET_DEVICES={ucx}",
                     f"export UCX_TCP_INTERFACE={ucx}",
                     f"export VLLM_CACHE_ROOT={runtime_cache_dir}"]
        if lmcache_env:
            env_lines.append(f"export LMCACHE_CONFIG_FILE={lmcache_config_path}")
        if nixl_env:
            env_lines.append(f"export VLLM_NIXL_SIDE_CHANNEL_HOST={item['host']}")
            env_lines.append(f"export VLLM_NIXL_SIDE_CHANNEL_PORT={kv_port}")
        # ``$$`` is recorded before ``exec``, so the pid file names the server
        # process itself, which is what the stop path kills.
        local_script = f"/tmp/{name}.sh"
        open(local_script, "w").write("\n".join(
            ["#!/bin/bash", "set -euo pipefail", f'echo $$ > "{pid_path}"']
            + env_lines + [f"exec {serve}"]) + "\n")
        print(f'scp -q -P {spec.get("ssh_port", 22)} {yaml_path} '
              f'{spec["ssh"]}:{yaml_path}')
        # The instance scripts go to /tmp first: ``log_dir`` may not exist yet on
        # a host that has never run a P/D instance (scp cannot create it).
        print(f'scp -q -P {spec.get("ssh_port", 22)} {local_script} '
              f'{spec["ssh"]}:/tmp/{name}.sh')
        print(f'{ssh} "mkdir -p {host_config_dir}/casr-lmcache-pd {log_dir} {cache_dir} '
              f'&& mv {yaml_path} {remote_cfg} && mv /tmp/{name}.sh {script_path} '
              f'&& chmod +x {script_path}"')
        if item["id"] in preserve:
            # Leave the running instance alone: this host cannot create new
            # containers, and the instance is healthy as it stands.
            print(f'echo "  preserving existing instance {name}"')
            continue
        if launcher == "venv":
            local_runner = f"/tmp/{name}.run.sh"
            open(local_runner, "w").write("\n".join([
                "#!/bin/bash", "set -euo pipefail", f'cd "{log_dir}"',
                f'if [ -f "{pid_path}" ]; then kill "$(cat {pid_path})" 2>/dev/null || true; sleep 2; fi',
                f'pkill -f "{name}.sh" >/dev/null 2>&1 || true',
                f'nohup setsid bash "{script_path}" >> "{log_path}" 2>&1 < /dev/null &',
                "sleep 4",
                f'if [ -f "{pid_path}" ] && kill -0 "$(cat {pid_path})" 2>/dev/null; then',
                f'  echo "  started {name} (pid $(cat {pid_path})), log {log_path}";',
                "else",
                f'  echo "  FAILED to start {name}" >&2; tail -20 "{log_path}" >&2; exit 1;',
                "fi"]) + "\n")
            print(f'scp -q -P {spec.get("ssh_port", 22)} {local_runner} '
                  f'{spec["ssh"]}:/tmp/{name}.run.sh')
            print(f'{ssh} "mv /tmp/{name}.run.sh {log_dir}/{name}.run.sh"')
            print(f'{ssh} "bash {log_dir}/{name}.run.sh"')
            continue
        run = (
            f'{ssh} "docker rm -f {name} >/dev/null 2>&1 || true; '
            # ``docker rm -f`` returns before the daemon has released the name
            # on the slower 3090 hosts, so the launch right after it can fail
            # with "container name is already in use".  Retry the launch (with
            # another remove in between) instead of aborting the whole topology.
            f'for _try in 1 2 3 4 5; do '
            # ``--log-driver=none``: the container's stdout is redirected into
            # the mounted data-disk log file, so json-file would only duplicate
            # it under /var/lib/docker (system disk).
            f'docker run -d --name {name} --network host --ipc host --shm-size 16g '
            f"--log-driver=none "
            f"{gpu_flags} "
            f'-e PYTHONHASHSEED=123 -e UCX_TLS=cuda_copy,tcp '
            f'-e UCX_NET_DEVICES={ucx} -e UCX_TCP_INTERFACE={ucx} '
            f'{lmcache_env}{nixl_env}'
            f'-v {cache_dir}:/root/.cache/vllm '
            f'-v {log_dir}:/casr-logs '
            f'-v {model}:/model:ro -v {host_config_dir}/casr-lmcache-pd:/pd-config:ro '
            f'--entrypoint /bin/bash {instance_image} '
            f"-c 'exec bash /casr-logs/{name}.sh >> /casr-logs/{name}.log 2>&1' "
            f"&& break; "
            f'sleep 3; docker rm -f {name} >/dev/null 2>&1 || true; done; '
            f'docker inspect -f "{{{{.State.Status}}}}" {name} >/dev/null 2>&1 '
            f'|| {{ echo "failed to launch {name} after 5 attempts" >&2; exit 1; }}'
            + '"')
        print(run)
PY

if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY_RUN: launch plan written to $PLAN (nothing executed)"
  exit 0
fi
bash "$PLAN"
echo "Topology launched:"
python3 -c "import json;d=json.load(open('$CONFIG'));print('  prefills:',[p['id'] for p in d['prefills']]);print('  decodes:',[x['id'] for x in d['decodes']])"

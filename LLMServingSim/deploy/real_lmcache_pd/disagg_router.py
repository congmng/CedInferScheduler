"""Multi-domain LMCache P/D router with pluggable placement policies.

Unlike ``disagg_proxy_pd.py`` (a single fixed P/D pair), this router fronts
several Prefill and Decode instances spread over heterogeneous domains.  The
placement policy decides, per request, which Prefill computes the prompt and
which Decode receives the pushed KV cache.  It is the real-system counterpart
of the CASR affinity plan evaluated in the simulator.

Policies
--------
``rr``    round-robin over Prefill and Decode instances (isolation baseline).
``load``  least in-flight/capacity on both roles (vLLM-style baseline).
``casr``  sticky prefix-affinity Prefill plus a heterogeneous, network-aware
          Decode cost that mirrors the simulator's ``f_ijk`` cost terms.
``cache_aware``  longest-prefix-match routing with a load spill guard, i.e. the
          SGLang / vLLM production-stack cache-aware router: prefer the Prefill
          that already holds the longest matching prefix, but fall back to
          least-loaded once that Prefill passes a utilisation threshold.
``casr_lp``      slow-layer ``AffinityPlan`` produced by the *shared* CASR
                 control loop (``casr.flow_solver`` LP, ``casr.plan_builder``),
                 replayed by the fast router with deficit weighted routing.
``casr_full``    ``casr_lp`` plus the ``+P``/``-P`` structural lifecycle, which
                 starts/stops real Prefill containers through the Docker API.

Ablation policies reuse the same router and differ only in one cost term:

``random``           uniformly random placement (lower bound).
``casr_noservice``   ``casr`` minus the per-instance ``service_ms`` term, i.e.
                     network + utilization only.
``casr_noaffinity``  ``casr`` minus prefix affinity, i.e. cost-only placement.
``casr_nonetwork``   ``casr`` minus the domain-link term, i.e. affinity +
                     measured service time + utilization only.

The router never mutates instance state beyond the request it is forwarding;
all bookkeeping is local, which keeps the comparison fair across policies.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx


def disabled_instance_ids():
    """Instance ids to skip for this run.

    Topology outages (a contended shared node, a driver upgrade) are transient
    environment state, not part of the checked-in topology, so the runner can
    pass ``DISABLED_INSTANCES=p3090b,p4090`` instead of editing the config.
    """
    return {token.strip() for token in
            os.environ.get("DISABLED_INSTANCES", "").split(",") if token.strip()}


def instance_enabled(spec):
    if spec.get("id") in disabled_instance_ids():
        return False
    return bool(spec.get("enabled", True))


def parse_sse_line(line, state):
    """Fold one Server-Sent-Events line into a streaming request's ``state``.

    ``state`` carries ``first_token_ms``, ``tokens`` and ``usage``.  Only the
    first chunk that actually carries generated text counts as the first token
    (an SSE preamble or a role-only chunk must not), which is what makes the
    recorded value a real TTFT rather than a connection-setup time.
    """
    if not line.startswith("data:"):
        return state
    body = line[len("data:"):].strip()
    if not body or body == "[DONE]":
        return state
    try:
        event = json.loads(body)
    except ValueError:
        return state
    if event.get("usage"):
        state["usage"] = event["usage"]
    for choice in event.get("choices") or ():
        if choice.get("text"):
            state["tokens"] = int(state.get("tokens", 0)) + 1
            if state.get("first_token_ms") is None:
                state["first_token_ms"] = state.get("now_ms")
    return state


def request_slo(request):
    """Per-request SLO carried on the client headers, or ``(None, None)``.

    The bound travels with the request because it belongs to the workload
    (dataset row / application), not to the router: one policy run can mix
    requests with different TTFT/TPOT budgets.
    """
    def as_float(name):
        raw = request.headers.get(name)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    return as_float("x-slo-ttft-ms"), as_float("x-slo-tpot-ms")


def slo_verdict(ttft_ms, tpot_ms, slo_ttft_ms, slo_tpot_ms):
    """``True``/``False``/``None`` for met / violated / not measurable."""
    if slo_ttft_ms is not None:
        if ttft_ms is None:
            return None
        if ttft_ms > slo_ttft_ms:
            return False
    if slo_tpot_ms is not None:
        if tpot_ms is None:
            return None
        if tpot_ms > slo_tpot_ms:
            return False
    return True


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:  # the plan-based policies also need the shared CASR package
    from casr.prefix_profiler import derive_class_id
except ImportError:  # pragma: no cover - reported when a plan policy starts
    def derive_class_id(*_args, **_kwargs):
        raise RuntimeError("CASR package is not importable; set PYTHONPATH to the "
                           "repository's serving/ directory")

CHUNK_TOKENS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--config", required=True, help="router config JSON")
    parser.add_argument("--policy", default="casr",
                        choices=("rr", "load", "random", "casr",
                                 "casr_noservice", "casr_noaffinity",
                                 "casr_nonetwork", "cache_aware", "casr_lp",
                                 "casr_full"))
    parser.add_argument("--metrics", default="", help="per-request metrics JSONL path")
    parser.add_argument("--state-output", default="",
                        help="JSONL dump of the control plane's per-tick view "
                             "(same schema as the simulator's --casr-state-output), "
                             "so a real run can be replayed offline tick by tick")
    parser.add_argument("--health-interval-s", type=float, default=5.0)
    parser.add_argument("--scale-backend", default="",
                        choices=("", "noop", "docker"),
                        help="override the config's casr.scale_backend; 'docker' "
                             "enables the +P/-P container lifecycle")
    return parser.parse_args()


PLAN_POLICIES = ("casr_lp", "casr_full")


def env_float(name: str, default: float) -> float:
    """Read a numeric env var, treating empty/unset as the default.

    The run script forwards these knobs as ``-e NAME="${NAME:-}"``, so an
    empty string must not turn into ``float('')``.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


class KVLandingTracker:
    """Receives LMCache's "KV has landed" notifications for a request.

    LMCache's PD sender pushes a request's KV to the receiver and then sends a
    ``ProxyNotif(req_id)`` message over a ZMQ PUSH socket to
    ``pd_proxy_host:pd_proxy_port``.  The orchestrator is the proxy: it is
    supposed to hold the Decode leg back until that notification arrives.

    This is not decoration.  The receiver registers a chunk key at
    *allocation* time, i.e. before the bytes arrive, so a Decode sent right
    after the Prefill response reads whatever the slot held before and
    generates garbage (measured 2026-09-13: 64 newlines / immediate EOS, while
    health checks and ``LMCache hit tokens: N/N`` both looked fine).  Disabling
    the notification instead (``pd_skip_proxy_notification: true``) removes the
    assertion failure but keeps the early read.

    Binding this PULL socket and waiting on it is what makes the handoff
    correct *and* keeps the KV transfer itself alive.
    """

    def __init__(self, port: int, host: str = "0.0.0.0", wait_s: float = 60.0):
        self.port = int(port)
        self.host = host
        self.wait_s = float(wait_s)
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._socket = None
        self._running = False
        self._thread = None
        self.stats = {"received": 0, "waited": 0, "timeouts": 0,
                      "orphans": 0, "errors": 0}

    def start(self) -> None:
        import zmq

        context = zmq.Context.instance()
        self._socket = context.socket(zmq.PULL)
        self._socket.bind(f"tcp://{self.host}:{self.port}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="kv-landing")
        self._thread.start()
        print(f"[kv-landing] PULL socket bound on tcp://{self.host}:{self.port}",
              flush=True)

    def _loop(self) -> None:
        import msgspec

        while self._running:
            try:
                raw = self._socket.recv()
            except Exception:  # noqa: BLE001 - socket closed on shutdown
                return
            request_id = None
            try:
                message = msgspec.msgpack.decode(raw)
                if isinstance(message, dict):
                    request_id = message.get("req_id")
                else:
                    request_id = getattr(message, "req_id", None)
            except Exception:  # noqa: BLE001 - never kill the listener
                self.stats["errors"] += 1
            with self._lock:
                event = self._events.get(request_id)
                if event is None:
                    # Notification arrived before the handler started waiting.
                    event = self._events[request_id] = threading.Event()
                    self.stats["orphans"] += 1
                event.set()
                self.stats["received"] += 1

    def wait(self, request_id: str, timeout_s: float | None = None) -> bool:
        """Block until this request's KV has landed (or the timeout expires)."""
        timeout = self.wait_s if timeout_s is None else float(timeout_s)
        with self._lock:
            event = self._events.setdefault(request_id, threading.Event())
        landed = event.wait(timeout)
        with self._lock:
            self._events.pop(request_id, None)
        self.stats["waited"] += 1
        if not landed:
            self.stats["timeouts"] += 1
        return landed

    def stop(self) -> None:
        self._running = False
        if self._socket is not None:
            self._socket.close(linger=0)


class Instance:
    def __init__(self, spec, role):
        self.id = str(spec["id"])
        self.host = str(spec["host"])
        self.port = int(spec["port"])
        self.domain = str(spec.get("domain", self.host))
        self.role = role
        self.speed = float(spec.get("speed", 1.0))
        self.capacity = max(1.0, float(spec.get("capacity", 1.0)))
        # Concurrent-request budget the engine was actually started with
        # (``--max-num-seqs``).  The cache-aware spill guard compares against
        # this, not against ``capacity`` (a requests/s figure that an in-flight
        # count can never reach).
        self.max_inflight = max(1.0, float(spec.get("max_num_seqs", 16)))
        # Integer identity shared with the CASR control loop, so the plan's
        # ``prefill_weights`` / ``decode_weights`` key straight back to a real
        # HTTP endpoint without a second naming scheme.
        self.instance_id = int(spec.get("instance_id", 0))
        self.enabled = instance_enabled(spec)
        # Measured service time for the instance's role, in milliseconds.  This
        # is what makes the cost heterogeneity-aware: capacity only shapes the
        # queueing term, while service_ms captures the raw compute-speed gap
        # between accelerators (e.g. 5090 vs 3090 Decode).
        self.service_ms = float(spec.get("service_ms", 0.0))
        self.base_url = f"http://{self.host}:{self.port}/v1"
        self.health_url = f"http://{self.host}:{self.port}/health"
        self.init_port = int(spec.get("init_port", 55555))
        self.alloc_port = int(spec.get("alloc_port", 55556))
        self.query_port = int(spec.get("query_port", 55557))
        self.inflight = 0
        # Prefill-side in-flight only: incremented while this instance is
        # producing KV and decremented when the Prefill response returns.  The
        # generic ``inflight`` spans Prefill *and* Decode, so under load every
        # instance sits near the client's concurrency cap and an
        # ``inflight / max_num_seqs`` utilisation saturates for all of them --
        # the comparison then degenerates to the (lagging) latency EWMA and the
        # router keeps feeding an instance whose Prefill queue is already deep.
        self.prefill_inflight = 0
        self.healthy = True
        # EWMA of the observed end-to-end Prefill latency for this instance.
        # ``service_ms`` is a *calibration* constant measured on the deployment's
        # reference prompt length (a few hundred tokens for the ShareGPT/Dolly
        # traces); it is wrong by an order of magnitude for long prompts
        # (CNN/DailyMail averages 1250 tokens).  The measured latency carries
        # the prompt-length dependence for free, so the cost model uses it as
        # soon as it has seen a request and falls back to the constant before.
        self.prefill_ms_ewma = 0.0

    def as_dict(self):
        return {"id": self.id, "role": self.role, "host": self.host, "port": self.port,
                "domain": self.domain, "speed": self.speed, "capacity": self.capacity,
                "healthy": self.healthy, "inflight": self.inflight,
                "prefill_inflight": self.prefill_inflight,
                "service_ms": self.service_ms, "instance_id": self.instance_id}


def build_instances(specs, role):
    """Mirror ``casr_control.slot_instance_ids`` when numbering real endpoints.

    Disabled slots keep their index so the router and the CASR control loop
    agree on every instance id even when a shared node is switched off.
    """
    instances = []
    for index, spec in enumerate(specs or ()):
        if not instance_enabled(spec):
            continue
        instance = Instance(spec, role)
        instance.instance_id = int(spec.get("instance_id", index))
        instances.append(instance)
    return instances


def equalize_capacity(instances, env_name):
    """Flatten one role's fast-router capacities for an ablation run.

    The heuristic policies rank a Decode by ``inflight / capacity``; a slow
    instance also has a *smaller* capacity here (d3090a 110 vs d5090 230), so
    "remove the service term" alone still leaves an instance that looks busier.
    ``EQUALIZE_*_CAPACITY`` removes that second signal too, which is what makes
    a service-term ablation interpretable.  Returns the shared value or None.
    """
    raw = os.environ.get(env_name)
    if not raw or not instances:
        return None
    if raw.strip().lower() == "max":
        shared = max(inst.capacity for inst in instances)
    else:
        shared = float(raw)
    for inst in instances:
        inst.capacity = shared
    return shared


class Router:
    def __init__(self, config, policy, metrics_path, scale_backend="",
                 state_output=""):
        self.policy = policy
        # ``enabled: false`` lets a topology that is physically wired but
        # temporarily unavailable (e.g. a contended shared node) stay in the
        # config without the router health-checking or routing to it.
        self.prefills = build_instances(config.get("prefills"), "prefill")
        self.decodes = build_instances(config.get("decodes"), "decode")
        equalize_capacity(self.prefills, "EQUALIZE_PREFILL_CAPACITY")
        equalize_capacity(self.decodes, "EQUALIZE_DECODE_CAPACITY")
        self.controller = None
        # Optional LMCache PD proxy-notification channel.  Enabled by setting
        # PD_PROXY_PORT; the Prefill senders are configured (by the launcher)
        # to PUSH their "KV landed" notifications here.
        proxy_port = int(os.environ.get("PD_PROXY_PORT") or 0)
        self.kv_landings = None
        if proxy_port:
            self.kv_landings = KVLandingTracker(
                proxy_port,
                host=os.environ.get("PD_PROXY_BIND", "0.0.0.0"),
                wait_s=float(os.environ.get("PD_PROXY_WAIT_S", "60")),
            )
            self.kv_landings.start()
        self._plan_assignments = {}
        self._plan_version = None
        # Why a plan-based policy did or did not follow its plan, per request.
        self._plan_stats = {"hit": 0, "expired": 0, "class_missing": 0}
        self._load_picks = {}
        if policy in PLAN_POLICIES:
            from casr_control import build_controller
            options = dict(config.get("casr", {}))
            if scale_backend:
                options["scale_backend"] = scale_backend
            if state_output:
                options["state_output"] = state_output
            self.controller = build_controller(config, options)
            # The plan is expressed in the control loop's instance ids; make
            # sure the fast router's endpoints carry the same ones.
            for instance in (*self.prefills, *self.decodes):
                real = self.controller.real_by_id(instance.instance_id)
                if real is not None:
                    instance.instance_id = real.instance_id
        self.links = {}
        # Whether the router may serve a request by letting the Decode prefill
        # it locally instead of moving KV across the fabric.  Measured
        # 2026-09-13 on the native NixlConnector path: a handoff costs
        # 585 ms (same host) / 1351 ms (cross host) per 1000 prompt tokens,
        # while the Decode prefilling the same prompt itself costs ~93 ms.
        # Moving KV is therefore a *resource*, not a default --
        # ``LOCAL_PREFILL=never`` restores the always-transfer behaviour for
        # A/B comparison, ``always`` the opposite, ``auto`` decides per request
        # from the two modelled costs.
        self.local_prefill_mode = os.environ.get("LOCAL_PREFILL", "auto")
        # Baselines get the same "prefill locally or hand off" option as CASR.
        # Until 2026-09-14 they were pinned to ``never`` while every CASR policy
        # ran ``auto``, and because a same-host recompute costs 93 ms/1k tokens
        # against 585-1351 ms/1k for a handoff, the CASR arm took the local path
        # in every archived comparison (dolly-r1, saturate-fix-b2, xl-r1/r2/r4,
        # mla-chat-a2, elastic-a1) while ``load``/``cache_aware`` could not.
        # Any delta measured that way mixes "better scheduling" with "allowed to
        # use a path the baseline was denied".  ``LOCAL_PREFILL_BASELINES=never``
        # restores the old asymmetry for an explicit A/B of that effect.
        baseline_policies = ("rr", "load", "random", "cache_aware")
        if (self.policy in baseline_policies
                and os.environ.get("LOCAL_PREFILL_BASELINES", "auto") == "never"):
            self.local_prefill_mode = "never"
        self.local_prefill_ms_per_1k = env_float("LOCAL_PREFILL_MS_PER_1K", 93.0)
        self.transfer_ms_per_1k_local = env_float(
            "TRANSFER_MS_PER_1K_LOCAL", 585.0)
        self.transfer_ms_per_1k_cross = env_float(
            "TRANSFER_MS_PER_1K_CROSS", 1351.0)
        self.transfer_fixed_ms_cross = env_float(
            "TRANSFER_FIXED_MS_CROSS", 48.0)
        # Weight of the queue terms in ``kv_exchange_decision``.  ``0`` reduces
        # the comparison to the two constants (the pre-2026-09-14 behaviour,
        # which priced a local prefill as if the Decode it runs on were idle);
        # ``1`` charges both paths for the queue they actually join.
        self.local_prefill_queue_weight = env_float(
            "LOCAL_PREFILL_QUEUE_WEIGHT", 1.0)
        # Cap on the local-path externality multiple (see
        # ``kv_exchange_decision``).  An unbounded term is wrong in both
        # directions: the pre-2026-09-14 code charged nothing, while charging
        # ``local_ms x batch`` made the router hand the KV over even on the
        # regime where the local path measured 4x cheaper end to end
        # (``elastic-a1``: forced transfer 1394 ms vs local 346 ms).
        self.local_prefill_queue_cap = env_float("LOCAL_PREFILL_QUEUE_CAP", 3.0)
        # ``native`` = vLLM's own disaggregated path (NixlConnector): the
        # Prefill response carries kv_transfer_params (remote block ids) and
        # the Decode leg must receive them verbatim.  ``lmcache`` keeps the
        # historical behaviour (LMCache PD backend, params dropped).
        self.transfer_backend = os.environ.get("PD_TRANSFER_BACKEND", "lmcache")
        # vLLM's NixlConnector demands identical engine configurations on both
        # legs: across vLLM versions it either refuses the handshake
        # ("compatibility hash mismatch") or cannot even decode the peer
        # metadata ("missing field block_strides").  Keep only the largest
        # ``kv_group`` when running native, so the router never builds a pair
        # that is guaranteed to fail; the dropped instances are reported.
        self.kv_group = None
        self.kv_group_dropped: list[str] = []
        if self.transfer_backend == "native":
            host_groups = {name: str(spec.get("kv_group", ""))
                           for name, spec in (config.get("hosts") or {}).items()}
            groups: dict[str, list] = {}
            for instance in (*self.prefills, *self.decodes):
                groups.setdefault(host_groups.get(instance.domain, ""),
                                  []).append(instance)
            if len(groups) > 1:
                self.kv_group = max(groups.items(),
                                    key=lambda item: (len(item[1]), item[0]))[0]
                for group, members in groups.items():
                    if group == self.kv_group:
                        continue
                    for instance in members:
                        self.kv_group_dropped.append(instance.id)
                        (self.prefills if instance.role == "prefill"
                         else self.decodes).remove(instance)
                print(f"[kv] native mode keeps kv_group={self.kv_group!r}; "
                      f"dropped {', '.join(self.kv_group_dropped)} "
                      f"(mixed vLLM versions cannot hand KV to each other)",
                      flush=True)
        for link in config.get("links", ()):
            self.links[(str(link["src"]), str(link["dst"]))] = (
                float(link.get("bw_gbps", 1.0)), float(link.get("rtt_ms", 0.0)))
        # Bytes currently being pushed across each domain pair.  Measured
        # 2026-09-13 (docs/实验结果汇总.md §5.10): a 156 MB cross-domain
        # handoff costs +0.4 ms when the link is idle -- LMCache overlaps the
        # push with the Prefill -- but the same decision costs +31.8% once the
        # offered cross-domain traffic exceeds the link (the 3800-token档
        # needs 3.8 GB/s against ~2 GB/s).  So the price of a transfer is not
        # its own serialisation time, it is the time it has to wait for the
        # bytes already on that link.
        self.link_inflight_bytes = {}
        weights = config.get("weights", {})
        self.w_network = float(weights.get("network", 1.0))
        self.w_prefill = float(weights.get("prefill_load", 1.0))
        self.w_decode = float(weights.get("decode_load", 1.0))
        self.w_service = float(weights.get("service", 1.0))
        self.kv_bytes_per_token = float(weights.get("kv_bytes_per_token", 147456.0))
        # Characters per prompt token, used to size the KV payload before the
        # receiver's tokenizer is consulted.  The default 4.0 fits typical mixed
        # text; repetitive English prose tokenizes closer to 6.7, and using the
        # wrong value skews the network term enough to mis-route Decode.
        self.chars_per_token = max(1.0, float(weights.get("chars_per_token", 4.0)))
        # ``chars_per_token`` is a seed, not a constant: the Prefill leg reports
        # the tokenizer's own prompt-token count, so the router fits the ratio
        # from real traffic (see ``calibrate_prompt_tokens``).
        self.prompt_token_calibration_reqs = int(
            env_float("PROMPT_TOKEN_CALIBRATION_REQS", 8.0))
        self._prompt_chars_observed = 0.0
        self._prompt_tokens_observed = 0.0
        self._prompt_token_samples = 0
        self._prompt_tokens_calibrated = False
        # Downstream timeout.  Without it a stalled KV handoff leaves the
        # handler (and its in-flight counters) hung forever, which wedges every
        # later request and silently kills a whole phase.
        timeouts = config.get("timeouts", {})
        self.downstream_timeout_s = float(timeouts.get("downstream_s", 120.0))
        self.affinity = {}
        # ``cache_aware`` keeps a coarse per-Prefill prefix index.  The key is a
        # hash of the first N characters of the prompt; matching the longest
        # window approximates a radix-tree prefix match without storing prompts.
        self.cache_windows = (2048, 1024, 512, 128)
        self.cache_aware_threshold = float(weights.get("cache_aware_load_threshold", 0.5))
        self._cache_index = {}
        self._bucket_chains = {}
        self._counters = {"prefill": 0, "decode": 0, "requests": 0}
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self._clients = {}
        # -- prefix warm-up --
        # A plan change that adds a ``(class, Prefill)`` edge makes the next
        # real request on that edge pay a full cold prefill.  Warming the class's
        # prefix in the background moves that cost off the client's critical
        # path.  Off by default so non-CASR policies and simulator runs are
        # unaffected.
        casr_options = config.get("casr", {}) or {}
        self.warmup_enabled = bool(casr_options.get("warmup_enabled", False))
        self.warmup_concurrency = max(1, int(casr_options.get("warmup_concurrency", 2)))
        self.warmup_min_share = float(casr_options.get("warmup_min_share", 0.1))
        self._prompt_cache = {}
        self._warmed_edges = set()
        self._warm_inflight = set()
        self._warm_tasks = set()
        self._warm_ok = 0
        self._warm_failed = 0
        # "Warm before switch": once a class is being served from a Prefill
        # whose prefix is cached, a plan that points at a different Prefill is
        # held back until the background warmer has seeded the new edge (or a
        # short deadline passes).  Without it the next request for the class
        # pays a full cold prefill, which is exactly the tail that kept the LP
        # from beating the SGLang-style cache-aware baseline.
        self.warm_before_switch = (self.warmup_enabled and
                                   bool(casr_options.get("warm_before_switch", False)))
        self.warm_switch_timeout_s = float(casr_options.get("warm_switch_timeout_s", 3.0))
        self._serving_prefill = {}
        self._switch_deadline_ns = {}

    def client(self, inst):
        if inst.id not in self._clients:
            self._clients[inst.id] = httpx.AsyncClient(
                base_url=inst.base_url,
                timeout=httpx.Timeout(self.downstream_timeout_s, connect=10.0))
        return self._clients[inst.id]

    async def aclose(self):
        for client in self._clients.values():
            await client.aclose()

    # -- prefix warm-up ---------------------------------------------------
    def remember_prompt(self, class_id, endpoint, payload):
        """Keep one representative prompt per class, for warming its prefix.

        Only the first prompt of a class is kept: every request in a class shares
        the same prefix, so the cached one reproduces the KV the plan's other
        Prefills would have to recompute.
        """
        if not self.warmup_enabled or not class_id or class_id in self._prompt_cache:
            return
        cached = dict(payload)
        cached.pop("stream", None)
        cached.pop("kv_transfer_params", None)
        self._prompt_cache[class_id] = (endpoint, cached)

    def _warm_decode(self, plan, prefill_id, class_id):
        """A healthy Decode that can receive the warm Prefill's KV push."""
        candidates = list(plan.decode_for(prefill_id, class_id))
        candidates += [int(i) for i in plan.fallback_for(prefill_id, class_id)]
        for instance_id in candidates:
            instance = self.decode_by_id(int(instance_id))
            if instance is not None and instance.healthy:
                return instance
        for instance in self.decodes:
            if instance.healthy:
                return instance
        return None

    def warm_targets(self, class_id, plan):
        """``(key, prefill, decode)`` edges the plan names but we have not warmed."""
        targets = []
        for prefill_id, share in plan.prefill_for(class_id).items():
            if share < self.warmup_min_share:
                continue
            key = (class_id, int(prefill_id))
            if key in self._warmed_edges or key in self._warm_inflight:
                continue
            prefill = self.prefill_by_id(int(prefill_id))
            if prefill is None or not prefill.healthy:
                continue
            decode = self._warm_decode(plan, int(prefill_id), class_id)
            if decode is None:
                continue
            targets.append((key, prefill, decode))
        return targets

    def maybe_warm(self, class_id, now_ns, serving_prefill_id=None):
        """Schedule background prefills for a class's newly planned edges.

        ``serving_prefill_id`` is the Prefill the request that triggered this
        call is about to use: it warms its own prefix, so warming it here too
        would recompute the same prefix and push the same 302 MB twice.
        """
        if not self.warmup_enabled or self.controller is None:
            return 0
        if serving_prefill_id is not None:
            # The request we are about to dispatch warms this edge by
            # construction.  Recording it is what lets ``warm_before_switch``
            # hold a warm incumbent later: without it ``_warmed_edges`` stays
            # empty for a *single-homed* class (the plan names exactly the
            # Prefill that is already serving it, and ``warm_targets`` skips
            # the serving Prefill), so the hold could never engage --
            # measured 2026-09-13: ``warmed_edges: 0`` through a whole
            # 600-request run with ``warm_before_switch`` enabled.
            self._warmed_edges.add((class_id, int(serving_prefill_id)))
        if len(self._warm_inflight) >= self.warmup_concurrency:
            return 0
        prompt = self._prompt_cache.get(class_id)
        plan = self._plan(now_ns)
        if prompt is None or plan is None:
            return 0
        endpoint, payload = prompt
        scheduled = 0
        for key, prefill, decode in self.warm_targets(class_id, plan):
            if len(self._warm_inflight) >= self.warmup_concurrency:
                break
            if serving_prefill_id is not None and prefill.instance_id == serving_prefill_id:
                continue
            self._warm_inflight.add(key)
            task = asyncio.create_task(
                self._warm_edge(key, prefill, decode, payload, endpoint))
            self._warm_tasks.add(task)
            task.add_done_callback(self._warm_tasks.discard)
            scheduled += 1
        return scheduled

    async def _warm_edge(self, key, prefill, decode, payload, endpoint):
        """Recompute ``class``'s prefix on ``prefill`` and drop the KV at ``decode``.

        The LMCache handoff is two-legged, so a warm has to drive both the
        producer and the consumer; the consumer's one output token is discarded.
        Warm traffic deliberately bypasses ``observe``/``record`` so it never
        shows up as client demand or in the latency metrics.
        """
        class_id, prefill_id = key
        request_id = f"warm-{class_id}-{prefill_id}-{int(time.time() * 1e6)}"
        try:
            prefill_payload = dict(payload)
            prefill_payload["stream"] = False
            prefill_payload["max_tokens"] = 1
            prefill_payload.pop("max_completion_tokens", None)
            # ``stream_options`` is only valid together with ``stream=True``;
            # the Prefill leg is a non-streaming, max_tokens=1 warm-up.
            prefill_payload.pop("stream_options", None)
            prefill_payload["kv_transfer_params"] = {
                "do_remote_decode": True,
                "do_remote_prefill": False,
            }
            if router.transfer_backend != "native":
                # LMCache's PD backend needs the receiver's alloc/query ports
                # up front; vLLM's NixlConnector learns them from the Prefill
                # response instead.
                prefill_payload["kv_transfer_params"]["disagg_spec"] = {
                    "req_id": request_id,
                    "receiver_host": decode.host,
                    "receiver_init_port": [decode.init_port],
                    "receiver_alloc_port": [decode.alloc_port],
                    "receiver_query_port": [decode.query_port],
                }
            headers = {"X-Request-Id": request_id}
            response = await self.client(prefill).post(
                endpoint, json=prefill_payload, headers=headers)
            response.raise_for_status()
            decode_payload = dict(payload)
            decode_payload["max_tokens"] = 1
            decode_payload.pop("max_completion_tokens", None)
            response = await self.client(decode).post(
                endpoint, json=decode_payload, headers=headers)
            response.raise_for_status()
            self._warmed_edges.add(key)
            self._warm_ok += 1
        except Exception as exc:  # noqa: BLE001 - warming must never break routing
            self._warm_failed += 1
            print(f"[warm] {key} failed: {exc!r}", flush=True)
        finally:
            self._warm_inflight.discard(key)

    def warm_state(self):
        return {"enabled": self.warmup_enabled,
                "warmed_edges": len(self._warmed_edges),
                "inflight": len(self._warm_inflight),
                "ok": self._warm_ok, "failed": self._warm_failed,
                "cached_classes": len(self._prompt_cache)}

    # -- slow control loop ------------------------------------------------
    async def control_loop(self):
        """Run the shared CASR control tick off the event loop thread.

        ``tick`` performs blocking Prometheus scrapes and, for ``casr_full``,
        Docker start/stop calls.  Running it via ``to_thread`` keeps request
        handling responsive while the plan is rebuilt.
        """
        while True:
            try:
                await asyncio.to_thread(self.controller.tick, time.monotonic_ns())
            except Exception as exc:  # noqa: BLE001 - a failed tick must not kill the router
                print(f"[casr] control tick failed: {exc!r}", flush=True)
            await asyncio.sleep(max(0.2, self.controller.interval_s))

    # -- health -----------------------------------------------------------
    async def health_loop(self, interval_s):
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                for inst in (*self.prefills, *self.decodes):
                    try:
                        response = await client.get(inst.health_url)
                        inst.healthy = response.status_code == 200
                    except Exception:
                        inst.healthy = False
                await asyncio.sleep(interval_s)

    # -- instance lookup --------------------------------------------------
    def prefill_by_id(self, instance_id):
        return next((inst for inst in self.prefills
                     if inst.instance_id == int(instance_id)), None)

    def decode_by_id(self, instance_id):
        return next((inst for inst in self.decodes
                     if inst.instance_id == int(instance_id)), None)

    # -- policies ---------------------------------------------------------
    def _pick_rr(self, instances, role):
        healthy = [inst for inst in instances if inst.healthy]
        if not healthy:
            return None
        index = self._counters[role] % len(healthy)
        self._counters[role] += 1
        return healthy[index]

    def _pick_load(self, instances):
        healthy = [inst for inst in instances if inst.healthy]
        if not healthy:
            return None
        # ``(inflight + 1) / capacity`` is the load *after* taking this request,
        # which is the standard least-loaded formulation.  The previous
        # ``inflight / capacity`` left the denominator as the only signal when
        # every candidate was idle, and the ``inst.id`` tie-break is a *string*
        # comparison ("d3090a" < "d4090" < "d5090" < "d_a100"), so the policy
        # degenerated into statically pinning the slowest Decode and the
        # largest-capacity A100 never got picked (measured 2026-09-13:
        # load_picks p4090=50 p5090=7 p_a100=3 of 60).
        chosen = min(healthy,
                     key=lambda inst: ((inst.inflight + 1) / inst.capacity, inst.id))
        if os.environ.get("LOAD_DEBUG"):
            print("[load-pick] " + " ".join(
                f"{i.id}={i.inflight}/{i.capacity:.0f}" for i in healthy)
                + f" -> {chosen.id}", flush=True)
        # Audit counter: the ``load`` baseline must actually follow
        # least-loaded, and its per-instance shares should track capacity.
        # A single total cannot show that, so count every pick.
        self._load_picks[chosen.id] = self._load_picks.get(chosen.id, 0) + 1
        return chosen

    def _pick_random(self, instances):
        healthy = [inst for inst in instances if inst.healthy]
        if not healthy:
            return None
        return random.choice(healthy)

    @staticmethod
    def _prompt_text(payload):
        messages = payload.get("messages")
        if messages:
            return "".join(str(item.get("content", "")) for item in messages)
        return str(payload.get("prompt", ""))

    def _prefix_chain(self, payload):
        text = self._prompt_text(payload)
        chain = []
        for window in self.cache_windows:
            head = text[:window]
            if head:
                chain.append(hashlib.sha256(head.encode("utf-8", "ignore"))
                             .hexdigest()[:16])
        return chain

    def _remember_cache(self, chain, prefill):
        for key in chain:
            self._cache_index.setdefault(key, set()).add(prefill.id)

    def _pick_cache_aware(self, bucket):
        """Longest-prefix match with a load spill guard (SGLang-style)."""
        chain = self._bucket_chains.get(bucket) or [bucket]
        healthy = [inst for inst in self.prefills if inst.healthy]
        if not healthy:
            return None
        matched = None
        for key in chain:
            owners = [inst for inst in healthy
                      if inst.id in self._cache_index.get(key, ())]
            if owners:
                # Same formulation as ``_pick_load``: rank by the load the
                # request would create, so the ``inst.id`` string tie-break is
                # only ever reached between genuinely equivalent candidates.
                matched = min(owners, key=lambda inst: (
                    (inst.inflight + 1) / max(1.0, inst.capacity), inst.id))
                break
        if matched is not None:
            load_ratio = matched.inflight / max(1.0, matched.max_inflight)
            if load_ratio < self.cache_aware_threshold:
                self._remember_cache(chain, matched)
                return matched
        selected = self._pick_load(healthy)
        self._remember_cache(chain, selected)
        return selected

    def _pick_prefill(self, bucket):
        if self.policy == "rr":
            return self._pick_rr(self.prefills, "prefill")
        if self.policy == "random":
            return self._pick_random(self.prefills)
        # ``load`` and the no-affinity ablation both ignore prefix stickiness.
        if self.policy in ("load", "casr_noaffinity"):
            return self._pick_load(self.prefills)
        if self.policy == "cache_aware":
            return self._pick_cache_aware(bucket)
        pinned = self.affinity.get(bucket)
        if pinned is not None:
            inst = next((item for item in self.prefills if item.id == pinned), None)
            if inst is not None and inst.healthy and not self._pin_is_overloaded(inst):
                return inst
        inst = self._pick_load(self.prefills)
        if inst is not None:
            self.affinity[bucket] = inst.id
        return inst

    def _pin_is_overloaded(self, pinned):
        """Whether a pinned Prefill has become the more loaded choice.

        Prefix affinity is only worth its concentration cost while the instance
        holding the prefix is not itself the bottleneck.  Measured 2026-09-13
        on the strong-reuse (deep) tier: unconditional pinning put 73% of the
        traffic on one pair and made the trimmed mean *worse* than the
        no-affinity ablation, on both the 2-Prefill and the 4-Prefill topology
        -- the cache reuse is real, but it did not pay for the queue it created.
        Give the pin up when the instance it names carries a clearly deeper run
        queue than the least loaded healthy one.
        """
        others = [item for item in self.prefills if item.healthy and item is not pinned]
        if not others:
            return False
        lightest = min(item.prefill_inflight for item in others)
        return pinned.prefill_inflight > 1.5 * (lightest + 1.0)

    def _pair_cost(self, prefill, decode, kv_bytes):
        if prefill.domain == decode.domain:
            network = 0.0
        else:
            bw_gbps, rtt_ms = self.links.get((prefill.domain, decode.domain), (1.0, 0.0))
            bandwidth = max(1e-9, bw_gbps * 1e9 / 8.0)
            # Occupancy, not serialisation: this request waits behind whatever
            # is already on the link, and pays nothing extra when the link is
            # idle (the push overlaps with the Prefill compute).  Pricing the
            # full ``kv_bytes / bandwidth`` instead made the router refuse a
            # profitable cross-domain pairing on short prompts (measured
            # -5.6% when that term was removed) while under-pricing the
            # saturated case by an order of magnitude.
            queued_s = self.link_inflight_bytes.get(
                (prefill.domain, decode.domain), 0.0) / bandwidth
            network = rtt_ms / 1000.0 + queued_s
        # The no-network ablation keeps affinity and the measured service term
        # but pretends every domain pair is equally reachable, isolating how
        # much the domain-link model contributes on this fabric.
        if self.policy == "casr_nonetwork":
            network = 0.0
        # ``inflight / capacity`` is a *wait-time* estimate (Little's law:
        # L = lambda*W, so W = L/lambda with lambda ~= the calibrated
        # requests/s), which is what makes it commensurate with the service
        # term below.  Dividing by ``max_num_seqs`` instead would turn it into
        # a dimensionless utilisation and over-penalise the faster instance
        # (regression covered by
        # ``test_decode_fallback_uses_pair_cost_not_least_loaded``).
        load = (self.w_prefill * prefill.inflight / prefill.capacity +
                self.w_decode * decode.inflight / decode.capacity)
        # The no-service ablation keeps network + utilization but drops the
        # measured heterogeneous service term, isolating its contribution.
        if self.policy == "casr_noservice":
            service = 0.0
        else:
            service = self.w_service * (prefill.service_ms + decode.service_ms) / 1000.0
        return self.w_network * network + service + load

    def _pick_decode_cost(self, prefill, kv_bytes):
        """Cost-based Decode for a Prefill: network + service + utilisation."""
        healthy = [inst for inst in self.decodes if inst.healthy]
        if not healthy:
            return None
        if os.environ.get("DECODE_DEBUG"):
            table = ", ".join(
                f"{inst.id}:{self._pair_cost(prefill, inst, kv_bytes):.4f}"
                f"(inf={inst.inflight},cap={inst.capacity:.0f},svc={inst.service_ms})"
                for inst in sorted(healthy, key=lambda i: i.id))
            print(f"[decode-cost] pref={prefill.id} kv={kv_bytes} -> {table}", flush=True)
        return min(healthy, key=lambda inst: (
            self._pair_cost(prefill, inst, kv_bytes), inst.id))

    def _pick_decode(self, prefill, bucket, kv_bytes):
        if self.policy == "rr":
            return self._pick_rr(self.decodes, "decode")
        if self.policy == "random":
            return self._pick_random(self.decodes)
        if self.policy == "load":
            return self._pick_load(self.decodes)
        # A cache-aware router still spreads Decode by load; keeping the
        # prefix match on the Prefill side only avoids piling every handoff on
        # whichever Decode looks cheapest.
        if self.policy == "cache_aware":
            return self._pick_load(self.decodes)
        return self._pick_decode_cost(prefill, kv_bytes)

    # -- plan replay (casr_lp / casr_full) --------------------------------
    def _plan(self, now_ns):
        plan = self.controller.last_plan if self.controller is not None else None
        if plan is None or plan.is_expired(now_ns):
            return None
        if plan.version != self._plan_version:
            # Same as the simulator's ``install_affinity_plan``: a new plan
            # restarts the deficit counters so the replay tracks the new
            # weights instead of a stale ratio.
            self._plan_version = plan.version
            self._plan_assignments.clear()
        return plan

    def _select_weighted(self, candidates, weights, assignment_key):
        """Deterministic deficit routing over a fractional plan (same greedy
        replay the simulator's request router uses)."""
        total = sum(self._plan_assignments.get((assignment_key, inst.instance_id), 0)
                    for inst in candidates)

        def score(inst):
            observed = self._plan_assignments.get((assignment_key, inst.instance_id), 0)
            return weights[inst.instance_id] * (total + 1) - observed

        highest = max(score(inst) for inst in candidates)
        # Deficit routing is bursty by construction: a candidate stays in
        # "credit" until it is far enough ahead of its planned share, then takes
        # a run of consecutive requests.  On an idle cluster that is invisible;
        # under load the run lands on a queue.  Measured on the saturation round
        # (2026-09-13): ``casr_lp`` and ``casr`` had *identical* pair shares yet
        # the LP variant's p95 was 3973 ms against 3186 ms -- so the difference
        # is *when* a request is placed, not *where*.  The plan still sets the
        # ratios; among candidates within a small band of the best deficit
        # score, the current queue decides the order.
        band = max(1.0, abs(highest)) * 0.05
        tied = [inst for inst in candidates if score(inst) >= highest - band]
        selected = min(tied, key=lambda inst: (inst.inflight / inst.capacity, inst.id))
        selected = self._deflect_if_overloaded(candidates, selected)
        key = (assignment_key, selected.instance_id)
        self._plan_assignments[key] = self._plan_assignments.get(key, 0) + 1
        return selected

    def _peek_weighted(self, candidates, weights, assignment_key):
        """Same choice as ``_select_weighted`` without consuming the deficit."""
        total = sum(self._plan_assignments.get((assignment_key, inst.instance_id), 0)
                    for inst in candidates)

        def score(inst):
            observed = self._plan_assignments.get((assignment_key, inst.instance_id), 0)
            return weights[inst.instance_id] * (total + 1) - observed

        highest = max(score(inst) for inst in candidates)
        band = max(1.0, abs(highest)) * 0.05
        tied = [inst for inst in candidates if score(inst) >= highest - band]
        return min(tied, key=lambda inst: (inst.inflight / inst.capacity, inst.id))

    def _note_assignment(self, instance, assignment_key):
        key = (assignment_key, instance.instance_id)
        self._plan_assignments[key] = self._plan_assignments.get(key, 0) + 1

    def _deflect_if_overloaded(self, candidates, selected):
        """Let a plan target go when it is carrying a clearly deeper queue.

        Deficit replay works in runs: a class's plan weight is often a single
        instance, so the replayer keeps choosing it until the deficit closes,
        and each run lands on the same queue.  Measured 2026-09-13: on the
        saturated wide tier ``casr_lp`` produced clusters of 10-15 requests
        slower than 3.5 s every 8-9 s while the queue-aware heuristic produced
        2 in the whole run -- with an otherwise identical per-second placement
        pattern, and with the control loop effectively disabled (so it is the
        *replay*, not the plan rebuild).  The same overload guard the affinity
        pin uses (``_pin_is_overloaded``) applies here: keep the plan's ratio
        unless the chosen instance is >1.5x the lightest by Prefill in-flight.
        """
        others = [item for item in candidates if item is not selected]
        if not others:
            return selected
        lightest = min(others, key=lambda item: item.inflight)
        if selected.inflight > 1.5 * (lightest.inflight + 1.0):
            return lightest
        return selected

    def _release_decode(self, prefill, decode, kv_bytes):
        """Give back the Decode-side accounting for one finished request.

        Called from wherever a request *actually* finishes: the handler's
        ``finally`` on the non-streaming path, and the streaming generator's
        ``finally`` for streamed replays (the handler has long since returned
        by then; see the comment there).
        """
        decode.inflight -= 1
        link_key = (prefill.domain, decode.domain)
        if link_key[0] == link_key[1]:
            return
        remaining = (self.link_inflight_bytes.get(link_key, 0.0)
                     - float(kv_bytes))
        if remaining > 0.0:
            self.link_inflight_bytes[link_key] = remaining
        else:
            self.link_inflight_bytes.pop(link_key, None)

    def _warm_switch_hold(self, class_id, target, now_ns):
        """The warm Prefill to keep serving this class, or ``None`` to switch.

        ``None`` means "follow the plan": either the new edge is already warm,
        there is no warm incumbent to hold, or the switch deadline has passed.
        """
        target_key = (class_id, target.instance_id)
        if target_key in self._warmed_edges:
            self._serving_prefill[class_id] = target.instance_id
            self._switch_deadline_ns.pop(class_id, None)
            return None
        serving_id = self._serving_prefill.get(class_id)
        current = self.prefill_by_id(serving_id) if serving_id is not None else None
        if (current is None or not current.healthy
                or (class_id, current.instance_id) not in self._warmed_edges):
            # Nothing warm to hold on to: take the plan's choice and pay the
            # one cold prefill, exactly as before this option existed.
            self._serving_prefill[class_id] = target.instance_id
            self._switch_deadline_ns.pop(class_id, None)
            return None
        deadline = self._switch_deadline_ns.get(class_id)
        if deadline is None:
            deadline = now_ns + int(self.warm_switch_timeout_s * 1e9)
            self._switch_deadline_ns[class_id] = deadline
        if now_ns >= deadline:
            self._serving_prefill[class_id] = target.instance_id
            self._switch_deadline_ns.pop(class_id, None)
            return None
        return current

    def _pick_prefill_affine(self, bucket, class_id):
        """Prefix affinity for a class the plan cannot name.

        The heuristic ``casr`` policy sticks a bucket to the Prefill it first
        landed on, which is what makes repeat requests of the same prefix hit
        the KV cache.  The plan-based policies had no such rule: for every
        class the controller has not observed yet (``plan_stats`` measured
        ``class_missing`` on 519 of 600 requests on the wide trace) the
        fallback re-ranked *all* Prefills from scratch, so a class's requests
        scattered and every one of them re-prefilled the same prefix cold.
        Pin the first choice, then reuse it while it is healthy; the queue
        aware constant in ``_pick_prefill_cost`` still decides that first pick.
        """
        pinned = self.affinity.get(bucket)
        if pinned is not None:
            inst = next((item for item in self.prefills if item.id == pinned), None)
            if inst is not None and inst.healthy:
                return inst
        inst = self._pick_prefill_cost(class_id)
        if inst is not None:
            self.affinity[bucket] = inst.id
        return inst

    def _pick_prefill_planned(self, class_id, bucket, now_ns):
        plan = self._plan(now_ns)
        if plan is None:
            # Distinguishing "no plan at all" from "plan does not name this
            # class" matters: the first is a control-loop latency problem, the
            # second a class-labelling one, and they look identical in the
            # request distribution.
            self._plan_stats["expired"] += 1
            return self._pick_prefill_affine(bucket, class_id)
        weights = plan.prefill_for(class_id)
        candidates = [inst for inst in self.prefills
                      if inst.healthy and inst.instance_id in weights]
        if not candidates:
            self._plan_stats["class_missing"] += 1
            return self._pick_prefill_affine(bucket, class_id)
        self._plan_stats["hit"] += 1
        if not self.warm_before_switch:
            return self._select_weighted(candidates, weights, ("prefill", class_id))
        target = self._peek_weighted(candidates, weights, ("prefill", class_id))
        held = self._warm_switch_hold(class_id, target, now_ns)
        if held is not None:
            self._note_assignment(held, ("prefill", class_id))
            return held
        return self._select_weighted(candidates, weights, ("prefill", class_id))

    def _pick_prefill_cost(self, class_id):
        """Fallback Prefill for a class the plan does not name (yet).

        Least-loaded ignores the heterogeneity the plan exists to exploit, and
        this fallback carries most of the traffic: the ShareGPT replay forms
        ~537 classes for 600 requests, so nearly every request is a class the
        controller has never seen and the plan therefore cannot name.  Ranking
        by ``service_ms + overhead`` uses the same measured constants, and the
        class's own TTFT budget adds the SLO penalty exactly like the solver.
        """
        healthy = [inst for inst in self.prefills if inst.healthy]
        if not healthy:
            return None
        config = getattr(getattr(self.controller, "solver", None), "config", None)
        slo_ms = None
        if config is not None:
            slo_ms = (config.class_ttft_slo_ms.get(class_id)
                      or config.ttft_slo_ms or None)

        def cost(inst):
            overhead = float((config.prefill_overhead_ms.get(inst.instance_id, 0.0)
                              if config is not None else 0.0))
            # Measured latency once we have one, configured constant before:
            # the constant is calibrated at the reference prompt length and is
            # badly wrong for long prompts.
            base = (inst.prefill_ms_ewma if inst.prefill_ms_ewma > 0.0
                    else inst.service_ms + overhead)
            if slo_ms and base > slo_ms:
                base += float(config.slo_penalty)
            # Queueing-aware, and deliberately *not* using the calibrated
            # requests/s capacity: that number is a reference-prompt figure.
            # On the 2026-09-13 CNN/DailyMail round (1250-token prompts) it put
            # ``inflight/capacity`` at 0.06 while the Prefill was in fact
            # running at ~10x its sustainable work rate, so ``casr_lp`` sent
            # 100% of the traffic to one pair and its TTFT P50 doubled.  The
            # engine's ``max_num_seqs`` is the one budget that does not depend
            # on prompt length, so the utilisation comes from that instead.
            utilization = min(0.95, inst.prefill_inflight / max(1.0, inst.max_inflight))
            return (base / max(0.05, 1.0 - utilization),
                    inst.prefill_inflight / max(1.0, inst.max_inflight), inst.id)

        return min(healthy, key=cost)

    def _pick_decode_planned(self, prefill, class_id, kv_bytes, now_ns):
        plan = self._plan(now_ns)
        if plan is not None:
            weights = plan.decode_for(prefill.instance_id, class_id)
            candidates = [inst for inst in self.decodes
                          if inst.healthy and inst.instance_id in weights]
            if candidates:
                return self._select_weighted(candidates, weights,
                                             ("decode", prefill.instance_id, class_id))
        # No plan opinion (unknown class, or the plan left this pair at zero):
        # choose with the cost model over *all* healthy Decodes.  The plan's
        # ``fallback_decode_ids`` only lists the Decodes the solver deliberately
        # left at zero, so using it here forced traffic onto whatever the cost
        # model had just avoided (on this cluster d3090a, 421 ms vs 145/157 ms).
        chosen = self._pick_decode_cost(prefill, kv_bytes)
        return chosen if chosen is not None else self._pick_load(self.decodes)

    def pick_prefill(self, bucket, class_id, now_ns):
        if self.policy in PLAN_POLICIES:
            return self._pick_prefill_planned(class_id, bucket, now_ns)
        return self._pick_prefill(bucket)

    def pick_decode(self, prefill, bucket, class_id, kv_bytes, now_ns):
        if self.policy in PLAN_POLICIES:
            return self._pick_decode_planned(prefill, class_id, kv_bytes, now_ns)
        return self._pick_decode(prefill, bucket, kv_bytes)

    # -- routing ----------------------------------------------------------
    def classify(self, payload):
        """Bucket identical shared prefixes together for affinity routing."""
        text = self._prompt_text(payload)
        head = text[:512]
        bucket = hashlib.sha256(head.encode("utf-8", "ignore")).hexdigest()[:16]
        self._bucket_chains[bucket] = self._prefix_chain(payload)
        return bucket

    def prompt_tokens(self, payload):
        messages = payload.get("messages")
        if messages:
            chars = sum(len(str(item.get("content", ""))) for item in messages)
        else:
            chars = len(str(payload.get("prompt", "")))
        return max(1, int(chars / self.chars_per_token))

    def calibrate_prompt_tokens(self, payload, prefill_response):
        """Fit ``chars_per_token`` to the engine's own prompt-token count.

        ``usage.prompt_tokens`` on the Prefill leg's ``max_tokens=1`` response
        is the tokenizer's answer for this exact prompt, so the ratio of the
        router's character count to it is a direct per-request measurement of
        the constant every downstream estimate depends on (KV bytes for the
        link budget, prompt-length work for the LP, SLO classes).

        The first ``PROMPT_TOKEN_CALIBRATION_REQS`` requests are pooled (a
        single request can straddle a token boundary), after which the ratio
        is EWMA-updated so a model or prompt-language change is tracked.
        """
        try:
            usage = (prefill_response.json() or {}).get("usage") or {}
            observed_tokens = float(usage.get("prompt_tokens") or 0.0)
        except Exception:  # noqa: BLE001 - a missing usage block is not fatal
            return
        self.observe_prompt_tokens(payload, observed_tokens)

    def observe_prompt_tokens(self, payload, observed_tokens):
        """Fold one engine-reported prompt-token count into the ratio.

        Called from *either* leg, because on this fabric the local path has no
        Prefill call at all: with ``LOCAL_PREFILL=auto`` and a local recompute
        6-14x cheaper than a handoff, calibrating only on the Prefill response
        meant the ratio never moved for exactly the runs that dominate the
        comparison (measured: 12 requests through a ``load`` router, all
        ``exchange=local``, ``samples=0``).  The Decode leg's usage carries the
        same tokenizer count.
        """
        observed_tokens = float(observed_tokens or 0.0)
        if observed_tokens <= 0.0:
            return
        messages = payload.get("messages")
        if messages:
            chars = sum(len(str(item.get("content", ""))) for item in messages)
        else:
            chars = len(str(payload.get("prompt", "")))
        if chars <= 0:
            return
        self._prompt_chars_observed += float(chars)
        self._prompt_tokens_observed += observed_tokens
        self._prompt_token_samples += 1
        if self._prompt_token_samples < self.prompt_token_calibration_reqs:
            return
        measured = self._prompt_chars_observed / self._prompt_tokens_observed
        # Clamp to a sane band: a degenerate sample (an empty or a giant prompt
        # served from cache) must not poison every later estimate.
        measured = min(max(1.5, measured), 12.0)
        if not self._prompt_tokens_calibrated:
            self.chars_per_token = measured
            self._prompt_tokens_calibrated = True
        else:
            self.chars_per_token = (0.9 * self.chars_per_token + 0.1 * measured)

    def kv_exchange_decision(self, prefill, decode, kv_bytes):
        """Decide whether to move the KV or let the Decode recompute it.

        Only the *incremental* cost of the two options is compared: the
        generation leg (and its queueing) is paid either way, so adding
        ``decode.inflight / decode.capacity`` to the local option double-counts
        it.  Measured 2026-09-13 on the Zamba2 hybrid fabric: with the queue
        term in, the decision flipped 58/64 to transfer and cost 3161 ms mean,
        while forcing local gave 1554 ms -- i.e. the term was wrong, not the
        direction.  What differs is where the Prefill compute runs and what it
        costs to move the KV across the two domains.  Returns
        ``(use_local, local_ms, transfer_ms)``.
        """
        tokens = float(kv_bytes) / max(1.0, float(self.kv_bytes_per_token))
        thousands = tokens / 1000.0
        if prefill.domain == decode.domain:
            transfer_ms = self.transfer_ms_per_1k_local * thousands
        else:
            transfer_ms = (self.transfer_fixed_ms_cross
                           + self.transfer_ms_per_1k_cross * thousands)
        local_ms = (self.local_prefill_ms_per_1k * thousands
                    / max(0.1, decode.speed))
        # Unified queue terms.  Both paths wait, and until 2026-09-14 only the
        # transfer path's constant costs were counted:
        #
        # * ``transfer`` waits for the Prefill instance.  ``prefill_ms_ewma`` is
        #   the measured wall time of this instance's own Prefill calls (queue
        #   included, prompt length included), so it is the honest estimate of
        #   what a request dispatched now will pay before its KV is pushed.  The
        #   calibrated ``service_ms`` is the fallback before any measurement.
        # * ``local`` pays its own recompute (``local_ms``) *and* a queueing
        #   externality: the inserted work lengthens the Decode's batch, and by
        #   the M/G/1 marginal-wait law that costs ``local_ms x rho/(1-rho)``
        #   on top, where ``rho`` is how much of the Decode's concurrency budget
        #   is in use.  An idle Decode (rho = 0) pays exactly ``local_ms`` --
        #   which is why local wins on an idle cluster -- and the term is capped
        #   so a busy batch cannot make an otherwise 6-14x cheaper local
        #   recompute look worse than the handoff.
        #
        # ``max_num_seqs`` (the engine's real concurrency budget) is the
        # denominator rather than the router's heuristic ``capacity``, which
        # was declared 220 req/s for a Prefill that measures ~11-72 req/s.
        weight = self.local_prefill_queue_weight
        prefill_wait_ms = (prefill.prefill_ms_ewma if prefill.prefill_ms_ewma > 0.0
                           else prefill.service_ms)
        batch_ratio = min(1.0, decode.inflight / max(1.0, decode.max_inflight))
        externality = 0.0
        if batch_ratio > 0.0:
            externality = min(self.local_prefill_queue_cap,
                              batch_ratio / max(1e-6, 1.0 - batch_ratio))
        transfer_total = transfer_ms + prefill_wait_ms * weight
        local_total = local_ms * (1.0 + externality * weight)
        if self.local_prefill_mode == "never":
            return False, local_total, transfer_total
        if self.local_prefill_mode == "always":
            return True, local_total, transfer_total
        return local_total <= transfer_total, local_total, transfer_total

    def output_tokens(self, payload):
        return max(1, int(payload.get("max_tokens")
                          or payload.get("max_completion_tokens") or 128))

    def class_id_for(self, payload, bucket):
        """The control loop's class label for this request.

        ``bucket`` plays the role of the simulator's block-aligned prefix id;
        token counts are the router's own estimate, which is the only token
        view a real gateway has before the engine tokenizes the prompt.
        """
        return derive_class_id(self.controller.model_name, bucket,
                               self.prompt_tokens(payload),
                               self.output_tokens(payload))

    def estimate_kv_bytes(self, payload):
        prompt_tokens = self.prompt_tokens(payload)
        chunks = max(1, math.ceil(prompt_tokens / CHUNK_TOKENS))
        return chunks * CHUNK_TOKENS * self.kv_bytes_per_token

    def record(self, entry):
        if self.metrics_path is None:
            return
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")


def build_app(args, config):
    router = Router(config, args.policy, args.metrics, args.scale_backend,
                    state_output=args.state_output)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = router
        tasks = [asyncio.create_task(router.health_loop(args.health_interval_s))]
        if router.controller is not None:
            tasks.append(asyncio.create_task(router.control_loop()))
        yield
        for task in tasks:
            task.cancel()
        for task in list(router._warm_tasks):
            task.cancel()
        await router.aclose()

    app = FastAPI(title="LMCache multi-domain P/D router", lifespan=lifespan)

    async def handle(request: Request, endpoint: str):
        payload = await request.json()
        request_id = request.headers.get("x-request-id", f"r{int(time.time()*1e6)}")
        slo_ttft_ms, slo_tpot_ms = request_slo(request)
        bucket = router.classify(payload)
        kv_bytes = router.estimate_kv_bytes(payload)
        now_ns = time.monotonic_ns()
        class_id = (router.class_id_for(payload, bucket)
                    if router.controller is not None else bucket)

        prefill = router.pick_prefill(bucket, class_id, now_ns)
        if prefill is None:
            return JSONResponse({"error": "no healthy prefill"}, status_code=503)
        decode = router.pick_decode(prefill, bucket, class_id, kv_bytes, now_ns)
        if decode is None:
            return JSONResponse({"error": "no healthy decode"}, status_code=503)
        if router.controller is not None:
            # Feed the slow control loop the same (class, prefill) arrival the
            # simulator's profiler records at admission time.
            real = router.controller.prefill_by_id(prefill.instance_id)
            if real is not None:
                router.controller.observe(class_id, real, router.prompt_tokens(payload),
                                          kv_bytes, now_ns,
                                          slo_ttft_ms=slo_ttft_ms)
            router.remember_prompt(class_id, endpoint, payload)
            router.maybe_warm(class_id, now_ns, prefill.instance_id)

        exchange_local, local_est_ms, transfer_est_ms = (
            router.kv_exchange_decision(prefill, decode, kv_bytes))
        prefill.inflight += 1
        decode.inflight += 1
        streamed = bool(payload.get("stream", False))
        link_key = (prefill.domain, decode.domain)
        if not exchange_local and link_key[0] != link_key[1]:
            router.link_inflight_bytes[link_key] = (
                router.link_inflight_bytes.get(link_key, 0.0) + float(kv_bytes))
        router._counters["requests"] += 1
        exchange_key = ("exchange_local" if exchange_local
                        else "exchange_transfer")
        router._counters[exchange_key] = router._counters.get(exchange_key, 0) + 1
        started = time.perf_counter()
        try:
            if exchange_local:
                # No Prefill leg and no KV movement: the Decode prefills the
                # prompt itself.  Measured 8-15x cheaper than a handoff on this
                # fabric unless the Decode is the busy side.
                prefill_ms = 0.0
                prefill_response = None
                headers = {"X-Request-Id": request_id}
            else:
                prefill_payload = dict(payload)
                prefill_payload["stream"] = False
                prefill_payload["max_tokens"] = 1
                prefill_payload.pop("max_completion_tokens", None)
                # ``stream_options`` is only valid with ``stream=True``; the
                # Prefill leg is a non-streaming, ``max_tokens=1`` KV-producing
                # call, so a streamed client request would otherwise be
                # rejected with 400.
                prefill_payload.pop("stream_options", None)
                prefill_payload["kv_transfer_params"] = {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                }
                if router.transfer_backend != "native":
                    # LMCache's PD backend needs the receiver's alloc/query
                    # ports up front; vLLM's NixlConnector learns them from the
                    # Prefill response instead.
                    prefill_payload["kv_transfer_params"]["disagg_spec"] = {
                        "req_id": request_id,
                        "receiver_host": decode.host,
                        "receiver_init_port": [decode.init_port],
                        "receiver_alloc_port": [decode.alloc_port],
                        "receiver_query_port": [decode.query_port],
                    }
                headers = {"X-Request-Id": request_id}
                prefill_started = time.perf_counter()
                prefill.prefill_inflight += 1
                try:
                    prefill_response = await router.client(prefill).post(
                        endpoint, json=prefill_payload, headers=headers)
                finally:
                    prefill.prefill_inflight -= 1
                prefill_response.raise_for_status()
                prefill_ms = (time.perf_counter() - prefill_started) * 1000.0
                # Self-calibrating: the next scheduling decision sees how long
                # this Prefill actually took for *this* prompt length.
                prefill.prefill_ms_ewma = (
                    prefill_ms if prefill.prefill_ms_ewma <= 0.0
                    else 0.8 * prefill.prefill_ms_ewma + 0.2 * prefill_ms)
                # Same idea for the prompt-size estimate.  ``chars_per_token``
                # was a constant (6.7) inherited from the synthetic traces, but
                # the deployment's real prompts are ~4.5 chars/token: every
                # token count the router derived was ~50% low, which
                # under-stated the KV bytes the link budget is denominated in.
                # The Prefill leg's own ``usage.prompt_tokens`` is ground truth
                # for this exact prompt, so calibrate against it instead of
                # trusting the constant.
                router.calibrate_prompt_tokens(payload, prefill_response)

                # Hold the Decode leg back until LMCache reports the KV has
                # landed.  Without this the Decode reads a staging slot that
                # may still hold the previous request and generates garbage at
                # full speed (docs/五台异构实验环境部署记录.md, "P/D 输出正确性缺陷").
                if router.kv_landings is not None:
                    landed = await asyncio.to_thread(
                        router.kv_landings.wait, request_id)
                    if not landed:
                        router._counters["kv_notify_timeout"] = (
                            router._counters.get("kv_notify_timeout", 0) + 1)
                        print(f"[kv-landing] timeout waiting for {request_id}",
                              flush=True)

            decode_payload = dict(payload)
            decode_payload.pop("kv_transfer_params", None)
            if router.transfer_backend == "native" and prefill_response is not None:
                # vLLM's NixlConnector needs the Prefill's
                # kv_transfer_params (remote_block_ids / engine id / host /
                # port) on the Decode request; dropping them makes the Decode
                # re-prefill.  This is the official disaggregated flow.
                remote_params = None
                try:
                    remote_params = (prefill_response.json() or {}).get(
                        "kv_transfer_params")
                except Exception:  # noqa: BLE001 - non-JSON error body
                    remote_params = None
                if remote_params:
                    decode_payload["kv_transfer_params"] = remote_params
                else:
                    router._counters["kv_params_missing"] = (
                        router._counters.get("kv_params_missing", 0) + 1)
            decode_started = time.perf_counter()
            if payload.get("stream", False):
                # The streaming path must still write a metrics row: the
                # aggregator reads ``metrics-<policy>.jsonl``, and returning
                # the StreamingResponse early used to leave a streaming run
                # with an empty metrics file.  TTFT is measured from *before*
                # the Prefill call (``started``), so it is the client-visible
                # time to first token: prefill + KV handoff + decode queue.
                async def stream():
                    state = {"first_token_ms": None, "tokens": 0, "usage": {}}
                    buffer = ""
                    status = 200
                    try:
                        async with router.client(decode).stream(
                                "POST", endpoint, json=decode_payload,
                                headers=headers) as upstream:
                            status = upstream.status_code
                            upstream.raise_for_status()
                            async for chunk in upstream.aiter_bytes():
                                buffer += chunk.decode("utf-8", "ignore")
                                while "\n" in buffer:
                                    line, buffer = buffer.split("\n", 1)
                                    state["now_ms"] = (
                                        (time.perf_counter() - started) * 1000.0)
                                    parse_sse_line(line, state)
                                yield chunk
                    finally:
                        # Release the Decode-side accounting *here*, not in the
                        # handler's own ``finally``: the handler returns the
                        # ``StreamingResponse`` before this generator ever runs,
                        # so the outer ``finally`` fires at dispatch time and
                        # left ``decode.inflight`` (and the link occupancy)
                        # pinned near zero for the whole request.  The decode
                        # load term is what stops the cost model from putting
                        # every request on the fastest Decode -- measured
                        # 2026-09-13: with the early release ``casr`` sent
                        # 186/200 requests to d5090 and its decode p50 was
                        # 1497 ms against the baseline's 863 ms.
                        router._release_decode(prefill, decode, kv_bytes)
                        total_ms = (time.perf_counter() - started) * 1000.0
                        ttft = state["first_token_ms"]
                        tokens = int((state["usage"] or {}).get("completion_tokens")
                                     or state["tokens"])
                        # Streaming is the default replay path, so its usage
                        # block is the only place the tokenizer count appears.
                        router.observe_prompt_tokens(
                            payload, (state["usage"] or {}).get("prompt_tokens"))
                        entry = {
                            "request_id": request_id, "policy": router.policy,
                            "bucket": bucket, "prefill": prefill.id,
                            "decode": decode.id,
                            "exchange": ("local" if exchange_local
                                         else "transfer"),
                            # Prompt size and the calibrated ratio behind it --
                            # the same two fields the non-streaming path
                            # records, because everything downstream (KV bytes
                            # for the link budget, prompt-length work for the
                            # LP) is derived from them.
                            "prompt_tokens": router.prompt_tokens(payload),
                            "chars_per_token": round(router.chars_per_token, 3),
                            "local_est_ms": round(local_est_ms, 2),
                            "transfer_est_ms": round(transfer_est_ms, 2),
                            # Plan version this request was dispatched under.
                            # Diagnostic only: used to check whether the plan
                            # policies' transient latency spikes line up with
                            # plan-version boundaries (see the "casr_lp 尾部"
                            # investigation in docs/异构环境搭建进展.md).
                            "plan_version": (router.controller.last_plan.version
                                             if router.controller is not None
                                             and router.controller.last_plan else None),
                            "prefill_ms": round(prefill_ms, 3),
                            "decode_ms": round(total_ms - prefill_ms, 3),
                            "ttft_ms": round(ttft, 3) if ttft is not None else None,
                            "completion_tokens": tokens,
                            "total_ms": round(total_ms, 3),
                            "status": status, "ts": time.time(),
                        }
                        if ttft is not None and tokens > 0:
                            entry["tpot_ms"] = round(
                                (total_ms - ttft) / max(1, tokens - 1), 3)
                        if slo_ttft_ms is not None or slo_tpot_ms is not None:
                            entry["slo_ttft_ms"] = slo_ttft_ms
                            entry["slo_tpot_ms"] = slo_tpot_ms
                            entry["slo_ok"] = slo_verdict(
                                ttft, entry.get("tpot_ms"), slo_ttft_ms, slo_tpot_ms)
                        router.record(entry)

                return StreamingResponse(stream(), media_type="text/event-stream")

            response = await router.client(decode).post(
                endpoint, json=decode_payload, headers=headers)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            # The decode leg reports the same tokenizer count, and it is the
            # only leg that exists on the local path.
            router.calibrate_prompt_tokens(payload, response)
            if router.warm_before_switch and response.status_code == 200:
                # This Prefill has now computed and cached the class prefix, so
                # the edge is warm even though it was not explicitly warmed.
                router._warmed_edges.add((class_id, prefill.instance_id))
            router.record({
                "request_id": request_id, "policy": router.policy, "bucket": bucket,
                "prefill": prefill.id, "decode": decode.id,
                "exchange": ("local" if exchange_local else "transfer"),
                # The prompt size the control plane actually used, plus the
                # calibrated ratio behind it: without both, a run cannot be
                # audited for the "every model estimate is length-blind" class
                # of bug (see docs/实验结果汇总.md §8).
                "prompt_tokens": router.prompt_tokens(payload),
                "chars_per_token": round(router.chars_per_token, 3),
                "local_est_ms": round(local_est_ms, 2),
                "transfer_est_ms": round(transfer_est_ms, 2),
                "prefill_ms": round(prefill_ms, 3), "decode_ms": round(decode_ms, 3),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "status": response.status_code, "ts": time.time(),
                # A non-streaming replay cannot observe TTFT, so a per-request
                # SLO is recorded as unmeasurable (``None``) rather than met.
                **({"slo_ttft_ms": slo_ttft_ms, "slo_tpot_ms": slo_tpot_ms,
                    "slo_ok": slo_verdict(None, None, slo_ttft_ms, slo_tpot_ms)}
                   if (slo_ttft_ms is not None or slo_tpot_ms is not None) else {}),
            })
            return Response(content=response.content, status_code=response.status_code,
                            media_type=response.headers.get("content-type"))
        except httpx.HTTPStatusError as exc:
            router.record({"request_id": request_id, "policy": router.policy,
                           "bucket": bucket, "prefill": prefill.id, "decode": decode.id,
                           "status": exc.response.status_code, "error": "upstream",
                           "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
                           "ts": time.time()})
            return JSONResponse({"error": "upstream", "status": exc.response.status_code},
                                status_code=exc.response.status_code)
        except Exception as exc:  # noqa: BLE001 - report routing failures verbatim
            router.record({"request_id": request_id, "policy": router.policy,
                           "bucket": bucket, "prefill": prefill.id, "decode": decode.id,
                           "status": 502, "error": repr(exc),
                           "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
                           "ts": time.time()})
            return JSONResponse({"error": repr(exc)}, status_code=502)
        finally:
            prefill.inflight -= 1
            if not streamed:
                # The streaming path returns its ``StreamingResponse`` before
                # the generator runs, so this handler-level ``finally`` cannot
                # be the place that gives the Decode accounting back -- see
                # ``stream()``'s ``finally``.
                router._release_decode(prefill, decode, kv_bytes)

    @app.get("/health")
    async def health():
        return {"status": "ok", "policy": router.policy}

    @app.get("/routing-state")
    async def routing_state():
        state = {
            "policy": router.policy,
            "prefills": [inst.as_dict() for inst in router.prefills],
            "decodes": [inst.as_dict() for inst in router.decodes],
            "link_inflight_bytes": {f"{src}->{dst}": value for (src, dst), value
                                    in router.link_inflight_bytes.items()},
            "affinity_buckets": len(router.affinity),
            "counters": router._counters,
            "plan_stats": dict(router._plan_stats),
            "load_picks": dict(router._load_picks),
            "warmup": router.warm_state(),
            "kv_landings": (dict(router.kv_landings.stats)
                            if router.kv_landings is not None else None),
            "kv_group": router.kv_group,
            "kv_group_dropped": router.kv_group_dropped,
            "local_prefill": {
                "mode": router.local_prefill_mode,
                "local_ms_per_1k": router.local_prefill_ms_per_1k,
                "transfer_ms_per_1k_same_domain":
                    router.transfer_ms_per_1k_local,
                "transfer_ms_per_1k_cross_domain":
                    router.transfer_ms_per_1k_cross,
                "transfer_fixed_ms_cross_domain":
                    router.transfer_fixed_ms_cross,
                "queue_weight": router.local_prefill_queue_weight,
            },
            # Every model estimate starts from these two numbers, so publish
            # both: the seed from the config and the value fitted from the
            # engine's own ``usage.prompt_tokens``.
            "prompt_tokens": {
                "chars_per_token": round(router.chars_per_token, 3),
                "samples": router._prompt_token_samples,
                "calibrated": router._prompt_tokens_calibrated,
                "observed_chars": router._prompt_chars_observed,
                "observed_tokens": router._prompt_tokens_observed,
            },
        }
        if router.controller is not None:
            state["casr"] = router.controller.as_dict()
        return state

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle(request, "/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle(request, "/chat/completions")

    return app


if __name__ == "__main__":
    import uvicorn

    parsed = parse_args()
    with open(parsed.config, encoding="utf-8") as stream:
        router_config = json.load(stream)
    uvicorn.run(build_app(parsed, router_config), host=parsed.host, port=parsed.port)

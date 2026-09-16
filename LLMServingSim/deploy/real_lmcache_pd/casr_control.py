"""Design-faithful CASR control plane for the real multi-domain P/D router.

The real deployment must run the *same* algorithm as the simulator, so this
module deliberately imports the simulator's CASR package instead of
re-implementing a cost function:

* ``casr.flow_solver``    - greedy / OR-Tools LP ``f_ijk`` solver
* ``casr.evaluator``      - counterfactual ``+P`` / ``-P`` structural decision
* ``casr.plan_builder``   - ``f_ijk`` to ``AffinityPlan`` weight tables
* ``casr.state``          - Prometheus text parsing for the observations

What is genuinely new here is only the *observation and actuation* layer:

* rows are built from vLLM's Prometheus endpoint (queue depth, prefix-cache
  queries/hits) plus the router's own per-request bookkeeping;
* structural edits are executed against Docker through the Engine API over the
  unix socket, so ``+P`` starts a stopped container and ``-P`` drains then
  stops one.

Everything else - the objective, the capacity and shared-link constraints, the
cache-aware ``work[p, class]`` term, the gain thresholds and the warm/cold
counterfactual - is shared code with the simulator.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Dict, Tuple
from urllib.request import Request, build_opener, ProxyHandler

from casr.evaluator import StructuralEvaluator
from casr.affinity import AffinityPlan
from casr.flow_solver import CapacityAwareFlowSolver, FlowSolverConfig
from casr.plan_builder import build_affinity_plan
from casr.prefix_profiler import derive_class_id
from casr.state import parse_prometheus_text


def disabled_instance_ids():
    """Instance ids switched off for this run.

    ``DISABLED_INSTANCES`` is the runtime counterpart of the config's
    ``enabled`` flag: a shared node that is contended or has mismatched
    drivers belongs to the run, not to the checked-in topology.  Both the fast
    router (``disagg_router.instance_enabled``) and the slow control loop must
    honour it with identical semantics, otherwise the structural counterfactual
    proposes a ``+P`` on a host that is deliberately out of service.
    """
    return {token.strip() for token in
            os.environ.get("DISABLED_INSTANCES", "").split(",") if token.strip()}


def instance_enabled(spec):
    if str(spec.get("id")) in disabled_instance_ids():
        return False
    return bool(spec.get("enabled", True))


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over the Docker unix socket, using only the standard library."""

    def __init__(self, socket_path, timeout=15.0):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


class DockerRuntime:
    """Start/stop the containers that back real Prefill instances.

    Containers on the router's own host are driven through the Docker unix
    socket; containers on a worker are driven with ``ssh <target> docker ...``
    using the same credentials the deploy scripts use.  Without the SSH path
    ``+P``/``-P`` could only ever touch the head node, which is not the
    multi-domain topology CASR is meant to control.

    ``state`` returns ``running: None`` when the host could not be reached, so
    a transient SSH failure never looks like "the engine crashed" and triggers
    a spurious scale-out.
    """

    def __init__(self, socket_path="/var/run/docker.sock", name_prefix="casr-md-",
                 timeout=20.0, ssh_timeout=30.0, local_domains=()):
        self.socket_path = socket_path
        self.name_prefix = name_prefix
        self.timeout = timeout
        self.ssh_timeout = ssh_timeout
        self.local_domains = frozenset(str(domain) for domain in local_domains)

    # -- local Docker socket ---------------------------------------------
    def _call(self, method, path):
        connection = _UnixHTTPConnection(self.socket_path, self.timeout)
        try:
            connection.request(method, path, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    # -- remote Docker over SSH -------------------------------------------
    def _local(self, instance):
        return instance.domain in self.local_domains

    def _ssh(self, instance, argv):
        target = str(getattr(instance, "ssh", "") or "")
        if not target:
            return False, f"no ssh target configured for {instance.id}"
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                   "-o", "StrictHostKeyChecking=no"]
        port = int(getattr(instance, "ssh_port", 22) or 22)
        if port != 22:
            command += ["-p", str(port)]
        command += [target] + list(argv)
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=self.ssh_timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        if completed.returncode != 0:
            return False, (completed.stderr or completed.stdout).strip()[:200]
        return True, completed.stdout.strip()

    def _name(self, instance):
        return f"{self.name_prefix}{instance.id}"

    # -- public API --------------------------------------------------------
    def state(self, instance):
        name = self._name(instance)
        if self._local(instance):
            status, body = self._call("GET", f"/containers/{name}/json")
            if status != 200:
                return {"running": False, "status": "missing"}
            state = json.loads(body).get("State", {})
            return {"running": bool(state.get("Running")),
                    "status": str(state.get("Status", "unknown"))}
        ok, out = self._ssh(instance, ["docker", "inspect", "--format",
                                       "{{.State.Running}}", name])
        if not ok:
            return {"running": None, "status": "unreachable", "detail": out}
        running = out.strip() == "true"
        return {"running": running, "status": "running" if running else "stopped"}

    def start(self, instance):
        name = self._name(instance)
        if self._local(instance):
            status, body = self._call("POST", f"/containers/{name}/start")
            # 304 means "already started", which is success for our purposes.
            return status in (204, 304), body.decode("utf-8", "replace")[:200]
        return self._ssh(instance, ["docker", "start", name])

    def stop(self, instance):
        name = self._name(instance)
        if self._local(instance):
            status, body = self._call("POST", f"/containers/{name}/stop?t=10")
            return status in (204, 304), body.decode("utf-8", "replace")[:200]
        return self._ssh(instance, ["docker", "stop", "-t", "10", name])


class MetricsScraper:
    """Best-effort vLLM Prometheus scrape, bypassing the control-plane proxy."""

    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self._opener = build_opener(ProxyHandler({}))

    def scrape(self, host, port):
        url = f"http://{host}:{port}/metrics"
        try:
            request = Request(url, headers={"Accept": "text/plain"})
            with self._opener.open(request, timeout=self.timeout) as response:
                return parse_prometheus_text(response.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 - a missing scrape must not break routing
            return {}


class RealInstance:
    """Duck-typed scheduler surface expected by ``casr.flow_solver``.

    The solver only reads ``instance_id``, ``start_npu``, ``max_num_seqs``,
    ``running`` and ``waiting``; the lifecycle/evaluator additionally read
    ``admission_state``.  Keeping this adapter tiny is what lets the simulator
    solver run unmodified against real containers.
    """

    def __init__(self, spec, role, instance_id):
        self.id = str(spec["id"])
        self.instance_id = int(instance_id)
        self.pd_type = role
        self.host = str(spec["host"])
        self.port = int(spec["port"])
        self.domain = str(spec.get("domain", self.host))
        # Runtime outages (``DISABLED_INSTANCES``) must disable the control
        # plane's view of this slot too, not just the config's ``enabled``.
        self.enabled = instance_enabled(spec)
        self.capacity = max(1.0, float(spec.get("capacity", 1.0)))
        # The simulator's ``prefill_capacity`` / ``decode_capacity`` keys map to
        # these real instances.  ``capacity`` is the fast router's normalized
        # load denominator; a calibrated solver capacity can be supplied
        # separately (e.g. proportional to 1/measured service time).
        self.solver_capacity = max(1.0, float(spec.get("solver_capacity", self.capacity)))
        self.service_ms = float(spec.get("service_ms", 0.0))
        self.max_num_seqs = max(1, int(spec.get("max_num_seqs", self.capacity)))
        self.start_npu = int(spec.get("domain_index", 0))
        # Container address for the +P/-P lifecycle; filled from the config's
        # ``hosts`` block by ``build_controller``.
        self.ssh = str(spec.get("ssh", ""))
        self.ssh_port = int(spec.get("ssh_port", 22))
        self.admission_state = "ACTIVE" if self.enabled else "INACTIVE"
        self.healthy = self.enabled
        self.running = []
        self.waiting = []
        self.queue_depth = 0.0
        self.prefix_queries = 0.0
        self.prefix_hits = 0.0

    @property
    def accepts_new_requests(self):
        return self.admission_state == "ACTIVE" and self.healthy

    def set_admission_state(self, state):
        self.admission_state = state

    def as_dict(self):
        return {"id": self.id, "instance_id": self.instance_id, "role": self.pd_type,
                "domain": self.domain, "admission_state": self.admission_state,
                "healthy": self.healthy, "running": len(self.running),
                "waiting": len(self.waiting), "service_ms": self.service_ms,
                "solver_capacity": self.solver_capacity}


@dataclass
class ClassState:
    """Per (prefill instance, prefix class) cache/demand observation."""

    arrival_rate_ewma: float = 0.0
    requested_tokens: float = 0.0
    # ``hit_tokens_ewma`` is per request, so the matching denominator must be a
    # per-request EWMA too; dividing by the cumulative counter makes the hit
    # ratio decay like 1/N and erases prefix affinity from the solver input.
    requested_tokens_ewma: float = 0.0
    hit_tokens_ewma: float = 0.0
    kv_bytes_per_request: float = 0.0
    last_arrival_ns: int = 0
    # Arrivals seen since the last control-tick sample; the rate is a count over
    # the window, never a reciprocal gap.
    arrivals_since_sample: int = 0
    # ``False`` until the first completed sampling window, so the EWMA starts at
    # the first measured rate instead of ramping up from zero (which would make
    # the LP under-provision during exactly the busiest ramp-up seconds).
    arrival_sampled: bool = False
    pending: list = field(default_factory=list)


class RealCASRController:
    """Slow control loop: observe -> solve f_ijk -> publish AffinityPlan."""

    def __init__(self, prefill, decode, links, options, docker=None):
        self.prefill = list(prefill)
        self.decode = list(decode)
        self.links = dict(links)
        self.options = dict(options or {})
        self.docker = docker
        self.docker_prefix = self.options.get("container_prefix", "casr-md-")
        self.model_name = str(self.options.get("model_name", "qwen3-8b"))
        self.alpha = float(self.options.get("ewma_alpha", 0.2))
        # Deprecated: arrival decay now comes from the per-tick sampling window
        # in ``sample_arrivals``.  The option is still accepted so existing
        # router configs keep parsing unchanged.
        self.hit_half_life_s = float(self.options.get("hit_half_life_s", 30.0))
        self.interval_s = float(self.options.get("control_interval_s", 10.0))
        self.plan_ttl_s = float(self.options.get("plan_ttl_s", 2 * self.interval_s))
        self.scale_backend = str(self.options.get("scale_backend", "noop")).lower()
        self.warmup_s = float(self.options.get("warmup_s", 90.0))
        self.min_active = int(self.options.get("min_active_prefill", 1))
        self.states: Dict[Tuple[int, str], ClassState] = {}
        self.scraper = MetricsScraper(float(self.options.get("scrape_timeout_s", 2.0)))
        self.solver = CapacityAwareFlowSolver(self._solver_config())
        # The structural evaluator has to price the same start-up the WARMING
        # gate waits for, otherwise "gain over the horizon" ignores the 45 s the
        # new container needs before it can serve a single request.
        structural_options = dict(self.options.get("structural") or {})
        structural_options.setdefault("startup_s", self.warmup_s)
        if ("max_active_prefill" not in structural_options
                and self.options.get("max_active_prefill")):
            structural_options["max_active_prefill"] = int(
                self.options["max_active_prefill"])
        self.evaluator = StructuralEvaluator(structural_options)
        self.version = 0
        self.last_plan = None
        self.last_tick_ns = 0
        self.last_structural = {}
        self.last_objective = 0.0
        self.actions = []
        # Optional JSONL dump of the control plane's view, one line per tick,
        # using the *simulator's* snapshot schema (``prefix_states`` / ``flows``
        # / ``instances`` / ``solver`` / ``time_ns``).  A real run that carries
        # this file can be replayed offline against the simulator tick by tick
        # instead of only comparing end-of-run latency aggregates; the
        # simulator already writes the same shape via ``--casr-state-output``.
        self.state_output = str(self.options.get("state_output") or "")
        # Plan-version transitions, with the objective that justified each one.
        # The incumbent is only replaced when it is worse by the hysteresis
        # margin, so this is the audit trail for "why did routing move?".
        self.plan_events = []
        self._last_sample_ns = 0
        self._warming_deadline = {}
        self._last_action_ns = -1
        # ``ACTIVE`` instances only need a periodic liveness probe: a remote
        # ``docker inspect`` costs ~0.3 s of SSH per instance, and doing it on
        # every 1 s tick stretches the control tick enough to move the
        # arrival-rate sampling window.  Instances that are mid-transition
        # (``WARMING``/``DRAINING``) are still probed every tick.
        self.state_refresh_s = float(self.options.get("state_refresh_s", 10.0))
        self._last_docker_probe = {}
        # class_id -> tightest TTFT budget the workload declared for it.  The
        # solver reads this as ``class_ttft_slo_ms``; empty means "one global
        # bound", i.e. the pre-data-driven behaviour.
        self._class_slo = {}
        # Backlog-aware demand: see ``_sample_backlog``.  ``_last_backlog_waiting``
        # is ``None`` until the first sample so the initial queue depth (often
        # non-zero when the controller starts) is not counted as growth.
        self._backlog_rps = 0.0
        self._last_backlog_waiting = None
        self._backlog_active = False
        self._backlog_multiple = max(
            float(self.options.get("backlog_rps_multiple", 4.0)), 0.0)

    # -- configuration ----------------------------------------------------
    def _solver_config(self):
        """Build the solver config from the same dict the simulator consumes.

        The router config's ``casr`` block mirrors the simulator's cluster
        ``casr`` block (``solver``, ``overflow_penalty``, ``shared_links``,
        ``prefill_capacity`` ...), so ``FlowSolverConfig.from_dict`` is the one
        shared parser.  The only real-system-specific part is the P/D link
        table, which is derived from the measured domain links.
        """
        config = FlowSolverConfig.from_dict(self.options)
        pair_costs = {}
        for p in self.prefill:
            for d in self.decode:
                if p.domain == d.domain:
                    rtt_ms, bandwidth = 0.0, 0.0
                else:
                    bw_gbps, rtt_ms = self.links.get((p.domain, d.domain), (1.0, 0.0))
                    # Measured 2026-09-13 (docs/实验结果汇总.md §5.10/§5.11): a
                    # cross-domain handoff costs +0.4 ms on an idle link -- the
                    # push overlaps with the Prefill compute -- so the per-pair
                    # price is the RTT.  Capacity is priced once, by the
                    # ``shared_links`` budget below (each Prefill's egress is
                    # 2 GB/s, measured), which is the occupancy constraint the
                    # saturated case actually needs.  Charging
                    # ``kv_bytes / bandwidth`` *per pair* as well double-counts
                    # it and, worse, over-prices an idle link -- which is what
                    # made the LP refuse profitable cross-domain pairings on
                    # short prompts.
                    bandwidth = 0.0
                pair_costs[p.instance_id, d.instance_id] = {
                    "rtt_ms": float(rtt_ms), "bandwidth_bytes_per_s": bandwidth}
        prefill_capacity = {int(key): float(value)
                            for key, value in config.prefill_capacity.items()}
        decode_capacity = {int(key): float(value)
                           for key, value in config.decode_capacity.items()}
        for instance in self.prefill:
            prefill_capacity.setdefault(instance.instance_id, instance.solver_capacity)
        for instance in self.decode:
            decode_capacity.setdefault(instance.instance_id, instance.solver_capacity)
        # Feed the measured compute speed into the shared objective.  Capacity
        # alone cannot express it: without a per-unit compute cost the LP will
        # happily place overflow on the slowest same-domain pair.
        prefill_service = {int(key): float(value)
                           for key, value in config.prefill_service_ms.items()}
        decode_service = {int(key): float(value)
                          for key, value in config.decode_service_ms.items()}
        for instance in self.prefill:
            prefill_service.setdefault(instance.instance_id, instance.service_ms)
        for instance in self.decode:
            decode_service.setdefault(instance.instance_id, instance.service_ms)
        return dataclass_replace(
            config,
            pair_costs=pair_costs,
            prefill_capacity=prefill_capacity,
            decode_capacity=decode_capacity,
            prefill_service_ms=prefill_service,
            decode_service_ms=decode_service)

    # -- instance lookup --------------------------------------------------
    def real_by_id(self, instance_id):
        """Map a plan / solver instance id back to its real endpoint."""
        target = int(instance_id)
        for instance in (*self.prefill, *self.decode):
            if instance.instance_id == target:
                return instance
        return None

    def prefill_by_id(self, instance_id):
        target = int(instance_id)
        for instance in self.prefill:
            if instance.instance_id == target:
                return instance
        return None

    # -- observation ------------------------------------------------------
    @staticmethod
    def class_id(model_name, bucket, prompt_tokens, output_tokens):
        return derive_class_id(model_name, bucket, prompt_tokens, output_tokens)

    def observe(self, class_id, prefill_instance, prompt_tokens, kv_bytes, now_ns,
                slo_ttft_ms=None):
        state = self.states.setdefault((prefill_instance.instance_id, class_id), ClassState())
        state.arrivals_since_sample += 1
        state.last_arrival_ns = now_ns
        state.kv_bytes_per_request = max(state.kv_bytes_per_request, float(kv_bytes))
        state.requested_tokens_ewma = (self.alpha * float(prompt_tokens) +
                                       (1.0 - self.alpha) * state.requested_tokens_ewma)
        state.pending.append(float(prompt_tokens))
        # The workload declares its own per-class budget (trace row / client
        # fallback).  Keep the tightest bound seen for a class: if two requests
        # of one class disagree, the binding one is what the router must honour.
        if slo_ttft_ms:
            current = self._class_slo.get(class_id)
            value = float(slo_ttft_ms)
            if current is None or value < current:
                self._class_slo[class_id] = value

    def refresh_metrics(self):
        """Pull queue depth and prefix-cache counters for every instance."""
        for instance in (*self.prefill, *self.decode):
            if not instance.enabled:
                # A deliberately absent host (shared node) must not cost a
                # connection timeout on every control tick.
                instance.healthy = False
                continue
            metrics = self.scraper.scrape(instance.host, instance.port)
            if not metrics:
                instance.healthy = False
                continue
            instance.healthy = True
            instance.queue_depth = float(metrics.get("vllm:num_requests_waiting", 0.0))
            instance.waiting = [None] * int(instance.queue_depth)
            instance.running = [None] * int(metrics.get("vllm:num_requests_running", 0.0))
            if instance.pd_type == "prefill":
                instance.prefix_hits = float(metrics.get("vllm:prefix_cache_hits_total", 0.0))
                instance.prefix_queries = float(metrics.get("vllm:prefix_cache_queries_total", 0.0))

    def attribute_hits(self):
        """Spread each Prefill's cache-hit delta over the classes it served."""
        for instance in self.prefill:
            pending = [(key, state) for key, state in self.states.items()
                       if key[0] == instance.instance_id and state.pending]
            if not pending:
                continue
            total_requested = sum(sum(state.pending) for _, state in pending)
            delta = max(0.0, instance.prefix_hits - getattr(instance, "_hits_seen", 0.0))
            if total_requested <= 0:
                continue
            for _, state in pending:
                share = sum(state.pending) / total_requested
                class_hits = min(sum(state.pending), delta * share)
                for requested in state.pending:
                    per_request = class_hits * (requested / max(1e-9, sum(state.pending)))
                    state.hit_tokens_ewma = (self.alpha * per_request +
                                             (1 - self.alpha) * state.hit_tokens_ewma)
                    state.requested_tokens += requested
                state.pending.clear()
        for instance in self.prefill:
            instance._hits_seen = instance.prefix_hits

    def sample_arrivals(self, now_ns):
        """Fold each class's arrival count into its rate once per control tick.

        The previous estimator was ``1 / (now - last_arrival)`` smoothed with an
        EWMA.  The reciprocal of an inter-arrival gap is a biased -- and
        unbounded -- estimate of a request rate: two requests dispatched in the
        same millisecond produce a spike of ~1000 req/s, and the mean of the
        reciprocal gap is always larger than the reciprocal of the mean gap.  The
        LP then believed the offered load was several times what the clients were
        actually sending, over-spread every class onto every Prefill and dragged
        prefix caches onto cold instances.  Counting arrivals over the sampled
        window is unbiased and bounded by the window length.
        """
        elapsed_s = 0.0
        if self._last_sample_ns and now_ns > self._last_sample_ns:
            elapsed_s = (now_ns - self._last_sample_ns) / 1e9
        for state in self.states.values():
            if elapsed_s > 0.0:
                instant = state.arrivals_since_sample / elapsed_s
                if state.arrival_sampled:
                    state.arrival_rate_ewma = (self.alpha * instant +
                                               (1.0 - self.alpha) * state.arrival_rate_ewma)
                else:
                    state.arrival_rate_ewma = instant
                    state.arrival_sampled = True
            state.arrivals_since_sample = 0
        self._sample_backlog(elapsed_s)
        self._last_sample_ns = int(now_ns)

    def _sample_backlog(self, elapsed_s):
        """Recover the *offered* load from the Prefill backlogs.

        The controller only sees the requests the router accepted, so under
        backpressure the observed arrival rate collapses to the served rate:
        the LP is told "demand == capacity" exactly when the system is
        overloaded, and ``prefill_overflow`` stays 0 through a run whose real
        queue is dozens deep (measured 2026-09-14: ``elastic-long-a1`` reported
        ``prefill_overflow {"0": 0.0}`` while ``p5090`` held 48 queued
        requests and ``prefill_ms`` P95 was 34 s).  The unserved part of the
        offered load is sitting in the engine's waiting queue, so its growth
        rate is the missing demand.

        The queue is not class-attributed, so ``rows()`` adds this rate back
        proportionally to each class's observed share -- a mean-field
        approximation.  Only *growth* is counted: a draining queue must not
        subtract demand, and the EWMA decays the term once the backlog stops
        building.
        """
        waiting = 0.0
        for instance in self.prefill:
            if not instance.enabled:
                continue
            waiting += float(len(instance.waiting))
        growth = 0.0
        if elapsed_s > 0.0 and self._last_backlog_waiting is not None:
            growth = max(0.0, waiting - self._last_backlog_waiting) / elapsed_s
        self._last_backlog_waiting = waiting
        if not self._backlog_active:
            self._backlog_rps = growth
            self._backlog_active = True
        else:
            self._backlog_rps = (self.alpha * growth +
                                 (1.0 - self.alpha) * self._backlog_rps)

    def rows(self):
        # Distribute the backlog growth over the observed classes, keeping the
        # relative shares the EWMA already established.
        observed = sum(state.arrival_rate_ewma for state in self.states.values())
        backlog = 0.0
        if observed > 0.0 and self._backlog_rps > 0.0:
            cap = self._backlog_multiple * observed
            backlog = min(self._backlog_rps, cap)
        scale = 1.0 + (backlog / observed if observed > 0.0 else 0.0)
        rows = []
        for (prefill_id, class_id), state in sorted(self.states.items()):
            rows.append({
                "class_id": class_id,
                "arrival_rate_ewma": state.arrival_rate_ewma * scale,
                "prefill_instance_id": int(prefill_id),
                "hit_tokens_ewma": state.hit_tokens_ewma,
                "requested_tokens": state.requested_tokens,
                "requested_tokens_ewma": state.requested_tokens_ewma,
                "kv_bytes_per_request": state.kv_bytes_per_request,
            })
        return rows

    def snapshot(self, now_ns):
        return {"time_ns": int(now_ns), "prefix_states": self.rows()}

    # -- structural control ----------------------------------------------
    def _sync_container_states(self, now_ns):
        """Reconcile the CASR admission state with the real container state.

        ``WARMING`` -> ``ACTIVE`` once the restarted engine answers metrics
        again; ``DRAINING`` -> ``INACTIVE`` once its queue is empty; and a
        container that stopped outside CASR (crash, manual stop, an external
        reclaimer) is demoted to ``INACTIVE`` so the evaluator can bring it
        back with ``+P`` instead of silently routing to a dead socket.
        """
        for instance in self.prefill:
            in_docker = None
            if self.docker is not None and instance.enabled:
                state = instance.admission_state
                last_probe = self._last_docker_probe.get(instance.instance_id, -1)
                if (state == "ACTIVE" and last_probe >= 0 and
                        (now_ns - last_probe) < self.state_refresh_s * 1e9):
                    continue
                in_docker = self.docker.state(instance)
                self._last_docker_probe[instance.instance_id] = now_ns
            # True / False / None (host unreachable, so keep the current state).
            running = in_docker.get("running") if in_docker is not None else None
            state = instance.admission_state
            if state == "WARMING":
                deadline = self._warming_deadline.get(instance.instance_id, now_ns)
                if now_ns >= deadline and instance.healthy:
                    instance.set_admission_state("ACTIVE")
                elif running is False and self.docker is not None:
                    # The container vanished while warming; make it eligible
                    # for another scale-out attempt.
                    instance.set_admission_state("INACTIVE")
            elif state == "DRAINING":
                drained = not instance.running and not instance.waiting
                if running is False or drained:
                    instance.set_admission_state("INACTIVE")
                    if self.docker is not None and running is not False:
                        self.docker.stop(instance)
            elif state == "ACTIVE" and running is False:
                instance.set_admission_state("INACTIVE")

    def apply_structural(self, decision, now_ns):
        if self.scale_backend != "docker" or self.docker is None or decision.action == "keep":
            return []
        results = []
        active_ids = {s.instance_id for s in self.prefill if s.admission_state == "ACTIVE"}
        wanted = set(decision.wanted_ids)
        for instance in self.prefill:
            if decision.action == "+P" and instance.instance_id in wanted - active_ids:
                ok, detail = self.docker.start(instance)
                if ok:
                    instance.set_admission_state("WARMING")
                    self._warming_deadline[instance.instance_id] = int(now_ns + self.warmup_s * 1e9)
                results.append({"action": "scale_out", "instance": instance.id, "ok": ok, "detail": detail})
            elif decision.action == "-P" and instance.instance_id in active_ids - wanted:
                instance.set_admission_state("DRAINING")
                results.append({"action": "scale_in_drain", "instance": instance.id, "ok": True, "detail": ""})
        self._last_action_ns = int(now_ns)
        return results

    # -- control tick -----------------------------------------------------
    def tick(self, now_ns):
        self.refresh_metrics()
        self.attribute_hits()
        self.sample_arrivals(now_ns)
        self._sync_container_states(now_ns)
        # Per-class budgets come from the workload, not the config, so refresh
        # them every tick before the solver runs.  With no observed budgets the
        # map stays empty and the global ``ttft_slo_ms`` alone applies.
        if self._class_slo:
            self.solver.config = dataclass_replace(
                self.solver.config, class_ttft_slo_ms=dict(self._class_slo))

        active = [s for s in self.prefill if s.accepts_new_requests]
        decode = [s for s in self.decode if s.accepts_new_requests]
        if not active or not decode:
            return self.last_plan
        snapshot = self.snapshot(now_ns)
        if not snapshot["prefix_states"]:
            return self.last_plan

        # A disabled host is wired but deliberately out of service, so it must
        # never be proposed as a ``+P`` candidate (the container is not there).
        eligible = [s for s in self.prefill if s.enabled]
        decision = self.evaluator.evaluate(
            snapshot, active, eligible, decode, self.solver, now_ns,
            self._last_action_ns, self.min_active)
        self.last_structural = decision.as_dict()
        executions = self.apply_structural(decision, now_ns)
        if executions:
            self.last_structural["executions"] = executions
            active = [s for s in self.prefill if s.accepts_new_requests]
        if decision.action != "keep":
            # Keep an audit trail of real scale events; the routing-state
            # endpoint only ever shows the most recent tick.
            self.actions.append({"tick": self.version, "now_ns": int(now_ns),
                                 **self.last_structural})
            del self.actions[:-50]

        rows = self.rows()
        flows = self.solver.solve(rows, active, decode)
        self.last_objective = float(self.solver.diagnostics.get("objective", 0.0))
        expires_at_ns = int(now_ns + self.plan_ttl_s * 1e9)
        candidate = build_affinity_plan(flows, decode, self.version + 1,
                                        expires_at_ns)
        self.last_plan = self._publish_plan(candidate, rows, active, decode,
                                            expires_at_ns)
        # A kept plan retains its version, which is what stops the fast router
        # from resetting its deficit counters on every control tick.
        self.version = self.last_plan.version
        self.last_tick_ns = int(now_ns)
        self._dump_state(now_ns, rows, flows)
        return self.last_plan

    def _dump_state(self, now_ns, rows, flows):
        """Append this tick's control-plane view, in the simulator's schema.

        Field names follow ``serving/casr/controller.py``'s
        ``--casr-state-output``: ``time_ns``, ``affinity_plan_version``,
        ``prefix_states`` (the rows the solver saw), ``flows`` (its assignment),
        ``instances`` (per-instance load) and ``solver`` (diagnostics).  Offline
        alignment can then diff the real and simulated control decisions instead
        of only their end-of-run latency.
        """
        if not self.state_output:
            return
        instances = {}
        for instance in (*self.prefill, *self.decode):
            instances[str(instance.instance_id)] = {
                "id": instance.id,
                "pd_type": instance.pd_type,
                "domain": instance.domain,
                "enabled": bool(instance.enabled),
                "accepts_new_requests": bool(getattr(instance, "accepts_new_requests", True)),
                "capacity": instance.capacity,
                "solver_capacity": instance.solver_capacity,
                "service_ms": instance.service_ms,
                "inflight": int(getattr(instance, "inflight", 0)),
                "running": len(getattr(instance, "running", ()) or ()),
                "waiting": len(getattr(instance, "waiting", ()) or ()),
            }
        diagnostics = self.solver.diagnostics or {}
        record = {
            "time_ns": int(now_ns),
            "affinity_plan_version": self.last_plan.version if self.last_plan else None,
            "prefix_states": rows,
            "flows": [
                {"class_id": item.class_id, "prefill_id": int(item.prefill_id),
                 "decode_id": int(item.decode_id), "flow": float(item.flow),
                 "cost": float(item.cost), "link_ids": list(item.link_ids or ())}
                for item in flows
            ],
            "instances": instances,
            "solver": {key: diagnostics.get(key) for key in (
                "backend", "objective", "total_demand_rps", "total_capacity_rps",
                "prefill_overflow", "decode_overflow", "link_overflow",
                "single_homed_classes", "capped_classes", "slo_violating_pairs")},
            "structural": self.last_structural,
        }
        try:
            with open(self.state_output, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # never take the data plane down for telemetry
            print(f"[casr] state dump failed: {exc}", file=sys.stderr, flush=True)

    def _publish_plan(self, candidate, rows, active, decode, expires_at_ns):
        """Apply plan hysteresis: keep the incumbent unless it is clearly worse.

        Re-solving every tick produces near-equivalent optima, and swapping a
        class between Prefills because of a rounding-level objective difference
        throws away the prefix cache the router has already warmed.  Requiring a
        minimum relative (or absolute) gain keeps the plan -- and the router's
        deficit counters, since the version only moves when the plan really
        changes -- stable, while still following a real shift in demand.
        """
        incumbent = self.last_plan
        relative = float(self.options.get("plan_gain_threshold_rel", 0.0))
        absolute = float(self.options.get("plan_gain_threshold_abs", 0.0))
        if incumbent is None or (relative <= 0.0 and absolute <= 0.0):
            return candidate
        if not self._plan_instances_available(incumbent, active, decode):
            return candidate
        incumbent_cost = self.solver.plan_objective(incumbent, rows, active, decode)
        candidate_cost = self.solver.plan_objective(candidate, rows, active, decode)
        if candidate_cost + absolute >= incumbent_cost * (1.0 - relative):
            # Same plan, new deadline: the router keeps its assignment counters
            # because the version is unchanged.
            return AffinityPlan(version=incumbent.version, expires_at_ns=expires_at_ns,
                                prefill_weights=incumbent.prefill_weights,
                                decode_weights=incumbent.decode_weights,
                                fallback_decode_ids=incumbent.fallback_decode_ids)
        self.plan_events.append({
            "version": candidate.version,
            "incumbent_version": incumbent.version,
            "incumbent_cost": round(float(incumbent_cost), 6),
            "candidate_cost": round(float(candidate_cost), 6),
            "prefill": {class_id: dict(weights)
                        for class_id, weights in candidate.prefill_weights.items()},
        })
        del self.plan_events[:-40]
        return candidate

    @staticmethod
    def _plan_instances_available(plan, active, decode):
        active_ids = {instance.instance_id for instance in active}
        decode_ids = {instance.instance_id for instance in decode}
        for class_id, weights in plan.prefill_weights.items():
            for prefill_id in weights:
                if int(prefill_id) not in active_ids:
                    return False
                for decode_id in plan.decode_weights.get((int(prefill_id), class_id), {}):
                    if int(decode_id) not in decode_ids:
                        return False
        return True

    def plan_dict(self):
        """Readable view of the published plan, for `/routing-state`."""
        plan = self.last_plan
        if plan is None:
            return None
        name = {instance.instance_id: instance.id
                for instance in (*self.prefill, *self.decode)}
        return {
            "version": plan.version,
            "expires_at_ns": plan.expires_at_ns,
            "prefill": {
                class_id: {name.get(instance_id, str(instance_id)): round(weight, 4)
                           for instance_id, weight in weights.items()}
                for class_id, weights in plan.prefill_weights.items()},
            "decode": {
                f"{name.get(prefill_id, str(prefill_id))}|{class_id}": {
                    name.get(instance_id, str(instance_id)): round(weight, 4)
                    for instance_id, weight in weights.items()}
                for (prefill_id, class_id), weights in plan.decode_weights.items()},
        }

    def as_dict(self):
        return {
            "version": self.version,
            "solver": self.solver.backend,
            "objective": self.last_objective,
            "plan": self.plan_dict(),
            "classes": len({class_id for _, class_id in self.states}),
            "observations": len(self.states),
            "structural": self.last_structural,
            "actions": self.actions,
            "plan_events": self.plan_events,
            "diagnostics": {k: v for k, v in self.solver.diagnostics.items()
                            if k in ("backend", "overflow_penalty", "use_cache_capacity",
                                     "network_weight", "utilization_weight", "compute_weight",
                                     "work", "prefill_overflow", "decode_overflow",
                                     "link_overflow",
                                     "ttft_slo_ms", "slo_penalty", "slo_violating_pairs",
                                     "fallback")},
            "class_slo": {"classes": len(self._class_slo),
                          "tightest_ms": (min(self._class_slo.values())
                                          if self._class_slo else None)},
            # Offered-load recovery: the arrival EWMA alone is post-backpressure.
            "demand": {"backlog_rps": round(self._backlog_rps, 4),
                       "waiting": self._last_backlog_waiting},
            "prefills": [s.as_dict() for s in self.prefill],
        }


def slot_instance_ids(items):
    """Stable integer instance ids for one role's config list.

    Disabled slots keep their index, so enabling a host later never renumbers
    the others.  An explicit ``instance_id`` in the config wins, which is how
    the router config mirrors the simulator's ``prefill_capacity`` /
    ``decode_capacity`` keys.
    """
    return [int(item.get("instance_id", index)) for index, item in enumerate(items)]


def build_controller(config, options=None):
    """Build a ``RealCASRController`` from the same router config the fast
    router reads.

    The ``casr`` block of the config is passed through verbatim as solver
    options, so the real deployment and the simulator share one parser
    (``FlowSolverConfig.from_dict``) and one policy surface.
    ``SLO_TTFT_MS`` / ``SLO_PENALTY`` override the SLO terms at run time, so a
    sweep can change the bound without editing the checked-in topology.
    """
    options = dict(options or {})
    for env_key, option_key in (("SLO_TTFT_MS", "ttft_slo_ms"),
                                ("SLO_PENALTY", "slo_penalty"),
                                ("PREFILL_CLASS_LIMIT", "prefill_class_limit"),
                                # ``PLAN_TTL_S`` keeps a plan usable for longer
                                # when the control loop stalls.  Measured on the
                                # 2026-09-13 saturation round: with the default
                                # 4 s TTL, 70 of 600 requests hit an *expired*
                                # plan and fell back to the cost-ordered path --
                                # a different placement rule -- which showed up
                                # as a p95 tail (the LP variant's p95 was
                                # 3973 ms against 3186 ms for the heuristic with
                                # identical pair shares).  The controller's own
                                # gain hysteresis is what stops plan churn, so a
                                # longer TTL cannot make it switch more often.
                                ("PLAN_TTL_S", "plan_ttl_s"),
                                # ``CONTROL_INTERVAL_S`` stretches the gap
                                # between plan rebuilds.  Used to test whether
                                # the plan-based policies' transient latency
                                # spikes line up with plan-version switches
                                # (measured 2026-09-13: `casr_lp` had 68 requests
                                # slower than 3.5 s against 2 for the heuristic,
                                # clustered in time, with an otherwise identical
                                # per-second placement pattern).
                                ("CONTROL_INTERVAL_S", "control_interval_s"),
                                # ``OVERFLOW_PENALTY`` prices one unit of
                                # capacity overflow.  The default 10 is far
                                # below the cost of a saturated link, so under
                                # load the LP happily over-commits the fast
                                # Prefill (plan diagnostics show
                                # ``prefill_overflow`` of ~7x capacity) and
                                # spills across domains.  A sweep can raise it
                                # to force a capacity-feasible plan.
                                # ``WARMUP_S`` is how long a freshly started
                                # Prefill container must answer ``/metrics``
                                # before routing to it.  The default (150 s) is
                                # a conservative placeholder; measured vLLM
                                # restarts of an already-created container are
                                # ~30-40 s, and a scale-out experiment whose
                                # overload phase is shorter than this cannot see
                                # any benefit from the added instance.
                                ("WARMUP_S", "warmup_s"),
                                ("MAX_ACTIVE_PREFILL", "max_active_prefill"),
                                ("OVERFLOW_PENALTY", "overflow_penalty")):
        value = os.environ.get(env_key)
        if value:
            options[option_key] = (int(value) if option_key == "prefill_class_limit"
                                   else float(value))
    # ``structural`` is a nested block, so its knobs cannot ride the flat loop
    # above.  Both matter for honest scale-out behaviour: ``startup_cost`` is
    # what stops ``+P`` from firing on a nearly idle pool (with the default
    # ``0.0`` any tiny modelled improvement clears the threshold, so a topology
    # that ships a stopped spare scales out on the first tick), and
    # ``evaluation_window_ms`` is the horizon whose benefit has to amortise the
    # container boot (with the default 1 s no real startup cost can ever be
    # repaid, because a vLLM restart needs tens of seconds).
    structural_overrides = {}
    for env_key, option_key in (("STRUCTURAL_STARTUP_COST", "startup_cost"),
                                ("STRUCTURAL_EVAL_WINDOW_MS", "evaluation_window_ms"),
                                ("STRUCTURAL_DWELL_MS", "dwell_time_ms"),
                                ("STRUCTURAL_STARTUP_S", "startup_s"),
                                ("STRUCTURAL_HOLDING_COST", "holding_cost"),
                                ("STRUCTURAL_IDLE_COST_FRACTION", "idle_cost_fraction")):
        value = os.environ.get(env_key)
        if value:
            structural_overrides[option_key] = float(value)
    if structural_overrides:
        structural = dict(options.get("structural") or {})
        structural.update(structural_overrides)
        options["structural"] = structural
    prefills = list(config.get("prefills", ()))
    decodes = list(config.get("decodes", ()))
    prefill_instances = [RealInstance(spec, "prefill", instance_id)
                         for spec, instance_id in zip(prefills, slot_instance_ids(prefills))]
    decode_instances = [RealInstance(spec, "decode", instance_id)
                        for spec, instance_id in zip(decodes, slot_instance_ids(decodes))]
    # Ablation knob: ``EQUALIZE_{PREFILL,DECODE}_CAPACITY`` makes every instance
    # of that role share one capacity, so the only remaining difference between
    # two instances is their measured ``service_ms``.  Without it the capacity
    # map and the service time both penalise a slow instance, and an ablation
    # that removes one of them cannot tell which one was doing the work.
    # ``max`` uses the largest configured value (a fair "everyone is equally
    # capable" baseline).
    for role, items in (("PREFILL", prefill_instances), ("DECODE", decode_instances)):
        raw = os.environ.get(f"EQUALIZE_{role}_CAPACITY")
        if not raw:
            continue
        key = f"{role.lower()}_capacity"
        configured = dict(options.get(key) or {})
        values = [float(configured.get(str(item.instance_id),
                                       item.solver_capacity))
                  for item in items]
        shared = max(values) if raw.strip().lower() == "max" else float(raw)
        options[key] = {str(item.instance_id): shared for item in items}
    # Distance in the simulator's pair cost is a node-id gap, so give every
    # real domain a stable index.  Two instances on one host share it.
    domain_index = {}
    for instance in (*prefill_instances, *decode_instances):
        domain_index.setdefault(instance.domain, len(domain_index))
    for instance in (*prefill_instances, *decode_instances):
        instance.start_npu = domain_index[instance.domain]

    links = {}
    for link in config.get("links", ()):
        links[(str(link["src"]), str(link["dst"]))] = (
            float(link.get("bw_gbps", 1.0)), float(link.get("rtt_ms", 0.0)))

    hosts = config.get("hosts", {}) or {}
    for instance in (*prefill_instances, *decode_instances):
        host = hosts.get(instance.domain) or {}
        if host.get("ssh"):
            instance.ssh = str(host["ssh"])
            instance.ssh_port = int(host.get("ssh_port", 22) or 22)

    docker = None
    if str(options.get("scale_backend", "noop")).lower() == "docker":
        docker = DockerRuntime(
            name_prefix=str(options.get("container_prefix", "casr-md-")),
            local_domains=options.get("local_domains", ()))
    return RealCASRController(prefill_instances, decode_instances, links, options,
                              docker=docker)

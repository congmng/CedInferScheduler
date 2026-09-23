"""Simulated Prefill worker lifecycle for CASR structural actions."""

from __future__ import annotations

import math

from .resources import ResourceOrchestrator


class PrefillLifecycle:
    """Coordinate worker lifecycle with a node-level resource pool."""

    def __init__(self, config=None):
        config = config or {}
        self.min_active = max(0, int(config.get("min_active_prefill", 1)))
        self.max_active = config.get("max_active_prefill")
        # Whether the lifecycle may resize the pool from observed demand.  The
        # real deployment decides pool size only through the router's structural
        # actions, so an alignment arm that models "no scaling" must switch this
        # off; the default stays on to preserve the existing elastic behaviour.
        self.scale_on_demand = bool(config.get("scale_on_demand", True))
        self.warmup_ns = max(0, int(float(config.get("warmup_ms", 0)) * 1_000_000))
        self.resources = ResourceOrchestrator(config.get("resources"))
        self._bootstrapped = False
        #: Set once the first ``update`` has run.  Only the *first* tick may
        #: prune the pool down to a recorded/declared set; after that the pool
        #: an operator is running is the floor and only an explicit ``-P``
        #: shrinks it (see the ``scale_on_demand`` branch in ``update``).
        self._pruned_once = False
        self.capacity = {int(key): float(value)
                         for key, value in config.get("prefill_capacity", {}).items()}
        # The pool the deployment was actually running, when the caller knows
        # it (a replay reads it off the recorded run).  The demand heuristic
        # below can only rank workers by load or declared capacity, and neither
        # of those is what an operator's ``STOP_BEFORE_RUN`` decided.
        self.initial_active = {int(value)
                               for value in (config.get("initial_active_prefill") or ())}
        # Length-aware capacity.  ``prefill_capacity`` is a requests/s figure
        # measured at ONE prompt length; the real router scales it by the
        # observed prompt size (``prefill_tokens_per_s``) and the simulator has
        # to do the same, otherwise a KV-heavy workload looks like "one worker
        # is plenty" and the pool never grows -- measured 2026-09-15: 23 req/s
        # of 254-token prompts against declared capacities of 63/45/29/59
        # produced ``desired = 1`` while an always-on 4-Prefill pool was 3.4x
        # faster (2210 ms vs 7506 ms).
        self.tokens_per_s = {int(key): float(value)
                             for key, value in (config.get("prefill_tokens_per_s") or {}).items()
                             if not str(key).startswith("_")}
        self.reference_tokens = float(config.get("capacity_reference_tokens", 1024) or 1024)
        # Producer-side KV egress, the constraint that actually binds for long
        # prompts: one Prefill can push ~0.26 GB/s (measured), and a 1250-token
        # request costs 184 MB of KV, so ~1.4 req/s per worker.  Without this
        # term the demand heuristic only saw compute capacity (11.2/5.8 req/s)
        # and kept a single worker while the run was link-bound -- measured
        # 2026-09-15: elastic == static at 335.6 s mean latency, while an
        # always-on 4-worker pool was 2.4x faster.
        egress = config.get("kv_egress_gbps")
        if egress in (None, ""):
            caps = [float(item.get("capacity_bytes_per_s", 0.0) or 0.0)
                    for item in (config.get("shared_links") or ())]
            caps = [value for value in caps if value > 0.0]
            egress = (min(caps) / 1e9) if caps else 0.0
        self.egress_bytes_per_s = max(0.0, float(egress or 0.0) * 1e9)
        # Bytes of KV one prompt token occupies, for the egress term below.
        # The simulator's snapshots leave ``kv_bytes_per_request`` at 0 (only
        # the real controller fills it), so the lifecycle derives the request's
        # KV from its token count.  147456 B/token is Qwen3-8B bf16 measured on
        # the deployment (144 KiB, matching the profiler bundles).
        self.bytes_per_token = float(config.get("kv_bytes_per_token", 0.0) or 0.0) \
            or 147456.0
        self._warming_until = {}
        self.last_wanted = set()
        # Ticks an evaluator decision stays in force.  Without this the
        # demand heuristic recomputes its own (length-blind) target every tick
        # and drains the worker the counterfactual just asked for.
        self.override_hold_ns = int(float(config.get("override_hold_ms", 5000)) * 1_000_000)
        self._override = ()
        self._override_until_ns = -1
        # A shrinking pool under a growing backlog is a death spiral: the
        # workers that leave make the backlog worse, which lowers the observed
        # arrival rate further (it is measured *after* back-pressure), which
        # shrinks the pool again.  When the controller reports a backlog the
        # demand heuristic may only grow, never shrink.
        self.hold_on_backlog = bool(config.get("hold_on_backlog", True))

    def _effective_capacity(self, scheduler, rows):
        """Requests/s this Prefill can absorb for the *observed* prompt mix."""
        declared = max(1.0, float(self.capacity.get(
            scheduler.instance_id, scheduler.max_num_seqs)))
        tokens = [float(row.get("requested_tokens_ewma") or 0.0) for row in rows]
        tokens = [value for value in tokens if value > 0.0]
        ceiling = float(self.tokens_per_s.get(scheduler.instance_id, 0.0) or 0.0)
        kv_bytes = [float(row.get("kv_bytes_per_request") or 0.0) for row in rows]
        kv_bytes = [value for value in kv_bytes if value > 0.0]
        if not tokens:
            return declared
        average_tokens = sum(tokens) / len(tokens)
        limit = declared
        if ceiling > 0.0:
            limit = min(limit, ceiling / max(1.0, average_tokens))
        if self.egress_bytes_per_s > 0.0:
            per_request = (sum(kv_bytes) / len(kv_bytes)) if kv_bytes \
                else average_tokens * self.bytes_per_token
            if per_request > 0.0:
                limit = min(limit, self.egress_bytes_per_s / per_request)
        return max(0.05, limit)

    def update(self, current_ns, rows, schedulers, wanted_override=None,
               backlog_rps: float = 0.0, action=None):
        all_schedulers = list(schedulers)
        schedulers = [s for s in all_schedulers if s.pd_type == "prefill"]
        if not schedulers:
            return ()
        events = [event.as_dict() for event in self.resources.bootstrap(all_schedulers, current_ns)] if not self._bootstrapped else []
        self._bootstrapped = True
        startup_ns = max(self.warmup_ns, self.resources.startup_ns)
        demand = sum(max(float(row["arrival_rate_ewma"]), 0.0) for row in rows)
        average_capacity = max(1.0, sum(self._effective_capacity(s, rows)
                                        for s in schedulers) / len(schedulers))
        if self.scale_on_demand:
            desired = max(self.min_active, int(math.ceil(demand / average_capacity)))
        else:
            # The deployment has no demand-driven autoscaler of its own: the
            # pool only changes when the router takes a structural action
            # (``casr_full`` with ``scale_backend=docker``).  Leaving this
            # heuristic on made a "structural actions disabled" arm grow anyway
            # -- measured 2026-09-16 on the small-cluster elasticity A/B, where
            # the spare Prefill served 101 requests in the arm that was
            # supposed to keep it stopped.
            desired = self.min_active
        if self.max_active is not None:
            desired = min(desired, int(self.max_active))
        desired = min(desired, len(schedulers))
        if self.hold_on_backlog and float(backlog_rps or 0.0) > 0.05:
            # Never shrink while the offered load is known to exceed what was
            # served; growing is still allowed (the ceiling above bounds it).
            in_pool = sum(1 for s in schedulers
                          if s.admission_state != "INACTIVE")
            desired = max(desired, in_pool)
            if self.max_active is not None:
                desired = min(desired, int(self.max_active))
        ranked = sorted(schedulers, key=lambda s: (
            0 if s.admission_state == "ACTIVE" else 1,
            len(s.running) + len(s.waiting), s.instance_id))
        # Keep an evaluator decision in force long enough to be used: the
        # counterfactual pays a 60 s horizon (minus the 45 s boot) for adding a
        # worker, so undoing it on the very next tick wastes the whole boot.
        if wanted_override is not None:
            # A structural edit is a *request*, not a licence: the pool bounds
            # still belong to the operator.  This branch used to bypass them
            # entirely, which is how a ``max_active_prefill=1`` pool reached
            # three workers (measured 2026-09-15 in the elasticity replay).
            wanted = {int(instance_id) for instance_id in wanted_override}
            wanted &= {scheduler.instance_id for scheduler in schedulers}
            # A *scale-out* request must not evict what is already up.  The
            # evaluator re-issues its wanted set every tick, and the set it
            # computed 45 s ago does not contain the worker that boot has just
            # brought online -- so the newcomer was drained the moment it
            # finished warming and the next tick started its boot again.
            # Measured 2026-09-24 on the Qwen3 WAN 240 s peak: ACTIVE at 46.1 s,
            # back to WARMING at 46.4 s, cycling every 45 s, and only 0.5 of the
            # plan's flow ever landing on it -- i.e. a peak five times the boot
            # time still gained nothing.  ``-P`` keeps its meaning: only a
            # scale-in request may shrink the pool.
            if action == "+P":
                wanted |= {scheduler.instance_id for scheduler in schedulers
                           if scheduler.admission_state != "INACTIVE"}
            limit = len(schedulers) if self.max_active is None else int(self.max_active)
            if len(wanted) > limit:
                kept = []
                for scheduler in ranked:
                    if scheduler.instance_id in wanted:
                        kept.append(scheduler.instance_id)
                        if len(kept) >= max(0, limit):
                            break
                wanted = set(kept)
            floor = min(self.min_active, len(schedulers))
            if len(wanted) < floor:
                wanted |= {scheduler.instance_id for scheduler in ranked[:floor]}
            wanted = set(wanted)
            self._override = tuple(sorted(wanted))
            # The hold has to outlast the boot, or the worker the counterfactual
            # asked for is drained before it can serve anything.
            self._override_until_ns = int(current_ns + max(self.override_hold_ns, startup_ns))
        else:
            # Hold the workers the pool already had and only fill the rest from
            # ``ranked``.  The load term above ranks whichever worker just took a
            # request as the *busiest* one, so re-picking the whole set every
            # tick drains exactly the worker that was just given work --
            # measured 2026-09-16 on the small-cluster elasticity arm: p5090 was
            # drained on the very tick it received request 0, so an arm meant to
            # mirror the real run ("p4090 stopped, p5090 + p3090a working") ran
            # p3090a + p4090 instead and replayed 247 of 300 handoffs across
            # domains that the cluster pushed over its intra-node link.
            held = [scheduler.instance_id for scheduler in schedulers
                    if scheduler.instance_id in self.last_wanted
                    and scheduler.admission_state != "INACTIVE"]
            wanted = set(held[:desired])
            if len(wanted) < desired:
                # First tick, or genuine growth: fill from a *stable* order.
                # Capacity describes the hardware; ``ranked`` describes the
                # last request, and seeding a pool from it is what drained the
                # worker above.
                seed = [scheduler.instance_id for scheduler in schedulers
                        if scheduler.instance_id in self.initial_active
                        and scheduler.admission_state != "INACTIVE"]
                busy = [scheduler.instance_id for scheduler in schedulers
                        if scheduler.instance_id not in seed
                        and (scheduler.running or scheduler.waiting)]
                order = seed + busy + [scheduler.instance_id for scheduler in sorted(
                    schedulers,
                    key=lambda s: (-self.capacity.get(s.instance_id, 0.0),
                                   s.instance_id))
                    if scheduler.instance_id not in seed + busy]
                for instance_id in order:
                    if len(wanted) >= desired:
                        break
                    wanted.add(instance_id)
            if self._override and current_ns < self._override_until_ns:
                wanted = set(self._override)
            elif self._override:
                self._override = ()
            if not self.scale_on_demand and self._pruned_once:
                # The pool an operator is running is the floor; only an explicit
                # ``-P`` (a ``wanted_override`` carrying that action) may shrink
                # it.  Without this, ``desired = min_active`` collapsed the pool
                # back to one worker on the first tick after the evaluator's
                # 45 s hold expired, and the next ``+P`` had to boot another
                # 45 s container -- measured 2026-09-24 on the Qwen3 WAN 240 s
                # peak, where two workers swapped ACTIVE/WARMING every 45 s and
                # the pool never held three (elastic 2 206 ms against the
                # always-on pool's 1 759).  The first tick is exempt so a
                # recorded pool is still pruned to what it should be.
                wanted |= {scheduler.instance_id for scheduler in schedulers
                           if scheduler.admission_state != "INACTIVE"}
                if self.max_active is not None and len(wanted) > int(self.max_active):
                    keep = [scheduler.instance_id for scheduler in ranked
                            if scheduler.instance_id in wanted]
                    wanted = set(keep[:int(self.max_active)])
        # Keep an acquired worker alive until startup completes.  Otherwise a
        # low-demand tick can immediately cancel a previous scale-out before
        # the new worker becomes eligible for the next plan.
        #
        # This must also hold when a structural *override* is in force, which is
        # why it is no longer gated on ``wanted_override is None``: the
        # evaluator re-issues its own wanted set every tick, and if that set
        # happens to name a different worker the WARMING one is dropped, its
        # ``_warming_until`` deadline is lost, and the boot restarts from
        # scratch.  Measured 2026-09-24 on the Qwen3 WAN 240 s peak with a 45 s
        # container start: the +P worker went WARMING -> INACTIVE -> WARMING
        # every second and never reached ACTIVE, so 230 +P decisions produced a
        # pool that never grew and elastic == static.
        wanted.update(scheduler.instance_id for scheduler in schedulers
                      if scheduler.admission_state == "WARMING")
        wanted = {scheduler.instance_id for scheduler in schedulers
                  if scheduler.instance_id in wanted}
        if wanted_override is None:
            self.last_wanted = set(wanted)
        for scheduler in schedulers:
            state = scheduler.admission_state
            if state == "WARMING" and current_ns >= self._warming_until.get(scheduler.instance_id, current_ns + startup_ns):
                scheduler.set_admission_state("ACTIVE")
                events.append({"instance_id": scheduler.instance_id, "action": "warm_complete"})
                state = "ACTIVE"
            if state == "DRAINING" and not scheduler.running and not scheduler.waiting:
                scheduler.set_admission_state("INACTIVE")
        resource_events = self.resources.reconcile(schedulers, wanted, current_ns)
        for event in resource_events:
            events.append(event.as_dict())
            if event.action in {"resource_acquire", "resource_reuse"}:
                scheduler = next((s for s in schedulers if s.instance_id == event.instance_id), None)
                if scheduler is not None and scheduler.admission_state == "WARMING":
                    self._warming_until[event.instance_id] = current_ns + startup_ns
                    events.append({"instance_id": event.instance_id, "action": "warm_start"})
        self.last_wanted = set(wanted)
        self._pruned_once = True
        return tuple(events)

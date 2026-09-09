"""Resource-level local worker orchestrator for CASR.

The simulator creates scheduler objects up front because ASTRA-Sim topology is
static for one run.  This module supplies the missing resource boundary: only
workers admitted by the orchestrator own GPU slots and memory, and activation
fails when the configured node pool cannot satisfy the request.  The same
interface can later be backed by Kubernetes or Ray without changing the
controller or router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ResourceEvent:
    instance_id: int
    action: str
    node_id: int
    gpu_ids: tuple[int, ...] = ()
    gpu_mem_gb: float = 0.0
    at_ns: int = 0
    reason: str = ""

    def as_dict(self):
        return {
            "instance_id": self.instance_id,
            "action": self.action,
            "node_id": self.node_id,
            "gpu_ids": list(self.gpu_ids),
            "gpu_mem_gb": self.gpu_mem_gb,
            "at_ns": self.at_ns,
            "reason": self.reason,
        }


@dataclass
class NodeResources:
    node_id: int
    gpu_count: int
    gpu_mem_capacities: tuple[float, ...]
    used_gpu_mem_gb: float = 0.0
    allocations: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def free_gpu_count(self):
        used = sum(len(ids) for ids in self.allocations.values())
        return self.gpu_count - used

    @property
    def free_gpu_mem_gb(self):
        used = set(gpu for ids in self.allocations.values() for gpu in ids)
        return sum(capacity for gpu, capacity in enumerate(self.gpu_mem_capacities)
                   if gpu not in used)

    def allocate(self, instance_id: int, gpu_count: int, gpu_mem_gb: float,
                 per_gpu_mem_gb: float | None = None):
        if self.free_gpu_count < gpu_count:
            return None
        used = {gpu for ids in self.allocations.values() for gpu in ids}
        required = gpu_mem_gb / max(1, gpu_count) if per_gpu_mem_gb is None else per_gpu_mem_gb
        gpu_ids = tuple(gpu for gpu in range(self.gpu_count)
                        if gpu not in used and self.gpu_mem_capacities[gpu] + 1e-9 >= required)[:gpu_count]
        if len(gpu_ids) != gpu_count:
            return None
        self.allocations[instance_id] = gpu_ids
        self.used_gpu_mem_gb += gpu_mem_gb
        return gpu_ids

    def release(self, instance_id: int, gpu_mem_gb: float):
        self.allocations.pop(instance_id, None)
        self.used_gpu_mem_gb = max(0.0, self.used_gpu_mem_gb - gpu_mem_gb)


class ResourceOrchestrator:
    """Allocate and release simulated worker resources deterministically."""

    def __init__(self, config=None):
        config = config or {}
        self.startup_ns = max(0, int(float(config.get("startup_ms", 0)) * 1_000_000))
        self.reclaim_ns = max(0, int(float(config.get("reclaim_ms", 0)) * 1_000_000))
        self.nodes: dict[int, NodeResources] = {}
        for key, raw in (config.get("nodes") or {}).items():
            node_id = int(key)
            raw_capacity = raw.get("gpu_mem_gb", 0)
            if isinstance(raw_capacity, (list, tuple)):
                capacities = tuple(max(0.0, float(value)) for value in raw_capacity)
            else:
                capacities = (max(0.0, float(raw_capacity)),) * max(0, int(raw.get("gpu_count", 0)))
            gpu_count = max(0, int(raw.get("gpu_count", len(capacities))))
            if len(capacities) != gpu_count:
                raise ValueError(f"resource node {node_id} gpu_mem_gb must have one value per GPU")
            self.nodes[node_id] = NodeResources(
                node_id=node_id,
                gpu_count=gpu_count,
                gpu_mem_capacities=capacities,
            )
        self._allocations: dict[int, tuple[int, float, tuple[int, ...]]] = {}
        self._pending_release: dict[int, int] = {}
        self.last_events: tuple[ResourceEvent, ...] = ()

    @property
    def enabled(self):
        return bool(self.nodes)

    def _limits_for(self, scheduler):
        node_id = int(scheduler.node_id)
        node = self.nodes.get(node_id)
        if node is None:
            return None, 0, 0.0
        gpu_count = max(1, int(getattr(scheduler, "num_npus", 1)))
        per_gpu = float(scheduler.memory.npu_mem) / (1024 ** 3)
        return node, gpu_count, per_gpu * gpu_count

    def bootstrap(self, schedulers, current_ns=0):
        """Register initially ACTIVE workers and account for their resources."""
        if not self.enabled:
            return ()
        events = []
        for scheduler in schedulers:
            if scheduler.admission_state != "ACTIVE" or scheduler.instance_id in self._allocations:
                continue
            node, gpu_count, mem = self._limits_for(scheduler)
            if node is None:
                continue
            per_gpu_mem = float(scheduler.memory.npu_mem) / (1024 ** 3)
            gpu_ids = node.allocate(scheduler.instance_id, gpu_count, mem, per_gpu_mem)
            if gpu_ids is None:
                scheduler.set_admission_state("INACTIVE")
                events.append(ResourceEvent(scheduler.instance_id, "resource_reject", scheduler.node_id,
                                             at_ns=current_ns, reason="initial GPU capacity unavailable"))
                continue
            self._allocations[scheduler.instance_id] = (node.node_id, mem, gpu_ids)
            scheduler.resource_gpu_ids = gpu_ids
            scheduler.resource_mem_gb = mem
            events.append(ResourceEvent(scheduler.instance_id, "resource_acquire", node.node_id,
                                         gpu_ids, mem, current_ns, "initial allocation"))
        self.last_events = tuple(events)
        return self.last_events

    def reconcile(self, schedulers, wanted_ids, current_ns):
        """Make resource ownership match the lifecycle's wanted worker set."""
        if not self.enabled:
            self.last_events = ()
            return self.last_events
        events = []
        by_id = {int(s.instance_id): s for s in schedulers}
        for instance_id, release_at in list(self._pending_release.items()):
            if current_ns < release_at:
                continue
            allocation = self._allocations.pop(instance_id, None)
            scheduler = by_id.get(instance_id)
            if allocation is not None:
                node_id, mem, _ = allocation
                self.nodes[node_id].release(instance_id, mem)
                if scheduler is not None:
                    scheduler.resource_gpu_ids = ()
                    scheduler.resource_mem_gb = 0.0
                events.append(ResourceEvent(instance_id, "resource_release", node_id,
                                             at_ns=current_ns, reason="drain complete"))
            self._pending_release.pop(instance_id, None)
            if scheduler is not None and scheduler.admission_state == "DRAINING":
                scheduler.set_admission_state("INACTIVE")
                events.append(ResourceEvent(instance_id, "deactivate", scheduler.node_id,
                                             at_ns=current_ns, reason="resources reclaimed"))

        for scheduler in schedulers:
            instance_id = int(scheduler.instance_id)
            if instance_id not in wanted_ids:
                if scheduler.admission_state in {"ACTIVE", "WARMING"}:
                    scheduler.set_admission_state("DRAINING" if scheduler.running or scheduler.waiting else "INACTIVE")
                    if scheduler.admission_state == "INACTIVE" and instance_id in self._allocations:
                        self._pending_release[instance_id] = current_ns + self.reclaim_ns
                    events.append(ResourceEvent(instance_id, "drain_start" if scheduler.admission_state == "DRAINING" else "deactivate",
                                                 scheduler.node_id, at_ns=current_ns, reason="scale in"))
                elif scheduler.admission_state == "INACTIVE" and instance_id in self._allocations and instance_id not in self._pending_release:
                    self._pending_release[instance_id] = current_ns + self.reclaim_ns
                    events.append(ResourceEvent(instance_id, "reclaim_scheduled", scheduler.node_id,
                                                 at_ns=current_ns, reason="scale-in resource reclaim"))
                continue
            if scheduler.admission_state != "INACTIVE":
                continue
            existing = self._allocations.get(instance_id)
            if existing is not None:
                node_id, mem, gpu_ids = existing
                self._pending_release.pop(instance_id, None)
                scheduler.resource_gpu_ids = gpu_ids
                scheduler.resource_mem_gb = mem
                scheduler.set_admission_state("WARMING" if self.startup_ns else "ACTIVE")
                events.append(ResourceEvent(instance_id, "resource_reuse", node_id,
                                             gpu_ids, mem, current_ns,
                                             "cancelled pending reclaim"))
                continue
            node, gpu_count, mem = self._limits_for(scheduler)
            if node is None:
                events.append(ResourceEvent(instance_id, "resource_reject", scheduler.node_id,
                                             at_ns=current_ns, reason="node is not configured"))
                continue
            per_gpu_mem = float(scheduler.memory.npu_mem) / (1024 ** 3)
            gpu_ids = node.allocate(instance_id, gpu_count, mem, per_gpu_mem)
            if gpu_ids is None:
                events.append(ResourceEvent(instance_id, "resource_reject", node.node_id,
                                             at_ns=current_ns, reason="GPU capacity unavailable"))
                continue
            self._allocations[instance_id] = (node.node_id, mem, gpu_ids)
            scheduler.resource_gpu_ids = gpu_ids
            scheduler.resource_mem_gb = mem
            self._pending_release.pop(instance_id, None)
            scheduler.set_admission_state("WARMING" if self.startup_ns else "ACTIVE")
            events.append(ResourceEvent(instance_id, "resource_acquire", node.node_id,
                                         gpu_ids, mem, current_ns, "scale out"))
            if not self.startup_ns:
                events.append(ResourceEvent(instance_id, "activate", node.node_id,
                                             gpu_ids, mem, current_ns, "resource ready"))
        self.last_events = tuple(events)
        return self.last_events

    def snapshot(self):
        return {
            "enabled": self.enabled,
            "startup_ns": self.startup_ns,
            "reclaim_ns": self.reclaim_ns,
            "allocations": {
                str(instance_id): {
                    "node_id": node_id,
                    "gpu_ids": list(gpu_ids),
                    "gpu_mem_gb": mem,
                }
                for instance_id, (node_id, mem, gpu_ids) in sorted(self._allocations.items())
            },
            "nodes": {
                str(node_id): {
                    "gpu_count": node.gpu_count,
                    "gpu_mem_gb": sum(node.gpu_mem_capacities),
                    "gpu_mem_capacities": list(node.gpu_mem_capacities),
                    "used_gpu_mem_gb": node.used_gpu_mem_gb,
                    "free_gpu_count": node.free_gpu_count,
                    "free_gpu_mem_gb": node.free_gpu_mem_gb,
                }
                for node_id, node in sorted(self.nodes.items())
            },
            "pending_release": dict(sorted(self._pending_release.items())),
        }

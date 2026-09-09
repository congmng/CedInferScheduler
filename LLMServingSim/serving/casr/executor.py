"""Optional external executor for CASR structure edits."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    action: str
    instance_id: int
    ok: bool
    detail: str

    def as_dict(self):
        return {"action": self.action, "instance_id": self.instance_id,
                "ok": self.ok, "detail": self.detail}


class ReconfigExecutor:
    """Run optional scale commands after local admission changes.

    Commands are disabled unless ``backend`` is ``command``. Values may be
    shell-like strings or argument lists and support ``{instance_id}`` and
    ``{mode}`` substitutions. This leaves Kubernetes/Ray-specific behavior at
    the deployment boundary.
    """

    def __init__(self, config=None):
        config = config or {}
        self.backend = str(config.get("backend", "noop")).lower()
        self.timeout_s = max(0.1, float(config.get("timeout_ms", 1000)) / 1000.0)
        self.commands = {name: config.get(name) for name in
                         ("scale_out", "scale_in", "relocate")}

    @staticmethod
    def _argv(command, instance_id, mode):
        values = shlex.split(command) if isinstance(command, str) else command
        return [str(value).format(instance_id=instance_id, mode=mode)
                for value in values]

    def _run(self, action, instance_id, mode):
        command = self.commands.get(action)
        if self.backend != "command" or not command:
            return ExecutionResult(action, instance_id, True, "noop")
        try:
            completed = subprocess.run(self._argv(command, instance_id, mode),
                                       check=True, capture_output=True, text=True,
                                       timeout=self.timeout_s)
            return ExecutionResult(action, instance_id, True,
                                   completed.stdout.strip() or "command completed")
        except (OSError, subprocess.SubprocessError) as exc:
            return ExecutionResult(action, instance_id, False, str(exc))

    def apply(self, decision, active_ids, blocked_ids=()):
        active_ids = {int(value) for value in active_ids}
        wanted_ids = {int(value) for value in decision.wanted_ids}
        blocked_ids = {int(value) for value in blocked_ids}
        if decision.action == "+P":
            return tuple(self._run("scale_out", instance_id, decision.mode)
                         for instance_id in sorted(wanted_ids - active_ids - blocked_ids))
        if decision.action == "-P":
            return tuple(self._run("scale_in", instance_id, decision.mode)
                         for instance_id in sorted(active_ids - wanted_ids))
        return ()

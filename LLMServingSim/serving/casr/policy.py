"""Public extension point for research CASR policies."""

from __future__ import annotations

import importlib


class PolicyError(ValueError):
    """Raised when a policy returns an invalid control decision."""


class BuiltinPolicy:
    """Delegate to the configured greedy or LP flow solver."""

    def __init__(self, config=None):
        self.config = config or {}

    def solve(self, snapshot, prefill, decode, solver):
        return solver.solve(snapshot["prefix_states"], prefill, decode)


def load_policy(spec, config):
    """Load ``package.module:Class`` and instantiate it with CASR config.

    A custom class only needs ``solve(snapshot, prefill, decode, solver)`` and
    must return an iterable of FlowAssignment objects (or dictionaries with
    equivalent fields).  This narrow contract keeps research policy code out
    of the event loop and makes its output independently testable.
    """
    if not spec or spec == "builtin":
        return BuiltinPolicy(config)
    try:
        module_name, class_name = spec.split(":", 1)
        policy_class = getattr(importlib.import_module(module_name), class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise PolicyError(f"Cannot load CASR policy {spec!r}: {exc}") from exc
    policy = policy_class(config)
    if not callable(getattr(policy, "solve", None)):
        raise PolicyError(f"CASR policy {spec!r} has no solve() method")
    return policy

"""CASR control-plane primitives for state-aware P/D simulation.

The package deliberately sits above ``serving.core``: schedulers retain their
vLLM-like batching and KV semantics, while CASR consumes observations and
publishes routing/lifecycle decisions at a slower control cadence.
"""

from .affinity import AffinityPlan
from .controller import CASRController
from .evaluator import StructuralDecision, StructuralEvaluator
from .flow_solver import CapacityAwareFlowSolver, FlowAssignment, FlowSolverConfig, SharedLink
from .lifecycle import PrefillLifecycle
from .policy import BuiltinPolicy, PolicyError
from .prefix_profiler import PrefixProfiler
from .resources import ResourceEvent, ResourceOrchestrator
from .state import PrometheusStateCollector, parse_prometheus_text

__all__ = ["AffinityPlan", "BuiltinPolicy", "CASRController", "CapacityAwareFlowSolver", "FlowAssignment", "FlowSolverConfig", "PolicyError", "PrefillLifecycle", "PrefixProfiler", "ResourceEvent", "ResourceOrchestrator", "SharedLink", "StructuralDecision", "StructuralEvaluator", "PrometheusStateCollector", "parse_prometheus_text"]

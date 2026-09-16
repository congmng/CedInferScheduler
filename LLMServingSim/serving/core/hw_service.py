"""Price the plan with the same engine costs the simulator executes.

The CASR policy's per-instance service times come from the *deployment's*
``router_config.json`` (``service_ms``, measured on the real cluster with 16
output tokens at concurrency 8).  The simulator, however, executes from the
profiler bundles, and the two disagree about how much slower a second card is:

===========  ===================  ===============  ==================
Decode        deployment           profiled step     cluster TPOT
instance      ``service_ms``        (1250-token        (1250-token
              implied per token     context, 1 seq)    prompt, 16 out)
===========  ===================  ===============  ==================
``d5090``     9.0 ms                14.8 ms           15.0 ms
``d4090``     9.8 ms                25.8 ms           24.6 ms
``d3090a``    26.4 ms               31.3 ms           43.1-47.1 ms
===========  ===================  ===============  ==================

The deployment's spread is nearly flat (1.09x between the 5090 and the 4090)
while both the profiler (1.74x) and the cluster's own measured TPOT (1.64x)
say the Decode step is a weight read and the 4090 is ~1.7x slower.  A plan fed
the flat numbers moves traffic onto a card the execution then charges 1.7x for
-- measured 2026-09-16 on the small-cluster A/B: the elastic arm routed 82 of
452 requests to ``d4090`` and its mean latency rose 17% (and the elasticity
acceptance gate came out at -17.2%).

So the relative spread is taken from the profiler (the model the run executes
with) and anchored on the deployment's absolute value for the fastest card, so
the plan keeps its measured scale while ranking instances the way the engine
will actually behave.  Source of the TPOT column:
``/mnt/home/casr/results/small3-elastic-heavy/metrics-*.jsonl`` and
``small3-long-xfer/metrics-load.jsonl``.
"""

from .trace_generator import _load_perf_db, _lookup_1d, _tp_tables, resolve_variant
from .utils import get_config


def step_cost_ns(hardware, model, tp=1, tokens=1, variant=None):
    """Profiled cost of one forward pass, in ns.

    Every dense layer once at ``tokens`` tokens plus ``lm_head``.  At
    ``tokens=1`` that is a Decode step, which is a weight read; at the prompt
    length it is the Prefill's compute.  Attention is deliberately left out --
    it is context dependent and ~0.01 ms at 1k context, three orders of
    magnitude below the dense term, so the ratio between two cards is carried
    by the weight read, which is what the cluster's TPOT and prefill times
    measure.
    """
    config = get_config(model)
    variant = variant or resolve_variant("bfloat16", "auto", config)
    db = _load_perf_db(hardware, model, variant, {tp}, config["model_type"])
    tables = _tp_tables(db, tp)
    total = 0
    for layer, table in (tables.get("dense") or {}).items():
        total += _lookup_1d(table["keys"], table["values"], tokens)
    for layer, table in (tables.get("per_sequence") or {}).items():
        if layer == "lm_head":
            total += _lookup_1d(table["keys"], table["values"], 1)
    return total


def rescale_service_times(casr_config, instances, key="decode_service_ms",
                          tokens=1, verbose=True):
    """Rewrite ``casr_config[key]`` from the profiler's relative card speeds.

    ``instances`` is the simulator's instance list (each carrying
    ``instance_id``, ``hardware``, ``model_name`` and ``tp_size``).  The
    fastest instance keeps its configured value; every other one is scaled by
    the profiled step ratio, which is a hardware property and therefore
    independent of the prompt mix that produced the deployment's absolute
    numbers.
    """
    # Cluster JSONs key these maps by string instance id; the instance list
    # uses ints.
    configured = {int(instance_id): float(value)
                  for instance_id, value in (casr_config.get(key) or {}).items()}
    if not configured:
        return {}
    steps = {}
    for instance in instances:
        instance_id = int(instance["instance_id"])
        if instance_id not in configured:
            continue
        try:
            steps[instance_id] = step_cost_ns(
                instance["hardware"], instance["model_name"],
                tp=int(instance.get("tp_size", 1) or 1), tokens=tokens)
        except (FileNotFoundError, KeyError, ImportError):
            # No profiler bundle for this card: keep the measured value.
            continue
    if not steps:
        return {}
    anchor = min(steps, key=lambda instance_id: steps[instance_id])
    anchor_value = float(configured[anchor])
    rescaled = {}
    for instance_id, step in steps.items():
        scale = step / max(1, steps[anchor])
        rescaled[instance_id] = round(anchor_value * scale, 3)
    for instance_id, value in configured.items():
        if int(instance_id) not in rescaled:
            rescaled[int(instance_id)] = float(value)
    casr_config[key] = {str(key_): float(value)
                        for key_, value in sorted(rescaled.items())}
    if verbose:
        print(f"  • {key} from profiler  : "
              + ", ".join(f"{key_} {value:.1f} ms"
                          for key_, value in sorted(rescaled.items())))
    return rescaled

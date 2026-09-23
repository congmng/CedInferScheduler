"""Price the plan with the same engine costs the simulator executes.

The CASR policy's per-instance service times come from the *deployment's*
``router_config.json`` (``service_ms``, measured on the real cluster with 16
output tokens at concurrency 8).  The simulator, however, executes from the
profiler bundles, and the two disagree about how much slower a second card is:

===========  ================  ==============  ==============  ========
Decode        profiled kernels  with the        cluster TPOT    Decode
instance      (0.29.0 bundle)   per-card scale  (1250-token     scale
                                                 prompt, 16 out)
===========  ================  ==============  ==============  ========
``d5090``     9.68 ms           14.9 ms         14.5-15.0 ms    1.55x
``d4090``     16.63 ms          24.6 ms         24.6 ms         1.48x
``d3090a``    19.92 ms          45.0 ms         43.1-47.1 ms    2.26x
===========  ================  ==============  ==============  ========

The third column is ``step_cost_ns(..., decode=True)``.  The gap the profile
leaves is attention + sampling + scheduler + host, which a layer-wise eager
profile does not time; it is calibrated **per card** because one constant does
not fit (the 3090 needs 2.26x where the other two need ~1.5x -- itself an open
finding, the 3090's deployment reaches only ~38% of its peak bandwidth where
the 5090 and 4090 reach ~60-64%).  See ``trace_generator.DECODE_STEP_SCALE``.

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

from .trace_generator import (_load_architecture, _load_perf_db, _lookup_1d,
                              _tp_tables, decode_step_scale,
                              plan_layer_sequences, resolve_variant,
                              timing_calibration)
from .utils import get_config


def step_cost_ns(hardware, model, tp=1, tokens=1, variant=None, decode=False):
    """Profiled cost of one forward pass, in ns.

    Prologue plus ``num_hidden_layers`` blocks plus the head, each layer read
    from the profiler at ``tokens`` tokens.  At ``tokens=1`` that is a Decode
    step, which is a weight read; at the prompt length it is the Prefill's
    compute.  Attention is deliberately left out: it is context dependent and
    ~0.01 ms at 1k context, three orders of magnitude below the dense term, so
    the ratio between two cards is carried by the weight read -- which is what
    the cluster's TPOT and prefill times measure.

    ``decode=True`` applies the Decode-side calibration the simulator executes
    with (``trace_generator.DECODE_STEP_SCALE``): the layer-wise profile times
    kernels only, while the cluster's TPOT also carries attention, sampling,
    scheduler and host.  Passing ``False`` gives the raw profiled figure --
    useful when the question is what the profile itself says.
    """
    config = get_config(model)
    variant = variant or resolve_variant("bfloat16", "auto", config)
    db = _load_perf_db(hardware, model, variant, {tp}, config["model_type"])
    tables = _tp_tables(db, tp)
    dense = tables.get("dense") or {}
    per_sequence = tables.get("per_sequence") or {}
    architecture = _load_architecture(config["model_type"])
    sequence = architecture["sequence"]
    blocks = int(config.get("num_hidden_layers", 1) or 1)
    total = 0

    def dense_time(layer, count):
        table = dense.get(layer)
        return count * _lookup_1d(table["keys"], table["values"], tokens) if table else 0

    def sequence_time(layer, count):
        table = per_sequence.get(layer)
        return count * _lookup_1d(table["keys"], table["values"], 1) if table else 0

    for layer in sequence.get("prologue") or ():
        total += dense_time(layer, 1)
    per_layer = plan_layer_sequences(config, architecture)
    if per_layer is not None:
        # A hybrid whose layers differ in shape: charge each layer its own
        # pipeline instead of the flat template's, once per layer.
        for layers in per_layer:
            for layer in layers:
                if layer == "attention":
                    continue      # context dependent; ~0.01 ms at 1k
                total += dense_time(layer, 1)
    else:
        for group in ("pre_attn", "post_attn", "mlp_dense", "mlp_moe"):
            for layer in sequence.get(group) or ():
                if layer == "attention":
                    continue      # context dependent; ~0.01 ms at 1k
                total += dense_time(layer, blocks)
    for layer in sequence.get("head") or ():
        total += sequence_time(layer, 1)
    if decode:
        override = timing_calibration().get("decode_scale")
        scale = decode_step_scale(hardware) if override is None else override
        total = int(total * scale)
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
                tp=int(instance.get("tp_size", 1) or 1), tokens=tokens,
                decode=(tokens == 1))
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


def rescale_capacities(casr_config, instances, verbose=True):
    """Set the per-instance capacities the plan prices from the profiled engine.

    The configured ``prefill_capacity``/``decode_capacity`` (63/45/59 and
    65/45/57 req/s on the small cluster) were calibrated at short prompts and
    8-16 way concurrency; the simulator *executes* a 1250-token prompt in
    ~128 ms on a 5090 and a Decode step in ~15 ms, so one engine serves
    ``1000 / (decode_reference_tokens x step)`` ~ 4 req/s of reference-length
    classes -- 16x below the declared figure.  With the inflated numbers the LP
    never sees congestion, single-homes the fastest pair, and the structural
    scale-out the elasticity acceptance run asks for can never pay (measured
    2026-09-16: 452 requests pinned to one pair, elastic == static).
    """
    decode_ref = int(casr_config.get("decode_reference_tokens", 16) or 16)
    prefill_ref = int(casr_config.get("capacity_reference_tokens", 1024) or 1024)
    decode_capacity = {}
    prefill_capacity = {}
    for instance in instances:
        instance_id = int(instance["instance_id"])
        role = str(instance.get("pd_type", "")).lower()
        if role not in ("decode", "prefill"):
            continue
        try:
            step_ms = step_cost_ns(instance["hardware"], instance["model_name"],
                                   tp=int(instance.get("tp_size", 1) or 1),
                                   tokens=1 if role == "decode" else prefill_ref,
                                   decode=(role == "decode")) / 1e6
        except (FileNotFoundError, KeyError, ImportError):
            continue
        if step_ms <= 0:
            continue
        if role == "decode":
            decode_capacity[instance_id] = 1000.0 / (decode_ref * step_ms)
        else:
            # The LP's capacity unit is "reference-length requests per second".
            prefill_capacity[instance_id] = 1000.0 / step_ms
    if prefill_capacity:
        casr_config["prefill_capacity"] = {
            str(key): round(value, 4) for key, value in sorted(prefill_capacity.items())}
    if decode_capacity:
        casr_config["decode_capacity"] = {
            str(key): round(value, 4) for key, value in sorted(decode_capacity.items())}
    if verbose:
        print("  • capacities from profiler : "
              f"prefill {casr_config.get('prefill_capacity')} "
              f"decode {casr_config.get('decode_capacity')}")
    return prefill_capacity, decode_capacity


def resolve_runtime_capacities(casr_config, instances, verbose=True):
    """Decide the capacity every part of the runtime will price, once.

    Regression guard for the 2026-09-23 audit: the run used to price three
    different answers for the same resource --

    * the router's load denominators came from the cluster config's *declared*
      numbers,
    * the solver's Prefill capacity was rewritten every tick from the egress
      bound,
    * the lifecycle computed its own length-aware variant,

    and ``rescale_capacities`` only ran when the config said
    ``capacity_from_profile: true``.  On the P-15B peak that left Decode on the
    deployment's placeholders (110/170/230 req/s = 510) while the bundle says
    4.5/9.3/12.5 (26.3) -- a 19x overstatement that made every "is Decode the
    bottleneck" conclusion unfalsifiable.

    This resolves both roles from the profiler bundles (default) and then caps
    the Prefill side by the producer's KV egress, so the number that lands here
    is the one the router, the solver and the lifecycle all start from.
    ``capacity_from_profile: false`` keeps the declared numbers (A/B only).

    Returns ``(report, resolved)`` where ``report`` is a printable dict and
    ``resolved`` says which sources/limits applied.
    """
    resolved = {"from_profile": False, "egress_limited": {}}
    if casr_config.get("capacity_from_profile", True):
        rescale_capacities(casr_config, instances, verbose=False)
        resolved["from_profile"] = True

    kv_per_token = float(casr_config.get("kv_bytes_per_token") or 0.0)
    links = casr_config.get("shared_links") or ()
    if kv_per_token > 0.0 and links:
        reference = int(casr_config.get("capacity_reference_tokens", 1024) or 1024)
        per_request_bytes = kv_per_token * reference
        budgets: dict[int, float] = {}
        for link in links:
            capacity = float(link.get("capacity_bytes_per_s") or 0.0)
            if capacity <= 0.0:
                continue
            for pair in (link.get("pairs") or ()):
                prefill_id = int(pair[0])
                budgets[prefill_id] = min(budgets.get(prefill_id, capacity),
                                          capacity)
        if budgets and per_request_bytes > 0.0:
            bounded = {}
            for prefill_id, capacity in budgets.items():
                key = str(prefill_id)
                declared = float((casr_config.get("prefill_capacity") or {})
                                 .get(key, capacity / per_request_bytes))
                bound = capacity / per_request_bytes
                bounded[key] = round(min(declared, bound), 4)
                if bound < declared:
                    resolved["egress_limited"][key] = round(bound, 4)
            casr_config["prefill_capacity"] = {**(casr_config.get("prefill_capacity") or {}),
                                              **bounded}
    report = {
        "prefill_capacity": casr_config.get("prefill_capacity"),
        "decode_capacity": casr_config.get("decode_capacity"),
        "kv_bytes_per_token": kv_per_token,
        **resolved,
    }
    if verbose:
        print("  • capacities in use      : "
              f"prefill {report['prefill_capacity']} "
              f"decode {report['decode_capacity']} "
              f"(kv {kv_per_token:.0f} B/token)"
              + (f", egress-capped {resolved['egress_limited']}"
                 if resolved["egress_limited"] else ""))
    return report, resolved

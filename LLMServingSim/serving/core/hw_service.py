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

from .trace_generator import (_layer_types, _load_architecture, _load_perf_db,
                              _lookup_1d, _lookup_attention, _tp_tables,
                              decode_step_scale, plan_layer_sequences,
                              resolve_variant, timing_calibration)
from .utils import get_config


#: Per-card overhead a *pair* pays on top of the profiled Prefill step, in ms.
#: The layer-wise charge is exact -- ``step_cost_ns`` and the trace generator
#: agree to <0.1% at every prompt length -- but a serving pair's realised
#: period is the charge plus a per-request cost the layer profile does not time:
#: the P/D handoff, the Decode's own first step, and scheduler/host bookkeeping.
#: Measured 2026-09-23 with a saturated 1P1D probe (64 x 1024-token prompts,
#: ``max-num-batched-tokens 1024``, one P and one D on the same card), comparing
#: the steady-state TTFT period against the charge:
#:
#:   card      charge(1024)   period   overhead
#:   RTX5090      58.14 ms    73.6 ms    15.5 ms
#:   RTX4090      89.74 ms   121.2 ms    31.5 ms
#:   RTX3090     164.89 ms   204.3 ms    39.4 ms
#:
#: It is per card, and not a constant fraction of the charge, for the same
#: reason ``DECODE_STEP_SCALE_BY_HARDWARE`` is: the 3090's Decode step is 2.7x
#: the 5090's (14.96 vs 5.64 ms) while its Prefill is 2.8x, and the two do not
#: scale the same way.  This is a *capacity* input only -- ``step_cost_ns``
#: keeps pricing the pure layer model, so card ranking still comes from the
#: profile.
#:
#: Without it the plan priced one 5090 at 17.2 reference req/s (14.1 requests/s
#: of 1250-token prompts) against an executed 9.2, so it overloaded the fast
#: worker by ~1.5x and the peak queued on it (measured 2026-09-23, the 16 rps
#: P-15B cell: plan 98% of capacity, execution ~8.6 of 9.2 req/s, TTFT p50
#: 8.7 s of pure queuing).
PREFILL_PIPELINE_OVERHEAD_MS_BY_HARDWARE = {
    "RTX5090": 15.5,
    "RTX4090": 31.5,
    "RTX3090": 39.4,
}

#: Overhead for a card with no measurement, in ms.  Mid-table on purpose: a
#: card priced without any overhead looks ~27% faster than it runs, which is
#: the failure this table exists to remove.  Measure it with
#: ``tests/probe_prefill_pipeline.py`` and add the entry.
PREFILL_PIPELINE_OVERHEAD_MS = 25.0


def prefill_pipeline_overhead_ms(hardware):
    """Per-request pair overhead for ``hardware``, in ms (0.0 to disable)."""
    return float(PREFILL_PIPELINE_OVERHEAD_MS_BY_HARDWARE.get(
        hardware, PREFILL_PIPELINE_OVERHEAD_MS))


def prefill_period_ms(hardware, model, tp=1, tokens=1024, reference=1024,
                      chunk=None, variant=None):
    """Executed pair period for one ``tokens``-token prompt, in ms.

    ``charge x chunks + overhead x chunks``: the layer charge is exact (the
    trace generator and ``step_cost_ns`` agree to <0.1%), and every chunk-step
    pays the per-step overhead measured in
    ``PREFILL_PIPELINE_OVERHEAD_MS_BY_HARDWARE``.  This is the quantity the plan
    has to price -- pricing the bare charge overstates a 5090 by 27% at the
    reference length and by 47% at 1250 tokens (measured 2026-09-23: period
    73.6 / 106.7 ms against charges of 58.1 / 72.2).
    """
    chunk = max(1, int(chunk or reference or tokens or 1))
    steps = max(1, -(-int(tokens) // chunk))
    charge = prefill_charge_ns(hardware, model, tp=tp, tokens=tokens,
                               chunk=chunk, variant=variant) / 1e6
    return charge + prefill_pipeline_overhead_ms(hardware) * steps


#: Prompt lengths (as multiples of ``capacity_reference_tokens``) at which the
#: period curve is published to the solver.  ``1.0`` and ``1.0009`` straddle the
#: first chunk boundary, which is where the curve steps: a prompt one token
#: longer than the chunk needs a second step and pays the overhead again.
PREFILL_PERIOD_ANCHOR_RATIOS = (1.0, 1.0009, 1.25, 1.5, 2.0, 4.0)


def prefill_charge_ns(hardware, model, tp=1, tokens=1024, chunk=None,
                      variant=None):
    """Layer-wise charge for a ``tokens``-token prompt, chunked like the run.

    The scheduler feeds a prompt in ``max_num_batched_tokens``-sized chunks and
    each chunk is its own step: the profiled prologue/head are paid again, and a
    later chunk's attention runs over what the earlier ones already computed.
    Reproduces the trace generator to <0.1%, which is what lets the capacity and
    the timeline quote the same number for the same prompt.
    """
    chunk = max(1, int(chunk or tokens or 1))
    config = get_config(model)
    variant = variant or resolve_variant("bfloat16", "auto", config)
    db = _load_perf_db(hardware, model, variant, {tp}, config["model_type"])
    architecture = _load_architecture(config["model_type"])
    total = 0
    computed = 0
    while computed < int(tokens):
        size = min(chunk, int(tokens) - computed)
        base = step_cost_ns(hardware, model, tp=tp, tokens=size,
                            variant=variant)
        if size > 1:
            base = (base
                    - attention_step_ns(db, tp, config, architecture, size)
                    + attention_step_ns(db, tp, config, architecture, size,
                                        computed))
        total += base
        computed += size
    return total


def attention_step_ns(db, tp, config, architecture, tokens, kv_prefill=0):
    """Per-layer attention cost for a ``tokens``-token prompt, in ns.

    One lookup per layer, at ``(prefill_chunk=tokens, kv_prefill=kv_prefill,
    n_decode=0, kv_decode=0)``.  ``kv_prefill=0`` is a prompt with no prior
    context, i.e. a first chunk; a later chunk of a long prompt attends over
    what the earlier chunks already computed.  The **same per-block-type
    dispatch the timeline uses**: a model that declares ``layers_block_type``
    (P-15B: full causal / window 4 / window 128) prices each layer against its
    own ``attention_<block>.csv``, falling back to the single ``attention``
    table when the bundle carries only one.  Returns 0 when the bundle has no
    attention profile at all, so the flat models that predate it keep working.
    """
    names = config.get("layers_block_type")
    types = architecture.get("layer_types") or {}
    blocks = int(config.get("num_hidden_layers", 1) or 1)
    total = 0
    for layer in range(blocks):
        table = None
        if names and types and 0 <= layer < len(names):
            name = str(names[layer])
            if name in types:
                table = f"attention_{name}"
        try:
            total += _lookup_attention(db, tp, tokens, kv_prefill, 0, 0,
                                       table=table)
        except KeyError:
            return 0
    return total


def step_cost_ns(hardware, model, tp=1, tokens=1, variant=None, decode=False):
    """Profiled cost of one forward pass, in ns.

    Prologue plus ``num_hidden_layers`` blocks plus the head, each layer read
    from the profiler at ``tokens`` tokens.  At ``tokens=1`` that is a Decode
    step, which is a weight read; at the prompt length it is the Prefill's
    compute.  Attention is included on the Prefill path.

    It used to be left out on the assumption that attention is ~0.01 ms at 1k
    context, three orders of magnitude below the dense term.  That holds for a
    dense model whose prefill really is a weight read (Qwen3-8B: 2.9 ms of
    attention against 68.4 ms of dense at 1024 tokens on a 5090), and it is
    badly false for a compressed-KV/hybrid model, where attention is the
    prefill: on the P-15B bundle at 1024 tokens, RTX5090, dense + MoE is 11.2 ms
    and attention is 47.0 ms -- 81% of the step, 4.2x the dense total.  Omitting
    it put ``prefill_capacity = 1000/step_ms`` at 89 req/s against a measured
    ~13.5 req/s, and the plan single-homed 740 requests on that one worker while
    the router stalled behind it (measured 2026-09-23, docs/实验数据集与对比基线说明.md
    6.19).  Capacity and timeline now read the same tables.

    The Decode path does **not** add attention here: ``decode_step_scale``
    already carries attention + sampling + scheduler + host for a 1-token step,
    calibrated per card against the deployment's TPOT, so charging it twice
    would break that anchor.

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
    # ``tokens == 1`` is a Decode step whatever the flag says, and that path's
    # attention is the one ``decode_step_scale`` carries; charge attention only
    # on a real prompt, so nothing is counted twice.
    if not decode and tokens > 1:
        total += attention_step_ns(db, tp, config, architecture, tokens)
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
            # The LP's capacity unit is "reference-length requests per second",
            # priced at the rate a *pair* executes -- the charge plus the
            # per-request handoff/Decode-first-step overhead (see
            # ``PREFILL_PIPELINE_OVERHEAD_MS_BY_HARDWARE``).
            reference_ms = prefill_period_ms(
                instance["hardware"], instance["model_name"],
                tp=int(instance.get("tp_size", 1) or 1), tokens=prefill_ref,
                reference=prefill_ref,
                chunk=int(casr_config.get("max_num_batched_tokens",
                                          prefill_ref) or prefill_ref))
            prefill_capacity[instance_id] = 1000.0 / reference_ms
    if prefill_capacity:
        casr_config["prefill_capacity"] = {
            str(key): round(value, 4) for key, value in sorted(prefill_capacity.items())}
    if decode_capacity:
        casr_config["decode_capacity"] = {
            str(key): round(value, 4) for key, value in sorted(decode_capacity.items())}
    # Make the *prompt-length* term live.  ``_prefill_length_work`` scales a
    # request by ``tokens / capacity_reference_tokens`` only when the instance
    # declares a token-rate ceiling; without it the solver falls back to 1.0,
    # and 53 of the 59 cluster configs -- including every three-domain matrix
    # config -- declare none.  The LP then priced a 1250-token prompt as one
    # 1024-token reference unit, 22% under, which is exactly the margin the
    # plan needs to keep off a worker that is already at its measured limit.
    # The ceiling is a restatement of the capacity just resolved, so derive it
    # here instead of leaving it to each config author.
    #
    # The Decode side has the same dead knob (``decode_ms_per_1k_tokens``) and is
    # deliberately left off: that capacity is a *serial* figure
    # (``1000 / (reference_tokens x step)``) which understates a batched engine
    # by roughly ``max_num_seqs``, so scaling the decode load by output length
    # on top of it would compound two errors instead of removing one.  Fixing
    # it needs the batched Decode capacity first.
    if prefill_capacity and not casr_config.get("prefill_tokens_per_s"):
        casr_config["prefill_tokens_per_s"] = {
            str(key): round(value * prefill_ref, 4)
            for key, value in sorted(prefill_capacity.items())}
    # Publish the *period curve* the LP scales a longer prompt by.  A linear
    # token ratio is the wrong shape once a prompt needs more than one chunk:
    # 1250 tokens is not 1.22 reference units, it is a second step (measured
    # period 106.7 ms against 73.6 ms, i.e. 1.45).  The curve is the same
    # ``prefill_period_ms`` the capacity came from, sampled across the first
    # chunk boundary, so the plan and the execution agree at the anchors.
    chunk = int(casr_config.get("max_num_batched_tokens", prefill_ref)
                or prefill_ref)
    curve = {}
    for instance in instances:
        instance_id = int(instance["instance_id"])
        if instance_id not in prefill_capacity:
            continue
        points = {}
        for ratio in PREFILL_PERIOD_ANCHOR_RATIOS:
            tokens = max(1, int(round(prefill_ref * ratio)))
            points[str(tokens)] = round(prefill_period_ms(
                instance["hardware"], instance["model_name"],
                tp=int(instance.get("tp_size", 1) or 1), tokens=tokens,
                reference=prefill_ref, chunk=chunk), 3)
        curve[str(instance_id)] = points
    if curve:
        casr_config["prefill_period_ms"] = curve
    if verbose:
        print("  • capacities from profiler : "
              f"prefill {casr_config.get('prefill_capacity')} "
              f"decode {casr_config.get('decode_capacity')}")
        print("  • prefill period @ref     : "
              + ", ".join(f"{key} {1000.0 / value:.1f} ms"
                          for key, value in sorted(prefill_capacity.items())))
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
                # A producer's *best* pairing.  Deployments price the same-host
                # push and the wire separately, and the plan's own per-link byte
                # constraints bound each pairing on its own; collapsing the two
                # with ``min`` charged same-domain handoffs at the cross-domain
                # rate, which capped the fastest Prefill below what it executes
                # (measured 2026-09-24: priced 6.33 req/s against 8.3 executed
                # on the P-15B WAN environment).
                budgets[prefill_id] = max(budgets.get(prefill_id, 0.0), capacity)
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
        "prefill_tokens_per_s": casr_config.get("prefill_tokens_per_s"),
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

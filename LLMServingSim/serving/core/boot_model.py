"""Startup-cost models for a worker boot (ServerlessLLM-style fast loading).

Every elastic arm pays a boot before a new worker can serve, and the arm's
result depends on that number as much as on the flow solver.  Until now the
number was a single ``startup_ms`` written into the config, which conflates two
different physical paths:

* **engine init** -- process start, engine import, CUDA context, KV pool and
  NIXL/UCX link setup.  Measured on the deployment: ``banner -> KV ready`` is a
  20 s median warm restart of which 2.8 s is weight loading, and 45 s cold of
  which 21.7 s is weight loading (``实验数据集与对比基线说明.md`` 6.25), so the
  engine part is ~17.2 s and does not move.
* **weight load** -- ``weights_bytes`` over the storage path's bandwidth.  That
  is the part ServerlessLLM attacks: prefetch/pipelined reading from the local
  device instead of serial reads through an uncached EXT4 mount.

``resolve_startup_ms`` derives the number from those two terms so an arm can be
"the same system with a faster loader" rather than an arbitrary discount.  The
defaults are this deployment's measurements; the caller overrides the storage
bandwidth (and optionally the overlap) to describe the loader under test.
"""

from __future__ import annotations

#: ``banner -> KV ready`` minus the weight load, from the 46 measured restarts.
ENGINE_INIT_MS = 17_200.0
#: 15.26 GiB of P-15B weights (5 safetensors shards).
P15B_WEIGHTS_BYTES = 15.26 * (1024 ** 3)


def resolve_startup_ms(config, fallback_ms):
    """Startup milliseconds for this run, or ``fallback_ms`` when unmodelled.

    ``config`` is the ``casr.boot`` block:

    ``model: "weight_load"``
        derive the number from engine init + ``weights_bytes`` /
        ``load_bandwidth_bytes_per_s``.
    ``overlap: true`` (default)
        the loader runs while the engine initialises, so the startup is the
        larger of the two instead of their sum (ServerlessLLM pipelines the
        checkpoint read against initialization).  ``overlap: false`` serialises
        them, which is what an unmodified container does.
    """
    spec = dict(config or {})
    if str(spec.get("model", "") or "").lower() not in ("weight_load", "serverless"):
        return float(fallback_ms)
    engine_ms = float(spec.get("engine_init_ms", ENGINE_INIT_MS))
    weights = float(spec.get("weights_bytes", P15B_WEIGHTS_BYTES) or 0.0)
    bandwidth = float(spec.get("load_bandwidth_bytes_per_s", 0.0) or 0.0)
    weight_ms = (weights / bandwidth * 1000.0) if bandwidth > 0.0 else 0.0
    if bool(spec.get("overlap", True)):
        return max(engine_ms, weight_ms)
    return engine_ms + weight_ms

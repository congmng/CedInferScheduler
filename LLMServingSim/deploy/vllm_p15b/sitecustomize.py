"""Register P-15B with vLLM at interpreter start.

Put this directory on ``PYTHONPATH`` and every vLLM process -- including the
worker vLLM spawns -- learns the ``P15BForCausalLM`` architecture before the
model config is read.  Same mechanism as ``design/dsv4_spec_fix``.

    PYTHONPATH=/work/deploy/vllm_p15b python3 -m profiler profile casr/P15B-r4 ...
"""

import pathlib
import sys

try:
    # ``sitecustomize`` is imported as a top-level module, so ``vllm_model``'s
    # own relative imports (``from .config import ...``) only resolve once its
    # parent directory is importable as a package.  Add it rather than
    # rewriting the modules to absolute imports -- keeping them a package is
    # what lets ``tests/check_p15b_shapes.py`` import them the same way.
    _parent = str(pathlib.Path(__file__).resolve().parents[1])
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from vllm_p15b.hf_config import register as register_config
    from vllm_p15b.vllm_model import register

    register_config()
    register()
    print("[p15b] config type + architecture registered", flush=True)
except Exception as exc:  # pragma: no cover - reported, never fatal
    print(f"[p15b] NOT registered: {type(exc).__name__}: {exc}", flush=True)

"""P-15B: the self-built DSV4-style block, shaped for vLLM and the profiler.

Spec: ``docs/DSV4-P15B设计.md``  Plan: ``docs/P15B-profile路径B实现计划.md``

Why this package exists: the official DeepSeek-V4 path only boots on sm90+
(mHC needs a Hopper/Blackwell deep_gemm kernel) and its o_proj is fp8-only, so
three of our four card types cannot load it at all.  This is the same design
with standard pre-norm residuals, plain bf16, and our own sparse attention --
loadable on 3090 / 4090 / 5090 / A100 alike.

Module names here are the *canonical* names the profiler's catalog binds
(``profiler/models/p15b.yaml``), which is why every RMSNorm and every linear
gets its own class instead of sharing a generic one: the catalog matches on
(class name, ancestor class name), so two ``nn.Linear`` under one parent would
collide.
"""

from .model import P15BForCausalLM                     # noqa: F401

__all__ = ["P15BForCausalLM"]

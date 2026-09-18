"""Portable reference implementation of a DeepSeek-V4-style block.

Why this exists: the released V4 checkpoints are 284B/1.6T and the official
vLLM path cannot run on our 3090/4090/A100 (mHC kernels exist for sm90/100/120
only, and the NVIDIA o_proj is fp8-DeepGEMM-only).  We want a small model of our
own on the same *attention* design, so this package is the executable
specification of that design: plain PyTorch, no custom kernels, runnable on any
CUDA GPU, and precise about the one thing we care most about -- how many KV
bytes each layer costs per token.

Design decisions taken here (see docs/DSV4_10-30B跨卡设计.md):

* **mHC is removed.**  Standard pre-norm residual
  ``x = x + Attn(norm(x)); x = x + FFN(norm(x))``.  mHC is worth 0.012% of the
  official model's parameters and 0.3% of its per-token FLOPs, and cutting it is
  what frees us from the sm90+ hyperconnection kernels.
* **MLA latent + CSA/HCA compression are kept.**  Per token a full layer stores
  ``head_dim + qk_rope_head_dim`` values; a compressed layer stores
  ``2 * coff * head_dim`` values every ``compress_ratio`` tokens, where
  ``coff = 2`` for ratio 4 (overlapping window) and 1 otherwise -- the same
  accounting the vLLM KV spec uses.
* **The compressor's internal MLP is our choice.**  The released config exposes
  the ratio, the state width and the op order (compress -> norm -> RoPE ->
  store) but not the projection itself, so we use a linear over the window.
"""

from .config import DSV4RefConfig, REF_CONFIGS                       # noqa: F401
from .model import DSV4RefModel                                      # noqa: F401


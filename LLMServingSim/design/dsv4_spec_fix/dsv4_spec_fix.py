"""Price the DSV4 compressor state cache the way the design intends.

`docs/DSV4_10-30B跨卡设计.md` §3.1.1 measured what vLLM 0.29.0 charges for the
compressor's state cache and why:

    CompressorStateCache.get_kv_cache_spec()   # compressor.py:173
        return SlidingWindowMLASpec(block_size=4 or 8, head_size=state_dim,
                                    dtype=torch.float32, ...)

`tokens_per_state` is left at its default of 1, so `num_states = block_size`
and the page is charged to `block_size` *tokens* -- while those `block_size`
states actually cover `block_size * compress_ratio` tokens.  A CSA layer is
therefore billed 8,208 B/token against an analytic 512 B, and HCA 4,104 B
against 8 B.

This module applies the accounting half of the fix (the storage half, fp32 ->
bf16, is a model change and is left to the design): one page holds one state
and is charged to `compress_ratio` tokens, which is what the cache really
stores.  Loading it via ``sitecustomize`` lets the probe measure the corrected
number without editing the image.

    PYTHONPATH=/work/design/dsv4_spec_fix python3 dsv4_kv_probe.py ...
"""

from __future__ import annotations

APPLIED: list[str] = []


def _ratio_of(cache) -> int:
    """Recover ``compress_ratio`` from the state the cache actually keeps.

    ``CompressorStateCache`` stores ``sliding_window = coff * compress_ratio``
    (``coff = 2`` for ratio 4, 1 otherwise) but not the ratio itself, and its
    log line is the only place the probe can read it back from.
    """
    window = cache.sliding_window
    return window // 2 if window == 8 else window


def patch() -> None:
    from vllm.models.deepseek_v4 import compressor as module
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    cls = module.CompressorStateCache
    if getattr(cls, "_dsv4_accounting_patched", False):
        return
    original = cls.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = original(self, vllm_config)
        ratio = _ratio_of(self)
        fixed = SlidingWindowMLASpec(
            block_size=ratio,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
            state_content_bytes=spec.state_content_bytes,
            alignment=spec.alignment,
            tokens_per_state=ratio,
        )
        print(
            "[spec-fix] compressor state cache ratio=%d head=%d dtype=%s: "
            "%.1f B/token (page %d) -> %.1f B/token (page %d)"
            % (ratio, spec.head_size, spec.dtype,
               spec.page_size_bytes / spec.block_size, spec.page_size_bytes,
               fixed.page_size_bytes / fixed.block_size, fixed.page_size_bytes),
            flush=True,
        )
        return fixed

    cls.get_kv_cache_spec = get_kv_cache_spec
    cls._dsv4_accounting_patched = True
    APPLIED.append("CompressorStateCache.get_kv_cache_spec")

"""Apply the DSV4 compressor KV-spec fix at interpreter start.

Python imports ``sitecustomize`` before user code, which is early enough that
every later ``import vllm`` sees the patched class.  Putting this directory on
``PYTHONPATH`` is the whole installation step; see ``dsv4_spec_fix.py``.
"""

try:
    from dsv4_spec_fix import patch

    patch()
    print("[spec-fix] applied", flush=True)
except Exception as exc:  # pragma: no cover - reported, never fatal
    print(f"[spec-fix] NOT applied: {type(exc).__name__}: {exc}", flush=True)

"""Every architecture yaml must bind the head to the module vLLM calls.

vLLM computes the head projection inside ``LogitsProcessor``
(``LogitsProcessor._apply_head`` calls ``lm_head.quant_method.apply``), so a
``ParallelLMHead`` module is never invoked and never appears in the profile
tree.  A catalog entry bound to it therefore produces **no rows at all** and
the simulator silently skips the layer -- which is exactly what happened to
P-15B: ``lm_head`` was bound to ``ParallelLMHead``, the row landed under the
``sampler`` name instead (that binding pointed at ``LogitsProcessor``), and
every other model's yaml had it right.

The gate is cheap and torch-free: it reads the yaml files directly.  Add a
model here only if its head genuinely goes through a different module.
"""

import pathlib
import sys
import unittest

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]

#: canonical name -> the module class vLLM actually invokes for it.
EXPECTED = {"lm_head": "LogitsProcessor", "sampler": "Sampler"}


class HeadBindingTests(unittest.TestCase):
    def _yamls(self):
        return sorted((REPO / "profiler" / "models").glob("*.yaml"))

    def test_at_least_one_yaml_declares_a_head(self):
        self.assertTrue(self._yamls(), "no architecture yamls found")

    def test_head_and_sampler_bind_their_real_modules(self):
        checked = 0
        for path in self._yamls():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            group = ((raw.get("catalog") or {}).get("per_sequence") or {})
            for canonical, expected in EXPECTED.items():
                entry = group.get(canonical)
                if entry is None:
                    continue
                checked += 1
                with self.subTest(yaml=path.name, layer=canonical):
                    self.assertEqual(
                        entry.get("vllm"), expected,
                        f"{path.name}: {canonical!r} must bind {expected!r} -- "
                        f"binding the weight-holder produces an empty table")
        self.assertGreater(checked, 0, "no per_sequence bindings found to check")


if __name__ == "__main__":
    unittest.main()

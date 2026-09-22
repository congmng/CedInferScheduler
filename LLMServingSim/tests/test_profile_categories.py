"""``--categories`` has to keep exactly the categories it names, or a
re-measurement silently rewrites the wrong CSV.

The flag exists for one job: refresh ``dense.csv`` inside a finished bundle
without touching a four-hour ``attention.csv``.  ``--force`` only rewrites the
CSVs the *running* categories own (``profiler/core/runner.py``), so what has to
hold is that ``categories_for(..., only=("dense",))`` returns nothing else.

``profiler.core.categories`` pulls in the engine (and therefore torch), so on a
bare host this module skips; run it in the image to exercise it:

    docker run --rm --entrypoint python3 -v "$PWD":/work -w /work \
        vllm/vllm-openai:casr029 -m pytest tests/test_profile_categories.py -q
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    from profiler.__main__ import _parse_categories              # noqa: E402
    from profiler.core.categories import categories_for          # noqa: E402
    from profiler.core.config import load_architecture           # noqa: E402
    HAVE_ENGINE = True
except ImportError:                                              # no torch
    HAVE_ENGINE = False

requires_engine = unittest.skipUnless(
    HAVE_ENGINE, "needs the profiler's engine (torch); run inside the image")


@requires_engine
class ParseCategoriesTests(unittest.TestCase):
    def test_comma_and_space_both_work(self):
        self.assertEqual(_parse_categories("dense,moe"), ("dense", "moe"))
        self.assertEqual(_parse_categories("dense, moe"), ("dense", "moe"))
        self.assertEqual(_parse_categories("dense"), ("dense",))

    def test_unknown_names_are_rejected(self):
        # Whitespace is a separator, so a trailing space is fine; an unknown
        # name and an empty list are not.
        for raw in ("dense,skew", "skew", ""):
            with self.subTest(raw=raw), self.assertRaises(
                    Exception):  # argparse.ArgumentTypeError
                _parse_categories(raw)


@requires_engine
class CategoriesForTests(unittest.TestCase):
    def setUp(self):
        arch_path = REPO / "profiler" / "models" / "p15b.yaml"
        self.arch = load_architecture(arch_path)

    def test_no_filter_keeps_every_category(self):
        names = [c.name for c in categories_for(self.arch, 1)]
        self.assertEqual(names, ["dense", "per_sequence", "attention", "moe"])

    def test_filter_keeps_only_the_named_ones(self):
        names = [c.name for c in categories_for(self.arch, 1, only=("dense",))]
        self.assertEqual(names, ["dense"])
        names = [c.name for c in categories_for(
            self.arch, 1, only=("attention", "moe"))]
        self.assertEqual(names, ["attention", "moe"])

    def test_filter_never_invents_a_category(self):
        """``moe`` is dropped at tp != 1 -- the filter must not resurrect it."""
        names = [c.name for c in categories_for(self.arch, 2, only=("moe",))]
        self.assertEqual(names, [])


if __name__ == "__main__":
    unittest.main()

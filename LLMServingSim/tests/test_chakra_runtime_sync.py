"""Guards for the two ways a simulator run lied to us on 2026-09-15.

1. **Tree edits that never reached the runtime.**  ``AGENTS.md`` documents
   ``pip3 install .`` inside ``astra-sim/extern/graph_frontend/chakra`` as the
   way to make converter changes take effect, and ``graph_generator.py`` warns
   that the import resolves to the *installed* chakra.  The 2026-09-12 deadlock
   fix was written in the tree and never installed, so every multi-domain run
   deadlocked at 0 tokens/s while the source looked correct.  ``pip3 install``
   itself no longer works in this environment (setup.cfg's ``build_grpc``
   metadata fails), so the sync is a one-file copy -- which makes a machine
   check mandatory rather than optional.

2. **Graphs whose send/recv pairs disagree.**  That mismatch is what the
   deadlock looked like: the Prefill's final output send named the legacy
   adjacent rank while the receiver ran elsewhere, and both sides waited
   forever.  ``tests/check_et_pairing.py`` catches it in milliseconds; this
   test proves the checker still detects a deliberately broken graph.
"""

import pathlib
import shutil
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAKRA_SRC = REPO / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "src"
for extra in (REPO, CHAKRA_SRC):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

sys.path.insert(0, str(REPO / "tests"))
import check_et_pairing                                              # noqa: E402

REPO_CONVERTER = (CHAKRA_SRC / "converter" / "llm_converter.py").resolve()

HEADER = ("PREFILL\t\tmodel_parallel_NPU_group: 1\t\tpp_stage_boundaries:"
          " 0\t\tpd_decode_npu_offset: 5")
ROWS = [[name, "100", "LOCAL", "16", "LOCAL", "16", "LOCAL", "16", "NONE",
         "2048", "NONE"]
        for _layer in range(2) for name in ("qkv_proj", "attn", "o_proj")]


def _convert(directory, converter_cls=None):
    import chakra.src.converter.llm_converter as module
    cls = converter_cls or module.LLMConverter
    cls("", str(pathlib.Path(directory) / "llm"), num_npus=1,
        npu_offset=3).convert_rows(HEADER, ROWS)
    return sorted(pathlib.Path(directory).glob("*.et"))


class RuntimeConverterIdentityTests(unittest.TestCase):
    """The loaded converter must be byte-identical to the checked-out one."""

    def test_runtime_converter_matches_the_repo_tree(self):
        import chakra.src.converter.llm_converter as module

        loaded = pathlib.Path(module.__file__).resolve()
        with self.subTest(loaded=str(loaded)):
            self.assertTrue(
                loaded.read_bytes() == REPO_CONVERTER.read_bytes(),
                f"the runtime loads {loaded}\n"
                f"but the tree has {REPO_CONVERTER}\n"
                "the 2026-09-12 handoff fixes are only in one of them; sync with:\n"
                f"  cp {REPO_CONVERTER} {loaded}")


class EtPairingTests(unittest.TestCase):
    def test_generated_graph_pairs_every_send_and_recv(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _convert(tmp)
            self.assertTrue(paths, "converter produced no graphs")
            violations = check_et_pairing.check(paths)
            self.assertEqual(violations, [], "\n".join(violations))

    def test_a_mismatched_send_is_detected(self):
        """Reproduce the deadlock shape: sender names a rank with no receiver."""
        import chakra.src.converter.llm_converter as module
        from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage

        with tempfile.TemporaryDirectory() as tmp:
            _convert(tmp)
            sender = pathlib.Path(tmp) / "llm.3.et"
            graphs = check_et_pairing.read_graph(sender)
            self.assertTrue(graphs[0], "expected a send node in the Prefill graph")

            # Rewrite every send's destination back to the legacy adjacent rank.
            buffer = pathlib.Path(tmp) / "patched.et"
            with sender.open("rb") as src, buffer.open("wb") as dst:
                decodeMessage(src, module.GlobalMetadata())
                encodeMessage(dst, module.GlobalMetadata())
                node = module.Node()
                while decodeMessage(src, node):
                    if node.type == module.COMM_SEND_NODE:
                        for attr in node.attr:
                            if attr.name == "comm_dst":
                                attr.int32_val = 4
                    encodeMessage(dst, node)
                    node = module.Node()
            shutil.move(buffer, sender)

            violations = check_et_pairing.check(sorted(pathlib.Path(tmp).glob("*.et")))
            self.assertTrue(violations, "the pairing checker missed a broken graph")
            self.assertTrue(any("send key" in item for item in violations),
                            violations)


if __name__ == "__main__":
    unittest.main()

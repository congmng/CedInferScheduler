"""P/D handoff cost model: producer egress cap + Decode staging budget.

Both numbers are measured on the real deployment, and both changed the
simulator's behaviour materially on 2026-09-15:

* the producer ceiling (115-314 MB/s, 184 MB per 1250-token handoff in
  585-731 ms) is 8x below the 2 GB/s wire, so charging the wire understated a
  cross-domain move by an order of magnitude;
* a Decode staging buffer (LMCache's 32 GiB PD buffer) stalls the sender when
  it fills -- without a model for it there is no way to express the multi-second
  handoff glitches the deployment shows.
"""

import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml                                                          # noqa: E402
from serving.core import config_builder                              # noqa: E402


class HandoffBandwidthTests(unittest.TestCase):
    def test_wire_value_is_kept_when_no_producer_ceiling_is_declared(self):
        self.assertEqual(config_builder._handoff_link_bandwidth(2.0, None), 2.0)

    def test_producer_ceiling_caps_the_handoff_path(self):
        # 0.26 GB/s = 260 MB/s, the deployment's measured NixlConnector push.
        self.assertAlmostEqual(
            config_builder._handoff_link_bandwidth(2.0, 0.26), 0.26)

    def test_a_fast_producer_does_not_raise_the_wire_rate(self):
        self.assertAlmostEqual(
            config_builder._handoff_link_bandwidth(0.88, 2.0), 0.88)

    def test_the_cap_reaches_the_generated_network_config(self):
        """The outermost dimension is the node boundary: that is the handoff."""
        instances = []
        for node in range(3):
            for slot, role in enumerate(("prefill", "decode")):
                instances.append({"instance_id": node * 2 + slot,
                                  "node_id": node, "pd_type": role,
                                  "num_npus": 1, "tp_size": 1, "pp_size": 1,
                                  "ep_size": 1})
        layout = config_builder._compute_domain_layout(instances, 3)
        handoff = config_builder._handoff_link_bandwidth(2.0, 0.26)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "network.yml"
            config_builder._create_network_config(
                str(path), instances, handoff, 300000.0,
                domain_layout=layout, intra_node_link=(50.0, 1000.0))
            data = yaml.safe_load(path.read_text())
        self.assertAlmostEqual(data["bandwidth"][-1], 0.26)
        self.assertGreater(data["bandwidth"][0], data["bandwidth"][-1],
                           "intra-node links must stay faster than the handoff")


class ConfigDeclaresTheKnobsTests(unittest.TestCase):
    """The deployment configs must state the regime they model."""

    def load(self, name):
        import json
        path = REPO / "configs" / "cluster" / name
        return json.loads(path.read_text())

    def test_short_prompt_config_keeps_the_wire_rate(self):
        config = self.load("casr_real_qwen3_8b_three_domain_aligned.json")
        # 254-token handoffs measure 40-90 ms, so the producer ceiling does not
        # bind there and the intranet rate is the right charge.
        self.assertEqual(config.get("kv_egress_gbps"), 2.0)
        self.assertIn("_handoff_comment", config)

    def test_kv_heavy_config_declares_the_measured_producer_ceiling(self):
        config = self.load("casr_real_qwen3_8b_three_domain_kvheavy.json")
        self.assertAlmostEqual(config.get("kv_egress_gbps"), 0.26)
        self.assertGreaterEqual(config.get("pd_buffer_bytes", 0), 2 ** 30)


if __name__ == "__main__":
    unittest.main()

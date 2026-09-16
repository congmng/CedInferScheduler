"""Domain-aware P/D handoff: topology, pricing, and the converter's pairing.

Before this change the simulator had no way to tell a same-node KV handoff from
a cross-node one: ``_create_network_config`` emitted a single flat
``[tp, instances]`` topology with one link, so ``p5090 -> d5090`` (same host)
and ``p4090 -> d5090`` (cross host) cost the same.  Worse, the router never
picked the Decode half of the pair, so the Prefill's per-layer KV egress went
to its own adjacent NPU whatever the plan said and the cross-domain link was
never charged at all.  See ``docs/模拟器与真机一致性核查.md`` 附七.
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from serving.core import config_builder                                  # noqa: E402
from serving.core.config_builder import (                                # noqa: E402
    _compute_domain_layout, _compute_network_dims, _create_network_config,
)
from serving.core.router import Router                                   # noqa: E402


def three_domain_instances():
    """Two instances per node, mirroring the real 3x(1P+1D) deployment."""
    instances = []
    for node in range(3):
        instances.append({"instance_id": node * 2, "node_id": node,
                          "pd_type": "prefill", "num_npus": 1, "tp_size": 1,
                          "pp_size": 1, "ep_size": 1})
        instances.append({"instance_id": node * 2 + 1, "node_id": node,
                          "pd_type": "decode", "num_npus": 1, "tp_size": 1,
                          "pp_size": 1, "ep_size": 1})
    return instances


def write_cluster_config(cluster, name):
    """Write a throwaway cluster config where the simulator would find it."""
    path = REPO / "configs" / "cluster" / name
    path.write_text(json.dumps(cluster))
    return path


def build_cluster_from(cluster, name):
    """Call ``build_cluster_config`` the way ``python -m serving`` does.

    The simulator chdirs into ``astra-sim/`` and the builder prefixes ``../``,
    so the cluster path has to be repo-relative and the call has to run from
    the same directory.
    """
    generated = write_cluster_config(cluster, name)
    cwd = os.getcwd()
    try:
        os.chdir(REPO / "astra-sim")
        return config_builder.build_cluster_config(
            str(REPO / "astra-sim"), f"configs/cluster/{name}",
            inputs_root=str(REPO / "tmp__pd_handoff_test_inputs"))
    finally:
        os.chdir(cwd)
        generated.unlink(missing_ok=True)
        shutil.rmtree(REPO / "tmp__pd_handoff_test_inputs", ignore_errors=True)


def minimal_node(pd_type):
    return {
        "num_instances": 1,
        "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
        "instances": [
            {"model_name": "Qwen/Qwen3-8B", "hardware": "RTX5090",
             "pd_type": pd_type, "num_npus": 1,
             "npu_mem": {"mem_size": 32, "mem_bw": 1792, "mem_latency": 0}},
        ],
    }


class DomainLayoutTests(unittest.TestCase):
    def test_flat_topology_is_unchanged_without_a_domain_split(self):
        self.assertEqual(_compute_network_dims(three_domain_instances()), [1, 9])

    def test_uniform_nodes_split_the_instance_dimension(self):
        layout = _compute_domain_layout(three_domain_instances(), 3)
        self.assertEqual(layout["inner_dims"], [1])
        self.assertEqual(layout["slots_per_node"], 3)
        self.assertEqual(layout["num_nodes"], 3)
        self.assertEqual(_compute_network_dims(three_domain_instances(), layout),
                         [1, 3, 3])

    def test_single_node_needs_no_domain_dimension(self):
        instances = [i for i in three_domain_instances() if i["node_id"] == 0]
        self.assertIsNone(_compute_domain_layout(instances, 1))

    def test_unequal_nodes_fall_back(self):
        instances = three_domain_instances()
        instances.append({"instance_id": 6, "node_id": 0, "pd_type": "decode",
                          "num_npus": 1, "tp_size": 1, "pp_size": 1, "ep_size": 1})
        self.assertIsNone(_compute_domain_layout(instances, 3))

    def test_address_math_charges_the_node_link_across_nodes(self):
        """Pin the multi-dimensional semantics the C++ side implements.

        Dimension 0 is innermost, the last one outermost.  A point-to-point
        send is carried by the outermost dimension in which the two addresses
        differ, so a same-node pair pays dimension 1 (intra) and a cross-node
        pair pays dimension 2 (inter) even though their local rank differs too.
        """
        dims = _compute_network_dims(three_domain_instances(),
                                     _compute_domain_layout(three_domain_instances(), 3))

        def address(npu_id):
            # Mirrors MultiDimTopology::translate_address: dims[0] is the
            # innermost (fastest varying) index.
            leftover = npu_id
            denominator = 1
            for size in dims:
                denominator *= size
            out = [0] * len(dims)
            for dim in range(len(dims) - 1, -1, -1):
                denominator //= dims[dim]
                out[dim] = leftover // denominator
                leftover %= denominator
            return out

        def dim_to_transfer(src, dst):
            src_addr, dst_addr = address(src), address(dst)
            for dim in range(len(dims) - 1, -1, -1):
                if src_addr[dim] != dst_addr[dim]:
                    return dim
            raise AssertionError("same address")

        # node 2 owns p5090 (NPU 6, sender slot 7) and d5090 (NPU 8).
        self.assertEqual(address(6), [0, 0, 2])
        self.assertEqual(address(8), [0, 2, 2])
        self.assertEqual(dim_to_transfer(6, 8), 1)   # same node -> intra link
        # node 1 owns p4090 (NPU 3, sender slot 4); d5090 is on node 2.
        self.assertEqual(dim_to_transfer(3, 8), 2)   # cross node -> inter link

    def test_network_config_carries_both_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "network.yml"
            instances = three_domain_instances()
            layout = _compute_domain_layout(instances, 3)
            _create_network_config(path, instances, 2.0, 300000.0,
                                   domain_layout=layout,
                                   intra_node_link=(50.0, 1000.0))
            text = path.read_text()
        self.assertIn("npus_count: [1, 3, 3]", text)
        self.assertIn("bandwidth: [50.0, 50.0, 2.0]", text)
        self.assertIn("latency: [1000.0, 1000.0, 300000.0]", text)

    def test_non_uniform_layout_with_intra_link_is_rejected(self):
        cluster = {
            "num_nodes": 2, "link_bw": 2.0, "link_latency": 300000.0,
            "intra_node_link_bw": 50.0, "intra_node_link_latency": 1000.0,
            "nodes": [minimal_node("prefill"), minimal_node("decode"),
                      minimal_node("decode")],
        }
        with self.assertRaises(ValueError):
            build_cluster_from(cluster, "_tmp_test_nonuniform.json")

    def test_intra_link_keys_must_come_as_a_pair(self):
        cluster = {"num_nodes": 1, "link_bw": 2.0, "link_latency": 3e5,
                   "intra_node_link_bw": 50.0, "nodes": [minimal_node("prefill")]}
        with self.assertRaises(KeyError):
            build_cluster_from(cluster, "_tmp_test_intrapair.json")

    def test_uniform_nodes_build_a_domain_aware_cluster(self):
        cluster = {
            "num_nodes": 2, "link_bw": 2.0, "link_latency": 300000.0,
            "intra_node_link_bw": 50.0, "intra_node_link_latency": 1000.0,
            "nodes": [
                {"num_instances": 2,
                 "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
                 "instances": [minimal_node("prefill")["instances"][0],
                               minimal_node("decode")["instances"][0]]},
                {"num_instances": 2,
                 "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
                 "instances": [minimal_node("prefill")["instances"][0],
                               minimal_node("decode")["instances"][0]]},
            ],
        }
        built = build_cluster_from(cluster, "_tmp_test_uniform.json")
        self.assertEqual(built["domain_layout"]["num_nodes"], 2)
        # Two slots per node: the Prefill owns a compute NPU and the KV-sender
        # NPU, the Decode owns one.
        self.assertEqual(built["domain_layout"]["slots_per_node"], 3)

    def test_without_intra_link_the_topology_stays_flat(self):
        cluster = {
            "num_nodes": 2, "link_bw": 2.0, "link_latency": 300000.0,
            "nodes": [
                {"num_instances": 2,
                 "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
                 "instances": [minimal_node("prefill")["instances"][0],
                               minimal_node("decode")["instances"][0]]},
                {"num_instances": 2,
                 "cpu_mem": {"mem_size": 128, "mem_bw": 256, "mem_latency": 0},
                 "instances": [minimal_node("prefill")["instances"][0],
                               minimal_node("decode")["instances"][0]]},
            ],
        }
        built = build_cluster_from(cluster, "_tmp_test_flat.json")
        self.assertIsNone(built["domain_layout"])


class FakeSched:
    def __init__(self, instance_id):
        self.instance_id = instance_id

    @property
    def accepts_new_requests(self):
        return True


class DecodeAssignmentTests(unittest.TestCase):
    def test_arriving_request_is_paired_with_a_decode_instance(self):
        router = Router.__new__(Router)
        router.decode_schedulers = [FakeSched(1), FakeSched(3)]
        router.affinity_plan = None
        router._select_instance = lambda eligible, role: 1
        self.assertEqual(router._decode_instance_id_for(object(), 0), 3)

    def test_no_decode_instances_leaves_the_pair_unset(self):
        router = Router.__new__(Router)
        router.decode_schedulers = []
        router.affinity_plan = None
        self.assertIsNone(router._decode_instance_id_for(object(), 0))

    def test_policy_is_used_when_the_plan_has_nothing_to_say(self):
        router = Router.__new__(Router)
        router.decode_schedulers = [FakeSched(1), FakeSched(3)]
        router.affinity_plan = None
        calls = []
        router._select_instance = lambda eligible, role: calls.append(role) or 0
        self.assertEqual(router._decode_instance_id_for(object(), 0), 1)
        self.assertEqual(calls, ["decode"])


def _chakra_available():
    try:
        import chakra.src.converter.llm_converter  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_chakra_available(), "chakra converter is not importable")
class ConverterPairingTests(unittest.TestCase):
    """The receiver graph must name the same NPU the send is addressed to."""

    def test_final_output_send_targets_the_selected_decode_npu(self):
        import chakra.src.converter.llm_converter as lc
        from chakra.src.third_party.utils.protolib import decodeMessage

        header = ("PREFILL\t\tmodel_parallel_NPU_group: 1\t\tpp_stage_boundaries:"
                  " 0\t\tpd_decode_npu_offset: 5")
        rows = []
        for _layer in range(2):
            for name in ("qkv_proj", "attn", "o_proj"):
                rows.append([name, "100", "LOCAL", "16", "LOCAL", "16",
                             "LOCAL", "16", "NONE", "2048", "NONE"])
        with tempfile.TemporaryDirectory() as tmp:
            converter = lc.LLMConverter("", str(pathlib.Path(tmp) / "llm"),
                                        num_npus=1, npu_offset=3)
            converter.convert_rows(header, rows)
            send_dsts, recv_dsts = set(), set()
            for npu in (3, 5):
                with open(f"{tmp}/llm.{npu}.et", "rb") as handle:
                    decodeMessage(handle, lc.GlobalMetadata())
                    node = lc.Node()
                    while decodeMessage(handle, node):
                        attrs = {a.name: a for a in node.attr}
                        if node.type == lc.COMM_SEND_NODE:
                            send_dsts.add(attrs["comm_dst"].int32_val)
                        elif node.type == lc.COMM_RECV_NODE:
                            recv_dsts.add(attrs["comm_dst"].int32_val)
                        node = lc.Node()
        # Every send on the Prefill NPU is addressed to the selected Decode NPU
        # -- including the final output hand-off, which used to keep the legacy
        # adjacent rank and deadlocked both sides.
        self.assertEqual(send_dsts, {5})
        self.assertEqual(recv_dsts, {5})


if __name__ == "__main__":
    unittest.main()

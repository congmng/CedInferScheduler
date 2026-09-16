"""The generated simulator topology must stay in step with the deployment.

Hand-written ``casr_real_*`` configs drifted from the fabric they claimed to
mirror (the "aligned" three-domain config models six RTX3090 instances while
the deployment is a 5090/3090/4090 mix).  The layout, links and budgets are now
derived by ``tests/gen_sim_config_from_real.py``; these tests fail when the
checked-in file no longer matches what the deployment config produces.
"""

import json
import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import gen_sim_config_from_real as gen                            # noqa: E402

GENERATED = REPO / "configs" / "cluster" / "casr_real_qwen3_8b_generated.json"
KVHEAVY = REPO / "configs" / "cluster" / "casr_real_qwen3_8b_generated_kvheavy.json"


class GeneratedConfigTests(unittest.TestCase):
    def test_checked_in_config_matches_the_deployment(self):
        fresh = json.dumps(gen.build(), ensure_ascii=False, indent=2) + "\n"
        self.assertEqual(
            GENERATED.read_text(encoding="utf-8"), fresh,
            "regenerate with: python3 tests/gen_sim_config_from_real.py "
            f"--out {GENERATED.relative_to(REPO)}")

    def test_kv_heavy_variant_differs_only_in_the_handoff_charge(self):
        default = gen.build()
        heavy = gen.build(kv_heavy=True)
        self.assertEqual(default["nodes"], heavy["nodes"])
        self.assertEqual(default["link_bw"], heavy["link_bw"])
        self.assertAlmostEqual(heavy["kv_egress_gbps"], 0.26)
        self.assertNotEqual(default["kv_egress_gbps"], heavy["kv_egress_gbps"])

    def test_layout_is_uniform_so_the_domain_model_applies(self):
        config = gen.build()
        slots = {node["num_instances"] for node in config["nodes"]}
        self.assertEqual(len(slots), 1,
                         "a non-uniform layout silently disables the intra-node link")
        self.assertIn("intra_node_link_bw", config)
        # What was dropped has to be stated, not implied.
        self.assertIn("_excluded_for_uniformity", config)
        self.assertTrue(any("RTX3090" in item
                            for item in config["_excluded_for_uniformity"]))

    def test_every_deployment_domain_now_has_a_profile_bundle(self):
        """The A100 gap closed on 2026-09-15.

        ``profiler/perf/A100/Qwen/Qwen3-8B/bf16`` was missing (its GPUs were
        busy with the owner's training jobs), which made the generated config
        either name a domain the simulator cannot load or silently drop it.
        The bundle now exists, so no domain may be skipped for that reason.
        """
        config = gen.build()
        self.assertNotIn("_skipped_domains", config)
        self.assertTrue(gen.has_profile("A100"))
        hardware = {instance["hardware"]
                    for node in config["nodes"] for instance in node["instances"]}
        self.assertIn("A100", hardware)

    def test_links_come_from_the_deployment_not_from_a_guess(self):
        config = gen.build()
        # 0.88 Gbps standard cross-domain link / 8 = 0.11 GB/s.
        self.assertAlmostEqual(config["link_bw"], 0.11, places=3)
        # Measured same-host handoff: 585 ms per 1000 prompt tokens, and one
        # token is 147 456 B of KV -> 252 MB/s (the deployment record's table
        # says 257 MB/s for this pair).  0.3 was the 1250-token KV size divided
        # by the 1000-token time, i.e. 17% too fast.
        self.assertAlmostEqual(config["intra_node_link_bw"], 0.257, places=3)

    def test_service_times_are_remapped_and_cover_every_instance(self):
        """Per-instance compute cost has to follow the deployment's numbering.

        The simulator's config inherited the *template's* ``service_ms`` maps
        verbatim, so every instance was priced with another domain's number
        (measured 2026-09-16: the 4-domain config gave sim instance 0 -- the
        5090 -- the 3090a's 68.0 ms, and the A100/4090 instances had no entry
        at all).  The real control loop seeds the solver from each instance's
        measured ``service_ms``, so the generator has to as well.
        """
        real = json.loads(gen.REAL.read_text(encoding="utf-8"))
        config, maps = gen.build(return_maps=True)
        expected = {}
        for key, role in (("prefills", "prefill"), ("decodes", "decode")):
            for item in real[key]:
                source_id = int(item["instance_id"])
                if item.get("service_ms") and source_id in maps[role]:
                    expected[str(maps[role][source_id])] = float(item["service_ms"])
        got = {**{k: float(v) for k, v in config["casr"]["prefill_service_ms"].items()},
               **{k: float(v) for k, v in config["casr"]["decode_service_ms"].items()}}
        self.assertEqual(got, expected)
        self.assertEqual(len(got), sum(node["num_instances"] for node in config["nodes"]),
                         "every modelled instance needs its own service time")

    def test_kv_heavy_file_is_generated_too(self):
        heavy = json.loads(KVHEAVY.read_text(encoding="utf-8"))
        self.assertAlmostEqual(heavy["kv_egress_gbps"], 0.26)

    def test_pair_costs_follow_the_deployment(self):
        """The LP needs a per-pair price; the real control loop always supplies one.

        Without ``pair_costs`` the simulator's ``_pair_cost`` sees rtt=0 and
        bandwidth=0 for every pair and prices the network with
        ``|NPU index distance| * 0.001`` -- an arbitrary proxy that made the
        cheapest edge depend on the order instances happen to be listed in
        (measured 2026-09-16: the generated config had no pair costs at all,
        while ``casr_control.py`` builds them from the deployment's measured
        links: same-domain free, cross-domain = RTT, capacity elsewhere).
        """
        real = json.loads(gen.REAL.read_text(encoding="utf-8"))
        config, maps = gen.build(return_maps=True)
        pair_costs = config["casr"]["pair_costs"]
        self.assertTrue(pair_costs, "the LP has no per-pair network price")
        domain = {}
        for key in ("prefills", "decodes"):
            for item in real[key]:
                domain[int(item["instance_id"])] = item["domain"]
        rtt = {(link["src"], link["dst"]): float(link["rtt_ms"])
               for link in real["links"]}
        for key, value in pair_costs.items():
            p_id, d_id = (int(part) for part in key.split(","))
            source = next(k for k, v in maps["prefill"].items() if v == p_id)
            target = next(k for k, v in maps["decode"].items() if v == d_id)
            if domain[source] == domain[target]:
                self.assertEqual(value["rtt_ms"], 0.0, "a same-host handoff is free")
            else:
                self.assertAlmostEqual(
                    value["rtt_ms"], rtt[(domain[source], domain[target])], places=3)
            self.assertEqual(value["bandwidth_bytes_per_s"], 0.0,
                             "capacity belongs to shared_links, not to the pair")


if __name__ == "__main__":
    unittest.main()

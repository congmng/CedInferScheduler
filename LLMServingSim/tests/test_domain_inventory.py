"""Ray domain-inventory checks for the heterogeneous experiment control plane."""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
RAY_DIR = REPO / "deploy" / "ray"
if str(RAY_DIR) not in sys.path:
    sys.path.insert(0, str(RAY_DIR))

from domain_inventory import node_labels, node_state  # noqa: E402


class DomainInventoryTests(unittest.TestCase):
    def test_dead_registration_keeps_labels_but_is_not_alive(self):
        node = {
            "node_ip": "10.212.67.68",
            "state": "DEAD",
            "state_message": "health check failed due to missing too many heartbeats",
            "resources_total": {
                "node:10.212.67.68": 1.0,
                "domain:3090": 1.0,
                "gpu_type:RTX3090": 1.0,
                "CPU": 35.0,
            },
        }
        state, alive = node_state(node)
        self.assertEqual(state, "DEAD")
        self.assertFalse(alive)
        # The stale entry still carries the old label; callers must filter it.
        self.assertEqual(node_labels(node), (["3090"], ["RTX3090"]))

    def test_alive_node_exposes_domain_and_gpu_type(self):
        node = {
            "node_ip": "10.212.67.68",
            "state": "ALIVE",
            "resources_total": {"domain:3090a": 1.0, "gpu_type:RTX3090": 1.0},
        }
        self.assertEqual(node_state(node), ("ALIVE", True))
        self.assertEqual(node_labels(node), (["3090a"], ["RTX3090"]))

    def test_missing_state_defaults_to_alive(self):
        self.assertEqual(node_state({"node_ip": "x"}), ("ALIVE", True))


if __name__ == "__main__":
    unittest.main()

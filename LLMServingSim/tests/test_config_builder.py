import unittest

from serving.core.config_builder import _collective_prefix_dims, _compute_network_dims


class ConfigBuilderTopologyTests(unittest.TestCase):
    def test_mixed_tp_uses_exact_factorized_topology(self):
        instances = [
            {"tp_size": 2, "pp_size": 1, "num_npus": 2, "pd_type": "prefill"},
            {"tp_size": 4, "pp_size": 1, "num_npus": 4, "pd_type": "decode"},
        ]
        dims = _compute_network_dims(instances)
        self.assertEqual(dims, [2, 2, 2])
        self.assertEqual(_collective_prefix_dims(2, dims), [True, False, False])
        self.assertEqual(_collective_prefix_dims(4, dims), [True, True, False])

    def test_mixed_tp_rejects_unrepresentable_npu_count(self):
        instances = [
            {"tp_size": 2, "pp_size": 1, "num_npus": 2, "pd_type": "prefill"},
            {"tp_size": 4, "pp_size": 1, "num_npus": 4, "pd_type": "decode"},
            {"tp_size": 2, "pp_size": 1, "num_npus": 2, "pd_type": "decode"},
        ]
        with self.assertRaises(ValueError):
            _compute_network_dims(instances)


if __name__ == "__main__":
    unittest.main()

"""The controller must price the *offered* load, not the served rate.

The profiler only sees requests the router already admitted, so under
back-pressure the observed arrival rate collapses to the served rate and every
``+P`` counterfactual looks worthless.  Measured 2026-09-16 on the small-cluster
elasticity A/B: the simulator reported 108 req/s of capacity against 9 req/s of
demand while the run sat 23 s deep in backlog, and the "elastic" arm came out
bit-identical to the arm with structural actions disabled.  The real controller
recovers the missing rate from the Prefills' waiting-queue growth
(``casr_control._sample_backlog``); these tests pin the same behaviour here.
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.casr.controller import CASRController                   # noqa: E402


class _Sched:
    def __init__(self, waiting=0):
        self.waiting = [None] * waiting


def controller(alpha=0.2, multiple=4.0):
    return CASRController(1_000_000,
                          policy={"ewma_alpha": alpha, "backlog_rps_multiple": multiple})


class BacklogSamplingTests(unittest.TestCase):
    def test_queue_growth_becomes_demand(self):
        ctl = controller()
        ctl._sample_backlog(0, [_Sched(0)])
        self.assertEqual(ctl._backlog_rps, 0.0)
        ctl._sample_backlog(int(10e9), [_Sched(20)])
        # 20 more queued requests over 10 s = 2 req/s of unserved offered load,
        # smoothed by the same EWMA the real controller applies (alpha 0.2).
        self.assertAlmostEqual(ctl._backlog_rps, 0.2 * 2.0, places=6)

    def test_a_draining_queue_does_not_subtract_demand(self):
        ctl = controller()
        ctl._sample_backlog(0, [_Sched(30)])
        ctl._sample_backlog(int(10e9), [_Sched(10)])
        self.assertEqual(ctl._backlog_rps, 0.0)

    def test_the_term_decays_once_the_queue_stops_growing(self):
        ctl = controller(alpha=0.5)
        ctl._sample_backlog(0, [_Sched(0)])
        ctl._sample_backlog(int(10e9), [_Sched(20)])       # growth 2.0 -> 1.0
        self.assertAlmostEqual(ctl._backlog_rps, 1.0, places=6)
        ctl._sample_backlog(int(20e9), [_Sched(20)])       # growth 0 -> decays
        self.assertAlmostEqual(ctl._backlog_rps, 0.5, places=6)


class DemandInflationTests(unittest.TestCase):
    def rows(self):
        return [{"class_id": "a", "arrival_rate_ewma": 2.0},
                {"class_id": "b", "arrival_rate_ewma": 1.0}]

    def test_every_class_scales_by_the_same_factor(self):
        ctl = controller()
        ctl._backlog_rps = 1.5
        rows = ctl._inflate_demand(self.rows())
        # observed 3.0 + 1.5 backlog -> 1.5x on every class.
        self.assertAlmostEqual(sum(r["arrival_rate_ewma"] for r in rows), 4.5, places=6)
        self.assertAlmostEqual(rows[0]["arrival_rate_ewma"], 3.0, places=6)
        self.assertAlmostEqual(rows[1]["arrival_rate_ewma"], 1.5, places=6)

    def test_the_inflation_is_capped(self):
        ctl = controller(multiple=2.0)
        ctl._backlog_rps = 100.0
        rows = ctl._inflate_demand(self.rows())
        self.assertAlmostEqual(sum(r["arrival_rate_ewma"] for r in rows), 9.0,
                               places=6, msg="cap = observed x multiple")

    def test_no_backlog_leaves_the_rows_alone(self):
        ctl = controller()
        rows = ctl._inflate_demand(self.rows())
        self.assertEqual([r["arrival_rate_ewma"] for r in rows], [2.0, 1.0])

    def test_no_observed_demand_cannot_be_inflated(self):
        ctl = controller()
        ctl._backlog_rps = 5.0
        rows = ctl._inflate_demand([{"class_id": "a", "arrival_rate_ewma": 0.0}])
        self.assertEqual(rows[0]["arrival_rate_ewma"], 0.0)


if __name__ == "__main__":
    unittest.main()

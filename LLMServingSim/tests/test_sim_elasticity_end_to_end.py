"""Gated end-to-end acceptance test for structural elasticity.

``tests/run_casr_elasticity_long.py`` runs two simulator arms (static pool vs
elastic pool) on a sustained-overload trace and requires the elastic arm to
win -- the criterion that the 2026-09-15 plumbing fixes made true after the
simulator had matched the static pool millisecond for millisecond.

It is gated behind ``SIM_ELASTICITY_E2E=1`` because it drives two full
ASTRA-Sim replays (minutes of wall clock), which does not belong in the default
unit-test loop.  Run it in the acceptance sweep:

    SIM_ELASTICITY_E2E=1 python3 -m pytest tests/test_sim_elasticity_end_to_end.py -q
"""

import os
import pathlib
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPO / "tests" / "run_casr_elasticity_long.py"


@unittest.skipUnless(os.environ.get("SIM_ELASTICITY_E2E") == "1",
                     "set SIM_ELASTICITY_E2E=1 to run the two-arm acceptance replay")
class ElasticityAcceptanceTests(unittest.TestCase):
    def test_elastic_pool_beats_the_static_pool(self):
        proc = subprocess.run([sys.executable, str(RUNNER)],
                              cwd=REPO, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         f"runner failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
        self.assertIn("ELASTICITY ACCEPTANCE OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()

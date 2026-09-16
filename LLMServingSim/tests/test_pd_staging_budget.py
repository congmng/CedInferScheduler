"""The Decode's P/D staging budget must be charged once per *request*.

A Prefill pushes each request's KV into its Decode's handoff buffer, and a full
buffer stalls the sender -- that is the measured back-pressure.  The budget is
reserved when the request is admitted and returned when the Decode actually
takes the request over (``add_decode``).

Charging it per *batch* instead of per request leaks one ``pd_kv_bytes`` for
every extra prefill chunk: a 3750-token prompt over a 2048-token budget is
built twice but handed over once.  Measured 2026-09-17 on the pack-3 arena,
375 MB per request filled the 32 GB buffer after ~85 requests, admission
stopped, and the run sat at "Running 0, Waiting 104, 0.0 tokens/s" while the
simulated clock ran to 3973 s with 205 requests still queued -- a livelock that
made every long-prompt experiment silently useless.

These tests pin both halves: the charge is once per request however many chunks
it takes, and the release clears exactly that charge.
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from serving.core.request import Request                          # noqa: E402
from serving.core.scheduler import Scheduler                      # noqa: E402

MODEL = "Qwen/Qwen3-8B"
MB = 1024 * 1024
DECODE_ID = 3


def _scheduler(role, staging, instance_id, buffer_mb=4096):
    # 4 GB: a 4000-token Qwen3-8B handoff is ~590 MB (36 layers x 147 KB/token),
    # so the buffer has to be comfortably larger for admission to be the thing
    # under test rather than the gate.
    scheduler = Scheduler(
        model=MODEL, node_id=0, instance_id=instance_id, max_num_seqs=8,
        max_num_batched_tokens=2048, num_npus=1, tp_size=1, pp_size=1,
        npu_mem=80, cpu_mem=0, start_npu=0, pd_type=role, fp=16,
        block_size=16, req_num=1, enable_prefix_caching=True,
        enable_prefix_sharing=False, prefix_pool=None, prefix_storage=None,
        enable_chunked_prefill=True,
        pd_buffer_bytes=buffer_mb * MB, pd_staging=staging,
    )
    # What serving/__main__.py wires up for the P/D pairing: the Prefill needs
    # the Decode's NPU offset to build a receiver-homogeneous batch.
    scheduler.decode_npu_offsets = {DECODE_ID: 1}
    scheduler.decode_npu_counts = {DECODE_ID: 1}
    return scheduler


def _request(req_id, tokens):
    req = Request(req_id, MODEL, tokens, 16, 0, 0,
                  input_hash_ids=list(range(tokens)),
                  output_hash_ids=list(range(16)))
    req.decode_instance_id = DECODE_ID
    return req


class StagingBudgetTests(unittest.TestCase):
    def test_a_chunked_prefill_charges_the_budget_once(self):
        staging = {}
        prefill = _scheduler("prefill", staging, instance_id=0)
        req = _request(0, 4000)                 # two chunks at a 2048 budget
        prefill.waiting.append(req)

        prefill._schedule_waiting(0, [], 2048, pd_target=DECODE_ID)
        charged = staging[DECODE_ID]
        self.assertAlmostEqual(charged, prefill.memory.pd_kv_bytes(4000))

        # Second chunk: schedule the running request, then build its batch.
        # The batch is where the budget used to be charged again.
        scheduled = []
        prefill._schedule_running(scheduled, [], 2048, DECODE_ID)
        self.assertTrue(scheduled, "the rest of the prompt must be scheduled")
        prefill._build_batch(1, 0, scheduled)
        self.assertAlmostEqual(
            staging[DECODE_ID], charged,
            msg="the second chunk must not reserve the Decode's buffer again")

    def test_the_decode_releases_the_charge_it_was_given(self):
        staging = {}
        prefill = _scheduler("prefill", staging, instance_id=0)
        decode = _scheduler("decode", staging, instance_id=DECODE_ID)
        req = _request(0, 4000)
        prefill.waiting.append(req)
        prefill._schedule_waiting(0, [], 2048, pd_target=DECODE_ID)
        prefill._schedule_running([], [], 2048, DECODE_ID)
        self.assertGreater(staging[DECODE_ID], 0.0)

        self.assertTrue(decode.add_decode(req))
        self.assertEqual(staging.get(DECODE_ID, 0.0), 0.0,
                         "a completed handoff must return the whole reservation")

    def test_the_budget_still_stops_admission_when_it_is_genuinely_full(self):
        """The gate the fix must not weaken: a full buffer still back-pressures."""
        staging = {DECODE_ID: 512 * MB}
        prefill = _scheduler("prefill", staging, instance_id=0, buffer_mb=512)
        prefill.waiting.append(_request(0, 4000))
        prefill._schedule_waiting(0, [], 2048, pd_target=DECODE_ID)
        self.assertEqual(len(prefill.running), 0,
                         "a Decode with no room must not be sent more KV")


if __name__ == "__main__":
    unittest.main()

import bisect
import pandas as pd
from time import time
import csv
import os

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .kv_cache_manager import request_block_hashes
from .graph_generator import *
from .trace_generator import *
from .logger import print_markup, print_rule
from .pim_model import *
import numpy as np

# class that shedules request of astra-sim
class Scheduler:
    """vLLM V1-style continuous batching over a per-tier block pool.

    Shape follows ``vllm/v1/core/sched/scheduler.py`` (v0.19.0): a persistent
    ``running`` set is served first, preempting only from its own tail, then the
    ``waiting`` queue is admitted while budget and slots remain -- never by
    preempting. A step that preempted skips admission entirely, which is what
    keeps the running set from oscillating.

    There is one ``schedule()`` for prefix caching on and off. The pool handles
    ``enable_caching=False`` the way vLLM does (allocate through the same free
    list, never index), so the two separate schedulers this file used to carry
    had no reason to exist -- and their drifting apart was its own source of bugs.
    """

    def __init__(self, model, node_id, instance_id, max_num_seqs, max_num_batched_tokens,
                 num_npus, tp_size, pp_size, npu_mem, cpu_mem,
                 start_npu, pd_type, fp, block_size, req_num,
                 enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage,
                 enable_chunked_prefill=False,
                 long_prefill_token_threshold=0, cxl_mem=0, ep_size=1, kv_cache_dtype='auto',
                 npu_memory_utilization=1.0, reserve_full_isl=True, prefix_profiler=None):
        self.model = model
        self.config = get_config(model)
        self.node_id = node_id
        self.instance_id = instance_id
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_batched_tokens = min(max_num_batched_tokens, self.config['max_position_embeddings'])
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefix_storage = prefix_storage
        # vLLM's scheduler_reserve_full_isl, True by default there: admit a
        # request only if its whole sequence fits, not merely its first chunk.
        self.reserve_full_isl = reserve_full_isl
        self.prefix_profiler = prefix_profiler
        # CASR models scale actions as admission-state transitions before it
        # grows a real orchestration backend around worker processes.
        self.admission_state = "ACTIVE"
        self.resource_gpu_ids = ()
        self.resource_mem_gb = 0.0
        self.decode_npu_offsets = {}
        self.decode_npu_counts = {}

        # Requests admitted and still generating. Persistent across steps: this
        # is what gives the scheduler a notion of "already running", which the
        # old single re-derived pool did not have.
        self.running = []
        # Not yet admitted, sorted by (arrival, id). A preempted request is
        # prepended, as in vLLM's waiting.prepend_request().
        self.waiting = []
        self.inflight = []
        self.done = []
        self.batch_ids = -1

        # Tokens recomputed because a request was preempted, and how many
        # preemptions happened. Both are reported: in the prefix-caching-off mode
        # a large recompute count is expected and is what that mode costs, while
        # in the prefix-caching modes it should stay near zero.
        self.recompute_tokens = 0
        self.num_preemptions = 0

        self.memory = MemoryModel(model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem,
                                  block_size, fp, enable_prefix_caching, enable_prefix_sharing,
                                  prefix_pool, prefix_storage, cxl_mem, ep_size=ep_size,
                                  pp_size=pp_size, kv_cache_dtype=kv_cache_dtype,
                                  npu_memory_utilization=npu_memory_utilization)
        self.kv = self.memory.kv

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    # ==================== scheduling ====================

    def schedule(self, current, sys, batch_id=-1):
        if sys != self.start_npu:
            return self._schedule_existing(sys, batch_id)

        # The start NPU is a joiner too, and it has to try joining *before* the
        # pipeline-depth cap below. With DP groups any NPU of the instance may
        # open a round -- the idle-member dummy in particular -- and ``add_done``
        # only completes a batch once ``start_npu`` has run it, so a batch opened
        # by another NPU would otherwise never finish and the group's collective
        # would block forever. Gating the join on a *full* pipeline missed
        # exactly that case: a dummy opened by a non-start NPU leaves
        # ``pp_size - 1`` slots free, so the start NPU took the build path
        # instead, found nothing to build (its member is idle) and answered
        # "pass" forever. Joining never adds a batch, so it cannot breach the cap.
        #
        # For a non-DP instance every in-flight batch was built here, so
        # ``start_npu`` is already in ``fired``, this returns None, and the cap
        # below decides exactly as it did before.
        existing = self._schedule_existing(sys, batch_id)
        if existing is not None:
            return existing

        # One batch in flight per pipeline stage: vLLM's ``batch_queue``, whose
        # maxlen is ``max_concurrent_batches`` == ``pipeline_parallel_size``.
        if len(self.inflight) >= self.pp_size:
            return None

        token_budget = self.max_num_batched_tokens
        # (request, tokens scheduled, num_computed_tokens before this step)
        scheduled = []
        preempted = []

        pd_target = self._select_pd_target(current)
        token_budget = self._schedule_running(scheduled, preempted, token_budget, pd_target)
        # vLLM skips the whole waiting phase on any step that preempted
        # (`if not preempted_reqs:`). Without that the running set oscillates
        # preempt -> refill -> preempt.
        if not preempted:
            token_budget = self._schedule_waiting(current, scheduled, token_budget, pd_target)

        if not scheduled:
            return None
        return self._build_batch(current, sys, scheduled)

    def _select_pd_target(self, current):
        """Select one Decode target for this P batch.

        ASTRA-Sim traces have one receiver graph per P batch.  Keeping the
        batch target-homogeneous makes that physical dependency explicit while
        allowing different queued requests on the same P worker to choose
        different Decode workers over successive batches.
        """
        if self.pd_type != "prefill":
            return None
        if self.running:
            return self.running[0].decode_instance_id
        for request in self.waiting:
            if request.arrival <= current:
                return request.decode_instance_id
            break
        return None

    @staticmethod
    def _matches_pd_target(request, target):
        return request.decode_instance_id == target

    def _schedule_running(self, scheduled, preempted, token_budget, pd_target=None):
        """Phase A: serve requests already running, preempting from the tail."""
        i = 0
        while i < len(self.running) and token_budget > 0:
            req = self.running[i]
            if self.pd_type == "prefill" and not self._matches_pd_target(req, pd_target):
                i += 1
                continue
            num_new = self._num_new_tokens(req, token_budget)
            if num_new <= 0:
                # Nothing left to compute for this request yet. vLLM continues
                # rather than breaking here, so a later request is not blocked.
                # Legitimate only while another batch is in flight and about to
                # advance this request (pp_size > 1); otherwise nothing will ever
                # move it and the run cannot terminate, so say so loudly.
                if not any(req in b.requests for b in self.inflight):
                    raise RuntimeError(
                        f"[Scheduler] [node_id={self.node_id},inst={self.instance_id}] "
                        f"request {req.id} is running with nothing to schedule and no "
                        f"batch in flight: num_computed_tokens="
                        f"{req.num_computed_tokens}, num_tokens_reached="
                        f"{req.num_tokens_reached}, output={req.output}. This deadlocks "
                        f"the run -- num_tokens_reached was not advanced when a token "
                        f"was produced."
                    )
                i += 1
                continue

            blocks = None
            while True:
                blocks = self.kv.allocate_slots(req, num_new)
                if blocks is not None:
                    break
                # Preempt the lowest-priority running request. Under FCFS that
                # is the most recently admitted, i.e. the tail.
                victim_index = next((index for index in range(len(self.running) - 1, -1, -1)
                                     if self.pd_type != "prefill" or
                                     self._matches_pd_target(self.running[index], pd_target)), None)
                if victim_index is None:
                    break
                victim = self.running.pop(victim_index)
                self._preempt_request(victim)
                preempted.append(victim)
                if victim is req:
                    break

            if blocks is None:
                break

            scheduled.append((req, num_new, req.num_computed_tokens))
            token_budget -= num_new
            i += 1
        return token_budget

    def _schedule_waiting(self, current, scheduled, token_budget, pd_target=None):
        """Phase B: admit from the waiting queue. Never preempts to admit."""
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.max_num_seqs:
                break
            matching_index = next((index for index, candidate in enumerate(self.waiting)
                                   if candidate.arrival <= current and
                                   (self.pd_type != "prefill" or
                                    self._matches_pd_target(candidate, pd_target))), None)
            if matching_index is None:
                break
            req = self.waiting[matching_index]

            num_computed = req.num_computed_tokens
            hit_blocks, num_npu_hit, num_lower_hit = [], 0, 0
            if num_computed == 0:
                hit_blocks, num_npu_hit, num_lower_hit = self.kv.get_computed_blocks(req)
                req.npu_cache_hit = num_npu_hit
                req.storage_cache_hit = num_npu_hit + num_lower_hit
                req.prefix_cache_hit = req.storage_cache_hit
                if self.prefix_profiler is not None:
                    self.prefix_profiler.observe_lookup(
                        req, self.instance_id, current, num_npu_hit,
                        req.storage_cache_hit)
                num_computed = num_npu_hit + num_lower_hit

            num_new = req.num_tokens - num_computed
            threshold = self.long_prefill_token_threshold
            if 0 < threshold < num_new:
                num_new = threshold
            if not self.enable_chunked_prefill and num_new > token_budget:
                # Cannot split this prefill, and it does not fit. Stop here
                # rather than skipping ahead, to keep FCFS.
                break
            num_new = min(num_new, token_budget)
            if num_new <= 0:
                break

            if self.reserve_full_isl and not self.kv.can_fit_full_sequence(
                    req, hit_blocks, num_npu_hit, num_lower_hit):
                # Its first chunk would fit but the whole sequence would not, so
                # admitting it now only defers a preemption. vLLM breaks here.
                break

            blocks = self.kv.allocate_slots(req, num_new, hit_blocks,
                                            num_npu_hit, num_lower_hit)
            if blocks is None:
                # vLLM breaks here: a waiting request never causes a preemption.
                break

            self.waiting.pop(matching_index)
            if req.num_preemptions > 0:
                # Resuming. Whatever neither tier could return has to be
                # computed again; with no lower tier that is the whole sequence.
                self.recompute_tokens += max(0, req.num_tokens_reached - num_computed)
                self.logger.info("Resuming request #%d (%d of %d tokens recovered)",
                                 req.id, num_computed, req.num_tokens_reached)
            req.num_computed_tokens = num_computed
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            self.memory.record_prefix_stats(req)

            scheduled.append((req, num_new, num_computed))
            token_budget -= num_new
        return token_budget

    def _num_new_tokens(self, req, token_budget):
        """Tokens to schedule for ``req`` this step, vLLM's uniform rule.

        No prefill/decode branch: a request simply catches up to the length it
        has reached. In steady-state decode that yields 1; for a resumed request
        with ``num_computed_tokens`` reset to 0 it yields the whole sequence,
        chunked by the budget.
        """
        num_new = req.num_tokens - req.num_computed_tokens
        threshold = self.long_prefill_token_threshold
        if 0 < threshold < num_new:
            num_new = threshold
        return min(num_new, token_budget)

    def _preempt_request(self, req):
        """Give up a running request's blocks so someone else can use them.

        vLLM verbatim, including resetting ``num_computed_tokens``: that is not
        "throw it away and re-prefill", it means "forget where you were and
        re-derive it from the caches". ``free_blocks`` keeps the blocks' hashes,
        so on re-admission ``get_computed_blocks`` finds whatever is still
        resident, a lower tier returns what was written down, and only the
        remainder is recomputed. Nothing here needs a special "preserve the
        decode state" path -- the tiers are what preserve it.
        """
        self.kv.preempt(req)
        req.status = RequestStatus.PREEMPTED
        req.num_computed_tokens = 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        # vLLM prepends, so a preempted request is first in line to come back.
        self.waiting.insert(0, req)
        self.logger.info("Preemption of the request #%d (count %d)", req.id, req.num_preemptions)

    def _build_batch(self, current, sys, scheduled):
        """Assemble the Batch the trace generator consumes.

        Prefill-vs-decode is decided by the *scheduled token count*, not by any
        request phase flag: >1 token is a chunk, exactly 1 is a decode. That is
        what the attention profile axes want (prefill_chunk / kv_prefill vs
        n_decode / kv_decode) and how the varlen kernel sees the batch anyway. It
        is also the only classification that survives a resumed request, whose
        recomputation must be traced as a chunk even though it is past its
        original prompt length.
        """
        total_len = 0
        kv_len = 0
        num_prefill = 0
        num_decode = 0
        q_list = []
        k_list = []
        prefill_q_list = []
        prefill_k_list = []
        decode_k_list = []
        scheduled_tokens = {}
        pd_kv_send_tokens = 0

        for req, num_new, computed_before in scheduled:
            scheduled_tokens[req.id] = num_new
            total_len += num_new
            q_list.append(num_new)
            k_list.append(computed_before)
            if num_new > 1:
                num_prefill += 1
                prefill_q_list.append(num_new)
                prefill_k_list.append(computed_before)
            else:
                num_decode += 1
                kv_len += computed_before
                decode_k_list.append(computed_before)
            if req.is_init:
                req.set_que_delay(current)
            if self.pd_type == "prefill":
                # The paired decode instance needs this chunk's KV, plus the KV
                # of any prefix-cache hit -- it was never computed here, but the
                # decode side still needs it.
                pd_kv_send_tokens += num_new
                if computed_before == 0:
                    pd_kv_send_tokens += req.prefix_cache_hit

            # vLLM advances num_computed_tokens at schedule time
            # (_update_after_schedule), not at completion. With pp_size > 1 two
            # batches can be in flight, and advancing late would let the same
            # tokens be scheduled twice.
            req.num_computed_tokens = computed_before + num_new

        recall_bytes, write_through_bytes = self.kv.take_traffic()

        batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list,
                      num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list,
                      current, self.kv.npu_used_bytes(), 0, recall_bytes,
                      pd_kv_send_tokens=pd_kv_send_tokens)
        batch.fired.append(sys)
        batch.requests.extend(req for req, _, _ in scheduled)
        if self.pd_type == "prefill":
            targets = {req.decode_instance_id for req, _, _ in scheduled
                       if req.decode_instance_id is not None}
            if len(targets) > 1:
                raise RuntimeError(
                    "A Prefill batch contains multiple Decode targets. "
                    "CASR must form target-homogeneous P batches before graph generation.")
            if targets:
                target = targets.pop()
                batch.pd_decode_npu_offset = self.decode_npu_offsets.get(target)
                if batch.pd_decode_npu_offset is None:
                    raise RuntimeError(f"Unknown Decode instance {target} for P/D handoff")
                batch.pd_decode_npu_count = self.decode_npu_counts.get(target, 0)
                if batch.pd_decode_npu_count != self.num_npus:
                    raise RuntimeError(
                        "CASR P/D receiver graphs require equal P and Decode NPU counts; "
                        f"P has {self.num_npus}, Decode instance {target} has "
                        f"{batch.pd_decode_npu_count}.")
        batch.scheduled_tokens = scheduled_tokens
        # Written down to a victim tier off the critical path, so it carries no
        # latency -- but the bytes still cost DRAM energy.
        batch.write_through = write_through_bytes
        self.inflight.append(batch)
        self.logger.info("Scheduling new batch #%d to NPU[%d]", batch.batch_id, sys)
        return batch

    def _schedule_existing(self, sys, batch_id):
        """Hand an already-formed batch to the next NPU of the instance."""
        for batch in self.inflight:
            if batch.batch_id == batch_id:
                if sys in batch.fired:
                    return None
                batch.fired.append(sys)
                self.logger.info("Scheduling existing batch #%d to NPU[%d]", batch.batch_id, sys)
                return batch
        return None

    # ==================== completion ====================

    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs

        batch = None
        idx = 0
        id -= 1
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        if batch is None or sys in batch.end:
            return prompt_t, gen_t, end_reqs

        batch.end.append(sys)
        # A prefill instance also waits for its paired decode NPUs, which
        # receive the streamed KV.
        last_npu = self.num_npus - 1
        if self.pd_type == "prefill" and batch.pd_decode_npu_offset is not None:
            completion_npu = batch.pd_decode_npu_offset + last_npu
        else:
            completion_npu = self.start_npu + (self.num_npus * (2 if self.pd_type == "prefill" else 1) - 1)
        if self.start_npu not in batch.end or completion_npu not in batch.end:
            return prompt_t, gen_t, end_reqs

        self.logger.info("Batch #%d is done", batch.batch_id)

        for req in batch.requests:
            if req.status == RequestStatus.FINISHED:
                # With pp_size > 1 a request is legitimately in more than one
                # in-flight batch, so it can finish on an earlier batch while a
                # later one is still running. vLLM V1 skips exactly this in
                # ``update_from_output`` -- "the request is already finished.
                # This can happen if the request is aborted while the model is
                # executing it (e.g., in pipeline parallelism)" -- and drops that
                # batch's output whole, tokens included: they are past the
                # request's target, so nothing wants them.
                #
                # Without this the completion path below runs once per in-flight
                # batch: duplicate rows in the per-request CSV, req_cnt counted
                # twice, end_time and latency overwritten with the later batch's
                # clock, and a KeyError in cache_blocks, whose req_to_blocks
                # entry the first pass already freed.
                continue
            num_new = batch.scheduled_tokens[req.id]
            # num_computed_tokens was already advanced at schedule time.
            prefill_done_now = req.is_init and req.num_computed_tokens >= req.original_input

            if prefill_done_now:
                # TTFT is recorded exactly once. A resumed request has is_init
                # cleared, so it can never overwrite its own TTFT.
                req.is_init = False
                req.set_ttft(finish)
                prompt_t += num_new + req.prefix_cache_hit
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)
                if self.pd_type == "prefill":
                    # The prefill instance ran through lm_head and the sampler, so
                    # the first output token exists: advance the reached length or
                    # the decode instance receives a request with nothing left to
                    # schedule (num_tokens_reached == num_computed_tokens) and
                    # deadlocks. gen_t is deliberately left to the decode side,
                    # which is where this token has always been counted.
                    req.num_tokens_reached += 1
                    req.pd_kv_bytes = self.memory.pd_kv_bytes(req.original_input)
                    self.logger.info("Request #%d is prefill done, sent to decode instance", req.id)
                    self.kv.free(req)
                    self._retire(req)
                    end_reqs.append(req)
                    continue
            elif num_new > 1:
                # Chunk of a prefill, or a resumed request catching up.
                prompt_t += num_new
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)

            # A token is produced exactly when the request has caught up to the
            # length it had reached. A resumed request recomputing its history
            # has not, so it stays silent until it does.
            if req.num_computed_tokens >= req.num_tokens_reached:
                req.num_tokens_reached += 1
                gen_t += 1
                if not prefill_done_now:
                    req.add_itl(finish)
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)

            if req.num_tokens_reached >= req.output:
                self.logger.info("Request #%d is done", req.id)
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)
                self.kv.free(req)
                req.add_latency(finish)
                self._retire(req)
                self.done.append(req)
                end_reqs.append(req)

        del self.inflight[idx]
        return prompt_t, gen_t, end_reqs

    def _retire(self, req):
        req.status = RequestStatus.FINISHED
        try:
            self.running.remove(req)
        except ValueError:
            pass

    # ==================== queue management ====================

    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    def add_request(self, req, is_init=True):
        new_req = Request(*(req), is_init=is_init)
        # Arrival order, which phase B relies on to stop at the first request
        # that has not arrived yet. Dynamically released agentic sub-requests
        # arrive mid-run, hence insort rather than append.
        bisect.insort(self.waiting, new_req, key=lambda r: (r.arrival, r.id))
        return new_req

    def add_decode(self, req):
        """Take over a request whose prefill ran on another instance.

        The KV transfer itself is already charged: the prefill instance's trace
        carries a per-layer send to the paired decode NPU. So this only claims
        the blocks -- reporting no load bytes, or the transfer would be billed
        twice.
        """
        req.instance_id = self.instance_id
        req.decode_instance_id = self.instance_id
        req.status = RequestStatus.RUNNING
        hit_blocks, num_npu_hit, num_lower_hit = self.kv.get_computed_blocks(req)
        num_computed = req.num_computed_tokens
        if self.kv.allocate_slots(req, 1, hit_blocks, num_npu_hit, num_lower_hit) is None:
            raise RuntimeError(
                f"[Scheduler] [node_id={self.node_id},inst={self.instance_id}] decode "
                f"instance cannot admit request {req.id}: {req.num_tokens_reached} tokens "
                f"need more blocks than the pool has free "
                f"({self.kv.npu_pool.get_num_free_blocks()} of {self.kv.npu_pool.num_blocks})"
            )
        req.num_computed_tokens = num_computed
        self.kv.take_traffic()          # a P/D handoff is not a recall
        self.running.append(req)

    @property
    def accepts_new_requests(self):
        return self.admission_state == "ACTIVE"

    def set_admission_state(self, state):
        state = str(state).upper()
        if state not in {"INACTIVE", "WARMING", "ACTIVE", "DRAINING"}:
            raise ValueError(f"Unknown admission state {state!r}")
        self.admission_state = state

    def warm_prefix(self, input_tokens, input_hash_ids):
        """Seed full prompt blocks as unpinned NPU cache entries.

        This represents a completed warm transfer.  Lifecycle timing models
        when the transfer becomes available; the copies themselves let normal
        cache lookup and eviction account for its subsequent benefit.
        """
        if not self.enable_prefix_caching or not input_hash_ids:
            return 0
        request = Request(-1, self.model, int(input_tokens), int(input_tokens),
                          0, self.instance_id, list(input_hash_ids), [])
        hashes = request_block_hashes(request, self.kv.block_size)
        full_blocks = min(int(input_tokens) // self.kv.block_size, len(hashes))
        warmed = sum(1 for block_hash in hashes[:full_blocks]
                     if self.kv.npu_pool.cache_copy(block_hash))
        return warmed * self.kv.npu_pool.bytes_per_block

    def is_request_empty(self):
        return not self.waiting and not self.running and not self.inflight

    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        def _render(title: str, values, num_space=0):
            print_rule(f"[sim.tagline]{title}[/]")
            if not values:
                print_markup(f"No {title.split()[0]} data available")
                return
            mean = np.mean(values) / 1_000_000
            median = np.median(values) / 1_000_000
            p99 = np.percentile(values, 99) / 1_000_000
            label = title.split()[-1] if title != "Time to First Token" else "TTFT"
            # Map to the metric short-name used in the detail rows.
            short = {
                "Time to First Token": "TTFT",
                "Time per Output Token (excl. 1st token)": "TPOT",
                "Inter-token Latency": "ITL",
            }[title]
            spacing = " " * num_space
            print_markup(f"Mean {short} (ms){spacing}:                                                     {mean:.2f}")
            print_markup(f"Median {short} (ms){spacing}:                                                   {median:.2f}")
            print_markup(f"P99 {short} (ms){spacing}:                                                      {p99:.2f}")

        _render("Time to First Token", ttft_values)
        _render("Time per Output Token (excl. 1st token)", tpot_values)
        _render("Inter-token Latency", itl_values, num_space=1)

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        if not os.path.isabs(output_file):
            output_file = f'../{output_file}'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow(['instance id', 'request id', 'model', 'input', 'output', 
                                'arrival', 'end_time', 'latency', 
                                'queuing_delay', 'TTFT', 'TPOT', 'ITL',
                                'class_id', 'prefix_id', 'prefill_instance_id',
                                'decode_instance_id', 'npu_hit_tokens',
                                'storage_hit_tokens', 'pd_kv_bytes', 'affinity_version'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.model,
                    req.input,
                    req.output - req.input,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot,
                    req.itl,
                    req.class_id,
                    req.prefix_id,
                    req.prefill_instance_id,
                    req.decode_instance_id,
                    req.npu_cache_hit,
                    req.storage_cache_hit,
                    req.pd_kv_bytes,
                    req.affinity_version,
                ])


def main():
    pass

if __name__ == "__main__":
    main()

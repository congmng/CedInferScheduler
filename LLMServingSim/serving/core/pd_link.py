"""The P/D KV handoff as a link resource, not as a pair of NPU-side graphs.

Real evidence, measured 2026-09-16 on the small cluster (4090 / 3090a / 5090,
forced handoff, 300 requests of 1259-token prompts, 8 concurrent clients):

* the Prefill engine is *not* blocked by the push.  p5090's own prefill time is
  195 ms p50 -- that is the 1250-token compute -- while the simulator charged
  the whole 184 MB handoff to the Prefill NPU's timeline as exposed
  communication: 10.97 s of its 11.77 s, so a Prefill instance spent 93% of its
  life inside a KV send.
* the Decode engine is not blocked either.  d5090 held a 14.9 ms p50 TPOT while
  receiving all 300 handoffs, against a 15.3 ms local Decode step.
* what *is* serialised is the push: 300 x 184 MB over the run's 240 s is
  0.23 GB/s, and the deployment's measured handoff is 585 ms per 1000 prompt
  tokens (~0.25 GB/s).  The KV link, not either engine, is the resource.

So a handoff is modelled here the way the deployment behaves: the batch's bytes
occupy the producer's egress for ``bytes / bandwidth``, the requests become
decodable one link latency later, and both engines keep running in the
meantime.  Previously the same handoff was a SEND on a Prefill rank plus a RECV
on a *Decode* rank, and running that RECV on the Decode's own NPU serialised it
with that instance's decode steps -- measured 2026-09-16 with the cluster's own
placement replayed, the Decode's TPOT came out at 401.8 ms against 14.9 ms on
the cluster (27x), while every batch in its trace cost the correct 15.3 ms.
"""

import heapq


class PdHandoffLink:
    """A FIFO KV push per Prefill instance, with arrivals on a heap.

    ``bandwidth_gbps`` and ``latency_ns`` are functions of
    ``(producer_node, consumer_node)``: a same-node handoff pays the
    deployment's measured intra-node link (0.3 GB/s, 1 us), a cross-domain one
    pays the inter-node link.  Bandwidth is GB/s, i.e. bytes/ns, which is the
    unit the cluster configs already use.
    """

    def __init__(self, bandwidth_gbps, latency_ns, logger=None):
        self._bandwidth_gbps = bandwidth_gbps
        self._latency_ns = latency_ns
        self._logger = logger
        # producer instance -> the time its egress becomes free again.
        self._free_at = {}
        self._due = []          # heap of (due_ns, seq, payload)
        self._seq = 0
        self.handoffs = 0
        self.bytes_shipped = 0
        self.egress_wait_ns = 0.0

    # -- scheduling -------------------------------------------------------

    def enqueue(self, producer_instance, producer_node, consumer_node,
                num_bytes, payload, now_ns):
        """Reserve the producer's egress and return when the KV lands.

        ``payload`` is opaque to this class: the frontend gets it back when the
        transfer completes.
        """
        bandwidth = max(1e-9, float(self._bandwidth_gbps(producer_node, consumer_node)))
        latency = max(0, int(self._latency_ns(producer_node, consumer_node)))
        transmit_ns = int(round(max(0, num_bytes) / bandwidth))
        start_ns = max(int(now_ns), self._free_at.get(producer_instance, 0))
        due_ns = start_ns + transmit_ns + latency
        self._free_at[producer_instance] = start_ns + transmit_ns
        self._seq += 1
        heapq.heappush(self._due, (due_ns, self._seq, payload))
        self.handoffs += 1
        self.bytes_shipped += int(num_bytes)
        self.egress_wait_ns += max(0, start_ns - int(now_ns))
        if self._logger is not None:
            self._logger.debug(
                "KV handoff %.1f MB from instance %s: egress waited %.1f ms, "
                "lands at %.1f ms",
                num_bytes / 1e6, producer_instance,
                (start_ns - int(now_ns)) / 1e6, due_ns / 1e6)
        return due_ns

    def pop_due(self, now_ns):
        """Everything whose KV has landed by ``now_ns``, oldest first."""
        landed = []
        while self._due and self._due[0][0] <= now_ns:
            landed.append(heapq.heappop(self._due)[2])
        return landed

    def next_due(self):
        """When the next handoff lands, or ``None`` when none is in flight."""
        return self._due[0][0] if self._due else None

    def in_flight(self):
        return len(self._due)

    def wait_ns(self, producer_instance, producer_node, consumer_node,
                num_bytes, now_ns):
        """How long a push enqueued *now* would take to land, in ns.

        The router prices a candidate pair with this the way the real control
        loop prices its links (``disagg_router._pair_cost``: RTT plus the bytes
        already queued on that link over its bandwidth).
        """
        if num_bytes is None or num_bytes <= 0:
            return 0
        bandwidth = max(1e-9, float(self._bandwidth_gbps(producer_node, consumer_node)))
        latency = max(0, int(self._latency_ns(producer_node, consumer_node)))
        transmit_ns = int(round(max(0, num_bytes) / bandwidth))
        queued_ns = max(0, self._free_at.get(producer_instance, 0) - int(now_ns))
        return queued_ns + transmit_ns + latency

    def stats(self):
        return {
            "handoffs": self.handoffs,
            "bytes_shipped": self.bytes_shipped,
            "mean_egress_wait_ms": (
                self.egress_wait_ns / self.handoffs / 1e6 if self.handoffs else 0.0),
            "in_flight": self.in_flight(),
        }

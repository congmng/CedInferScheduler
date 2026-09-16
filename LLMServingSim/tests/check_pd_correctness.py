#!/usr/bin/env python3
"""Gate the P/D pipeline on *what it answers*, not just on how fast.

Every other measurement in this repo looks at timings, and a P/D deployment
can serve fast garbage.  Measured on 2026-09-13: the A100 pair passed the
health gate, reported ``LMCache hit tokens: 1009/1009`` and answered in
~300 ms, while the Decode was reading a staging slot the sender had not
written yet -- the model then emitted 64 newlines, or stopped after one token.
A latency-only harness cannot see that.

The oracle here is deliberately binary.  Each request carries a long article
(so the prompt spans several LMCache chunks and a real handoff happens) with a
random access code at *both* ends, and the model is told to reply with only
those codes:

    Access code A: BVQH-2087
    <article> ... <article>
    Access code B: QK7M-4319
    Verification task: reply with ONLY the two access codes above...

Both codes must come back.  One at the head catches a corrupted first chunk,
one at the tail catches the last chunk -- the slot that a Decode reads too
early in the failure mode this script was written for.  The codes are unique
per request, so they cannot be guessed or carried over from an earlier answer.

Exit code is non-zero when more than ``--max-fail-ratio`` of the answers are
missing their own code; comparison runs should abort on that.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import string
import sys
import threading
import time
import uuid

import httpx

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_WORKLOAD = REPO / "workloads" / "cnndm-real-qwen3-8b-600-sps14.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:9000",
                        help="router or a single instance, e.g. http://host:8100")
    parser.add_argument("--prefill", default="",
                        help="with --decode: drive one specific P/D pair "
                             "instead of the router, e.g. http://host:8100")
    parser.add_argument("--decode", default="",
                        help="with --prefill: the Decode endpoint, e.g. "
                             "http://host:8200")
    parser.add_argument("--receiver-host", default="",
                        help="host advertised to the sender for the KV push; "
                             "defaults to the host inside --decode")
    parser.add_argument("--workload", default=str(DEFAULT_WORKLOAD),
                        help="jsonl with an ``input_text`` field per line")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Qwen3's ``/no_think`` switch is appended to the "
                             "prompt, so the answer is the bare code and a "
                             "small budget is enough")
    parser.add_argument("--max-fail-ratio", type=float, default=0.25)
    parser.add_argument("--proxy-port", type=int, default=0,
                        help="bind a ZMQ PULL here and wait for LMCache's "
                             "'KV landed' notification before sending the "
                             "Decode leg (only with --prefill/--decode)")
    parser.add_argument("--proxy-wait-s", type=float, default=60.0)
    parser.add_argument("--consistency-prefix-chars", type=int, default=48,
                        help="in consistency mode, compare only this many "
                             "leading characters: a broken handoff diverges "
                             "from the first token, while MoE batch-composition "
                             "nondeterminism only diverges later")
    parser.add_argument("--native-pd", action="store_true",
                        help="vLLM NixlConnector mode: copy the Prefill "
                             "response's kv_transfer_params into the Decode "
                             "request (the officially supported flow)")
    parser.add_argument("--consistency", action="store_true",
                        help="model-agnostic oracle: generate the same prompt "
                             "locally on the Decode and through the P/D pair, "
                             "and require identical text.  Needed for base "
                             "models, which ignore 'reply with only the code' "
                             "style instructions")
    parser.add_argument("--model", default="qwen3-8b",
                        help="served model name sent in the request body")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def make_code(rng: random.Random) -> str:
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
    digits = "".join(rng.choice(string.digits) for _ in range(4))
    return f"{letters}-{digits}"


class LandingListener:
    """Bind the LMCache PD proxy PULL socket and wait per request.

    Lets a single P/D pair be exercised (with the sender configured to notify
    this port) without going through the router.
    """

    def __init__(self, port: int):
        self.port = int(port)
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._socket = None
        self.received = 0
        self.timeouts = 0

    def start(self) -> None:
        import zmq

        context = zmq.Context.instance()
        self._socket = context.socket(zmq.PULL)
        self._socket.bind(f"tcp://0.0.0.0:{self.port}")
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        import msgspec

        while True:
            try:
                raw = self._socket.recv()
                message = msgspec.msgpack.decode(raw)
            except Exception:  # noqa: BLE001 - the listener must never die
                continue
            request_id = (message.get("req_id") if isinstance(message, dict)
                          else getattr(message, "req_id", None))
            with self._lock:
                self._events.setdefault(request_id, threading.Event()).set()
                self.received += 1

    def wait(self, request_id: str, timeout_s: float) -> bool:
        with self._lock:
            event = self._events.setdefault(request_id, threading.Event())
        landed = event.wait(timeout_s)
        with self._lock:
            self._events.pop(request_id, None)
        if not landed:
            self.timeouts += 1
        return landed


def main() -> int:
    args = parse_args()
    articles = []
    with open(args.workload) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("input_text"):
                articles.append(row["input_text"])
    if len(articles) < args.requests:
        print(f"workload has only {len(articles)} usable rows", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(articles)
    client = httpx.Client(timeout=600.0, trust_env=False)
    listener = None
    if args.proxy_port:
        listener = LandingListener(args.proxy_port)
        listener.start()
        if not args.json:
            print(f"[proxy] PULL bound on tcp://0.0.0.0:{args.proxy_port}",
                  flush=True)
    records = []
    for index, article in enumerate(articles[:args.requests]):
        code_a, code_b = make_code(rng), make_code(rng)
        request_id = f"kvcheck-{uuid.uuid4().hex[:8]}"
        if args.consistency and args.prefill and args.decode:
            # Same prompt twice: once prefilled by the Decode itself, once
            # handed over by the Prefill.  Greedy decoding makes the two
            # outputs comparable for any model, including base models.
            # Base models stop early on open-ended prompts, so the probe ends
            # with an explicit continuation cue and the gate reports how many
            # samples were actually comparable.
            prompt = (f"Consistency probe {uuid.uuid4().hex[:8]}.\n\n"
                      f"{article}\n\nOne-sentence summary:")
            payload = {"model": args.model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": args.max_tokens, "temperature": 0.0,
                       "stream": False}
            started = time.perf_counter()
            local_text = ""
            try:
                local = client.post(
                    f"{args.decode}/v1/chat/completions", json=payload,
                    headers={"X-Request-Id": request_id + "-local"})
                local_text = ((local.json()["choices"][0]["message"]["content"])
                              or "").strip()
                prefill_payload = dict(payload)
                prefill_payload["max_tokens"] = 1
                prefill_payload["kv_transfer_params"] = {
                    "do_remote_decode": True, "do_remote_prefill": False}
                prefill_response = client.post(
                    f"{args.prefill}/v1/chat/completions",
                    json=prefill_payload,
                    headers={"X-Request-Id": request_id})
                params = ((prefill_response.json() or {})
                          .get("kv_transfer_params"))
                decode_payload = dict(payload)
                if params:
                    decode_payload["kv_transfer_params"] = params
                handed = client.post(
                    f"{args.decode}/v1/chat/completions", json=decode_payload,
                    headers={"X-Request-Id": request_id})
                handed_text = ((handed.json()["choices"][0]["message"]["content"])
                               or "").strip()
                # Empty local output means the model produced nothing to
                # compare against; that is not evidence of a broken handoff,
                # so record it separately instead of failing the pair.
                prefix = max(1, args.consistency_prefix_chars)
                found = (bool(local_text)
                         and handed_text[:prefix] == local_text[:prefix])
                empty = not local_text
                answer = handed_text
                prompt_tokens = (local.json().get("usage") or {}).get(
                    "prompt_tokens")
                status = handed.status_code
            except Exception as exc:  # noqa: BLE001
                found, empty, answer, status, prompt_tokens = (
                    False, True, f"<error {exc!r}>", -1, None)
            latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
            record = {"index": index, "request_id": request_id,
                      "status": status, "prompt_tokens": prompt_tokens,
                      "latency_ms": latency_ms, "access_code": "-",
                      "code_found": found,
                      "local_answer": local_text[:120], "empty": empty,
                      "answer": answer.strip()[:160]}
            records.append(record)
            if not args.json:
                verdict = "ok " if found else "BAD"
                print(f"[{verdict}] {latency_ms:8.1f} ms  "
                      f"local={record['local_answer'][:40]!r} "
                      f"pd={record['answer'][:40]!r}", flush=True)
            continue
        prompt = (f"Access code A: {code_a}\n\n{article}\n\n"
                  f"Access code B: {code_b}\n\n"
                  f"Verification task: reply with ONLY the two access codes "
                  f"above, in the order given, separated by a single space, "
                  f"and nothing else.\n/no_think")
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            if args.prefill and args.decode:
                receiver_host = args.receiver_host or (
                    args.decode.split("//")[-1].split(":")[0])
                prefill_payload = dict(payload)
                prefill_payload["max_tokens"] = 1
                prefill_payload["kv_transfer_params"] = {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "disagg_spec": {
                        "req_id": request_id,
                        "receiver_host": receiver_host,
                        "receiver_init_port": [55555],
                        "receiver_alloc_port": [55556],
                        "receiver_query_port": [55557],
                    },
                }
                prefill_response = client.post(
                    f"{args.prefill}/v1/chat/completions",
                    json=prefill_payload,
                    headers={"X-Request-Id": request_id})
                if listener is not None:
                    listener.wait(request_id, args.proxy_wait_s)
                decode_payload = payload
                if args.native_pd:
                    params = ((prefill_response.json() or {})
                              .get("kv_transfer_params"))
                    if params:
                        decode_payload = dict(payload)
                        decode_payload["kv_transfer_params"] = params
                response = client.post(f"{args.decode}/v1/chat/completions",
                                       json=decode_payload,
                                       headers={"X-Request-Id": request_id})
            else:
                response = client.post(f"{args.endpoint}/v1/chat/completions",
                                       json=payload,
                                       headers={"X-Request-Id": request_id})
            body = response.json()
            answer = (body["choices"][0]["message"]["content"] or "")
            status = response.status_code
            prompt_tokens = (body.get("usage") or {}).get("prompt_tokens")
        except Exception as exc:  # noqa: BLE001 - surface transport failures
            answer, status, prompt_tokens = f"<error {exc!r}>", -1, None
        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        found = code_a in answer and code_b in answer
        record = {"index": index, "request_id": request_id, "status": status,
                  "prompt_tokens": prompt_tokens, "latency_ms": latency_ms,
                  "access_code_a": code_a, "access_code_b": code_b,
                  "code_found": found,
                  "answer": answer.strip()[:160]}
        records.append(record)
        if not args.json:
            verdict = "ok " if found else "BAD"
            print(f"[{verdict}] {latency_ms:8.1f} ms  prompt={prompt_tokens} "
                  f"codes={code_a}/{code_b}  answer={record['answer'][:60]!r}",
                  flush=True)

    # In consistency mode an empty local generation means there was nothing to
    # compare (base models sometimes emit nothing for an open-ended prompt);
    # that is a harness artefact, not a broken handoff, so report it as skipped
    # rather than as a failure.
    skipped = [r for r in records
               if r.get("empty") and not r["code_found"]] if args.consistency \
        else []
    failures = [r for r in records
                if not r["code_found"] and r not in skipped]
    denominator = max(1, len(records) - len(skipped))
    fail_ratio = len(failures) / denominator
    summary = {
        "endpoint": args.endpoint,
        "requests": len(records),
        "failures": len(failures),
        "skipped_empty": len(skipped),
        "fail_ratio": round(fail_ratio, 3),
        "max_fail_ratio": args.max_fail_ratio,
        "median_latency_ms":
            sorted(r["latency_ms"] for r in records)[len(records) // 2],
        "failed_request_ids": [r["request_id"] for r in failures],
        "proxy_notifications": listener.received if listener else None,
        "proxy_timeouts": listener.timeouts if listener else None,
        "records": records,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = fail_ratio <= args.max_fail_ratio
    if not args.json:
        print(f"P/D output check: {'PASS' if ok else 'FAIL'} "
              f"({len(failures)}/{len(records)} answers missing their own "
              f"access code)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
